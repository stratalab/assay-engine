"""The inference seam (E2.1, PRD §11): NL → candidate IR — propose → validate → execute.

Assay never calls a model provider directly (A-5): this package is the *only* place a
model is ever touched (engineering §4), and what it exports is a **seam** — a backend
interface plus validation no backend can bypass. A backend's whole job is to *propose*:
a candidate IR (task selection + input extraction + missing-flags), an ambiguity fork
(UX §5.3), or an out-of-scope refusal (UX §5.5). Every candidate is validated against
its task template's contract **before anything runs** (``validate_candidate``) — a
malformed or hallucinated IR is rejected, not executed. The model influences *which
template and inputs*, never the method, the facts, or the execution.

Adoption is staged like the store seam (E0.2): shipped today is
``DeterministicBackend`` — a rule-based reference backend (catalog-driven keyword
matching + dimension-keyed quantity extraction; no model, no network) that exercises
every honest state end to end. The binding to the embedded Strata inference layer
(llama.cpp serving a local Lithos model + optional provider routing) lands behind this
same interface when its SDK ships.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from functools import lru_cache
from string import Formatter
from typing import Any, Protocol

import pint
from pydantic import BaseModel, ConfigDict, Field

from assay import __version__
from assay.execute import declared_dimensionality, normalize_expression
from assay.ir import IR, Producer, Quantity
from assay.resolver import FactTable, builtin_tables
from assay.templates import FormulaMethod, SymbolicMethod, Template, expr_symbols

__all__ = [
    "Ambiguity",
    "Attribution",
    "CandidateIRError",
    "DeterministicBackend",
    "InferenceBackend",
    "OutOfScope",
    "Proposal",
    "ProposalError",
    "ProposedIR",
    "validate_candidate",
]


# The seam's name for ``assay.ir.Producer`` — one type, not a parallel one: every
# backend stamps its proposals' IRs with ``produced_by``, so the artifact names the
# NL→IR model (E2.7, engineering §11 Q1).
Attribution = Producer


class ProposedIR(BaseModel):
    """A candidate interpretation: an IR (not yet validated, never yet executed) and a
    one-line human ``reading`` of it — what the user confirms or corrects (UX §5.4)."""

    model_config = ConfigDict(extra="forbid")
    ir: IR
    reading: str
    attribution: Attribution


class Ambiguity(BaseModel):
    """More than one reading fits: the fork is surfaced, never silently chosen
    (UX §5.3). Each option is a complete proposal, ready to run once picked."""

    model_config = ConfigDict(extra="forbid")
    options: list[ProposedIR] = Field(min_length=2)
    attribution: Attribution


class OutOfScope(BaseModel):
    """No task template covers the question: refuse with the reason and what *is*
    covered — never guess (UX §5.5, A-12)."""

    model_config = ConfigDict(extra="forbid")
    reason: str
    covered: list[str]
    attribution: Attribution


Proposal = ProposedIR | Ambiguity | OutOfScope


class InferenceBackend(Protocol):
    """The seam every NL→IR producer stands behind (A-5). ``propose`` returns exactly
    one of the honest shapes; it never executes, resolves a fact, or invents an input.
    """

    attribution: Attribution

    def propose(self, question: str, catalog: Sequence[Template]) -> Proposal: ...


class CandidateIRError(ValueError):
    """The proposed IR violates its template's contract — rejected, not executed
    (PRD §11). ``reasons`` lists every violation at once (A-12)."""

    def __init__(self, reasons: Sequence[str]) -> None:
        self.reasons = list(reasons)
        super().__init__("candidate IR rejected: " + "; ".join(self.reasons))


class ProposalError(Exception):
    """The backend failed to produce a usable proposal (e.g. the model emitted
    malformed output) — stated plainly, never patched over with a guess (A-12)."""


@lru_cache(maxsize=1)
def _registry() -> pint.UnitRegistry[float]:
    return pint.UnitRegistry()


def validate_candidate(ir: IR, catalog: Sequence[Template]) -> Template:
    """The pre-execution gate of propose → validate → execute (A-5): check a candidate
    IR against its template's contract and return the template, or raise
    ``CandidateIRError`` listing every violation. Nothing a backend proposes reaches
    the executor without passing here — hallucinated tasks, undeclared or
    wrong-dimension inputs, and ungated expressions are rejected, not run.
    """
    templates = {template.id: template for template in catalog}
    template = templates.get(ir.task)
    if template is None:
        raise CandidateIRError(
            [f"no template {ir.task!r} is in the catalog — refusing to execute a guess"]
        )
    reasons: list[str] = []
    if ir.domain != template.domain:
        reasons.append(
            f"domain {ir.domain!r} contradicts template {template.id!r}"
            f" (domain {template.domain!r})"
        )
    declared = {inp.name: inp for inp in template.inputs}
    for field_name, names in (
        ("inputs", set(ir.inputs)),
        ("missing_inputs", set(ir.missing_inputs)),
        ("resolved", set(ir.resolved)),
    ):
        if unknown := sorted(names - set(declared)):
            reasons.append(
                f"{field_name} not declared by template {template.id!r}:"
                f" {', '.join(unknown)}"
            )
    if overlap := sorted(set(ir.inputs) & set(ir.missing_inputs)):
        reasons.append(f"inputs both supplied and flagged missing: {', '.join(overlap)}")
    required = {inp.name for inp in template.inputs if inp.required}
    accounted = set(ir.inputs) | set(ir.resolved) | set(ir.missing_inputs)
    if ir.solve_for is not None:  # a solve-for IR (E2.10): the target is the unknown
        if ir.solve_for not in {inp.name for inp in template.inputs}:
            reasons.append(f"solve_for names an undeclared input {ir.solve_for!r}")
        if ir.solve_for in set(ir.inputs) | set(ir.resolved):
            reasons.append(f"solve_for target {ir.solve_for!r} must not also be supplied")
        if ir.given_output is None:
            reasons.append("a solve_for IR needs given_output (the stated output value)")
        required = required - {ir.solve_for}
    elif ir.given_output is not None:
        reasons.append("given_output applies only to solve_for IRs")
    if unaccounted := sorted(required - accounted):
        reasons.append(
            f"required inputs neither supplied nor flagged missing: {', '.join(unaccounted)}"
            " — a candidate must account for every input; nothing is silently defaulted"
        )
    ureg = _registry()
    for name, supplied in sorted(ir.inputs.items()):
        if name not in declared:
            continue  # already reported as undeclared
        elements = supplied if isinstance(supplied, list) else [supplied]
        if declared[name].many != isinstance(supplied, list):
            reasons.append(
                f"input {name!r} is a {'list' if declared[name].many else 'scalar'}"
                " input — supplied shape mismatches"
            )
            continue
        expected = declared_dimensionality(declared[name].dimension, ureg)
        for quantity in elements:
            try:
                provided = ureg.Quantity(quantity.value, quantity.unit or "")
            except Exception:
                reasons.append(f"input {name!r}: unknown unit {quantity.unit!r}")
                break
            if provided.dimensionality != expected:
                reasons.append(
                    f"input {name!r} must have dimension {declared[name].dimension!r};"
                    f" got {quantity.unit!r}"
                )
                break
    if isinstance(template.method, SymbolicMethod):
        reasons += _symbolic_setup_reasons(template, ir.setup)
    if reasons:
        raise CandidateIRError(reasons)
    return template


# The setup key each symbolic operation carries its problem in — so the pre-execution
# gate (defense in depth; the executor gates everything again) checks the right one.
# ``equation`` (y'/y'' notation) and ``field`` skip the expr gate here — ``parse_ode``
# and the per-component parse do it at execution.
_SYMBOLIC_SETUP_KEY = {
    "solve": "expression", "integrate": "expression", "differentiate": "expression",
    "limit": "expression", "solve_inequality": "expression",
    "taylor_polynomial": "expression", "partial_derivative": "expression",
    "gradient": "expression", "directional_derivative": "expression",
    "integrate_multiple": "expression", "series_convergence": "term",
    "parametric_slope": "x_expression", "ode_solve": "equation",
    "divergence": "field", "curl": "curl",
}


def _symbolic_setup_reasons(template: Template, setup: Mapping[str, Any]) -> list[str]:
    assert isinstance(template.method, SymbolicMethod)
    operation = template.method.operation
    reasons: list[str] = []
    if operation in ("divergence", "curl"):
        field = setup.get("field")
        if not isinstance(field, list) or not field or not all(isinstance(c, str) for c in field):
            return [f"template {template.id!r} needs setup 'field' (component expressions)"]
        for component in field:
            try:
                expr_symbols(component)
            except ValueError as exc:
                reasons.append(f"setup field component rejected: {exc}")
        return reasons
    if operation == "parametric_slope":
        keys = ["x_expression", "y_expression"]
    else:
        keys = [_SYMBOLIC_SETUP_KEY.get(operation, "expression")]
    for key in keys:
        raw = setup.get(key)
        if not isinstance(raw, str) or not raw.strip():
            reasons.append(f"template {template.id!r} needs setup {key!r} (a non-empty string)")
            continue
        if key == "equation":  # y'/y'' notation — parse_ode gates it at execution
            continue
        for side in raw.split("="):
            try:
                expr_symbols(side.strip())  # the parse-only gate — before execution
            except ValueError as exc:
                reasons.append(f"setup {key} rejected: {exc}")
    return reasons


# --- the deterministic reference backend (the shipped stage of the seam) -------------

_SYNONYMS = {
    "antiderivative": "integrate",
    "integral": "integrate",
    "integrating": "integrate",
    "root": "solve",
    "roots": "solve",
    "solving": "solve",
    "derivative": "differentiate",
    "differentiating": "differentiate",
    "ohm": "resistor",
    "ohms": "resistor",
}

_VERB = re.compile(
    r"\b(?:solve|integrate|differentiate|antiderivative\s+of|integral\s+of"
    r"|derivative\s+of|roots\s+of)\b",
    re.IGNORECASE,
)

# The symbolic operations the deterministic backend can actually SERVE from prose: the
# ones whose problem is a single free expression, which ``_symbolic_setup`` extracts.
# Every other symbolic op needs structured setup the bag-of-tokens matcher can't build
# — a limit point, a variable list, a vector field, a derivative order, an ODE in
# primed notation — so those are reached through the structured API or the model-backed
# backend, never guessed from prose. Excluding them from the candidate pool also stops
# their generic family tokens (ode_solve's "solve", integrate_multiple's "integrate")
# from colliding with the routes that ARE served (round-9 tripwire).
_NL_SYMBOLIC_OPS = frozenset({"solve", "integrate", "differentiate"})

_TRAILING_VARIABLE = re.compile(r"\b(?:with\s+respect\s+to|for|in)\s+([A-Za-z]\w*)\s*$")

# "<number> <unit>", where the unit may carry one power (m^4) and one denominator
# (m/s, m/s**2) — the shapes questions actually use. A captured word that isn't a
# unit ("2 beams") fails Pint and is skipped.
_UNIT_TOKEN = r"[A-Za-z]+(?:(?:\^|\*\*)-?\d+)?"
_QUANTITY = re.compile(
    r"(?<![\w.])([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*"
    rf"({_UNIT_TOKEN}(?:/{_UNIT_TOKEN})?)\b"
)

# COUNT words that Pint happens to parse as ANGLES ("500 turns" → 500·2π rad — a
# wrong-by-2π answer that would still pass the dimension check, since radians are
# dimensionless). Never units for extraction: the input stays missing and the ask
# names it, rather than a silent ×2π. (Caught live on self_inductance.solenoid.)
_COUNT_WORDS = frozenset(
    {"turn", "turns", "rev", "revs", "revolution", "revolutions", "cycle", "cycles"}
)


# Carry no signal for task selection — stripped from both the question and the
# template-id token sets, so an id segment like "from_sample" can't match the "from"
# in "escape velocity from a planet".
_STOP_WORDS = frozenset(
    {"a", "an", "and", "at", "by", "for", "from", "in", "is", "of", "on", "the", "to",
     "what", "with"}
)


def _tokens(question: str) -> set[str]:
    raw = re.findall(r"[a-z0-9]+", question.lower())
    return {_SYNONYMS.get(token, token) for token in raw} - _STOP_WORDS


def _head_token(question: str) -> str | None:
    """The question's first meaningful word — in practice it names the requested
    quantity ("SPEED of a wave …", "PERIOD of a pendulum …"), so it earns a routing
    boost toward templates whose family or output carries it."""
    for raw in re.findall(r"[a-z]+", question.lower()):
        mapped: str = _SYNONYMS.get(raw, raw)
        if mapped not in _STOP_WORDS:
            return mapped
    return None


class DeterministicBackend:
    """The seam's reference implementation and test fake: rule-based NL→IR — catalog-
    driven keyword matching plus dimension-keyed quantity extraction. No model, no
    network; the same question always yields the same proposal.

    It exists to prove the seam and the honest states, and to keep ``assay ask``
    usable (and testable in CI) before the Strata SDK binding lands. Like every
    backend it only *extracts* — a value absent from the question is flagged missing,
    never invented (A-2).
    """

    attribution = Attribution(provider="assay", model="deterministic-rules/v0")

    def __init__(self, tables: Iterable[FactTable] | None = None) -> None:
        self._tables = {
            table.library: table
            for table in (builtin_tables() if tables is None else tables)
        }

    def propose(self, question: str, catalog: Sequence[Template]) -> Proposal:
        tokens = _tokens(question)
        head = _head_token(question)
        scored: list[tuple[int, Template]] = []
        for template in catalog:
            if (
                isinstance(template.method, SymbolicMethod)
                and template.method.operation not in _NL_SYMBOLIC_OPS
            ):
                continue  # setup isn't extractable from prose — not a matcher candidate
            segments = template.id.split(".")
            family = set(segments[0].split("_")) - _STOP_WORDS
            # a question token counts once: qualifier tokens that repeat a family
            # token carry no extra signal ("mach_angle.from_mach_number" must not
            # score "mach" twice and dodge the stray-keyword floor)
            qualifiers = {
                token for segment in segments[1:] for token in segment.split("_")
            } - _STOP_WORDS - family
            if not tokens & family:
                continue
            score = len(tokens & family) + len(tokens & qualifiers)
            output_tokens = set(template.output.name.split("_")) - _STOP_WORDS
            if head is not None and (head in family or head in output_tokens):
                # the head word names the requested quantity: "speed of a wave …"
                # asks for a speed, not for the wavelength that also appears later.
                score += 1
            scored.append((score, template))
        if not scored:
            return OutOfScope(
                reason=f"no task template matches: {question!r}",
                covered=sorted({template.domain for template in catalog}),
                attribution=self.attribution,
            )
        best = max(score for score, _ in scored)
        winners = [template for score, template in scored if score == best]
        options = [self._proposal(template, question, tokens) for template in winners]
        # Completeness: what the question actually specified (extracted expression /
        # bound inputs). Used twice below.
        completeness = [len(option.ir.setup) + len(option.ir.inputs) for option in options]
        fullest = max(completeness)
        if len(options) > 1:
            # Tie-break by completeness: prefer the reading the question actually
            # specifies over one whose problem spec would still be empty
            # ("integrate x^2" is the symbolic antiderivative, not an unspecified
            # definite integral). A genuine tie — same extraction either way — still
            # forks (UX §5.3).
            options = [
                option
                for option, count in zip(options, completeness, strict=True)
                if count == fullest
            ]
        if len(options) > 1:
            # Specificity tie-break: an UNQUALIFIED question prefers the unqualified
            # reading — "kinetic energy of a 2 kg mass" means the classical template,
            # not the relativistic one whose extra family token the question never
            # said. (Saying "relativistic" outscores the classical outright.)
            unmatched = [
                len(
                    (set(option.ir.task.split(".")[0].split("_")) - _STOP_WORDS)
                    - tokens
                )
                for option in options
            ]
            tightest = min(unmatched)
            options = [
                option
                for option, count in zip(options, unmatched, strict=True)
                if count == tightest
            ]
        by_id = {template.id: template for template in winners}
        if (
            best == 1
            and fullest == 0
            and all(
                isinstance(by_id[option.ir.task].method, FormulaMethod)
                for option in options
            )
        ):
            # One stray family token and nothing extractable is noise, not a reading:
            # a formula template with no bound inputs cannot be what "turbulent flow
            # over an airfoil" meant. Symbolic/solver templates stay exempt — their
            # problem rides in the prose ("minimize (x-2)^2 + 1 ...") and one verb is
            # a legitimate match.
            return OutOfScope(
                reason=f"no task template matches: {question!r}"
                " (a single stray keyword is not a reading)",
                covered=sorted({template.domain for template in catalog}),
                attribution=self.attribution,
            )
        if len(options) > 1:
            return Ambiguity(options=options, attribution=self.attribution)
        return options[0]

    def _proposal(self, template: Template, question: str, tokens: set[str]) -> ProposedIR:
        if isinstance(template.method, SymbolicMethod):
            setup = self._symbolic_setup(question)
        else:
            setup = self._material_setup(template, tokens)
        inputs, missing = self._extracted_inputs(template, question)
        ir = IR(
            assay_version=__version__,
            query=question,
            produced_by=self.attribution,
            domain=template.domain,
            task=template.id,
            setup=setup,
            inputs=inputs,
            missing_inputs=missing,
        )
        return ProposedIR(
            ir=ir, reading=template.description or template.id, attribution=self.attribution
        )

    @staticmethod
    def _symbolic_setup(question: str) -> dict[str, Any]:
        """The problem is the question minus the verb phrase, normalized — extraction,
        not interpretation; the safe-parse gate still judges the result."""
        match = _VERB.search(question)
        expression = question[match.end() :] if match else question
        expression = expression.strip().strip("?.!").strip()
        setup: dict[str, Any] = {}
        if trailing := _TRAILING_VARIABLE.search(expression):
            setup["variable"] = trailing.group(1)
            expression = expression[: trailing.start()].strip()
        setup["expression"] = normalize_expression(expression)
        return setup

    def _material_setup(self, template: Template, tokens: set[str]) -> dict[str, Any]:
        """Fill a resolve-hint placeholder (e.g. ``{material}``) from the curated
        tables' own vocabulary — mentioned in the question, matched to a table key,
        never free-typed."""
        setup: dict[str, Any] = {}
        for declared in template.inputs:
            if declared.resolve is None:
                continue
            table = self._tables.get(declared.resolve.library)
            if table is None:
                continue
            placeholders = [
                field for _, field, _, _ in Formatter().parse(declared.resolve.key) if field
            ]
            for placeholder in placeholders:
                if placeholder in setup:
                    continue
                ids = {key.rsplit(".", 1)[0] for key in table.facts}
                matched = sorted(name for name in ids if name.split(".")[0] in tokens)
                if len(matched) == 1:
                    setup[placeholder] = matched[0]
        return setup

    @staticmethod
    def _extracted_inputs(
        template: Template, question: str
    ) -> tuple[dict[str, Quantity], list[str]]:
        """Dimension-keyed extraction: each ``<number> <unit>`` in the question maps to
        the *one* declared input with that dimension — an ambiguous or absent quantity
        stays missing (flagged, never guessed)."""
        if isinstance(template.method, SymbolicMethod):
            return {}, []
        ureg = _registry()
        found: dict[Any, list[Quantity]] = {}
        for value, unit in _QUANTITY.findall(question):
            if unit.lower() in _COUNT_WORDS:
                continue  # a count, not an angle — asked for by name, never ×2π
            normalized = unit.replace("^", "**")
            try:
                dimensionality = ureg.Quantity(1.0, normalized).dimensionality
            except Exception:
                continue  # not a unit ("2 beams") — not an input
            found.setdefault(dimensionality, []).append(
                Quantity(value=float(value), unit=normalized)
            )
        by_dimension: dict[Any, list[str]] = {}
        for declared in template.inputs:
            dimensionality = declared_dimensionality(declared.dimension, ureg)
            by_dimension.setdefault(dimensionality, []).append(declared.name)
        inputs: dict[str, Quantity] = {}
        for dimensionality, quantities in found.items():
            names = by_dimension.get(dimensionality, [])
            if len(quantities) == 1 and len(names) == 1:
                inputs[names[0]] = quantities[0]
        missing = [
            declared.name
            for declared in template.inputs
            if declared.required and declared.name not in inputs
        ]
        return inputs, missing
