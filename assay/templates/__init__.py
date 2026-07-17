"""Task templates (PRD §7): the declarative contract + the shared validator (E1.1).

A template is hand-authored (or Chisel-emitted) **data** — a symbolic formula with
dimensioned inputs and outputs, assumptions, verification hooks, fixtures, and provenance.
The template owns the method and the plan (A-3); the generic executor (E1.2) runs it.

``validate_template()`` is the Chisel seam (A-15, chisel-alignment §10): standalone and
dependency-light — importing this module pulls pydantic + stdlib only — so Chisel imports
it and can never emit a template Assay would reject. Two independent tier axes travel in
``provenance`` (chisel-alignment §7): ``license_tier`` (*may we use the source?*) and
``status`` (*has the template proven itself?* — ``candidate`` → ``verified``, the A-14
gate; promotion itself is E2.2). ``TemplateRegistry`` serves only ``verified`` templates
by default.
"""

from __future__ import annotations

import ast
import json
import re
import string
from collections.abc import Mapping
from importlib import resources
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

__all__ = [
    "SAFE_CONSTANTS",
    "SAFE_FUNCTIONS",
    "Bounds",
    "CandidateTemplateError",
    "CasesMethod",
    "INEQUALITY_OPERATORS",
    "INTERVAL_NOTATION",
    "ORDER_REDUCERS",
    "PAIRED_REDUCERS",
    "REDUCERS",
    "ExpectedValue",
    "split_inequality",
    "FactRef",
    "Fixture",
    "FormulaMethod",
    "LicenseTier",
    "Provenance",
    "SolverMethod",
    "Step",
    "SymbolicMethod",
    "Template",
    "TemplateInput",
    "TemplateOutput",
    "TemplateRegistry",
    "TemplateStatus",
    "TemplateValidationError",
    "VerificationHooks",
    "expr_symbols",
    "golden_template",
    "chisel_fixture_attachments",
    "chisel_templates",
    "golden_templates",
    "coverage",
    "domain_placement",
    "series_term_symbols",
    "taxonomy",
    "validate_template",
]

# The PINNED Lithos/Chisel tier vocabulary (chisel-alignment §7 / round 2), vendored
# verbatim so three repos can't drift — do not extend or rename without cross-repo
# alignment. "unknown" exists so foreign records can be *named*, but the seam REFUSES
# it (an unknown tier is a bug by contract, not a default — validate_template fails
# closed; see _enforce_contract).
LicenseTier = Literal["open", "lawful", "restricted", "synthetic-verified", "unknown"]
TemplateStatus = Literal["candidate", "verified"]

# The one expression namespace, shared: the validator gates against it here, and the
# executor (E1.2, engineering §7) imports it and supplies the implementations — the two
# can never drift. Widening this set is a SECURITY CHANGE (sandboxing.md): each name
# was added deliberately (asin/acos/atan + log10 + abs: chisel round 4 §3/§4 — the RLC
# phase-angle family and the decibel/beat encodings; erf: E2.13 — the normal CDF for
# the statistics corpus, dimensionless-in/dimensionless-out like the rest).
SAFE_FUNCTIONS = frozenset(
    {"sqrt", "sin", "cos", "tan", "asin", "acos", "atan", "exp", "log", "log10", "abs", "erf"}
)
SAFE_CONSTANTS = frozenset({"pi"})
# Reducers (schema_version 2, E2.11; the statistics vocabulary, E2.13): consume a LIST
# input (`many: true`) — and only that. Structurally constrained at the gate: a
# reducer call takes exactly one bare input name. The sum-like reducers expand to
# plain whitelisted arithmetic over the list's elements (sum(R_i) → R_i__0 + R_i__1 +
# …; count/mean/sum_sq likewise), so evaluation gains no new machinery. The
# ORDER_REDUCERS (order statistics — no arithmetic expansion exists) are computed
# deterministically at input-binding time and bound as reserved `name__min`-style
# scalars, so the evaluation walk is again untouched.
ORDER_REDUCERS = frozenset({"min", "max", "median"})
REDUCERS = frozenset({"sum", "sum_inverse", "count", "mean", "sum_sq"}) | ORDER_REDUCERS
# Paired reducers (E2.13): element-wise over TWO equal-length list inputs —
# sum_product(x_i, y_i) → x_i__0*y_i__0 + … (the regression/correlation shape). The
# equal-length rule is enforced at binding time with a named refusal.
PAIRED_REDUCERS = frozenset({"sum_product"})

_NAME = r"^[A-Za-z_][A-Za-z0-9_]*$"

# The solve_inequality answer vocabulary (E2.13): canonical interval notation Assay
# renders and verifies itself (stable across SymPy versions, unlike sstr of Sets) —
# "(-oo, 5)", "[3, oo)", "(-oo, -2] U [2, oo)", "{0}", "empty".
_INTERVAL_BOUND = r"-?(?:oo|\d+(?:\.\d+)?(?:/\d+)?)"
_INTERVAL_PIECE = (
    rf"[\[(]{_INTERVAL_BOUND}, {_INTERVAL_BOUND}[\])]"
    rf"|\{{{_INTERVAL_BOUND}(?:, {_INTERVAL_BOUND})*\}}"
)
INTERVAL_NOTATION = rf"^(?:empty|(?:{_INTERVAL_PIECE})(?: U (?:{_INTERVAL_PIECE}))*)$"

# Relational operators a solve_inequality expression must contain exactly one of
# (checked longest-first so "<=" is not misread as "<").
INEQUALITY_OPERATORS = ("<=", ">=", "<", ">")


def _is_bound(value: Any) -> bool:
    """A limit point / integration bound: a real number, or the infinity strings."""
    if isinstance(value, int | float) and not isinstance(value, bool):
        return True
    return isinstance(value, str) and value.strip() in {"oo", "-oo", "inf", "-inf"}


def split_inequality(expression: str) -> tuple[str, str, str]:
    """Split a relational expression into ``(lhs, operator, rhs)`` — exactly one
    relational operator, both sides non-empty. Raises ``ValueError`` with the reason.
    Longest-first matching so ``<=`` is never misread as ``<`` followed by ``=``."""
    found = [op for op in ("<=", ">=") if op in expression]
    stripped = expression
    for op in found:
        stripped = stripped.replace(op, "\x00")
    found += [op for op in ("<", ">") if op in stripped]
    if len(found) != 1 or expression.count(found[0]) > 1:
        raise ValueError(
            "a solve_inequality expression needs exactly one relational operator"
            " (one of: <=, >=, <, >)"
        )
    lhs, _, rhs = expression.partition(found[0])
    if not lhs.strip() or not rhs.strip():
        raise ValueError("both sides of the inequality must be non-empty expressions")
    return lhs.strip(), found[0], rhs.strip()


def _check_inequality_expression(expression: str, where: str) -> list[str]:
    try:
        lhs, _op, rhs = split_inequality(expression)
    except ValueError as exc:
        return [f"{where}.setup expression: {exc}"]
    problems = []
    for side in (lhs, rhs):
        try:
            expr_symbols(side)
        except ValueError as exc:
            problems.append(f"{where}.setup expression: {exc}")
    return problems

_EXPR_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Constant,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd, ast.Load,
)
_DIMENSION_NODES = (
    ast.Expression, ast.BinOp, ast.Mult, ast.Div, ast.Pow, ast.Name, ast.Constant, ast.Load,
)


def expr_symbols(expr: str) -> set[str]:
    """Free symbols of a formula, via a parse-only stdlib ``ast`` walk (never evaluated).

    Accepts the numeric-expression grammar only — names, numbers, ``+ - * / **``, unary
    sign, and calls to the safe math functions. Anything else (attributes, subscripts,
    strings, comparisons, lambdas) is rejected with a reason, so a crafted "formula"
    fails at validation, long before the executor's safe parse (E1.2, engineering §7).
    """
    return _gated_symbols(expr, frozenset())


def series_term_symbols(expr: str) -> set[str]:
    """The series-term grammar (E2.16): the ordinary gate plus ``factorial(...)`` —
    a SCOPED extension for ``series_convergence`` terms only. The factorial stays
    symbolic (the ratio-test limit is symbolic manipulation, never a float
    evaluation), so the integer-semantics objection to a general widening does not
    apply here (sandboxing.md, the widening ledger)."""
    return _gated_symbols(expr, frozenset({"factorial"}))


def _gated_symbols(expr: str, extra_calls: frozenset[str]) -> set[str]:
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"not a valid expression: {exc.msg}") from None
    names: set[str] = set()
    # Function-position names are consumed as calls, not symbols — so a v1 input
    # legitimately named `count` (the shipped chemistry corpus) stays an ordinary
    # symbol; only `count(...)` is the reducer (E2.13 backward compatibility).
    function_positions = {
        id(node.func)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    for node in ast.walk(tree):
        if not isinstance(node, _EXPR_NODES):
            raise ValueError(f"disallowed construct {type(node).__name__!r} in expression")
        if isinstance(node, ast.Call):
            allowed_calls = SAFE_FUNCTIONS | REDUCERS | PAIRED_REDUCERS | extra_calls
            if not (isinstance(node.func, ast.Name) and node.func.id in allowed_calls):
                raise ValueError(
                    "only calls to the safe math functions are allowed: "
                    + ", ".join(sorted(allowed_calls))
                )
            if node.keywords:
                raise ValueError("keyword arguments are not allowed in expressions")
            if (
                isinstance(node.func, ast.Name)
                and node.func.id in extra_calls
                and len(node.args) != 1
            ):
                raise ValueError(f"{node.func.id}() takes exactly one argument")
            # structural rules: a reducer takes exactly one bare input name; a paired
            # reducer exactly two DISTINCT bare input names
            if isinstance(node.func, ast.Name) and node.func.id in REDUCERS:
                if len(node.args) != 1 or not isinstance(node.args[0], ast.Name):
                    raise ValueError(
                        f"{node.func.id}() takes exactly one input name (a list input)"
                    )
            elif (
                isinstance(node.func, ast.Name)
                and node.func.id in PAIRED_REDUCERS
                and (
                    len(node.args) != 2
                    or not all(isinstance(arg, ast.Name) for arg in node.args)
                    or node.args[0].id == node.args[1].id  # type: ignore[attr-defined]
                )
            ):
                raise ValueError(
                    f"{node.func.id}() takes exactly two distinct input names"
                    " (paired list inputs)"
                )
        elif isinstance(node, ast.Constant):
            if type(node.value) not in (int, float):
                raise ValueError(f"disallowed literal {node.value!r} in expression")
        elif isinstance(node, ast.Name) and id(node) not in function_positions:
            names.add(node.id)
    return names - SAFE_FUNCTIONS - SAFE_CONSTANTS


def _reducer_usage(expr: str) -> tuple[set[str], set[str]]:
    """Names used inside reducer calls vs outside them (E2.11): the contract says a
    list input appears ONLY as a reducer's argument, and reducers take only list
    inputs. Assumes ``expr`` already passed ``expr_symbols`` (reducer args are bare
    names by the gate's structural rule)."""
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return set(), set()
    inside: set[str] = set()
    function_positions: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            function_positions.add(id(node.func))
            if node.func.id in REDUCERS | PAIRED_REDUCERS:
                for arg in node.args:
                    if isinstance(arg, ast.Name):
                        inside.add(arg.id)
                        function_positions.add(id(arg))
    outside = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and id(node) not in function_positions
    }
    return inside, outside


# The attested dimension-token vocabulary (chisel round 4 §5): every token here is
# executor-proven (Pint resolves it; 500+ corpus fixtures ran through it). VENDORED so
# the seam stays dependency-light — extend by the same note-and-confirm process as the
# tier vocabulary. A token outside this set fails at VALIDATE time, not at first ask
# (the `magnetic_flux_density` lesson: teslas are `magnetic_field`).
_DIMENSION_TOKENS = frozenset(
    {
        "acceleration", "area", "capacitance", "charge", "current", "density",
        "dimensionless", "electric_potential", "energy", "force", "frequency",
        "inductance", "length", "luminosity", "magnetic_field", "mass", "power",
        "pressure", "resistance", "speed", "substance", "temperature", "time",
        "velocity", "viscosity", "volume",
    }
)


def _check_dimension(dimension: str) -> str | None:
    """Grammar + vocabulary check for a dimension — ``length``, ``length**4``,
    ``force/length**2``.

    The algebraic shape (names, ``* / **``, int exponents) is checked by ``ast``; the
    identifiers must come from the attested, executor-proven token set above — a typo
    dies here, at the seam, not at the first ask (round 4 §5). The validator stays
    dependency-light: the vocabulary is vendored data, not a Pint import.
    """
    try:
        tree = ast.parse(dimension, mode="eval")
    except SyntaxError:
        return f"invalid dimension {dimension!r}"
    for node in ast.walk(tree):
        if not isinstance(node, _DIMENSION_NODES):
            return f"invalid dimension {dimension!r} ({type(node).__name__!r} not allowed)"
        if isinstance(node, ast.Constant) and type(node.value) is not int:
            return f"invalid dimension {dimension!r} (only integer exponents)"
        if isinstance(node, ast.Name) and node.id not in _DIMENSION_TOKENS:
            return (
                f"unknown dimension token {node.id!r} — not in the attested vocabulary"
                " (round 4 §5; e.g. teslas are 'magnetic_field')"
            )
    return None


class FactRef(BaseModel):
    """Where an input resolves from (PRD §8): a trusted table + a key pattern.

    ``{placeholders}`` in ``key`` fill from the IR's ``setup`` (e.g.
    ``"{material}.E"`` + ``setup: {material: steel.structural}`` →
    ``steel.structural.E``). The *template* names the source, the *resolver* honors it,
    and the model never supplies the value (A-2) — template-owns-resolution, the same
    doctrine as template-owns-method (A-3).
    """

    model_config = ConfigDict(extra="forbid")
    library: str = Field(min_length=1)
    key: str = Field(min_length=1)

    @model_validator(mode="after")
    def _check_key_pattern(self) -> FactRef:
        try:
            parsed = list(string.Formatter().parse(self.key))
        except ValueError as exc:
            raise ValueError(f"malformed key pattern {self.key!r}: {exc}") from exc
        for _, field, spec, conversion in parsed:
            if field is None:
                continue
            if not field.isidentifier() or spec not in (None, "") or conversion is not None:
                raise ValueError(
                    f"key pattern {self.key!r}: placeholders must be plain setup names"
                    " (no positions, dots, conversions, or format specs)"
                )
        return self


class TemplateInput(BaseModel):
    """One declared input: a name (the canonical symbol, PRD §6) and its dimension.

    ``resolve`` optionally names the input's trusted source; an input without it can
    only be user-supplied (the resolver will leave it missing, fail-clear — A-8).
    ``many: true`` (schema_version 2, E2.11) declares a LIST input — N same-dimension
    values consumed only through a whitelisted reducer (``sum``/``sum_inverse``), the
    series/parallel-circuit shape.
    """

    model_config = ConfigDict(extra="forbid")
    many: bool = False
    name: str = Field(pattern=_NAME)
    dimension: str
    required: bool = True
    resolve: FactRef | None = None


class TemplateOutput(BaseModel):
    """The declared output quantity and its expected dimension (the dimensional hook, §9)."""

    model_config = ConfigDict(extra="forbid")
    name: str = Field(pattern=_NAME)
    dimension: str


class Step(BaseModel):
    """One named assignment in a multi-step formula (E2.9): ``name = expr``, where
    ``expr`` may reference the declared inputs and any *earlier* step — never a later
    one. Each step passes the same safe-parse gate as a single-expression method."""

    model_config = ConfigDict(extra="forbid")
    name: str = Field(pattern=_NAME)
    expr: str


class FormulaMethod(BaseModel):
    """``kind: formula`` — symbolic math the generic executor evaluates (A-13): either
    a single ``expr``, or an ordered DAG of named ``steps`` (E2.9, round 2 — the
    combined-loading / Mohr's-circle shapes) whose **last step is the result**.
    Exactly one of the two forms; units ride through every intermediate either way."""

    model_config = ConfigDict(extra="forbid")
    kind: Literal["formula"]
    expr: str | None = None
    steps: list[Step] = []


class SolverMethod(BaseModel):
    """``kind: solver`` — an Assay-authored code binding for procedural cases (E3.5).

    ``binding`` is a *name* resolved against the executor's curated registry
    (``assay.execute.solvers``) — never an import path to execute. An unknown binding
    fails clear at execution, so it can never pass the fixture gate.
    """

    model_config = ConfigDict(extra="forbid")
    kind: Literal["solver"]
    binding: str = Field(min_length=1)


class SymbolicMethod(BaseModel):
    """``kind: symbolic`` — a curated symbolic operation the executor owns (E1.5).

    The problem (``expression``, optional ``variable``) arrives in the IR's/fixture's
    ``setup``, passes the same safe-parse gate as formulas, and produces dimensionless
    symbolic results (roots, an antiderivative, a derivative). Verification is built in
    per operation (substitution / derivative / difference-quotient), so the declarative
    hooks don't apply.

    E2.13 (the mathematics corpus): ``limit`` (setup ``point``, optional ``direction``
    — verified by a numeric approach sequence), ``integrate`` with optional setup
    ``limits`` ``[lo, hi]`` (definite/improper — ``"oo"``/``"-oo"`` allowed; verified
    by independent quadrature), and ``solve_inequality`` (a relational expression;
    the answer is canonical interval notation, verified by test points).
    """

    model_config = ConfigDict(extra="forbid")
    kind: Literal["symbolic"]
    operation: Literal[
        "solve", "integrate", "differentiate", "limit", "solve_inequality",
        # chisel round 8, finding 3: dy/dx = (dy/dt)/(dx/dt) for parametric curves
        # (polar tangents encode as x = r(θ)·cos(θ), y = r(θ)·sin(θ) — same op)
        "parametric_slope",
        # E2.16 (round-8 exhibits): the series operations — Taylor/Maclaurin
        # polynomial to order n (setup expression/center/order), and radius +
        # interval of convergence by the ratio test (setup term/center; the term may
        # use factorial(...) of the summation index — a SCOPED grammar extension,
        # symbolic-only, see sandboxing.md)
        "taylor_polynomial", "series_convergence",
    ]


class CasesMethod(BaseModel):
    """``kind: cases`` (schema_version 2, E2.11) — one physical quantity, one
    discriminator, per-case closed forms (the moment-of-inertia-table shape,
    chisel round 5 §2.2). The IR's/fixture's ``setup[discriminator]`` selects the
    case — template-owns-vocabulary, the same doctrine as resolve hints. Each case
    expression passes the same safe gate; case-dependent inputs are declared
    ``required: false`` and the *selected case* defines what it needs."""

    model_config = ConfigDict(extra="forbid")
    kind: Literal["cases"]
    discriminator: str = Field(pattern=_NAME)
    cases: dict[str, str]  # case name -> gated expression


Method = Annotated[
    FormulaMethod | SolverMethod | SymbolicMethod | CasesMethod,
    Field(discriminator="kind"),
]

# A fixture's expectation: a numeric quantity ``[value, unit]`` (formula templates), a
# real root set ``[2, 3]`` (symbolic solve), or an expression string (symbolic results).
ExpectedValue = tuple[float, str] | list[float] | str


class Bounds(BaseModel):
    """A plausibility range for the output (PRD §9) — rejects the absurd before reporting."""

    model_config = ConfigDict(extra="forbid")
    min: float | None = None
    max: float | None = None
    unit: str = ""

    @model_validator(mode="after")
    def _check_range(self) -> Bounds:
        if self.min is None and self.max is None:
            raise ValueError("bounds must set min, max, or both")
        if self.min is not None and self.max is not None and self.min >= self.max:
            raise ValueError("bounds min must be < max")
        return self


class VerificationHooks(BaseModel):
    """Template-declared verification (PRD §7, §9); consumed by the verify stage (E1.4)."""

    model_config = ConfigDict(extra="forbid")
    bounds: Bounds | None = None
    cross_method: str | None = None  # an independent expression to agree with, within tol


class Fixture(BaseModel):
    """A worked example with a known answer — the correctness proof (the fixture gate).

    Formula templates: ``inputs`` holds compact ``[value, unit]`` pairs (the Chisel
    emission form). Symbolic templates: the problem lives in ``setup`` (``expression``,
    optional ``variable``) and ``inputs`` stays empty. ``expect`` is keyed by the
    declared output name; ``tol`` is **relative** (E2.9, round 2 — matching the
    task-bank semantics, default 1e-6, with a 1e-12 absolute floor for zero-expected
    values); symbolic strings are compared by algebraic equivalence instead.
    """

    model_config = ConfigDict(extra="forbid")
    setup: dict[str, Any] = {}
    # scalar inputs: one [value, unit] pair; list inputs (many, schema_version 2):
    # a list of [value, unit] pairs
    inputs: dict[str, tuple[float, str] | list[tuple[float, str]]] = {}
    expect: dict[str, ExpectedValue]
    tol: float = Field(default=1e-6, gt=0)
    # Solve-for fixtures (E2.10): recover `solve_for` (a declared input) from the
    # other inputs plus the stated `output`; `expect` is keyed by the TARGET, and
    # `tol` carries the recovered input's print precision (tolerance provenance,
    # round 5 §1). Plain fixtures leave both unset.
    solve_for: str | None = None
    output: tuple[float, str] | None = None


class Provenance(BaseModel):
    """Where the template came from + the two independent tiers (chisel-alignment §7).

    Defaults fail closed: an undeclared license defaults to ``unknown``, and the seam
    *refuses* ``unknown`` outright (round 2: a bug by contract, not a servable state) —
    so a record that never declared its tier cannot validate. Every template starts as
    ``candidate``; only the fixture gate (E2.2) flips it to ``verified``.
    """

    model_config = ConfigDict(extra="forbid")
    source: str = Field(min_length=1)
    license_tier: LicenseTier = "unknown"
    status: TemplateStatus = "candidate"


class Template(BaseModel):
    """The declarative task template (PRD §7.1) — the unit of curated domain knowledge.

    The template owns the method and the plan (A-3); the model only ever maps a question
    to ``task`` + ``inputs`` + ``missing_inputs``. ``schema_version`` is the template
    schema's own version (engineering §10) — a validator honors exactly the versions it
    knows, rather than mis-reading a future one.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1, 2] = 1
    id: str = Field(pattern=r"^[a-z0-9_]+(\.[a-z0-9_]+)*$")
    domain: str = Field(pattern=r"^[a-z0-9_]+$")
    description: str = ""
    inputs: list[TemplateInput] = []  # empty for symbolic templates (problem is in setup)
    method: Method
    output: TemplateOutput
    # Extra outputs (E2.13, schema_version 2): a steps-DAG template may declare named
    # intermediate steps as additional outputs (slope AND intercept; mean AND std) —
    # each dimension-checked and reported alongside the primary. The last step stays
    # the primary output's value; extras name EARLIER steps.
    extra_outputs: list[TemplateOutput] = []
    assumptions: list[str] = []
    execution_plan: list[str] = []
    verification: VerificationHooks = Field(default_factory=VerificationHooks)
    fixtures: list[Fixture] = Field(min_length=1)
    provenance: Provenance

    @model_validator(mode="after")
    def _enforce_contract(self) -> Template:
        problems: list[str] = []
        if self.provenance.license_tier == "unknown":
            problems.append(
                "provenance.license_tier 'unknown' is refused at the seam — declare the"
                " source's tier (an unknown tier is a bug by contract, not a default;"
                " chisel-alignment round 2)"
            )
        names = [inp.name for inp in self.inputs]
        if duplicates := sorted({n for n in names if names.count(n) > 1}):
            problems.append(f"duplicate input names: {', '.join(duplicates)}")
        input_names = set(names)
        if self.output.name in input_names:
            problems.append(f"output {self.output.name!r} collides with an input name")
        # v1 keeps its E2.11 reserved set exactly (the seam pin: shipped verbatim
        # corpora legitimately use names like `count` as scalar inputs); v2 reserves
        # the full reducer vocabulary so emitted expressions stay unambiguous to
        # human readers.
        namespace = SAFE_FUNCTIONS | SAFE_CONSTANTS | frozenset({"sum", "sum_inverse"})
        if self.schema_version >= 2:
            namespace |= REDUCERS | PAIRED_REDUCERS
        if shadowed := sorted(input_names & namespace):
            problems.append(
                f"input names shadow the safe expression namespace: {', '.join(shadowed)}"
            )
        if self.schema_version == 1:  # v1 semantics stay frozen for the seam pin
            if any(inp.many for inp in self.inputs):
                problems.append("list inputs (many: true) require schema_version 2")
            if isinstance(self.method, CasesMethod):
                problems.append("kind: cases requires schema_version 2")
            if self.extra_outputs:
                problems.append("extra_outputs require schema_version 2")
        if any("__" in name for name in input_names):
            problems.append("input names may not contain '__' (reserved for expansion)")

        for inp in self.inputs:
            if (problem := _check_dimension(inp.dimension)) is not None:
                problems.append(f"input {inp.name!r}: {problem}")
            if inp.many and inp.resolve is not None:
                problems.append(
                    f"input {inp.name!r}: a list input cannot carry a resolve hint"
                    " (curated facts are scalars)"
                )
        if (problem := _check_dimension(self.output.dimension)) is not None:
            problems.append(f"output {self.output.name!r}: {problem}")
        problems += self._extra_outputs_contract(input_names)

        expect_keys = {self.output.name} | {out.name for out in self.extra_outputs}
        if (
            isinstance(self.method, SymbolicMethod)
            and self.method.operation == "series_convergence"
        ):
            expect_keys |= {"radius"}  # the operation's second result label (E2.16)
        for index, fixture in enumerate(self.fixtures):
            if fixture.solve_for is not None:
                if set(fixture.expect) != {fixture.solve_for}:
                    problems.append(
                        f"fixtures[{index}].expect must be keyed by exactly the"
                        f" solve_for target {fixture.solve_for!r}"
                    )
            elif self.output.name not in fixture.expect or not set(
                fixture.expect
            ) <= expect_keys:
                problems.append(
                    f"fixtures[{index}].expect must include the declared output"
                    f" {self.output.name!r} and only declared outputs"
                    f" ({', '.join(sorted(expect_keys))})"
                )

        if isinstance(self.method, SymbolicMethod):
            problems += self._symbolic_contract()
        else:
            problems += self._quantitative_contract(input_names)
        if problems:
            raise ValueError("; ".join(problems))
        return self

    def _quantitative_contract(self, input_names: set[str]) -> list[str]:
        """The formula/solver rules: gated expressions over declared inputs (plus, for
        multi-step methods, earlier step names), and fixtures that cover everything
        the method needs."""
        problems: list[str] = []
        used: set[str] = set()  # inputs the expressions actually reference
        expressions: list[tuple[str, str, set[str]]] = []  # (where, expr, allowed names)
        if isinstance(self.method, FormulaMethod):
            if (self.method.expr is None) == (not self.method.steps):
                problems.append(
                    "method needs exactly one of expr or steps (E2.9: a single"
                    " expression, or an ordered DAG of named assignments)"
                )
            if self.method.expr is not None:
                expressions.append(("method.expr", self.method.expr, input_names))
            seen_steps: set[str] = set()
            for index, step in enumerate(self.method.steps):
                where = f"method.steps[{index}] ({step.name})"
                if step.name in seen_steps:
                    problems.append(f"{where}: duplicate step name")
                if step.name in input_names:
                    problems.append(f"{where}: step name shadows an input")
                if step.name in SAFE_FUNCTIONS | SAFE_CONSTANTS:
                    problems.append(f"{where}: step name shadows the safe namespace")
                expressions.append((where, step.expr, input_names | seen_steps))
                seen_steps.add(step.name)
        case_symbols: dict[str, set[str]] = {}
        if isinstance(self.method, CasesMethod):  # schema_version 2 (E2.11)
            if len(self.method.cases) < 2:
                problems.append("kind: cases needs at least two cases (else use a formula)")
            if self.method.discriminator in input_names:
                problems.append("the cases discriminator must not collide with an input")
            for case_name, case_expr in self.method.cases.items():
                if not case_name.isidentifier():
                    problems.append(f"case name {case_name!r} must be an identifier")
                where = f"method.cases[{case_name}]"
                expressions.append((where, case_expr, input_names))
                try:
                    case_symbols[case_name] = expr_symbols(case_expr) & input_names
                except ValueError:
                    case_symbols[case_name] = set()  # the gate loop reports the reason
        if self.verification.cross_method is not None:
            expressions.append(
                ("verification.cross_method", self.verification.cross_method, input_names)
            )
        many_names = {inp.name for inp in self.inputs if inp.many}
        for where, expr, allowed in expressions:
            try:
                symbols = expr_symbols(expr)
            except ValueError as exc:
                problems.append(f"{where}: {exc}")
                continue
            if unknown := sorted(symbols - allowed):
                problems.append(
                    f"{where} references undeclared names: {', '.join(unknown)}"
                    " (inputs and earlier steps only)"
                )
            inside, outside = _reducer_usage(expr)
            if stray := sorted((symbols & many_names) & outside):
                problems.append(
                    f"{where}: list inputs may appear only inside a reducer:"
                    f" {', '.join(stray)}"
                )
            if scalar := sorted(inside - many_names):
                problems.append(
                    f"{where}: reducers apply only to list inputs (many: true):"
                    f" {', '.join(scalar)}"
                )
            used |= symbols & input_names

        required = {inp.name for inp in self.inputs if inp.required}
        for index, fixture in enumerate(self.fixtures):
            provided = set(fixture.inputs)
            if unknown := sorted(provided - input_names):
                problems.append(
                    f"fixtures[{index}] provides undeclared inputs: {', '.join(unknown)}"
                )
            for provided_name, provided_value in fixture.inputs.items():
                if provided_name not in input_names:
                    continue
                is_list = isinstance(provided_value, list)
                if provided_name in many_names and not is_list:
                    problems.append(
                        f"fixtures[{index}]: {provided_name!r} is a list input —"
                        " supply a list of [value, unit] pairs"
                    )
                if provided_name not in many_names and is_list:
                    problems.append(
                        f"fixtures[{index}]: {provided_name!r} is a scalar input —"
                        " supply one [value, unit] pair"
                    )
            if isinstance(self.method, CasesMethod):
                selected = fixture.setup.get(self.method.discriminator)
                if not isinstance(selected, str) or selected not in self.method.cases:
                    problems.append(
                        f"fixtures[{index}].setup must select a case via"
                        f" {self.method.discriminator!r} (one of:"
                        f" {', '.join(sorted(self.method.cases))})"
                    )
                    continue
                if missing := sorted(case_symbols.get(selected, set()) - provided):
                    problems.append(
                        f"fixtures[{index}] is missing inputs for case {selected!r}:"
                        f" {', '.join(missing)}"
                    )
            needed = required | used
            if isinstance(self.method, CasesMethod):
                needed = set()  # per-case coverage was checked above
            if fixture.solve_for is not None:  # a solve-for fixture (E2.10)
                if not isinstance(self.method, FormulaMethod):
                    problems.append(
                        f"fixtures[{index}]: solve_for applies to formula templates only"
                    )
                if fixture.solve_for in many_names:
                    problems.append(
                        f"fixtures[{index}]: solve_for cannot target a list input"
                    )
                if fixture.solve_for not in input_names:
                    problems.append(
                        f"fixtures[{index}].solve_for names an undeclared input"
                        f" {fixture.solve_for!r}"
                    )
                if fixture.solve_for in provided:
                    problems.append(
                        f"fixtures[{index}]: the solve_for target must not also be supplied"
                    )
                if fixture.output is None:
                    problems.append(
                        f"fixtures[{index}]: a solve-for fixture needs the stated"
                        " output as [value, unit]"
                    )
                needed = needed - {fixture.solve_for}
            elif fixture.output is not None:
                problems.append(
                    f"fixtures[{index}]: 'output' applies only to solve-for fixtures"
                )
            if missing := sorted(needed - provided):
                problems.append(f"fixtures[{index}] is missing inputs: {', '.join(missing)}")
            for value in fixture.expect.values():
                if not isinstance(value, tuple):
                    problems.append(
                        f"fixtures[{index}].expect must be a [value, unit] pair for a"
                        f" {self.method.kind} template"
                    )
        return problems

    def _extra_outputs_contract(self, input_names: set[str]) -> list[str]:
        """Extra outputs (E2.13): steps-DAG formula templates only — each extra output
        names an EARLIER step (the last step stays the primary output's value)."""
        if not self.extra_outputs:
            return []
        problems: list[str] = []
        if not (isinstance(self.method, FormulaMethod) and self.method.steps):
            problems.append(
                "extra_outputs apply only to formula templates with steps"
                " (each extra output names a step)"
            )
            return problems
        step_names = [step.name for step in self.method.steps]
        seen: set[str] = set()
        for out in self.extra_outputs:
            if (problem := _check_dimension(out.dimension)) is not None:
                problems.append(f"extra output {out.name!r}: {problem}")
            if out.name == self.output.name or out.name in input_names:
                problems.append(
                    f"extra output {out.name!r} collides with the primary output"
                    " or an input"
                )
            if out.name in seen:
                problems.append(f"duplicate extra output {out.name!r}")
            seen.add(out.name)
            if out.name not in step_names[:-1]:
                problems.append(
                    f"extra output {out.name!r} must name an earlier step"
                    " (the last step is the primary output)"
                )
        return problems

    def _symbolic_contract(self) -> list[str]:
        """The symbolic-operation rules (E1.5): the problem lives in ``setup``, results
        are dimensionless, and the built-in checks replace the declarative hooks."""
        problems: list[str] = []
        if self.inputs:
            problems.append(
                "symbolic templates take no dimensioned inputs (the problem lives in setup)"
            )
        if self.output.dimension not in ("1", "dimensionless"):
            problems.append("a symbolic template's output must be dimensionless")
        if self.verification.bounds is not None or self.verification.cross_method is not None:
            problems.append(
                "verification hooks don't apply to symbolic operations"
                " (substitution/derivative checks are built in)"
            )
        assert isinstance(self.method, SymbolicMethod)
        operation = self.method.operation
        for index, fixture in enumerate(self.fixtures):
            if fixture.inputs:
                problems.append(f"fixtures[{index}]: symbolic fixtures take no inputs")
            if fixture.solve_for is not None:
                problems.append(
                    f"fixtures[{index}]: solve_for applies to formula templates only"
                )
            if operation == "series_convergence":  # a term, no 'expression' (E2.16)
                term = fixture.setup.get("term")
                if not isinstance(term, str) or not term.strip():
                    problems.append(
                        f"fixtures[{index}].setup needs 'term' (the summand as a"
                        " function of the index and the variable)"
                    )
                else:
                    try:
                        series_term_symbols(term)
                    except ValueError as exc:
                        problems.append(f"fixtures[{index}].setup term: {exc}")
                center = fixture.setup.get("center")
                if center is not None and not (
                    isinstance(center, int | float) and not isinstance(center, bool)
                ):
                    problems.append(f"fixtures[{index}].setup 'center' must be a number")
                for key, value in fixture.expect.items():
                    if key == self.output.name:  # the interval, canonical notation
                        if not isinstance(value, str) or not re.match(
                            INTERVAL_NOTATION, value
                        ):
                            problems.append(
                                f"fixtures[{index}].expect[{key}]: expected canonical"
                                " interval notation (e.g. '(-1, 1)', '[1, 3)',"
                                " '(-oo, oo)', '{0}')"
                            )
                    elif key == "radius":  # a number, or "oo"
                        numeric = isinstance(value, list) and len(value) == 1
                        infinite = isinstance(value, str) and value.strip() == "oo"
                        if not (numeric or infinite):
                            problems.append(
                                f"fixtures[{index}].expect[radius]: expected [R]"
                                " or 'oo'"
                            )
                continue
            if operation == "taylor_polynomial":  # 'order' + optional 'center' (E2.16)
                order = fixture.setup.get("order")
                if not isinstance(order, int) or isinstance(order, bool) or order < 0:
                    problems.append(
                        f"fixtures[{index}].setup needs 'order' (an integer >= 0)"
                    )
                center = fixture.setup.get("center")
                if center is not None and not (
                    isinstance(center, int | float) and not isinstance(center, bool)
                ):
                    problems.append(f"fixtures[{index}].setup 'center' must be a number")
                # the expression itself falls through to the standard checks below
            if operation == "parametric_slope":  # two expressions, no 'expression'
                for key in ("x_expression", "y_expression"):
                    raw_value = fixture.setup.get(key)
                    if not isinstance(raw_value, str) or not raw_value.strip():
                        problems.append(
                            f"fixtures[{index}].setup needs {key!r} (a non-empty string)"
                        )
                        continue
                    try:
                        expr_symbols(raw_value)
                    except ValueError as exc:
                        problems.append(f"fixtures[{index}].setup {key}: {exc}")
                point = fixture.setup.get("point")
                if point is not None and not (
                    isinstance(point, int | float) and not isinstance(point, bool)
                ):
                    problems.append(
                        f"fixtures[{index}].setup 'point' must be a number"
                        " (omit it for the symbolic slope)"
                    )
                variable = fixture.setup.get("variable")
                if variable is not None and (
                    not isinstance(variable, str) or not variable.isidentifier()
                ):
                    problems.append(
                        f"fixtures[{index}].setup 'variable' must be a simple name"
                    )
                for value in fixture.expect.values():
                    if isinstance(value, tuple):
                        problems.append(
                            f"fixtures[{index}].expect: a [value, unit] pair is for"
                            " formula templates"
                        )
                continue
            expression = fixture.setup.get("expression")
            if not isinstance(expression, str) or not expression.strip():
                problems.append(
                    f"fixtures[{index}].setup needs 'expression' (a non-empty string)"
                )
            elif operation == "solve_inequality":
                problems += _check_inequality_expression(expression, f"fixtures[{index}]")
            else:
                sides = [side.strip() for side in expression.split("=")]
                if len(sides) > 2 or any(not side for side in sides):
                    problems.append(f"fixtures[{index}]: malformed equation {expression!r}")
                else:
                    for side in sides:
                        try:
                            expr_symbols(side)
                        except ValueError as exc:
                            problems.append(f"fixtures[{index}].setup expression: {exc}")
            variable = fixture.setup.get("variable")
            if variable is not None and (
                not isinstance(variable, str) or not variable.isidentifier()
            ):
                problems.append(f"fixtures[{index}].setup 'variable' must be a simple name")
            if operation == "limit":
                point = fixture.setup.get("point")
                if not _is_bound(point):
                    problems.append(
                        f"fixtures[{index}].setup needs 'point' — a number or"
                        " 'oo'/'-oo' (the limit operation, E2.13)"
                    )
                direction = fixture.setup.get("direction")
                if direction is not None and direction not in ("+", "-"):
                    problems.append(
                        f"fixtures[{index}].setup 'direction' must be '+' or '-'"
                    )
            if operation == "integrate":
                limits = fixture.setup.get("limits")
                if limits is not None and (
                    not isinstance(limits, list | tuple)
                    or len(limits) != 2
                    or not all(_is_bound(bound) for bound in limits)
                ):
                    problems.append(
                        f"fixtures[{index}].setup 'limits' must be [lo, hi] — numbers"
                        " or 'oo'/'-oo' (definite/improper integration, E2.13)"
                    )
            for value in fixture.expect.values():
                if isinstance(value, tuple):
                    problems.append(
                        f"fixtures[{index}].expect: a [value, unit] pair is for formula"
                        " templates; expect a root list or an expression string"
                    )
                elif isinstance(value, str):
                    if operation == "solve_inequality":
                        if not re.match(INTERVAL_NOTATION, value):
                            problems.append(
                                f"fixtures[{index}].expect: {value!r} is not canonical"
                                " interval notation (e.g. '(-oo, 5)', '[3, oo)',"
                                " '(-oo, -2] U [2, oo)', '{0}', 'empty')"
                            )
                    else:
                        try:
                            expr_symbols(value)
                        except ValueError as exc:
                            problems.append(f"fixtures[{index}].expect: {exc}")
        return problems


class TemplateValidationError(ValueError):
    """A template failed validation; the message lists every reason (the seam's answer)."""


def validate_template(record: Mapping[str, Any]) -> Template:
    """Validate ``record`` against the template contract — the function Chisel imports (A-15).

    Returns the validated ``Template``, or raises ``TemplateValidationError`` naming every
    violation. Chisel calls this before emitting, so it can never emit a template Assay
    would reject (chisel-alignment §10).
    """
    try:
        return Template.model_validate(record)
    except ValidationError as exc:
        reasons = "\n".join(
            f"  - {'.'.join(str(part) for part in error['loc']) or '<template>'}: {error['msg']}"
            for error in exc.errors()
        )
        raise TemplateValidationError(
            f"template {record.get('id', '<no id>')!r} is invalid:\n{reasons}"
        ) from exc


def golden_template() -> Template:
    """The golden beam template (chisel-alignment §10) — the canonical worked example,
    shipped in the package and mirrored byte-identical on the Chisel side."""
    text = (
        resources.files("assay.templates")
        / "golden"
        / "beam_deflection.simply_supported.center_point.json"
    ).read_text(encoding="utf-8")
    record: dict[str, Any] = json.loads(text)
    return validate_template(record)


def golden_templates() -> tuple[Template, ...]:
    """All shipped golden templates (E1.5: solve_equation, integrate, beam_deflection) —
    their fixtures green under the generic executor is the v0 correctness proof."""
    directory = resources.files("assay.templates") / "golden"
    return tuple(
        validate_template(json.loads(entry.read_text(encoding="utf-8")))
        for entry in sorted(directory.iterdir(), key=lambda entry: entry.name)
        if entry.name.endswith(".json")
    )


def _chisel_lines() -> list[dict[str, Any]]:
    directory = resources.files("assay.templates") / "chisel"
    lines: list[dict[str, Any]] = []
    for entry in sorted(directory.iterdir(), key=lambda entry: entry.name):
        if not entry.name.endswith(".jsonl"):
            continue
        for raw in entry.read_text(encoding="utf-8").splitlines():
            if raw.strip():
                lines.append(json.loads(raw))
    return lines


def chisel_templates() -> tuple[Template, ...]:
    """The Chisel-emitted batches shipped with Assay (E2.3, staged: today's batches are
    hand-extracted and Assay-reviewed; the emitter takes over the same files later).

    Each ``.jsonl`` under ``templates/chisel/`` is an emission envelope kept
    **verbatim** (provenance intact). ``template`` lines validate through the A-15
    seam; ``fixture-attachment`` lines targeting templates *in this corpus* merge here
    (more independent proof per template — the promotion-floor path). Everything is a
    ``candidate``: shipping grants presence, and the fixture gate still stands between
    every one of these and serving (A-14).
    """
    templates: dict[str, Template] = {}
    for line in _chisel_lines():
        if line.get("kind") == "template":
            template = validate_template(line["record"])
            if template.id in templates:
                raise TemplateValidationError(
                    f"duplicate template id {template.id!r} across chisel batches"
                )
            templates[template.id] = template
    for line in _chisel_lines():
        if line.get("kind") == "fixture-attachment":
            target = templates.get(str(line.get("template_id")))
            if target is None:
                continue  # targets a golden or plugin: merged by full_catalog (E2.4)
            merged = target.model_copy(
                update={
                    "fixtures": [
                        *target.fixtures,
                        *(Fixture.model_validate(f) for f in line["fixtures"]),
                    ]
                }
            )
            templates[target.id] = validate_template(merged.model_dump(mode="json"))
    return tuple(templates.values())


def chisel_fixture_attachments() -> dict[str, tuple[Fixture, ...]]:
    """Fixture attachments whose target lives *outside* the chisel corpus — independent
    proof landing on shipped goldens (or plugins). ``full_catalog`` merges them; each
    merged template revalidates and its fixtures still gate serving.

    Attachments whose target does not exist anywhere yet are **retained and re-tried
    on every load** by construction (batches ship verbatim and are re-read), and are
    surfaced — never silently dropped — by ``assay domains`` (the round-3 §1 ruling)."""
    corpus = {line["record"]["id"] for line in _chisel_lines() if line.get("kind") == "template"}
    attachments: dict[str, list[Fixture]] = {}
    for line in _chisel_lines():
        if line.get("kind") != "fixture-attachment":
            continue
        target = str(line.get("template_id"))
        if target in corpus:
            continue  # merged inside chisel_templates()
        attachments.setdefault(target, []).extend(
            Fixture.model_validate(f) for f in line["fixtures"]
        )
    return {target: tuple(fixtures) for target, fixtures in attachments.items()}


def taxonomy() -> dict[str, Any]:
    """The catalog taxonomy (subject → topic → domains): curated Assay-side data over
    the flat ``domain`` strings — the batches themselves are untouched. A domain not
    placed here cannot ship (the lockstep test); extensions are deliberate curation,
    the same posture as the resolver tables."""
    text = (resources.files("assay.templates") / "taxonomy.json").read_text(encoding="utf-8")
    result: dict[str, Any] = json.loads(text)
    return result


def coverage() -> dict[str, Any]:
    """The coverage map (E2.14): subject → field → topic with pending / in-progress /
    complete status — the PLANNING counterpart of ``taxonomy()``. Growth is targeted:
    a topic names its intended source (and, where blocked, the engine gate) before
    any extraction starts; this is not pretraining-style bulk ingestion. The lockstep
    test keeps the map honest against the shipped catalog."""
    text = (resources.files("assay.templates") / "coverage.json").read_text(encoding="utf-8")
    result: dict[str, Any] = json.loads(text)
    return result


def domain_placement() -> dict[str, tuple[str, str]]:
    """Reverse lookup: ``domain -> (subject, topic)``. Raises if any domain is placed
    twice — one home per domain."""
    placement: dict[str, tuple[str, str]] = {}
    for subject, subject_node in taxonomy()["subjects"].items():
        for topic, topic_node in subject_node["topics"].items():
            for domain in topic_node["domains"]:
                if domain in placement:
                    raise TemplateValidationError(
                        f"taxonomy places domain {domain!r} twice"
                        f" ({placement[domain]} and {(subject, topic)})"
                    )
                placement[domain] = (subject, topic)
    return placement


class CandidateTemplateError(LookupError):
    """A ``candidate`` template was requested without opting in (A-14: refuse by default)."""


class TemplateRegistry:
    """The template registry (A-9): id-keyed, serving only ``verified`` by default (A-14).

    Registration takes an already-validated ``Template``; discovery/packaging is the
    plugin SDK (E2.4), and candidate → verified promotion is the fixture gate (E2.2).
    """

    def __init__(self) -> None:
        self._by_id: dict[str, Template] = {}

    def register(self, template: Template) -> None:
        if template.id in self._by_id:
            raise ValueError(f"template {template.id!r} is already registered")
        self._by_id[template.id] = template

    def get(self, template_id: str, *, allow_candidate: bool = False) -> Template:
        try:
            template = self._by_id[template_id]
        except KeyError:
            raise KeyError(f"unknown template {template_id!r}") from None
        if template.provenance.status != "verified" and not allow_candidate:
            raise CandidateTemplateError(
                f"template {template_id!r} is a candidate (unverified) and does not serve"
                " by default; pass allow_candidate=True to opt in (A-14)"
            )
        return template

    def ids(self, *, status: TemplateStatus | None = None) -> list[str]:
        return sorted(
            template.id
            for template in self._by_id.values()
            if status is None or template.provenance.status == status
        )

    def __contains__(self, template_id: object) -> bool:
        return template_id in self._by_id

    def __len__(self) -> int:
        return len(self._by_id)
