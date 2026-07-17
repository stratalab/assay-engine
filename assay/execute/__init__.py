"""The generic executor (E1.2, PRD §7.1): run any declarative template, safely (A-13).

One domain-agnostic path for every ``kind: formula`` template: **safe parse** —
``parse_expr`` over a restricted transformation set and a locked namespace, never
``sympify``/``eval`` on a raw string (engineering §7), behind the shared ``ast`` gate
from the template validator — then **unit-bound evaluation** (inputs become Pint
quantities and the arithmetic happens *in* the unit system, so a wrong formula usually
cannot even evaluate), then a **dimension check** of the result against the template's
declared output. A failed check raises; the executor never returns a value it cannot
stand behind (PRD §9 posture; the full verify stage is E1.4).

``kind: symbolic`` templates (E1.5/E2.5) run curated symbolic operations — solve,
integrate, differentiate —
on a ``setup`` problem (``expression`` + optional ``variable``) that passes the *same*
gate and locked parse; results are dimensionless symbolic values (roots, an
antiderivative), printed deterministically and sorted.

Nothing executes except a validated IR (A-1, ``execute_ir``) or a validated template's
own fixtures (``run_fixtures`` — the proof the promotion gate E2.2 trusts). ``kind:
solver`` templates (E3.5) dispatch to the curated binding registry in ``solvers.py`` —
names looked up in a whitelist, never import paths. Pure and deterministic
(engineering §4, NFR-1): no wall-clock, no RNG, no network.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping
from functools import lru_cache
from typing import Any

import pint
import sympy
from pydantic import BaseModel, ConfigDict, Field
from sympy.core.function import AppliedUndef
from sympy.parsing.sympy_parser import auto_number, auto_symbol, parse_expr

from assay.answer import TraceStep
from assay.ir import IR
from assay.templates import (
    ORDER_REDUCERS,
    PAIRED_REDUCERS,
    REDUCERS,
    SAFE_CONSTANTS,
    SAFE_FUNCTIONS,
    CasesMethod,
    ExpectedValue,
    FormulaMethod,
    SolverMethod,
    SymbolicMethod,
    Template,
    expr_symbols,
    split_inequality,
)

__all__ = [
    "DimensionError",
    "ExecutedValue",
    "ExecutionError",
    "ExecutionResult",
    "FixtureResult",
    "InputError",
    "MissingInputError",
    "UnsafeExpressionError",
    "declared_dimensionality",
    "execute_ir",
    "execute_template",
    "ir_input_pairs",
    "normalize_expression",
    "parse_bound",
    "parse_formula",
    "parse_problem",
    "render_solution_set",
    "run_fixtures",
    "symbolic_problem",
    "symbolically_equal",
]


class ExecutionError(Exception):
    """The executor refused or failed; the message says exactly why (A-12)."""


class UnsafeExpressionError(ExecutionError):
    """The formula fell outside the safe expression grammar — rejected, never run."""


class InputError(ExecutionError):
    """The provided inputs don't satisfy the template's input schema."""


class MissingInputError(InputError):
    """A required input is absent — fail-clear; it will not be fabricated (A-2, A-8)."""


class DimensionError(ExecutionError):
    """A dimension is unknown, or the result's dimension contradicts the declaration."""


class ExecutedValue(BaseModel):
    """One labelled value of an execution — numeric with a unit, or symbolic (a string).
    Mirrors the answer object's ``ResultValue`` (E0.3), which E1.6 assembles from it."""

    model_config = ConfigDict(extra="forbid")
    label: str
    value: float | str
    unit: str = ""


class ExecutionResult(BaseModel):
    """The executed output: one value for a formula (canonical base SI units,
    deterministic across runs), possibly several for a symbolic solve (the roots)."""

    model_config = ConfigDict(extra="forbid")
    output: str
    values: list[ExecutedValue] = Field(min_length=1)
    # The execution trace (E2.15): the literal record of what was evaluated, in
    # order — every DAG step with its computed base-unit value. Not a narration.
    trace: list[TraceStep] = []

    @property
    def value(self) -> float | str:
        """Single-value convenience; raises when the result is a value set (use
        ``values``)."""
        if len(self.values) != 1:
            raise ValueError(f"{self.output!r} has {len(self.values)} values; use .values")
        return self.values[0].value

    @property
    def unit(self) -> str:
        if len(self.values) != 1:
            raise ValueError(f"{self.output!r} has {len(self.values)} values; use .values")
        return self.values[0].unit


class FixtureResult(BaseModel):
    """One fixture's verdict — numeric ``computed`` is stated in the expected unit."""

    model_config = ConfigDict(extra="forbid")
    index: int
    ok: bool
    computed: list[float | str] | None = None
    expected: ExpectedValue
    tol: float
    detail: str = ""


@lru_cache(maxsize=1)
def _registry() -> pint.UnitRegistry[float]:
    return pint.UnitRegistry()


# Implementations for the shared safe-function namespace (assay.templates.SAFE_FUNCTIONS).
# ``sqrt`` needs no entry: SymPy canonicalizes it to ``Pow(x, 1/2)``, which propagates
# units. ``log10`` needs none either: it parses to ``log(x)/log(10)`` (the blessed
# decibel encoding, now native). ``abs`` is handled structurally in ``_evaluate`` —
# unlike the rest it takes a *dimensioned* argument (|Δf| is legitimate). Everything
# here requires a dimensionless argument (Pint enforces it); ``asin``/``acos`` raise
# on out-of-domain inputs and the executor fails clear (A-12).
_FUNCTIONS: dict[Any, Callable[[float], float]] = {
    sympy.sin: math.sin,
    sympy.cos: math.cos,
    sympy.tan: math.tan,
    sympy.asin: math.asin,
    sympy.acos: math.acos,
    sympy.atan: math.atan,
    sympy.exp: math.exp,
    sympy.log: math.log,
    sympy.erf: math.erf,  # the normal-CDF building block (E2.13, sandboxing.md)
}


def normalize_expression(text: str) -> str:
    """Deterministic input normalization (UX §5.1): ``^`` → ``**`` and implicit
    multiplication (``3x``, ``2(x+1)``, ``)(``) made explicit — scientific notation
    (``5e9``) is preserved. The result still passes the full safe-parse gate."""
    text = text.replace("^", "**")
    text = re.sub(r"(\d)(?![eE][-+]?\d)\s*(?=[A-Za-z_(])", r"\1*", text)
    text = re.sub(r"\)\s*(?=[A-Za-z_0-9(])", r")*", text)
    return text


def declared_dimensionality(dimension: str, ureg: pint.UnitRegistry[float]) -> Any:
    """Resolve a template dimension (``length``, ``force/length**2``) to Pint's base form.

    Template dimensions are bare names; Pint's are bracketed (``[length]``) — wrap each
    identifier and let Pint expand derived dimensions to base ones.
    """
    if dimension.strip() in {"1", "dimensionless"}:
        return ureg.Quantity(1).dimensionality
    bracketed = re.sub(r"[A-Za-z_][A-Za-z0-9_]*", lambda m: f"[{m.group(0)}]", dimension)
    try:
        return ureg.get_dimensionality(bracketed)
    except Exception as exc:
        raise DimensionError(f"unknown dimension {dimension!r}: {exc}") from exc


def parse_formula(expr: str, input_names: frozenset[str] | set[str]) -> Any:
    """Safe symbolic parse (engineering §7): the shared ``ast`` gate, then ``parse_expr``
    with a restricted transformation set and a locked namespace — never ``sympify``."""
    try:
        symbols = expr_symbols(expr)  # parse-only stdlib gate; rejects anything exotic
    except ValueError as exc:
        raise UnsafeExpressionError(f"formula rejected: {exc}") from exc
    if unknown := sorted(symbols - set(input_names)):
        raise UnsafeExpressionError(
            f"formula references undeclared inputs: {', '.join(unknown)}"
        )
    local_dict: dict[str, Any] = {name: sympy.Symbol(name) for name in input_names}
    local_dict |= {name: getattr(sympy, name, None) for name in SAFE_FUNCTIONS | SAFE_CONSTANTS}
    local_dict["abs"] = sympy.Abs  # sympy spells it Abs
    local_dict["log10"] = _log10  # sympy has no log10: parses to log(x)/log(10)
    # parse_expr resolves names through these dicts only. The global dict carries just
    # the token constructors its transformations emit — plus an explicitly EMPTY
    # __builtins__: eval() silently injects the real builtins into any globals dict
    # missing that key (engineering §7: locked namespace, no builtins).
    global_dict: dict[str, Any] = {
        "__builtins__": {},
        "Integer": sympy.Integer,
        "Float": sympy.Float,
        "Rational": sympy.Rational,
        "Symbol": sympy.Symbol,
        "Function": sympy.Function,
    }
    try:
        parsed = parse_expr(
            expr,
            local_dict=local_dict,
            global_dict=global_dict,
            transformations=(auto_symbol, auto_number),
        )
    except Exception as exc:
        raise UnsafeExpressionError(f"formula failed to parse: {exc}") from exc
    if not parsed.free_symbols <= {sympy.Symbol(name) for name in input_names}:
        raise UnsafeExpressionError("formula contains symbols outside the declared inputs")
    if parsed.atoms(AppliedUndef):
        raise UnsafeExpressionError("formula contains unknown functions")
    return parsed


def parse_problem(
    raw: str, variable: str | None = None, *, allow_equation: bool = True
) -> tuple[Any, Any]:
    """Gate + safely parse an expression (or ``lhs = rhs`` equation → ``lhs - rhs``)
    with univariate variable inference. Returns ``(expression, variable_symbol)``.
    The one path every user-supplied expression takes — solve, integrate, and the
    render primitives (E1.8) all sample/operate on what *this* returns."""
    sides = [side.strip() for side in raw.split("=")]
    if len(sides) > 2 or any(not side for side in sides):
        raise InputError(f"malformed equation {raw!r} — expected 'expr' or 'lhs = rhs'")
    if len(sides) == 2 and not allow_equation:
        raise InputError("this operation takes an expression, not an equation")
    names: set[str] = set()
    for side in sides:
        try:
            names |= expr_symbols(side)
        except ValueError as exc:
            raise UnsafeExpressionError(f"expression rejected: {exc}") from exc
    if variable is not None and (not isinstance(variable, str) or not variable.isidentifier()):
        raise InputError("setup 'variable' must be a simple name")
    if variable is None:
        if len(names) != 1:
            raise InputError(
                "cannot infer the variable"
                f" (free symbols: {', '.join(sorted(names)) or 'none'});"
                " specify setup 'variable'"
            )
        variable = next(iter(names))
    if extra := sorted(names - {variable}):
        raise InputError(
            f"only the variable {variable!r} may appear; unexpected symbols:"
            f" {', '.join(extra)}"
        )
    parsed = [parse_formula(side, {variable}) for side in sides]
    expression = parsed[0] - parsed[1] if len(parsed) == 2 else parsed[0]
    return expression, sympy.Symbol(variable)


def symbolic_problem(template: Template, setup: Mapping[str, Any]) -> tuple[Any, Any]:
    """Parse a symbolic task's problem from ``setup``, safely — the same gate + locked
    parse as formulas. Returns ``(expression, variable_symbol)``; an equation
    ``lhs = rhs`` becomes ``lhs - rhs``. Univariate only (v0)."""
    if not isinstance(template.method, SymbolicMethod):
        raise ExecutionError(f"template {template.id!r} is not a symbolic template")
    raw = setup.get("expression")
    if not isinstance(raw, str) or not raw.strip():
        raise InputError(
            f"template {template.id!r} needs setup 'expression' (a non-empty string)"
        )
    variable = setup.get("variable")
    if variable is not None and not isinstance(variable, str):
        raise InputError("setup 'variable' must be a simple name")
    allow_equation = template.method.operation == "solve"
    return parse_problem(raw, variable, allow_equation=allow_equation)


def parse_bound(value: Any) -> Any:
    """A limit point / integration bound from setup: a number, or ``"oo"``/``"-oo"``
    (``"inf"``/``"-inf"`` accepted) → the SymPy number. Fails clear otherwise."""
    if isinstance(value, int | float) and not isinstance(value, bool):
        return sympy.Float(value) if isinstance(value, float) else sympy.Integer(value)
    if isinstance(value, str) and value.strip() in {"oo", "inf"}:
        return sympy.oo
    if isinstance(value, str) and value.strip() in {"-oo", "-inf"}:
        return -sympy.oo
    raise InputError(f"expected a number or 'oo'/'-oo', got {value!r}")


def _render_interval_bound(value: Any) -> str:
    if value == sympy.oo:
        return "oo"
    if value == -sympy.oo:
        return "-oo"
    if isinstance(value, sympy.Rational):  # Integer included — exact stays exact
        return str(sympy.sstr(value))
    return format(float(value), "g")


def render_solution_set(solution: Any) -> str:
    """A SymPy real set in Assay's canonical interval notation (E2.13) — rendered by
    Assay, not ``sstr``, so the artifact's answer is stable across SymPy versions:
    ``(-oo, 5)``, ``[3, oo)``, ``(-oo, -2] U [2, oo)``, ``{0}``, ``empty``."""
    if solution is sympy.S.EmptySet or solution == sympy.S.EmptySet:
        return "empty"
    if isinstance(solution, sympy.FiniteSet):
        rendered = sorted(
            (float(v), _render_interval_bound(v)) for v in solution.args
        )
        return "{" + ", ".join(text for _key, text in rendered) + "}"
    if isinstance(solution, sympy.Interval):
        left = "(" if solution.left_open else "["
        right = ")" if solution.right_open else "]"
        return (
            f"{left}{_render_interval_bound(solution.start)},"
            f" {_render_interval_bound(solution.end)}{right}"
        )
    if isinstance(solution, sympy.Union):
        pieces = sorted(solution.args, key=lambda part: float(part.inf))
        return " U ".join(render_solution_set(piece) for piece in pieces)
    raise ExecutionError(
        "the solution set cannot be stated in interval notation — refusing to"
        " approximate it (A-12)"
    )


_RELATIONS: dict[str, Any] = {
    "<": sympy.Lt, "<=": sympy.Le, ">": sympy.Gt, ">=": sympy.Ge,
}


def _execute_inequality(template: Template, setup: Mapping[str, Any]) -> ExecutionResult:
    """``solve_inequality`` (E2.13): both sides pass the ordinary safe gate; the
    solution set is rendered in Assay's canonical interval notation."""
    raw = setup.get("expression")
    if not isinstance(raw, str) or not raw.strip():
        raise InputError(
            f"template {template.id!r} needs setup 'expression' (a non-empty string)"
        )
    try:
        lhs_text, operator, rhs_text = split_inequality(raw)
    except ValueError as exc:
        raise InputError(str(exc)) from None
    names: set[str] = set()
    for side in (lhs_text, rhs_text):
        try:
            names |= expr_symbols(side)
        except ValueError as exc:
            raise UnsafeExpressionError(f"expression rejected: {exc}") from exc
    variable = setup.get("variable")
    if variable is None:
        if len(names) != 1:
            raise InputError(
                "cannot infer the variable"
                f" (free symbols: {', '.join(sorted(names)) or 'none'});"
                " specify setup 'variable'"
            )
        variable = next(iter(names))
    if not isinstance(variable, str) or not variable.isidentifier():
        raise InputError("setup 'variable' must be a simple name")
    if extra := sorted(names - {variable}):
        raise InputError(
            f"only the variable {variable!r} may appear; unexpected symbols:"
            f" {', '.join(extra)}"
        )
    lhs = parse_formula(lhs_text, {variable})
    rhs = parse_formula(rhs_text, {variable})
    symbol = sympy.Symbol(variable)
    try:
        solution = sympy.solve_univariate_inequality(
            _RELATIONS[operator](lhs, rhs), symbol, relational=False
        )
    except (NotImplementedError, ValueError) as exc:
        raise ExecutionError(
            f"cannot solve this inequality symbolically: {exc} — refusing to guess (A-12)"
        ) from exc
    return ExecutionResult(
        output=template.output.name,
        values=[ExecutedValue(label=variable, value=render_solution_set(solution))],
    )


def _execute_symbolic(template: Template, setup: Mapping[str, Any]) -> ExecutionResult:
    """Run a curated symbolic operation (E1.5). Deterministic: solutions are sorted
    (numeric ascending, then symbolic lexicographic); printing uses SymPy's ``sstr``."""
    assert isinstance(template.method, SymbolicMethod)  # execute_template dispatches
    if template.method.operation == "solve_inequality":  # relational grammar (E2.13)
        return _execute_inequality(template, setup)
    expression, symbol = symbolic_problem(template, setup)
    if template.method.operation == "limit":  # E2.13
        if "point" not in setup:
            raise InputError("the limit operation needs setup 'point'")
        point = parse_bound(setup["point"])
        direction = setup.get("direction", "+")
        if direction not in ("+", "-"):
            raise InputError("setup 'direction' must be '+' or '-'")
        try:
            value = sympy.limit(expression, symbol, point, direction)
        except (NotImplementedError, ValueError) as exc:
            raise ExecutionError(f"cannot evaluate this limit: {exc}") from exc
        if value.has(sympy.Limit) or value is sympy.nan or value.has(sympy.zoo):
            raise ExecutionError(
                "the limit does not evaluate to a stated value — refusing to guess (A-12)"
            )
        result: float | str = (
            float(value)
            if isinstance(value, sympy.Rational | sympy.Float)
            else sympy.sstr(value)
        )
        return ExecutionResult(
            output=template.output.name,
            values=[ExecutedValue(label=template.output.name, value=result)],
        )
    if template.method.operation == "solve":
        solutions = sympy.solve(expression, symbol)
        if not solutions:
            raise ExecutionError(
                "the equation has no solutions (or none SymPy can find) — nothing to return"
            )
        numeric: list[float] = []
        symbolic: list[str] = []
        for solution in solutions:
            # Rational covers Integer; Float is the parsed-decimal case (e.g. 5e9).
            # Exact irrationals (sqrt(2)) stay symbolic — exactness is the brand.
            if isinstance(solution, sympy.Rational | sympy.Float):
                numeric.append(float(solution))
            else:
                symbolic.append(sympy.sstr(solution))
        values = [
            ExecutedValue(label=str(symbol), value=value)
            for value in [*sorted(numeric), *sorted(symbolic)]
        ]
        return ExecutionResult(output=template.output.name, values=values)
    if template.method.operation == "differentiate":
        derivative = sympy.diff(expression, symbol)
        return ExecutionResult(
            output=template.output.name,
            values=[ExecutedValue(label=template.output.name, value=sympy.sstr(derivative))],
        )
    limits = setup.get("limits")
    if limits is not None:  # definite/improper integration (E2.13)
        if not isinstance(limits, list | tuple) or len(limits) != 2:
            raise InputError("setup 'limits' must be [lo, hi]")
        low, high = parse_bound(limits[0]), parse_bound(limits[1])
        try:
            value = sympy.integrate(expression, (symbol, low, high))
        except (NotImplementedError, ValueError) as exc:
            raise ExecutionError(f"cannot integrate this symbolically: {exc}") from exc
        if value.has(sympy.Integral):
            raise ExecutionError(
                "no closed form for this definite integral — refusing to guess (A-12)"
            )
        if value is sympy.nan or value.has(sympy.zoo):
            raise ExecutionError(
                "the integral does not converge to a stated value — refusing (A-12)"
            )
        stated: float | str = (
            float(value)
            if isinstance(value, sympy.Rational | sympy.Float)
            else sympy.sstr(value)  # exact irrationals stay exact; "oo" states divergence
        )
        return ExecutionResult(
            output=template.output.name,
            values=[ExecutedValue(label=template.output.name, value=stated)],
        )
    antiderivative = sympy.integrate(expression, symbol)
    if antiderivative.has(sympy.Integral):
        raise ExecutionError(
            "no closed-form antiderivative found — refusing to guess (A-12)"
        )
    return ExecutionResult(
        output=template.output.name,
        values=[ExecutedValue(label=template.output.name, value=sympy.sstr(antiderivative))],
    )


def symbolically_equal(left: str, right: str) -> bool:
    """Algebraic equivalence of two gated expression strings: simplify(a - b) == 0."""
    allowed = expr_symbols(left) | expr_symbols(right)
    difference = parse_formula(left, allowed) - parse_formula(right, allowed)
    return bool(sympy.simplify(difference) == 0)


def _log10(argument: Any) -> Any:
    """The parse-time expansion of the safe ``log10`` name: ``log(x, 10)`` — SymPy
    canonicalizes it to ``log(x)/log(10)``, which the evaluation walk already handles."""
    return sympy.log(argument, 10)


def _dimensionless_magnitude(quantity: Any, function_name: str) -> float:
    try:
        return float(quantity.to("dimensionless").magnitude)
    except pint.DimensionalityError as exc:
        raise DimensionError(
            f"{function_name}() requires a dimensionless argument; got {quantity.units}"
        ) from exc


def _evaluate(node: Any, bindings: Mapping[str, Any], ureg: pint.UnitRegistry[float]) -> Any:
    """Evaluate a parsed formula over Pint quantities by an explicit tree walk.

    No codegen, no ``eval`` — each SymPy node type is handled structurally, and anything
    outside the whitelist is rejected. Units ride through the arithmetic, so the result
    carries its dimension.
    """
    if isinstance(node, sympy.Symbol):
        if node.name not in bindings:
            raise MissingInputError(
                f"input {node.name!r} is referenced by the formula but was not provided"
            )
        return bindings[node.name]
    if not node.args and node.is_number:  # Integer / Float / Rational / pi
        return ureg.Quantity(float(node), "")
    if isinstance(node, sympy.Add):
        values = [_evaluate(arg, bindings, ureg) for arg in node.args]
        try:
            total = values[0]
            for value in values[1:]:
                total = total + value
        except pint.DimensionalityError as exc:
            raise DimensionError(f"cannot add quantities of different dimensions: {exc}") from exc
        return total
    if isinstance(node, sympy.Mul):
        product = ureg.Quantity(1.0, "")
        for arg in node.args:
            product = product * _evaluate(arg, bindings, ureg)
        return product
    if isinstance(node, sympy.Pow):
        exponent = node.exp
        if exponent.args or not exponent.is_number:
            raise UnsafeExpressionError("exponents must be numeric")
        return _evaluate(node.base, bindings, ureg) ** float(exponent)
    if node.func is sympy.Abs and len(node.args) == 1:
        # |x| keeps its dimension — the one safe function with a dimensioned argument
        return abs(_evaluate(node.args[0], bindings, ureg))
    if isinstance(node, sympy.Function) and node.func in _FUNCTIONS:
        if len(node.args) != 1:
            raise UnsafeExpressionError(f"{node.func}() takes exactly one argument")
        argument = _evaluate(node.args[0], bindings, ureg)
        magnitude = _dimensionless_magnitude(argument, str(node.func))
        return ureg.Quantity(_FUNCTIONS[node.func](magnitude), "")
    raise UnsafeExpressionError(f"cannot evaluate {type(node).__name__!r} node")


def _expand_reducers(expr: str, counts: Mapping[str, int]) -> str:
    """Rewrite reducer calls into plain whitelisted arithmetic (E2.11):
    ``sum(R_i)`` → ``(R_i__0 + R_i__1 + …)``, ``sum_inverse(R_i)`` → the reciprocal
    sum — a deterministic ``ast`` rewrite, so the unit-carrying evaluation walk needs
    no new machinery. Element names use the reserved ``__`` suffix."""
    import ast as _ast

    try:
        tree = _ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        # This rewrite runs BEFORE the safe parse gate; an unparseable "formula" must
        # still surface as a rejection, never a raw SyntaxError (defense in depth —
        # caught by the Windows CI matrix, where the hostile-payload test's tmp_path
        # backslashes make the payload itself unparseable).
        raise UnsafeExpressionError(f"formula rejected: {exc.msg}") from None

    class _Rewriter(_ast.NodeTransformer):
        def visit_Call(self, node: _ast.Call) -> _ast.AST:  # noqa: N802
            self.generic_visit(node)
            if not isinstance(node.func, _ast.Name):
                return node
            reducer = node.func.id
            if reducer in PAIRED_REDUCERS and len(node.args) == 2:  # sum_product (E2.13)
                first, second = node.args
                if not (isinstance(first, _ast.Name) and isinstance(second, _ast.Name)):
                    return node
                n, m = counts.get(first.id, 0), counts.get(second.id, 0)
                if n < 1 or m < 1:
                    raise InputError(
                        f"paired list inputs {first.id!r} and {second.id!r} each need"
                        " at least one element"
                    )
                if n != m:
                    raise InputError(
                        f"paired list inputs must have the same length:"
                        f" {first.id!r} has {n} elements, {second.id!r} has {m}"
                    )
                rewritten = " + ".join(
                    f"{first.id}__{k} * {second.id}__{k}" for k in range(n)
                )
                return _ast.parse(f"({rewritten})", mode="eval").body
            if reducer in REDUCERS and node.args and isinstance(node.args[0], _ast.Name):
                name = node.args[0].id
                count = counts.get(name, 0)
                if count < 1:
                    raise InputError(f"list input {name!r} needs at least one element")
                if reducer in ORDER_REDUCERS:  # bound at input-binding time (E2.13)
                    return _ast.Name(id=f"{name}__{reducer}", ctx=_ast.Load())
                elements = [f"{name}__{k}" for k in range(count)]
                if reducer == "sum":
                    rewritten = " + ".join(elements)
                elif reducer == "sum_inverse":
                    rewritten = " + ".join(f"1 / {element}" for element in elements)
                elif reducer == "count":
                    rewritten = str(count)
                elif reducer == "mean":
                    rewritten = f"({' + '.join(elements)}) / {count}"
                else:  # sum_sq
                    rewritten = " + ".join(f"{element}**2" for element in elements)
                return _ast.parse(f"({rewritten})", mode="eval").body
            return node

    return _ast.unparse(_Rewriter().visit(tree))


def _bind_inputs(
    template: Template,
    inputs: Mapping[str, tuple[float, str] | list[tuple[float, str]]],
    ureg: pint.UnitRegistry[float],
    *,
    enforce_required: bool = True,
) -> tuple[dict[str, Any], dict[str, int]]:
    """Bind named ``(value, unit)`` pairs to the template's input schema, checking each
    provided unit against the declared dimension (the "validate units" step, PRD §6).

    List inputs (``many``, E2.11) take a list of pairs — each element is
    dimension-checked and bound as ``name__k``; the element counts are returned so
    reducer expansion knows the arity. ``enforce_required=False`` is the ``cases``
    path, where the selected case defines what it needs (an unbound reference fails
    clear during evaluation)."""
    declared = {inp.name: inp for inp in template.inputs}
    if unknown := sorted(set(inputs) - set(declared)):
        raise InputError(
            f"inputs not declared by template {template.id!r}: {', '.join(unknown)}"
        )
    if enforce_required:
        required = {inp.name for inp in template.inputs if inp.required}
        if missing := sorted(required - set(inputs)):
            raise MissingInputError(
                f"missing required inputs for template {template.id!r}: {', '.join(missing)}"
                " — supply or resolve them; nothing will be fabricated (A-2)"
            )
    bindings: dict[str, Any] = {}
    counts: dict[str, int] = {}

    def _one(name: str, value: float, unit: str, bind_as: str) -> None:
        try:
            quantity = ureg.Quantity(value, unit or "")
        except Exception as exc:
            raise InputError(f"input {name!r}: unknown unit {unit!r}") from exc
        expected = declared_dimensionality(declared[name].dimension, ureg)
        if quantity.dimensionality != expected:
            raise InputError(
                f"input {name!r} must have dimension {declared[name].dimension!r}"
                f" ({expected}); got {unit!r} ({quantity.dimensionality})"
            )
        bindings[bind_as] = quantity

    for name, provided in inputs.items():
        if declared[name].many:
            if not isinstance(provided, list):
                raise InputError(
                    f"input {name!r} is a list input — supply a list of [value, unit] pairs"
                )
            counts[name] = len(provided)
            for k, (value, unit) in enumerate(provided):
                _one(name, value, unit, f"{name}__{k}")
            if provided:  # order statistics (E2.13): computed HERE, deterministically,
                # on base-unit magnitudes (elements may arrive in different units of
                # the same dimension) — the reserved `__` names keep the evaluation
                # walk untouched.
                base = sorted(
                    bindings[f"{name}__{k}"].to_base_units() for k in range(len(provided))
                )
                bindings[f"{name}__min"] = base[0]
                bindings[f"{name}__max"] = base[-1]
                middle = len(base) // 2
                bindings[f"{name}__median"] = (
                    base[middle]
                    if len(base) % 2
                    else (base[middle - 1] + base[middle]) / 2
                )
        else:
            if isinstance(provided, list):
                raise InputError(
                    f"input {name!r} is a scalar input — supply one [value, unit] pair"
                )
            value, unit = provided
            _one(name, value, unit, name)
    return bindings, counts


def _execute_quantity(
    template: Template,
    inputs: Mapping[str, tuple[float, str] | list[tuple[float, str]]],
    setup: Mapping[str, Any] | None = None,
) -> tuple[Any, dict[str, Any], list[TraceStep]]:
    """Returns ``(primary quantity, extra-output quantities by name, trace)`` —
    extras (E2.13) come from a steps DAG's named intermediates, each
    dimension-checked; the trace (E2.15) is the literal computation record."""
    method = template.method
    assert isinstance(method, FormulaMethod | CasesMethod)  # execute_template dispatches
    ureg = _registry()
    input_names = {inp.name for inp in template.inputs}

    def _traced(label: str, quantity: Any, expr: str | None, note: str = "") -> TraceStep:
        base = quantity.to_base_units()
        return TraceStep(
            label=label, expr=expr, value=float(base.magnitude),
            unit=f"{base.units}", note=note,
        )

    if isinstance(method, CasesMethod):  # schema_version 2 (E2.11): setup selects
        selected = (setup or {}).get(method.discriminator)
        if not isinstance(selected, str) or selected not in method.cases:
            raise InputError(
                f"template {template.id!r} needs setup {method.discriminator!r} —"
                f" one of: {', '.join(sorted(method.cases))}"
            )
        bindings, counts = _bind_inputs(template, inputs, ureg, enforce_required=False)
        expanded = _expand_reducers(method.cases[selected], counts)
        allowed = input_names | set(bindings)
        try:
            computed = _evaluate(parse_formula(expanded, allowed), bindings, ureg)
        except (ZeroDivisionError, OverflowError, ValueError) as exc:
            raise ExecutionError(f"evaluation failed: {exc}") from exc
        trace = [
            _traced(
                template.output.name, computed, method.cases[selected],
                note=f"{method.discriminator} = {selected}",
            )
        ]
        return computed, {}, trace
    bindings, counts = _bind_inputs(template, inputs, ureg)
    allowed_names = input_names | set(bindings)
    trace = []
    try:
        if method.steps:  # the DAG of assignments (E2.9): last step = result
            allowed = set(allowed_names)
            for step in method.steps:
                expanded = _expand_reducers(step.expr, counts)
                bindings[step.name] = _evaluate(
                    parse_formula(expanded, allowed), bindings, ureg
                )
                allowed.add(step.name)
                trace.append(_traced(step.name, bindings[step.name], step.expr))
            computed = bindings[method.steps[-1].name]
        else:
            assert method.expr is not None  # the schema enforces exactly one
            expanded = _expand_reducers(method.expr, counts)
            computed = _evaluate(
                parse_formula(expanded, allowed_names), bindings, ureg
            )
            trace.append(_traced(template.output.name, computed, method.expr))
    except (ZeroDivisionError, OverflowError, ValueError) as exc:
        raise ExecutionError(f"evaluation failed: {exc}") from exc
    expected = declared_dimensionality(template.output.dimension, ureg)
    if computed.dimensionality != expected:
        raise DimensionError(
            f"result dimension ({computed.dimensionality}) contradicts the declared output"
            f" {template.output.dimension!r} ({expected}) — refusing to return it (PRD §9)"
        )
    extras: dict[str, Any] = {}
    for out in template.extra_outputs:  # named steps as outputs (E2.13)
        extra = bindings[out.name]
        declared = declared_dimensionality(out.dimension, ureg)
        if extra.dimensionality != declared:
            raise DimensionError(
                f"extra output {out.name!r} dimension ({extra.dimensionality})"
                f" contradicts its declaration {out.dimension!r} ({declared}) —"
                " refusing to return it (PRD §9)"
            )
        extras[out.name] = extra
    return computed, extras, trace


def execute_template(
    template: Template,
    inputs: Mapping[str, tuple[float, str] | list[tuple[float, str]]],
    setup: Mapping[str, Any] | None = None,
) -> ExecutionResult:
    """Execute a validated template: ``kind: formula`` on named ``(value, unit)``
    inputs (result in canonical base SI units), or ``kind: symbolic`` on a ``setup``
    problem. Raises an ``ExecutionError`` subclass (with the reason) rather than ever
    returning a value that failed its checks.
    """
    if isinstance(template.method, SymbolicMethod):
        if inputs:
            raise InputError(
                "symbolic templates take no dimensioned inputs (the problem lives in setup)"
            )
        return _execute_symbolic(template, setup or {})
    if isinstance(template.method, SolverMethod):
        from assay.execute.solvers import execute_solver  # SciPy stays a lazy import

        return execute_solver(template, inputs, setup or {})
    computed, extras, trace = _execute_quantity(template, inputs, setup)
    quantity = computed.to_base_units()
    values = [
        ExecutedValue(
            label=template.output.name,
            value=float(quantity.magnitude),
            unit=f"{quantity.units}",
        )
    ]
    for out in template.extra_outputs:  # declared order, after the primary (E2.13)
        extra = extras[out.name].to_base_units()
        values.append(
            ExecutedValue(
                label=out.name, value=float(extra.magnitude), unit=f"{extra.units}"
            )
        )
    return ExecutionResult(output=template.output.name, values=values, trace=trace)


def ir_input_pairs(
    ir: IR, template: Template
) -> dict[str, tuple[float, str] | list[tuple[float, str]]]:
    """The A-1 pre-checks shared by the execute and verify stages: the IR must name the
    template, have nothing missing (fail-clear, never fabricated — A-2, A-8), and its
    ``inputs``/``resolved`` must not conflict. Returns the merged ``(value, unit)`` pairs."""
    if ir.task != template.id:
        raise ExecutionError(f"IR task {ir.task!r} does not name template {template.id!r}")
    if ir.missing_inputs:
        raise MissingInputError(
            "missing required inputs: " + ", ".join(sorted(ir.missing_inputs))
            + " — resolve or supply them; nothing will be fabricated (A-2, A-8)"
        )
    if conflict := sorted(set(ir.inputs) & set(ir.resolved)):
        raise InputError(
            f"inputs appear both user-supplied and resolved: {', '.join(conflict)}"
        )
    pairs: dict[str, tuple[float, str] | list[tuple[float, str]]] = {}
    for name, provided in ir.inputs.items():
        if isinstance(provided, list):
            pairs[name] = [(quantity.value, quantity.unit) for quantity in provided]
        else:
            pairs[name] = (provided.value, provided.unit)
    pairs |= {name: (fact.value, fact.unit) for name, fact in ir.resolved.items()}
    return pairs


def execute_ir(ir: IR, template: Template) -> ExecutionResult:
    """Execute a validated IR against its template — the sole execution contract (A-1).

    The IR's ``inputs`` and ``resolved`` facts together must satisfy the template's input
    schema; declared-but-unresolved inputs fail clear, never fabricated (A-2, A-8).
    A ``solve_for`` IR (E2.10) recovers the target input from the others plus
    ``given_output`` instead of executing forward.
    """
    if ir.solve_for is not None:
        from assay.execute.solve_for import solve_for_input  # avoid import cycle

        if ir.given_output is None:
            raise InputError("a solve_for IR needs given_output (the stated output value)")
        if ir.missing_inputs and set(ir.missing_inputs) != {ir.solve_for}:
            raise MissingInputError(
                "missing required inputs: "
                + ", ".join(sorted(set(ir.missing_inputs) - {ir.solve_for}))
                + " — resolve or supply them; nothing will be fabricated (A-2, A-8)"
            )
        pairs: dict[str, tuple[float, str]] = {}
        for name, provided in ir.inputs.items():
            if isinstance(provided, list):
                raise InputError(
                    f"input {name!r}: solve-for over list inputs is not defined"
                )
            pairs[name] = (provided.value, provided.unit)
        pairs |= {name: (fact.value, fact.unit) for name, fact in ir.resolved.items()}
        return solve_for_input(
            template, ir.solve_for, pairs, (ir.given_output.value, ir.given_output.unit)
        )
    return execute_template(template, ir_input_pairs(ir, template), setup=ir.setup)


def _within_tolerance(computed: float, expected: float, tol: float) -> bool:
    """Fixture tolerance is RELATIVE (E2.9, round 2 — the task-bank semantics):
    |computed − expected| ≤ tol·|expected|, with a 1e-12 absolute floor so
    zero-expected values stay comparable."""
    return abs(computed - expected) <= max(tol * abs(expected), 1e-12)


def _run_fixture(template: Template, fixture: Any, index: int) -> FixtureResult:
    name, expected = next(iter(fixture.expect.items()))
    try:
        if fixture.solve_for is not None:  # a solve-for fixture (E2.10)
            from assay.execute.solve_for import solve_for_input  # avoid import cycle

            assert isinstance(expected, tuple)  # the schema enforces [value, unit]
            expected_value, expected_unit = expected
            result = solve_for_input(
                template, fixture.solve_for, fixture.inputs, fixture.output
            )
            recovered = [
                float(_registry().Quantity(v.value, v.unit or "").to(expected_unit or "").magnitude)
                for v in result.values
                if isinstance(v.value, float)
            ]
            ok = any(_within_tolerance(root, expected_value, fixture.tol) for root in recovered)
            detail = (
                ""
                if ok
                else f"{name}: no recovered root among {recovered!r} matches"
                f" {expected_value!r} within relative tol {fixture.tol!r}"
            )
            return FixtureResult(
                index=index, ok=ok, computed=list(recovered), expected=expected,
                tol=fixture.tol, detail=detail,
            )
        if isinstance(expected, tuple):  # numeric [value, unit] expectation(s)
            result = execute_template(template, fixture.inputs, setup=fixture.setup)
            by_label = {value.label: value for value in result.values}
            # the primary is always first — solver results label it by the variable
            by_label.setdefault(template.output.name, result.values[0])
            computed_all: list[float] = []
            failures: list[str] = []
            for expect_name, expect_value in fixture.expect.items():  # extras too (E2.13)
                assert isinstance(expect_value, tuple)  # the contract enforces pairs
                expected_value, expected_unit = expect_value
                value = by_label[expect_name]  # the contract pins keys to outputs
                if not isinstance(value.value, float):
                    raise ExecutionError(
                        f"{expect_name}: expected a numeric result, computed"
                        f" {value.value!r}"
                    )
                quantity = _registry().Quantity(value.value, value.unit or "").to(
                    expected_unit or ""
                )
                computed = float(quantity.magnitude)
                computed_all.append(computed)
                if expect_name == template.output.name:
                    bounds = template.verification.bounds
                    if bounds is not None:
                        low = bounds.min if bounds.min is not None else -math.inf
                        high = bounds.max if bounds.max is not None else math.inf
                        in_bounds = float(quantity.to(bounds.unit or "").magnitude)
                        if not low <= in_bounds <= high:
                            failures.append(
                                f"{expect_name}: {in_bounds:g} {bounds.unit} violates"
                                f" the template's own declared bounds"
                                f" [{low:g}, {high:g}] — the template contradicts"
                                " itself (bounds ruling, round 3 §5)"
                            )
                            continue
                if not _within_tolerance(computed, expected_value, fixture.tol):
                    failures.append(
                        f"{expect_name}: {computed!r} vs {expected_value!r}"
                        f" exceeds relative tol {fixture.tol!r}"
                    )
            return FixtureResult(
                index=index, ok=not failures, computed=computed_all, expected=expected,
                tol=fixture.tol, detail="; ".join(failures),
            )
        result = execute_template(template, fixture.inputs, setup=fixture.setup)
        computed_values: list[float | str] = [value.value for value in result.values]
        if isinstance(expected, str):  # a symbolic expectation, compared by equivalence
            is_interval = (
                isinstance(template.method, SymbolicMethod)
                and template.method.operation == "solve_inequality"
            )
            ok = (
                len(computed_values) == 1
                and isinstance(computed_values[0], str)
                and (
                    computed_values[0].strip() == expected.strip()  # canonical (E2.13)
                    if is_interval
                    else symbolically_equal(computed_values[0], expected)
                )
            )
            detail = (
                ""
                if ok
                else f"{name}: computed {computed_values!r} is not equivalent"
                f" to {expected!r}"
            )
        else:  # a real root set, order-insensitive, each within tol
            numeric = [value for value in computed_values if isinstance(value, float)]
            if len(numeric) != len(computed_values) or len(numeric) != len(expected):
                ok = False
                detail = (
                    f"{name}: computed {computed_values!r} does not match the"
                    f" {len(expected)} expected real roots {sorted(expected)!r}"
                )
            else:
                pairs = list(zip(sorted(numeric), sorted(expected), strict=True))
                ok = all(_within_tolerance(a, b, fixture.tol) for a, b in pairs)
                detail = (
                    ""
                    if ok
                    else f"{name}: {sorted(numeric)!r} vs {sorted(expected)!r}"
                    f" exceeds relative tol {fixture.tol!r}"
                )
        return FixtureResult(
            index=index, ok=ok, computed=computed_values, expected=expected,
            tol=fixture.tol, detail=detail,
        )
    except Exception as exc:  # a failing fixture is a verdict, not a crash
        return FixtureResult(
            index=index, ok=False, expected=expected, tol=fixture.tol, detail=str(exc)
        )


def run_fixtures(template: Template) -> list[FixtureResult]:
    """Run the template's worked examples — the correctness proof (PRD §7.1) that the
    candidate → verified promotion gate (E2.2) trusts. Numeric expectations compare
    within **relative** ``tol`` (in the expected unit, 1e-12 absolute floor — E2.9);
    root sets order-insensitively; symbolic expectations by algebraic equivalence. A
    fixture that errors fails with the reason; it never crashes the batch."""
    return [
        _run_fixture(template, fixture, index)
        for index, fixture in enumerate(template.fixtures)
    ]
