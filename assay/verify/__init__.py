"""Deterministic verification (E1.4, PRD §9): check the answer before returning it (A-6).

Three checks, all deterministic and recorded per-verdict (the ``Verification`` object the
answer carries, E0.3):

- **dimensional consistency** — the executor already refuses a result whose dimension
  contradicts the declared output; here that refusal becomes a recorded ✗ instead of a
  bare exception. A wrong formula usually fails on units.
- **plausibility bounds** — the template-declared range rejects the absurd (3 km of beam
  deflection) before it is reported.
- **cross-method agreement** — the template's independent expression (``cross_method``)
  is evaluated on the same inputs and must agree within tolerance. A disagreement is a
  template bug, not the user's input (UX §5.6).

Symbolic operations (E1.5/E2.5) carry **built-in** checks instead of the declarative
hooks: solve → **substitution** (every root substitutes back to zero, UX §5.1),
integrate → **derivative** (the antiderivative differentiates back to the integrand,
exactly), differentiate → **difference quotient** (central difference agrees at sample
points — numeric vs symbolic, a genuinely independent method).

A failed check **withholds** the answer with the reason: ``result`` is ``None`` unless
every check passed. The computed ``candidate`` stays explicitly reachable — the
``--unsafe`` escape (UX §5.6) — but is never handed out as the answer. Execution
*errors* (bad inputs, unsafe formula) raise as usual: they are failures to compute,
not verdicts about a computed value.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from functools import lru_cache
from typing import Any

import pint
import sympy
from pydantic import BaseModel, ConfigDict

from assay.answer import Verification, VerificationCheck
from assay.execute import (
    DimensionError,
    ExecutionError,
    ExecutionResult,
    execute_template,
    ir_input_pairs,
    parametric_problem,
    parse_bound,
    parse_formula,
    symbolic_problem,
)
from assay.ir import IR
from assay.templates import (
    FormulaMethod,
    SolverMethod,
    SymbolicMethod,
    Template,
    expr_symbols,
    split_inequality,
)

__all__ = ["VerifiedExecution", "verify_execution", "verify_ir"]

# Cross-method tolerance (PRD §9): both paths are float evaluation, so an honestly
# independent form of the same physics agrees to ~1e-15 relative; 1e-6 leaves float
# headroom while still catching any real formula or transcription bug.
_CROSS_RTOL = 1e-6
_CROSS_ATOL = 1e-12

# Substitution residual ceiling: exact roots substitute to exactly zero; float roots to
# rounding noise. Anything above this is a wrong root, not noise.
_SUBSTITUTION_ATOL = 1e-9


@lru_cache(maxsize=1)
def _registry() -> pint.UnitRegistry[float]:
    return pint.UnitRegistry()


class VerifiedExecution(BaseModel):
    """An execution plus its verdicts. ``result`` enforces the withhold rule; reading
    ``candidate`` on a failed verification is the caller's explicit ``--unsafe`` act."""

    model_config = ConfigDict(extra="forbid")
    verification: Verification
    candidate: ExecutionResult | None = None

    @property
    def result(self) -> ExecutionResult | None:
        """The answer — present only when every check passed (never silent, PRD §9)."""
        return self.candidate if self.verification.ok else None


def _single_magnitude(result: ExecutionResult) -> float:
    """The one numeric magnitude of a formula result (hooks only apply to those)."""
    value = result.value
    if not isinstance(value, float):
        raise ExecutionError(f"expected a numeric result, got {value!r}")
    return value


def _check_bounds(template: Template, result: ExecutionResult) -> VerificationCheck:
    bounds = template.verification.bounds
    assert bounds is not None  # caller guards
    low = "-inf" if bounds.min is None else f"{bounds.min:g}"
    high = "inf" if bounds.max is None else f"{bounds.max:g}"
    window = f"[{low}, {high}] {bounds.unit}".rstrip()
    try:
        magnitude = float(
            _registry()
            .Quantity(_single_magnitude(result), result.unit)
            .to(bounds.unit or "")
            .magnitude
        )
    except pint.DimensionalityError:
        return VerificationCheck(
            name="bounds",
            ok=False,
            detail=(
                f"bounds unit {bounds.unit!r} is incompatible with the result"
                f" ({result.unit}) — a template bug, not your input"
            ),
        )
    ok = (bounds.min is None or magnitude >= bounds.min) and (
        bounds.max is None or magnitude <= bounds.max
    )
    if ok:
        detail = f"{magnitude:g} {bounds.unit} within the plausible range {window}".strip()
    else:
        detail = (
            f"{magnitude:g} {bounds.unit} is outside the plausible range {window}"
            " — withholding the answer"
        )
    return VerificationCheck(name="bounds", ok=ok, detail=detail)


def _check_cross_method(
    template: Template,
    result: ExecutionResult,
    inputs: Mapping[str, tuple[float, str] | list[tuple[float, str]]],
) -> VerificationCheck:
    expr = template.verification.cross_method
    assert expr is not None  # caller guards
    independent = template.model_copy(
        update={"method": FormulaMethod(kind="formula", expr=expr)}
    )
    try:
        cross = execute_template(independent, inputs)
    except ExecutionError as exc:
        return VerificationCheck(
            name="cross-method",
            ok=False,
            detail=f"the independent method failed to evaluate: {exc} — a template bug",
        )
    mine, other = _single_magnitude(result), _single_magnitude(cross)
    if math.isclose(mine, other, rel_tol=_CROSS_RTOL, abs_tol=_CROSS_ATOL):
        detail = f"independent method agrees: {other:g} {cross.unit} (rel tol {_CROSS_RTOL:g})"
        return VerificationCheck(name="cross-method", ok=True, detail=detail)
    return VerificationCheck(
        name="cross-method",
        ok=False,
        detail=(
            f"method {mine:g} {result.unit} vs independent method"
            f" {other:g} {cross.unit} — disagreement exceeds rel tol {_CROSS_RTOL:g};"
            " a template bug, not your input"
        ),
    )


def _check_substitution(
    template: Template, result: ExecutionResult, setup: Mapping[str, Any]
) -> VerificationCheck:
    """UX §5.1: every root must substitute back into the equation to (near-)zero."""
    expression, symbol = symbolic_problem(template, setup)
    worst = 0.0
    for executed in result.values:
        try:
            root = (
                sympy.Float(executed.value)
                if isinstance(executed.value, float)
                else parse_formula(str(executed.value), {str(symbol)})
            )
            residual = abs(complex(sympy.N(expression.subs(symbol, root))))
        except Exception as exc:
            return VerificationCheck(
                name="substitution",
                ok=False,
                detail=f"could not verify root {executed.value!r} by substitution: {exc}",
            )
        worst = max(worst, residual)
        if residual > _SUBSTITUTION_ATOL:
            return VerificationCheck(
                name="substitution",
                ok=False,
                detail=(
                    f"root {executed.value!r} does not satisfy the equation"
                    f" (residual {residual:g}) — withholding the answer"
                ),
            )
    count = len(result.values)
    plural = "s" if count != 1 else ""
    return VerificationCheck(
        name="substitution",
        ok=True,
        detail=f"all {count} root{plural} substitute to 0 (worst residual {worst:g})",
    )


def _check_derivative(
    template: Template, result: ExecutionResult, setup: Mapping[str, Any]
) -> VerificationCheck:
    """The antiderivative differentiates back to the integrand, exactly (SymPy)."""
    expression, symbol = symbolic_problem(template, setup)
    try:
        antiderivative = parse_formula(str(result.value), {str(symbol)})
        difference = sympy.simplify(sympy.diff(antiderivative, symbol) - expression)
    except Exception as exc:
        return VerificationCheck(
            name="derivative", ok=False, detail=f"could not verify by differentiation: {exc}"
        )
    if difference == 0:
        return VerificationCheck(
            name="derivative",
            ok=True,
            detail=f"d/d{symbol} of the antiderivative equals the integrand",
        )
    return VerificationCheck(
        name="derivative",
        ok=False,
        detail=(
            f"d/d{symbol} of the antiderivative differs from the integrand by"
            f" {sympy.sstr(difference)} — withholding the answer"
        ),
    )


def _check_difference_quotient(
    template: Template, result: ExecutionResult, setup: Mapping[str, Any]
) -> VerificationCheck:
    """The computed derivative agrees with the numeric difference quotient of the
    original expression at sample points — a cross-method check (central difference vs
    symbolic differentiation), not a re-run of the same operation."""
    expression, symbol = symbolic_problem(template, setup)
    name = "difference-quotient"
    try:
        derivative = parse_formula(str(result.value), {str(symbol)})
    except Exception as exc:
        return VerificationCheck(
            name=name, ok=False, detail=f"could not parse the computed derivative: {exc}"
        )
    step = 1e-6
    compared = 0
    for x in (0.7, 1.3, -0.6):
        try:
            above = float(expression.subs(symbol, sympy.Float(x + step)))
            below = float(expression.subs(symbol, sympy.Float(x - step)))
            exact = float(derivative.subs(symbol, sympy.Float(x)))
        except (TypeError, ValueError):
            continue  # non-real at this sample point: not comparable here
        quotient = (above - below) / (2 * step)
        if abs(quotient - exact) > 1e-4 * max(1.0, abs(exact)):
            return VerificationCheck(
                name=name,
                ok=False,
                detail=(
                    f"at {symbol} = {x:g}: difference quotient {quotient:.6g} disagrees"
                    f" with the derivative {exact:.6g} — withholding the answer"
                ),
            )
        compared += 1
    if compared == 0:
        return VerificationCheck(
            name=name, ok=False, detail="no real sample point to check the derivative at"
        )
    return VerificationCheck(
        name=name,
        ok=True,
        detail=f"central difference matches the derivative at {compared} sample points",
    )


def _reported_number(text: str) -> float | None:
    """A reported symbolic value (``"pi/2"``, ``"2*sqrt(2)"``) as a float — through
    the same gated parse as everything; ``None`` when it isn't a plain number."""
    try:
        value = parse_formula(text, expr_symbols(text))
        return float(value)
    except Exception:
        return None


def _check_limit_approach(
    template: Template, result: ExecutionResult, setup: Mapping[str, Any]
) -> VerificationCheck:
    """The limit's independent check (E2.13): evaluate the expression along a numeric
    approach sequence — the values must head where the symbolic limit says."""
    name = "limit-approach"
    expression, symbol = symbolic_problem(template, setup)
    point = parse_bound(setup["point"])
    direction = setup.get("direction", "+")
    if point == sympy.oo:
        samples = [1e3, 1e5, 1e7]
    elif point == -sympy.oo:
        samples = [-1e3, -1e5, -1e7]
    else:
        sign = 1.0 if direction == "+" else -1.0
        samples = [float(point) + sign * step for step in (1e-3, 1e-5, 1e-7)]
    trail: list[float] = []
    for x in samples:
        try:
            trail.append(float(expression.subs(symbol, sympy.Float(x))))
        except (TypeError, ValueError):
            continue
    if len(trail) < 2:
        return VerificationCheck(
            name=name, ok=False, detail="no real approach points to check the limit at"
        )
    reported = result.value
    if isinstance(reported, str) and reported.strip() in {"oo", "-oo"}:
        heading = trail[-1] > 1e6 if reported.strip() == "oo" else trail[-1] < -1e6
        return VerificationCheck(
            name=name,
            ok=heading,
            detail=(
                f"the approach sequence heads to {reported.strip()} ({trail[-1]:.3g})"
                if heading
                else f"the approach sequence ({trail[-1]:.3g}) does not diverge to"
                f" {reported.strip()} — withholding the answer"
            ),
        )
    stated = reported if isinstance(reported, float) else _reported_number(str(reported))
    if stated is None:
        return VerificationCheck(
            name=name, ok=False,
            detail=f"cannot check the reported limit {reported!r} numerically",
        )
    last_gap = abs(trail[-1] - stated)
    ok = last_gap <= max(1e-3, 1e-3 * abs(stated)) and last_gap <= abs(
        trail[0] - stated
    ) + 1e-12
    return VerificationCheck(
        name=name,
        ok=ok,
        detail=(
            f"the approach sequence converges to {stated:.6g} (gap {last_gap:.3g})"
            if ok
            else f"the approach sequence ({trail[-1]:.6g}) does not converge to the"
            f" reported limit {stated:.6g} — withholding the answer"
        ),
    )


def _check_quadrature(
    template: Template, result: ExecutionResult, setup: Mapping[str, Any]
) -> VerificationCheck:
    """The definite/improper integral's independent check (E2.13): adaptive numeric
    quadrature (a genuinely different method than symbolic integration) must agree —
    and a stated divergence must show unbounded partial integrals."""
    name = "cross-method:quadrature"
    expression, symbol = symbolic_problem(template, setup)
    limits = setup.get("limits")
    assert isinstance(limits, list | tuple) and len(limits) == 2  # contract-checked
    low, high = parse_bound(limits[0]), parse_bound(limits[1])

    def function(x: float) -> float:
        return float(expression.subs(symbol, sympy.Float(x)))

    from scipy.integrate import quad  # lazy, like the solver bindings

    reported = result.value
    if isinstance(reported, str) and reported.strip() in {"oo", "-oo"}:
        sign = 1.0 if reported.strip() == "oo" else -1.0
        anchor = float(low) if low not in (sympy.oo, -sympy.oo) else -1e6
        try:
            near = quad(function, anchor, 1e3, limit=200)[0]
            far = quad(function, anchor, 1e6, limit=200)[0]
        except Exception as exc:
            return VerificationCheck(
                name=name, ok=False, detail=f"could not probe the divergence: {exc}"
            )
        growing = sign * (far - near) > 1.0 and sign * far > sign * near
        return VerificationCheck(
            name=name,
            ok=growing,
            detail=(
                f"partial integrals grow without bound ({near:.4g} → {far:.4g})"
                if growing
                else f"partial integrals do not diverge ({near:.4g} → {far:.4g}) —"
                " withholding the answer"
            ),
        )
    stated = reported if isinstance(reported, float) else _reported_number(str(reported))
    if stated is None:
        return VerificationCheck(
            name=name, ok=False,
            detail=f"cannot check the reported integral {reported!r} numerically",
        )
    lo = -math.inf if low == -sympy.oo else float(low)
    hi = math.inf if high == sympy.oo else float(high)
    try:
        numeric, _estimate = quad(function, lo, hi, limit=200)
    except Exception as exc:
        return VerificationCheck(
            name=name, ok=False, detail=f"quadrature failed: {exc}"
        )
    ok = abs(numeric - stated) <= max(1e-6 * abs(stated), 1e-8)
    return VerificationCheck(
        name=name,
        ok=ok,
        detail=(
            f"symbolic {stated:.10g} vs quadrature {numeric:.10g}"
            + ("" if ok else " — disagreement; withholding the answer")
        ),
    )


def _interval_pieces(text: str) -> list[tuple[Any, ...]]:
    """Parse Assay's canonical interval notation back into pieces — Assay renders it,
    so this parse is exact by construction."""
    if text.strip() == "empty":
        return []

    def bound(token: str) -> Any:
        token = token.strip()
        if token == "oo":
            return sympy.oo
        if token == "-oo":
            return -sympy.oo
        return sympy.Rational(token)

    pieces: list[tuple[Any, ...]] = []
    for part in text.split(" U "):
        part = part.strip()
        if part.startswith("{"):
            pieces.append(("finite", [bound(v) for v in part[1:-1].split(",")]))
        else:
            inner_low, inner_high = part[1:-1].split(",")
            pieces.append(
                ("interval", bound(inner_low), bound(inner_high),
                 part[0] == "(", part[-1] == ")")
            )
    return pieces


def _check_interval_testpoints(
    template: Template, result: ExecutionResult, setup: Mapping[str, Any]
) -> VerificationCheck:
    """The inequality's independent check (E2.13): membership in the reported set must
    agree with the original inequality at interior points, at every finite boundary,
    and just outside each boundary."""
    name = "interval-testpoints"
    raw = setup.get("expression")
    assert isinstance(raw, str)  # contract-checked
    lhs_text, operator, rhs_text = split_inequality(raw)
    names = expr_symbols(lhs_text) | expr_symbols(rhs_text)
    variable = setup.get("variable") or next(iter(names))
    lhs = parse_formula(lhs_text, {variable})
    rhs = parse_formula(rhs_text, {variable})
    symbol = sympy.Symbol(variable)
    comparators: dict[str, Callable[[float, float], bool]] = {
        "<": lambda a, b: a < b, "<=": lambda a, b: a <= b,
        ">": lambda a, b: a > b, ">=": lambda a, b: a >= b,
    }
    compare = comparators[operator]

    def truth(x: Any) -> bool | None:
        try:
            left = lhs.subs(symbol, x)
            right = rhs.subs(symbol, x)
            return bool(compare(float(left), float(right)))
        except (TypeError, ValueError):
            return None  # non-real here: not a usable probe

    reported = str(result.value)
    pieces = _interval_pieces(reported)

    def member(x: Any) -> bool:
        for piece in pieces:
            if piece[0] == "finite":
                if any(x == v for v in piece[1]):
                    return True
            else:
                _kind, low, high, low_open, high_open = piece
                above = x > low or (not low_open and x == low)
                below = x < high or (not high_open and x == high)
                if above and below:
                    return True
        return False

    probes: set[Any] = {sympy.Rational(p) for p in (-3, -1, 0, 1, 3)}
    for piece in pieces:
        if piece[0] == "finite":
            for v in piece[1]:
                probes |= {v, v - sympy.Rational(1, 10), v + sympy.Rational(1, 10)}
        else:
            _kind, low, high, _lo_open, _hi_open = piece
            step = sympy.Rational(1, 1000)
            if low != -sympy.oo:
                probes |= {low, low - step, low + step}
            if high != sympy.oo:
                probes |= {high, high - step, high + step}
            if low != -sympy.oo and high != sympy.oo:
                probes.add((low + high) / 2)
            elif low != -sympy.oo:
                probes.add(low + 1)
            elif high != sympy.oo:
                probes.add(high - 1)
    checked = 0
    for x in sorted(probes, key=float):
        actual = truth(x)
        if actual is None:
            continue
        if member(x) != actual:
            return VerificationCheck(
                name=name,
                ok=False,
                detail=(
                    f"at {variable} = {sympy.sstr(x)}: the inequality is"
                    f" {'true' if actual else 'false'} but the reported set"
                    f" {reported!r} says {'in' if member(x) else 'out'} —"
                    " withholding the answer"
                ),
            )
        checked += 1
    if checked == 0:
        return VerificationCheck(
            name=name, ok=False, detail="no usable test points to check the set with"
        )
    return VerificationCheck(
        name=name,
        ok=True,
        detail=f"set membership agrees with the inequality at {checked} test points",
    )


def _check_parametric_difference(
    template: Template, result: ExecutionResult, setup: Mapping[str, Any]
) -> VerificationCheck:
    """The parametric slope's independent check (chisel round 8): the central
    difference (Δy/Δx along the curve) must agree — numerically at the given point,
    or at sample parameter values for a symbolic slope."""
    name = "parametric-difference"
    x_expr, y_expr, symbol = parametric_problem(setup)
    step = 1e-6
    reported = result.value

    def chord(at: float) -> float | None:
        try:
            dx = float(x_expr.subs(symbol, sympy.Float(at + step))) - float(
                x_expr.subs(symbol, sympy.Float(at - step))
            )
            dy = float(y_expr.subs(symbol, sympy.Float(at + step))) - float(
                y_expr.subs(symbol, sympy.Float(at - step))
            )
        except (TypeError, ValueError):
            return None
        if abs(dx) < 1e-15:
            return None
        return dy / dx

    if isinstance(reported, float):
        point = setup.get("point")
        assert isinstance(point, int | float)  # the executor required it
        numeric = chord(float(point))
        if numeric is None:
            return VerificationCheck(
                name=name, ok=False,
                detail=f"no usable central difference at {symbol} = {point:g}",
            )
        ok = abs(numeric - reported) <= 1e-4 * max(1.0, abs(reported))
        return VerificationCheck(
            name=name,
            ok=ok,
            detail=(
                f"central difference {numeric:.6g} vs slope {reported:.6g}"
                + ("" if ok else " — disagreement; withholding the answer")
            ),
        )
    try:
        slope_expr = parse_formula(str(reported), {str(symbol)})
    except Exception as exc:
        return VerificationCheck(
            name=name, ok=False, detail=f"could not parse the computed slope: {exc}"
        )
    compared = 0
    for at in (0.7, 1.3, -0.6):
        numeric = chord(at)
        try:
            exact = float(slope_expr.subs(symbol, sympy.Float(at)))
        except (TypeError, ValueError):
            continue
        if numeric is None:
            continue
        if abs(numeric - exact) > 1e-4 * max(1.0, abs(exact)):
            return VerificationCheck(
                name=name,
                ok=False,
                detail=(
                    f"at {symbol} = {at:g}: central difference {numeric:.6g}"
                    f" disagrees with the slope {exact:.6g} — withholding the answer"
                ),
            )
        compared += 1
    if compared == 0:
        return VerificationCheck(
            name=name, ok=False, detail="no usable sample point to check the slope at"
        )
    return VerificationCheck(
        name=name,
        ok=True,
        detail=f"central difference matches the slope at {compared} sample points",
    )


def _check_taylor_coefficients(
    template: Template, result: ExecutionResult, setup: Mapping[str, Any]
) -> VerificationCheck:
    """The Taylor polynomial's independent check (E2.16): coefficient k must equal
    f⁽ᵏ⁾(a)/k! — checked by differentiating BOTH the reported polynomial and f at
    the center, exactly, independent of the series expansion that generated it."""
    name = "taylor-coefficients"
    expression, symbol = symbolic_problem(template, setup)
    order = setup.get("order")
    assert isinstance(order, int)  # the executor required it
    center = setup.get("center", 0)
    at = sympy.Integer(center) if isinstance(center, int) else sympy.Float(center)
    try:
        polynomial = parse_formula(str(result.value), {str(symbol)})
    except Exception as exc:
        return VerificationCheck(
            name=name, ok=False, detail=f"could not parse the reported polynomial: {exc}"
        )
    if sympy.degree(sympy.expand(polynomial), symbol) > order:
        return VerificationCheck(
            name=name, ok=False,
            detail=f"the reported polynomial exceeds degree {order} — withholding",
        )
    reported_k = polynomial
    function_k = expression
    for k in range(order + 1):
        try:
            difference = sympy.simplify(
                reported_k.subs(symbol, at) - function_k.subs(symbol, at)
            )
        except (TypeError, ValueError) as exc:
            return VerificationCheck(
                name=name, ok=False,
                detail=f"cannot evaluate derivative {k} at the center: {exc}",
            )
        if difference != 0:
            return VerificationCheck(
                name=name,
                ok=False,
                detail=(
                    f"derivative {k} at the center disagrees"
                    f" (difference {sympy.sstr(difference)}) — withholding the answer"
                ),
            )
        reported_k = sympy.diff(reported_k, symbol)
        function_k = sympy.diff(function_k, symbol)
    return VerificationCheck(
        name=name,
        ok=True,
        detail=f"all {order + 1} coefficients match f's derivative table at the center",
    )


def _check_term_behavior(
    template: Template, result: ExecutionResult, setup: Mapping[str, Any]
) -> VerificationCheck:
    """The convergence result's independent check (E2.16): sample the TERM
    numerically — inside the reported radius the terms must shrink; outside a
    finite radius they must not vanish. Endpoint inclusion rides on SymPy's
    convergence machinery (recorded; not re-checked numerically)."""
    name = "term-behavior"
    from assay.execute import parse_series_term

    raw = setup.get("term")
    assert isinstance(raw, str)
    variable = setup.get("variable", "x")
    index = setup.get("index", "n")
    term = parse_series_term(raw, {variable, index})
    x, n = sympy.Symbol(variable), sympy.Symbol(index)
    center = float(setup.get("center", 0))
    radius_value = result.values[1].value if len(result.values) > 1 else None

    def magnitude(x_at: float, n_at: int) -> float | None:
        try:
            return abs(float(term.subs({x: sympy.Float(x_at), n: sympy.Integer(n_at)})))
        except (TypeError, ValueError, OverflowError):
            return None

    def shrinks(x_at: float) -> bool | None:
        early, late = magnitude(x_at, 8), magnitude(x_at, 40)
        if early is None or late is None:
            return None
        return late <= early * 1e-3 or late <= 1e-9

    probes: list[tuple[float, bool]] = []  # (x, expected-to-shrink)
    if radius_value == "oo":
        probes = [(center + 5.0, True), (center - 5.0, True)]
    elif isinstance(radius_value, float) and radius_value == 0.0:
        probes = [(center + 1.0, False), (center - 1.0, False)]
    elif isinstance(radius_value, float):
        probes = [
            (center + radius_value / 2, True), (center - radius_value / 2, True),
            (center + 2 * radius_value, False), (center - 2 * radius_value, False),
        ]
    checked = 0
    for x_at, expect_shrink in probes:
        verdict = shrinks(x_at)
        if verdict is None:
            continue
        if verdict != expect_shrink:
            side = "inside" if expect_shrink else "outside"
            return VerificationCheck(
                name=name,
                ok=False,
                detail=(
                    f"at x = {x_at:g} ({side} the reported radius) the terms"
                    f" {'do not shrink' if expect_shrink else 'vanish'} —"
                    " contradicting the ratio test; withholding the answer"
                ),
            )
        checked += 1
    if checked == 0:
        return VerificationCheck(
            name=name, ok=False, detail="no usable probe points for the term"
        )
    return VerificationCheck(
        name=name,
        ok=True,
        detail=(
            f"term magnitudes agree with the reported radius at {checked} probes"
            " (endpoint inclusion per SymPy's convergence test)"
        ),
    )


def _verify_symbolic(
    template: Template, setup: Mapping[str, Any]
) -> VerifiedExecution:
    """Symbolic operations carry built-in checks (E1.5/E2.5/E2.13/E2.16) in place of
    the declarative hooks: solve → substitution, differentiate → difference quotient,
    integrate → derivative (or quadrature when definite), limit → approach sequence,
    solve_inequality → test points, parametric_slope → central difference,
    taylor_polynomial → the derivative table, series_convergence → term behavior."""
    result = execute_template(template, {}, setup=setup)
    assert isinstance(template.method, SymbolicMethod)  # dispatcher guards
    if template.method.operation == "solve":
        check = _check_substitution(template, result, setup)
    elif template.method.operation == "differentiate":
        check = _check_difference_quotient(template, result, setup)
    elif template.method.operation == "limit":
        check = _check_limit_approach(template, result, setup)
    elif template.method.operation == "solve_inequality":
        check = _check_interval_testpoints(template, result, setup)
    elif template.method.operation == "parametric_slope":
        check = _check_parametric_difference(template, result, setup)
    elif template.method.operation == "taylor_polynomial":
        check = _check_taylor_coefficients(template, result, setup)
    elif template.method.operation == "series_convergence":
        check = _check_term_behavior(template, result, setup)
    elif setup.get("limits") is not None:  # definite/improper integration (E2.13)
        check = _check_quadrature(template, result, setup)
    else:
        check = _check_derivative(template, result, setup)
    return VerifiedExecution(
        verification=Verification(ok=check.ok, checks=[check]), candidate=result
    )


def _verify_solver(template: Template, setup: Mapping[str, Any]) -> VerifiedExecution:
    """Solver bindings carry built-in independent checks (E3.5): substitution for
    roots, a second quadrature for integrals, local optimality for minima, a second
    integrator for ODEs — the same withhold rule as everywhere."""
    from assay.execute.solvers import verify_solver  # SciPy stays a lazy import

    result = execute_template(template, {}, setup=setup)
    check = verify_solver(template, result, setup)
    return VerifiedExecution(
        verification=Verification(ok=check.ok, checks=[check]), candidate=result
    )


def verify_execution(
    template: Template,
    inputs: Mapping[str, tuple[float, str] | list[tuple[float, str]]],
    setup: Mapping[str, Any] | None = None,
) -> VerifiedExecution:
    """Execute and verify: every declared check runs and is recorded; one failure
    withholds the answer with its reason (A-6). Never returns a silently-failed result."""
    if isinstance(template.method, SymbolicMethod):
        return _verify_symbolic(template, setup or {})
    if isinstance(template.method, SolverMethod):
        return _verify_solver(template, setup or {})
    dimension_check = f"dimension:{template.output.dimension}"
    try:
        result = execute_template(template, inputs, setup=setup)
    except DimensionError as exc:
        checks = [VerificationCheck(name=dimension_check, ok=False, detail=str(exc))]
        return VerifiedExecution(verification=Verification(ok=False, checks=checks))
    checks = [
        VerificationCheck(
            name=dimension_check,
            ok=True,
            detail=(
                "result dimension matches the declared output"
                f" ({result.values[0].unit})"  # extras are checked in the executor
            ),
        )
    ]
    if template.verification.bounds is not None:
        checks.append(_check_bounds(template, result))
    if template.verification.cross_method is not None:
        checks.append(_check_cross_method(template, result, inputs))
    ok = all(check.ok for check in checks)
    return VerifiedExecution(verification=Verification(ok=ok, checks=checks), candidate=result)


def _verify_solve_for(ir: IR, template: Template) -> VerifiedExecution:
    """The solve-for verification (E2.10): every recovered root runs back through the
    FORWARD template and must reproduce the stated output — genuinely independent of
    the symbolic inversion that produced it."""
    from assay.execute import execute_ir

    result = execute_ir(ir, template)
    assert ir.solve_for is not None and ir.given_output is not None  # execute enforced
    others: dict[str, tuple[float, str]] = {}
    for name, provided in ir.inputs.items():
        if not isinstance(provided, list):  # solve-for refuses lists upstream
            others[name] = (provided.value, provided.unit)
    others |= {name: (fact.value, fact.unit) for name, fact in ir.resolved.items()}
    ureg = _registry()
    stated = ureg.Quantity(ir.given_output.value, ir.given_output.unit or "").to_base_units()
    failures: list[str] = []
    for value in result.values:
        assert isinstance(value.value, float)
        forward = execute_template(
            template, {**others, ir.solve_for: (value.value, value.unit)}
        )
        assert isinstance(forward.value, float)
        recomputed = ureg.Quantity(forward.value, forward.unit or "").to_base_units()
        difference = abs(float((recomputed - stated).magnitude))
        ceiling = 1e-9 * max(abs(float(stated.magnitude)), 1e-30)
        if difference > ceiling:
            failures.append(
                f"{ir.solve_for} = {value.value:g} {value.unit} does not reproduce the"
                f" stated output ({forward.value:g} vs {float(stated.magnitude):g})"
            )
    plural = "s" if len(result.values) != 1 else ""
    check = VerificationCheck(
        name="forward-substitution",
        ok=not failures,
        detail=(
            f"all {len(result.values)} recovered root{plural} reproduce the stated"
            " output through the forward template"
            if not failures
            else "; ".join(failures) + " — withholding"
        ),
    )
    return VerifiedExecution(
        verification=Verification(ok=check.ok, checks=[check]), candidate=result
    )


def verify_ir(ir: IR, template: Template) -> VerifiedExecution:
    """Verify a validated IR's execution (A-1) — the pipeline's verify stage (PRD §5):
    the same pre-checks as ``execute_ir``, then every applicable verification check."""
    if ir.solve_for is not None:
        return _verify_solve_for(ir, template)
    return verify_execution(template, ir_input_pairs(ir, template), setup=ir.setup)
