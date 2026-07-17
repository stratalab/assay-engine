"""Curated solver bindings (E3.5, PRD §7.1): ``kind: solver``, done safely.

A solver template's ``binding`` is a **name looked up in this registry** — never an
import path to execute, never model-influenced code (engineering §7). Every binding is
Assay-authored: it parses its problem through the same safe gate as everything else
(``parse_problem``/``parse_formula``; expressions are evaluated by SymPy ``subs``, no
``eval``, no ``lambdify``), drives a SciPy routine with **pinned, deterministic
settings**, and carries a **built-in independent check** the verify stage runs —
substitution for roots, a second quadrature for integrals, local optimality for
minima, a second integrator for ODEs. Withhold-with-reason applies as everywhere
(A-6): a result whose check fails is never returned as the answer.

The v0 contract: a solver template declares no dimensioned inputs — the problem lives
in ``setup`` (like symbolic operations); results are dimensionless. Dimensioned solver
templates (Chisel-supplied fixtures over engineering charts) come later with
``kind: table``. SciPy imports are local to each binding so the deterministic verbs
never pay the import.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any

import sympy

from assay.answer import VerificationCheck
from assay.execute import (
    ExecutedValue,
    ExecutionError,
    ExecutionResult,
    InputError,
    parse_bound,
    parse_formula,
    parse_problem,
)
from assay.templates import SolverMethod, Template, expr_symbols

__all__ = ["execute_solver", "solver_bindings", "verify_solver"]


def _setup_pair(setup: Mapping[str, Any], key: str, what: str) -> tuple[float, float]:
    raw = setup.get(key)
    if (
        not isinstance(raw, list | tuple)
        or len(raw) != 2
        or not all(isinstance(v, int | float) and not isinstance(v, bool) for v in raw)
    ):
        raise InputError(f"setup {key!r} must be {what} — two numbers [lo, hi]")
    low, high = float(raw[0]), float(raw[1])
    if not high > low:
        raise InputError(f"setup {key!r} is empty: [{low:g}, {high:g}]")
    return low, high


def _scalar_problem(setup: Mapping[str, Any], *, allow_equation: bool) -> tuple[Any, Any]:
    raw = setup.get("expression")
    if not isinstance(raw, str) or not raw.strip():
        raise InputError("this solver needs setup 'expression' (a non-empty string)")
    variable = setup.get("variable")
    if variable is not None and not isinstance(variable, str):
        raise InputError("setup 'variable' must be a simple name")
    return parse_problem(raw, variable, allow_equation=allow_equation)


def _as_function(parsed: Any, symbol: Any) -> Callable[[float], float]:
    """The safe evaluation path: SymPy ``subs`` per point — no eval, no lambdify."""

    def function(x: float) -> float:
        return float(parsed.subs(symbol, sympy.Float(x)))

    return function


# --- root_find.brentq -----------------------------------------------------------


def _root_find(setup: Mapping[str, Any]) -> ExecutionResult:
    parsed, symbol = _scalar_problem(setup, allow_equation=True)
    low, high = _setup_pair(setup, "bracket", "the sign-change bracket")
    from scipy.optimize import brentq

    function = _as_function(parsed, symbol)
    if function(low) * function(high) > 0:
        raise ExecutionError(
            f"no sign change over the bracket [{low:g}, {high:g}] —"
            " a bracketing method needs one; refusing to guess (A-12)"
        )
    root = float(brentq(function, low, high, xtol=1e-14, maxiter=200))
    return ExecutionResult(
        output="root", values=[ExecutedValue(label=str(symbol), value=root)]
    )


def _check_root(
    template: Template, result: ExecutionResult, setup: Mapping[str, Any]
) -> VerificationCheck:
    parsed, symbol = _scalar_problem(setup, allow_equation=True)
    root = result.values[0].value
    assert isinstance(root, float)
    residual = abs(_as_function(parsed, symbol)(root))
    ok = residual <= 1e-8
    return VerificationCheck(
        name="substitution",
        ok=ok,
        detail=(
            f"|f({root:.10g})| = {residual:.3g}"
            + ("" if ok else " — the root does not satisfy the equation; withholding")
        ),
    )


# --- integrate.quad -------------------------------------------------------------


def _integration_limits(setup: Mapping[str, Any]) -> tuple[float, float]:
    """The integration limits, IMPROPER bounds included (chisel round 8, finding 1):
    each may be a number or ``"oo"``/``"-oo"`` — SciPy's quad handles infinite ranges
    natively, and the cross-check transforms them (see ``_check_integral``)."""
    raw = setup.get("limits")
    if not isinstance(raw, list | tuple) or len(raw) != 2:
        raise InputError("setup 'limits' must be the integration limits [lo, hi]")
    try:
        bounds = [parse_bound(value) for value in raw]
    except InputError as exc:
        raise InputError(f"setup 'limits': {exc}") from None
    low = -math.inf if bounds[0] == -sympy.oo else float(bounds[0])
    high = math.inf if bounds[1] == sympy.oo else float(bounds[1])
    if not high > low:
        raise InputError(f"setup 'limits' is empty: [{raw[0]}, {raw[1]}]")
    return low, high


def _integrate(setup: Mapping[str, Any]) -> ExecutionResult:
    parsed, symbol = _scalar_problem(setup, allow_equation=False)
    low, high = _integration_limits(setup)
    from scipy.integrate import quad

    value, _estimate = quad(
        _as_function(parsed, symbol), low, high, epsabs=1e-12, epsrel=1e-12, limit=200
    )
    return ExecutionResult(
        output="integral", values=[ExecutedValue(label="integral", value=float(value))]
    )


def _fixed_grid_simpson(function: Callable[[float], float], low: float, high: float) -> float:
    samples = 2001  # odd: Simpson needs an even interval count; fixed: deterministic
    step = (high - low) / (samples - 1)
    total = function(low) + function(high)
    for index in range(1, samples - 1):
        total += function(low + index * step) * (4 if index % 2 else 2)
    return total * step / 3


def _check_integral(
    template: Template, result: ExecutionResult, setup: Mapping[str, Any]
) -> VerificationCheck:
    """Cross-method: composite Simpson on a fixed grid must agree with the adaptive
    quadrature — two genuinely different rules. Improper ranges are transformed to a
    finite interval first (x = a + t/(1-t) for [a, oo); x = t/(1-t**2) for the doubly
    infinite case), so the same fixed-grid rule applies; the interval is clipped a
    hair short of the singular endpoint, so the check carries a slightly looser
    tolerance there — a slowly-decaying tail that genuinely exceeds it withholds,
    honestly (A-6)."""
    parsed, symbol = _scalar_problem(setup, allow_equation=False)
    low, high = _integration_limits(setup)
    function = _as_function(parsed, symbol)
    improper = math.isinf(low) or math.isinf(high)
    if not improper:
        simpson = _fixed_grid_simpson(function, low, high)
        tolerance = max(1e-6 * abs(simpson), 1e-9)
    else:
        clip = 1e-6  # t in [clip, 1-clip]: the tail beyond ~1/clip is the tolerance
        if math.isinf(low) and math.isinf(high):
            def transformed(t: float) -> float:
                x = t / (1 - t * t)
                return function(x) * (1 + t * t) / (1 - t * t) ** 2
            simpson = _fixed_grid_simpson(transformed, -1 + clip, 1 - clip)
        elif math.isinf(high):
            def transformed(t: float) -> float:
                return function(low + t / (1 - t)) / (1 - t) ** 2
            simpson = _fixed_grid_simpson(transformed, 0.0, 1 - clip)
        else:  # (-oo, b]
            def transformed(t: float) -> float:
                return function(high - t / (1 - t)) / (1 - t) ** 2
            simpson = _fixed_grid_simpson(transformed, 0.0, 1 - clip)
        tolerance = max(1e-5 * abs(simpson), 1e-7)
    value = result.values[0].value
    assert isinstance(value, float)
    ok = abs(value - simpson) <= tolerance
    return VerificationCheck(
        name="cross-method:simpson",
        ok=ok,
        detail=(
            f"adaptive quadrature {value:.10g} vs Simpson {simpson:.10g}"
            + (" (transformed to a finite interval)" if improper else "")
            + ("" if ok else " — disagreement; withholding")
        ),
    )


# --- minimize.scalar_bounded ----------------------------------------------------


def _minimize(setup: Mapping[str, Any]) -> ExecutionResult:
    parsed, symbol = _scalar_problem(setup, allow_equation=False)
    low, high = _setup_pair(setup, "bounds", "the search bounds")
    from scipy.optimize import minimize_scalar

    solution = minimize_scalar(
        _as_function(parsed, symbol),
        bounds=(low, high),
        method="bounded",
        options={"xatol": 1e-12},
    )
    if not solution.success:
        raise ExecutionError(f"minimization failed: {solution.message}")
    return ExecutionResult(
        output="minimizer",
        values=[ExecutedValue(label=str(symbol), value=float(solution.x))],
    )


def _check_minimum(
    template: Template, result: ExecutionResult, setup: Mapping[str, Any]
) -> VerificationCheck:
    """Local optimality: the reported minimizer beats its neighbors (within the
    bounds) — checked directly on the function, not on the solver's say-so."""
    parsed, symbol = _scalar_problem(setup, allow_equation=False)
    low, high = _setup_pair(setup, "bounds", "the search bounds")
    minimizer = result.values[0].value
    assert isinstance(minimizer, float)
    function = _as_function(parsed, symbol)
    step = 1e-6 * (high - low)
    at_min = function(minimizer)
    neighbors = [
        function(x) for x in (minimizer - step, minimizer + step) if low <= x <= high
    ]
    ok = low <= minimizer <= high and all(at_min <= n + 1e-9 * max(1.0, abs(n)) for n in neighbors)
    return VerificationCheck(
        name="local-optimality",
        ok=ok,
        detail=(
            f"f({minimizer:.10g}) = {at_min:.10g} ≤ its neighbors within the bounds"
            if ok
            else f"{minimizer:.10g} is not a local minimum on [{low:g}, {high:g}]; withholding"
        ),
    )


# --- ode.solve_ivp ---------------------------------------------------------------


def _ode_problem(setup: Mapping[str, Any]) -> tuple[Any, Any, Any, float, float, float]:
    raw = setup.get("expression")
    if not isinstance(raw, str) or not raw.strip():
        raise InputError("this solver needs setup 'expression' — dy/dt as f(t, y)")
    try:
        names = expr_symbols(raw)
    except ValueError as exc:
        raise ExecutionError(f"expression rejected: {exc}") from exc
    if extra := sorted(names - {"t", "y"}):
        raise InputError(
            f"an ODE right-hand side may reference only t and y; unexpected: {', '.join(extra)}"
        )
    parsed = parse_formula(raw, {"t", "y"})
    y0 = setup.get("y0")
    if not isinstance(y0, int | float) or isinstance(y0, bool):
        raise InputError("setup 'y0' (the initial value) must be a number")
    t0, t1 = _setup_pair(setup, "t_span", "the integration span")
    return parsed, sympy.Symbol("t"), sympy.Symbol("y"), float(y0), t0, t1


def _solve_ode(setup: Mapping[str, Any], method: str) -> float:
    parsed, t_sym, y_sym, y0, t0, t1 = _ode_problem(setup)

    def rhs(t: float, y: Any) -> list[float]:
        return [float(parsed.subs({t_sym: sympy.Float(t), y_sym: sympy.Float(y[0])}))]

    from scipy.integrate import solve_ivp

    solution = solve_ivp(rhs, (t0, t1), [y0], method=method, rtol=1e-10, atol=1e-12)
    if not solution.success:
        raise ExecutionError(f"ODE integration failed: {solution.message}")
    return float(solution.y[0, -1])


def _ode(setup: Mapping[str, Any]) -> ExecutionResult:
    value = _solve_ode(setup, "RK45")
    return ExecutionResult(
        output="y_end", values=[ExecutedValue(label="y(t1)", value=value)]
    )


def _check_ode(
    template: Template, result: ExecutionResult, setup: Mapping[str, Any]
) -> VerificationCheck:
    """Cross-method: a second integrator family (DOP853) must agree with RK45."""
    value = result.values[0].value
    assert isinstance(value, float)
    second = _solve_ode(setup, "DOP853")
    ok = abs(value - second) <= max(1e-6 * abs(second), 1e-9)
    return VerificationCheck(
        name="cross-method:dop853",
        ok=ok,
        detail=(
            f"RK45 {value:.10g} vs DOP853 {second:.10g}"
            + ("" if ok else " — the integrators disagree; withholding")
        ),
    )


# --- the registry -----------------------------------------------------------------

_Solver = Callable[[Mapping[str, Any]], ExecutionResult]
_Check = Callable[[Template, ExecutionResult, Mapping[str, Any]], VerificationCheck]

_SOLVERS: dict[str, tuple[_Solver, _Check]] = {
    "root_find.brentq": (_root_find, _check_root),
    "integrate.quad": (_integrate, _check_integral),
    "minimize.scalar_bounded": (_minimize, _check_minimum),
    "ode.solve_ivp": (_ode, _check_ode),
}


def solver_bindings() -> tuple[str, ...]:
    """The curated binding names — the whole universe of what ``kind: solver`` can run."""
    return tuple(sorted(_SOLVERS))


def _binding(template: Template) -> str:
    if not isinstance(template.method, SolverMethod):
        raise ExecutionError(f"template {template.id!r} is not a solver template")
    binding = template.method.binding
    if binding not in _SOLVERS:
        raise ExecutionError(
            f"unknown solver binding {binding!r} — curated solvers only"
            f" (available: {', '.join(solver_bindings())})"
        )
    return binding


def execute_solver(
    template: Template,
    inputs: Mapping[str, tuple[float, str] | list[tuple[float, str]]],
    setup: Mapping[str, Any],
) -> ExecutionResult:
    """Run a solver template's curated binding on its ``setup`` problem."""
    if inputs:
        raise InputError(
            "solver templates take no dimensioned inputs (v0) — the problem lives in setup"
        )
    solver, _check = _SOLVERS[_binding(template)]
    return solver(setup)


def verify_solver(
    template: Template, result: ExecutionResult, setup: Mapping[str, Any]
) -> VerificationCheck:
    """The binding's built-in independent check (A-6) — run by the verify stage."""
    _solver, check = _SOLVERS[_binding(template)]
    return check(template, result, setup)
