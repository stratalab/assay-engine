"""Chisel round 8 (the calculus walk): the two shipped findings.

Finding 1: ``integrate.quad`` now accepts improper bounds (``"oo"``/``"-oo"``) —
SciPy handles the infinite range; the Simpson cross-check transforms it to a finite
interval so the verification stays genuinely two-method. Finding 3: the
``parametric_slope`` operation — dy/dx = (dy/dt)/(dx/dt), symbolic or at a point,
verified by the central difference along the curve; polar tangents encode as
x = r(θ)cos(θ), y = r(θ)sin(θ) through the same operation.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from assay.execute import ExecutionError, execute_template, run_fixtures
from assay.templates import validate_template
from assay.verify import verify_execution

_PROV = {"source": "assay:demo", "license_tier": "open"}


def _quad(fixtures: list[dict[str, Any]]) -> Any:
    return validate_template({
        "id": "integrate.definite.numeric.exhibit",
        "domain": "calculus",
        "description": "definite integral by quadrature",
        "method": {"kind": "solver", "binding": "integrate.quad"},
        "output": {"name": "integral", "dimension": "dimensionless"},
        "fixtures": fixtures,
        "provenance": _PROV,
    })


def test_improper_integrals_through_the_quad_solver() -> None:
    """The parked-fixtures shape: infinite bounds on the SOLVER path, verified."""
    template = _quad([
        {"setup": {"expression": "exp(-x)", "limits": [0, "oo"]},
         "expect": {"integral": [1.0, ""]}, "tol": 1e-9},
        {"setup": {"expression": "1/x**2", "limits": [1, "oo"]},
         "expect": {"integral": [1.0, ""]}, "tol": 1e-9},
        {"setup": {"expression": "exp(x)", "limits": ["-oo", 0]},
         "expect": {"integral": [1.0, ""]}, "tol": 1e-9},
        {"setup": {"expression": "1/(1 + x**2)", "limits": ["-oo", "oo"]},
         "expect": {"integral": [math.pi, ""]}, "tol": 1e-9},
        {"setup": {"expression": "exp(-x**2)", "limits": ["-oo", "oo"]},
         "expect": {"integral": [math.sqrt(math.pi), ""]}, "tol": 1e-9},
    ])
    assert all(r.ok for r in run_fixtures(template))
    verified = verify_execution(
        template, {}, setup={"expression": "exp(-x)", "limits": [0, "oo"]}
    )
    assert verified.verification.ok
    assert "transformed to a finite interval" in verified.verification.checks[0].detail


def test_finite_quad_is_unchanged() -> None:
    template = _quad([
        {"setup": {"expression": "x**2", "limits": [0, 3]},
         "expect": {"integral": [9.0, ""]}, "tol": 1e-9},
    ])
    assert all(r.ok for r in run_fixtures(template))
    verified = verify_execution(template, {}, setup={"expression": "x**2", "limits": [0, 3]})
    assert verified.verification.ok
    assert "transformed" not in verified.verification.checks[0].detail


def _slope(fixtures: list[dict[str, Any]]) -> Any:
    return validate_template({
        "id": "slope.parametric_tangent",
        "domain": "calculus",
        "description": "Slope of the tangent to a parametric curve.",
        "method": {"kind": "symbolic", "operation": "parametric_slope"},
        "output": {"name": "slope", "dimension": "dimensionless"},
        "fixtures": fixtures,
        "provenance": _PROV,
    })


def test_parametric_slope_symbolic_and_at_a_point() -> None:
    template = _slope([
        # the cycloid x = t - sin(t), y = 1 - cos(t): dy/dx = sin(t)/(1 - cos(t))
        {"setup": {"x_expression": "t - sin(t)", "y_expression": "1 - cos(t)"},
         "expect": {"slope": "sin(t) / (1 - cos(t))"}},
        # the unit circle at t = pi/4: dy/dx = -cot(t) -> -1
        {"setup": {"x_expression": "cos(t)", "y_expression": "sin(t)",
                   "point": 0.7853981633974483},
         "expect": {"slope": [-1.0]}},
    ])
    assert all(r.ok for r in run_fixtures(template))
    verified = verify_execution(
        template, {},
        setup={"x_expression": "cos(t)", "y_expression": "sin(t)", "point": 0.7853981633974483},
    )
    assert verified.verification.ok
    assert verified.verification.checks[0].name == "parametric-difference"


def test_polar_tangent_through_the_same_operation() -> None:
    """The round-8 polar answer: r = 1 + cos(θ) (the cardioid) encodes as
    x = r cos θ, y = r sin θ — at θ = pi/2: dy/dx = 1."""
    template = _slope([
        {"setup": {
            "x_expression": "(1 + cos(t)) * cos(t)",
            "y_expression": "(1 + cos(t)) * sin(t)",
            "point": 1.5707963267948966,
        }, "expect": {"slope": [1.0]}},
    ])
    assert all(r.ok for r in run_fixtures(template))


def test_parametric_second_derivative_exhibits() -> None:
    """The round-8 follow-up (m53850, §7.2): order 2 is d²y/dx² =
    d/dt(dy/dx) / (dx/dt) — NOT (d²y/dt²)/(d²x/dt²), the standard trap."""
    template = _slope([
        # exhibit 1: x = t²−3, y = 2t−1 → -1/(2t³)
        {"setup": {"x_expression": "t**2 - 3", "y_expression": "2*t - 1", "order": 2},
         "expect": {"slope": "-1/(2*t**3)"}},
        # exhibit 2: x = t²−4t, y = 2t³−6t
        {"setup": {"x_expression": "t**2 - 4*t", "y_expression": "2*t**3 - 6*t",
                   "order": 2},
         "expect": {"slope": "(3*t**2 - 12*t + 3)/(2*(t - 2)**3)"}},
        # exhibit 3: a straight line — the exact-zero path
        {"setup": {"x_expression": "4*t", "y_expression": "3*t - 2",
                   "order": 2, "point": 3},
         "expect": {"slope": [0.0]}},
        # exhibit 4: x = 2 + sec θ, y = 1 + 2 tan θ at θ = π/6 → -6√3
        {"setup": {"x_expression": "2 + 1/cos(t)", "y_expression": "1 + 2*sin(t)/cos(t)",
                   "order": 2, "point": 0.5235987755982988},
         "expect": {"slope": [-10.392304845413264]}},
    ])
    assert all(r.ok for r in run_fixtures(template))
    verified = verify_execution(
        template, {},
        setup={"x_expression": "2 + 1/cos(t)", "y_expression": "1 + 2*sin(t)/cos(t)",
               "order": 2, "point": 0.5235987755982988},
    )
    assert verified.verification.ok  # the chord-of-chords second difference agrees


def test_order_is_gated() -> None:
    with pytest.raises(Exception, match="'order' must be 1"):
        _slope([
            {"setup": {"x_expression": "t", "y_expression": "t**2", "order": 3},
             "expect": {"slope": [0.0]}},
        ])


def test_vertical_tangent_refuses_by_name() -> None:
    template = _slope([
        {"setup": {"x_expression": "cos(t)", "y_expression": "sin(t)", "point": 0.7853981633974483},
         "expect": {"slope": [-1.0]}},
    ])
    with pytest.raises(ExecutionError, match="vertical tangent"):
        execute_template(
            template, {},
            setup={"x_expression": "cos(t)", "y_expression": "sin(t)", "point": 0.0},
        )
