"""E1.5: the three golden templates — solve_equation, integrate, beam_deflection.

The done-criterion: **all fixtures green under the generic executor.** Plus the symbolic
path itself: curated operations on a gated ``setup`` problem, deterministic results, and
the built-in substitution/derivative verification.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from assay.execute import (
    ExecutedValue,
    ExecutionError,
    ExecutionResult,
    InputError,
    UnsafeExpressionError,
    execute_ir,
    execute_template,
    run_fixtures,
)
from assay.ir import IR
from assay.templates import Template, golden_templates
from assay.verify import _check_derivative, _check_substitution, verify_execution

_GOLDENS = {template.id: template for template in golden_templates()}


def test_the_shipped_goldens() -> None:
    """The E1.5 three, plus the E2.5 nucleus breadth (an exact set: additions are
    deliberate, deletions are regressions)."""
    assert set(_GOLDENS) == {
        # the deterministic spine's three
        "beam_deflection.simply_supported.center_point",
        "solve_equation.univariate",
        "integrate.univariate",
        # nucleus breadth (demo catalog)
        "differentiate.univariate",
        "projectile.range",
        "kinetic_energy.point_mass",
        "gravitational_potential_energy.point_mass",
        "pendulum.period.simple",
        "ideal_gas.pressure",
        "resistor.voltage_drop",
        "wave.speed",
        "axial_stress.bar",
        "molarity.solution",
        "molar_mass.from_sample",
        "escape_velocity.surface",
        "schwarzschild.radius",
        "heat.sensible",
        "rc.time_constant",
        "spring_mass.natural_frequency",
        "max_rectangle_area.fixed_perimeter",
        # the multi-step (DAG-of-assignments) shape
        "principal_stress.plane.max",
        # the curated solver bindings
        "root_find.univariate.numeric",
        "integrate.definite.numeric",
        "minimize.univariate.numeric",
        "ode.initial_value.numeric",
        # schema v2 + E2.13 demos: lists, cases, multi-output, limit, inequality
        "equivalent_resistance.series",
        "area.standard_shapes",
        "sample_stats.dataset",
        "limit.univariate",
        "inequality.univariate",
        "relativistic_kinetic_energy.point_mass",
    }


@pytest.mark.parametrize("template_id", sorted(_GOLDENS))
def test_all_golden_fixtures_are_green(template_id: str) -> None:
    """The E1.5 done-criterion, per golden."""
    results = run_fixtures(_GOLDENS[template_id])
    assert results and all(r.ok for r in results), [r.detail for r in results if not r.ok]


def _solve(expression: str, variable: str | None = None) -> ExecutionResult:
    setup: dict[str, str] = {"expression": expression}
    if variable is not None:
        setup["variable"] = variable
    return execute_template(_GOLDENS["solve_equation.univariate"], {}, setup=setup)


def test_solve_quadratic_roots() -> None:
    result = _solve("x**2 - 5*x + 6", "x")
    assert [v.value for v in result.values] == [2.0, 3.0]
    assert all(v.label == "x" for v in result.values)


def test_solve_equation_form_with_inferred_variable() -> None:
    result = _solve("y**2 + 3*y - 4 = 0")
    assert [v.value for v in result.values] == [-4.0, 1.0]
    assert result.values[0].label == "y"


def test_solve_keeps_irrational_roots_exact() -> None:
    result = _solve("x**2 - 2")
    assert [v.value for v in result.values] == ["-sqrt(2)", "sqrt(2)"]


def test_solve_via_ir() -> None:
    ir = IR.model_validate(
        {
            "domain": "algebra",
            "task": "solve_equation.univariate",
            "setup": {"expression": "x**2 - 5*x + 6", "variable": "x"},
        }
    )
    result = execute_ir(ir, _GOLDENS["solve_equation.univariate"])
    assert [v.value for v in result.values] == [2.0, 3.0]


def test_integrate_polynomial() -> None:
    result = execute_template(
        _GOLDENS["integrate.univariate"], {}, setup={"expression": "x**2", "variable": "x"}
    )
    assert result.value == "x**3/3"


def test_verify_solve_by_substitution() -> None:
    verified = verify_execution(
        _GOLDENS["solve_equation.univariate"], {}, setup={"expression": "x**2 - 5*x + 6"}
    )
    assert verified.verification.ok
    check = verified.verification.checks[0]
    assert check.name == "substitution" and "roots substitute to 0" in check.detail
    assert verified.result is not None


def test_verify_integrate_by_derivative() -> None:
    verified = verify_execution(
        _GOLDENS["integrate.univariate"], {}, setup={"expression": "sin(x)**2"}
    )
    assert verified.verification.ok
    check = verified.verification.checks[0]
    assert check.name == "derivative" and "equals the integrand" in check.detail


def test_substitution_check_rejects_a_wrong_root() -> None:
    doctored = ExecutionResult(
        output="roots",
        values=[ExecutedValue(label="x", value=2.0), ExecutedValue(label="x", value=4.0)],
    )
    check = _check_substitution(
        _GOLDENS["solve_equation.univariate"], doctored, {"expression": "x**2 - 5*x + 6"}
    )
    assert not check.ok and "does not satisfy the equation" in check.detail


def test_derivative_check_rejects_a_wrong_antiderivative() -> None:
    doctored = ExecutionResult(
        output="antiderivative", values=[ExecutedValue(label="antiderivative", value="x**3")]
    )
    check = _check_derivative(
        _GOLDENS["integrate.univariate"], doctored, {"expression": "x**2"}
    )
    assert not check.ok and "differs from the integrand" in check.detail


def test_multivariate_expression_fails_clear() -> None:
    with pytest.raises(InputError, match="specify setup 'variable'"):
        _solve("x*y + 1")


def test_stray_symbols_fail_clear() -> None:
    with pytest.raises(InputError, match="unexpected symbols: y"):
        _solve("x*y + 1", "x")


def test_malformed_equation_fails_clear() -> None:
    with pytest.raises(InputError, match="malformed equation"):
        _solve("x == 2")


def test_integrate_rejects_an_equation() -> None:
    with pytest.raises(InputError, match="expression, not an equation"):
        execute_template(
            _GOLDENS["integrate.univariate"], {}, setup={"expression": "x**2 = 1"}
        )


def test_no_closed_form_refuses_to_guess() -> None:
    with pytest.raises(ExecutionError, match="no closed-form"):
        execute_template(
            _GOLDENS["integrate.univariate"], {}, setup={"expression": "x**x", "variable": "x"}
        )


def test_hostile_setup_expression_is_rejected_not_run(tmp_path: Path) -> None:
    canary = tmp_path / "pwned"
    with pytest.raises(UnsafeExpressionError):
        _solve(f"__import__('os').system('touch {canary}')", "x")
    assert not canary.exists()


def test_failing_symbolic_fixture_reports_the_difference() -> None:
    golden = _GOLDENS["solve_equation.univariate"]
    doctored_fixture = golden.fixtures[0].model_copy(update={"expect": {"roots": [2.0, 4.0]}})
    results = run_fixtures(golden.model_copy(update={"fixtures": [doctored_fixture]}))
    assert not results[0].ok and "exceeds relative tol" in results[0].detail


def test_goldens_stay_candidates_until_the_gate() -> None:
    assert all(t.provenance.status == "candidate" for t in _GOLDENS.values())
    assert all(isinstance(t, Template) for t in _GOLDENS.values())
