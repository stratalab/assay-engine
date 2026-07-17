"""E3.5: ``kind: solver`` — curated bindings, safely (PRD §7.1, engineering §7).

The discipline under test: a binding is a *name* in a whitelist (never an import path,
never model-influenced code); every problem passes the same safe-parse gate; every
binding carries a built-in independent check that withholds on disagreement; and the
quintic that the symbolic path honestly withheld now gets a verified numeric answer.
"""

from __future__ import annotations

import pytest

from assay.execute import ExecutionError, InputError, UnsafeExpressionError
from assay.execute.solvers import execute_solver, solver_bindings, verify_solver
from assay.templates import Template, golden_templates
from assay.verify import verify_execution

_GOLDENS = {template.id: template for template in golden_templates()}
_ROOT = _GOLDENS["root_find.univariate.numeric"]
_QUAD = _GOLDENS["integrate.definite.numeric"]
_MIN = _GOLDENS["minimize.univariate.numeric"]
_ODE = _GOLDENS["ode.initial_value.numeric"]


def test_the_registry_is_the_whole_universe() -> None:
    assert solver_bindings() == (
        "integrate.quad",
        "minimize.scalar_bounded",
        "ode.solve_ivp",
        "root_find.brentq",
    )


def test_the_quintic_now_answers_verified() -> None:
    """The E3.5 headline: x^5 - x + 1 = 0 has no closed form (the symbolic path
    withholds); the bracketing solver answers it, verified by substitution."""
    verified = verify_execution(
        _ROOT, {}, setup={"expression": "x**5 - x + 1 = 0", "bracket": [-2, 0]}
    )
    assert verified.verification.ok
    assert verified.result is not None
    root = verified.result.values[0].value
    assert isinstance(root, float)
    assert root == pytest.approx(-1.1673039782614187, rel=1e-9)
    assert verified.verification.checks[0].name == "substitution"


def test_no_sign_change_refuses_to_guess() -> None:
    with pytest.raises(ExecutionError, match="no sign change"):
        execute_solver(_ROOT, {}, {"expression": "x**2 + 1", "bracket": [-1, 1]})


def test_definite_integral_cross_checked_by_simpson() -> None:
    verified = verify_execution(
        _QUAD, {}, setup={"expression": "exp(-x**2)", "limits": [0, 1]}
    )
    assert verified.verification.ok
    assert verified.result is not None
    value = verified.result.values[0].value
    assert isinstance(value, float)
    assert value == pytest.approx(0.7468241328124271, rel=1e-9)  # erf-based reference
    assert verified.verification.checks[0].name == "cross-method:simpson"


def test_minimizer_verified_by_local_optimality() -> None:
    verified = verify_execution(
        _MIN, {}, setup={"expression": "sin(x) + x**2 / 10", "bounds": [-3, 3]}
    )
    assert verified.verification.ok
    assert verified.result is not None
    minimizer = verified.result.values[0].value
    assert isinstance(minimizer, float)
    assert verified.verification.checks[0].name == "local-optimality"


def test_ode_cross_checked_by_second_integrator() -> None:
    verified = verify_execution(
        _ODE, {}, setup={"expression": "-2 * t * y", "y0": 1, "t_span": [0, 1]}
    )
    assert verified.verification.ok  # y(1) = e^-1 for dy/dt = -2ty
    assert verified.result is not None
    value = verified.result.values[0].value
    assert isinstance(value, float)
    assert value == pytest.approx(0.36787944117144233, rel=1e-7)
    assert verified.verification.checks[0].name == "cross-method:dop853"


def test_hostile_expressions_are_gated_before_any_solver_runs() -> None:
    hostile = "__import__('os').system('true')"
    with pytest.raises(UnsafeExpressionError):
        execute_solver(_ROOT, {}, {"expression": hostile, "bracket": [-1, 1]})
    with pytest.raises((UnsafeExpressionError, ExecutionError)):
        execute_solver(_ODE, {}, {"expression": hostile, "y0": 1, "t_span": [0, 1]})


def test_ode_rhs_may_reference_only_t_and_y() -> None:
    with pytest.raises(InputError, match="only t and y"):
        execute_solver(_ODE, {}, {"expression": "-k * y", "y0": 1, "t_span": [0, 1]})


def test_malformed_setup_fails_clear() -> None:
    with pytest.raises(InputError, match="bracket"):
        execute_solver(_ROOT, {}, {"expression": "x - 1", "bracket": [1]})
    with pytest.raises(InputError, match="empty"):
        execute_solver(_ROOT, {}, {"expression": "x - 1", "bracket": [2, 2]})
    with pytest.raises(InputError, match="y0"):
        execute_solver(_ODE, {}, {"expression": "-y", "y0": "one", "t_span": [0, 1]})
    with pytest.raises(InputError, match="no dimensioned inputs"):
        execute_solver(_ROOT, {"x": (1.0, "m")}, {"expression": "x - 1", "bracket": [-2, 0]})


def test_a_doctored_result_is_withheld() -> None:
    """The check judges the result, not the solver's say-so."""
    from assay.execute import ExecutedValue, ExecutionResult

    doctored = ExecutionResult(output="root", values=[ExecutedValue(label="x", value=0.5)])
    check = verify_solver(
        _ROOT, doctored, {"expression": "x**5 - x + 1 = 0", "bracket": [-2, 0]}
    )
    assert not check.ok and "does not satisfy" in check.detail


def test_solver_templates_are_full_citizens(tmp_path: object) -> None:
    """Fixture gate + artifact + rerun — the same lifecycle as every other kind."""
    from pathlib import Path

    from assay.artifact import create_artifact, load_artifact, rerun, save_artifact
    from assay.ir import IR
    from assay.templates.promote import promote

    assert promote(_ROOT).provenance.status == "verified"
    ir = IR.model_validate(
        {
            "domain": "numerical_methods",
            "task": _ROOT.id,
            "setup": {"expression": "x**5 - x + 1 = 0", "bracket": [-2, 0]},
        }
    )
    artifact = create_artifact(ir, _ROOT)
    assert artifact.answer.verified.ok
    path = save_artifact(artifact, Path(str(tmp_path)) / "quintic.result.json")
    assert rerun(load_artifact(path)).status == "exact"


def test_every_solver_golden_is_gated_and_typed() -> None:
    for template in (_ROOT, _QUAD, _MIN, _ODE):
        assert isinstance(template, Template)
        assert template.method.kind == "solver"
        assert template.provenance.status == "candidate"  # trust is earned, as ever