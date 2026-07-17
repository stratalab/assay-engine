"""E1.2: the generic executor — safe parse, unit-bound evaluation, dimension check.

The done-criteria: the golden beam template evaluates to 0.50 mm; a formula string
attempting code execution is rejected, not run (the security fixture proves it).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest

from assay.execute import (
    DimensionError,
    ExecutionError,
    InputError,
    MissingInputError,
    UnsafeExpressionError,
    execute_ir,
    execute_template,
    run_fixtures,
)
from assay.ir import IR
from assay.templates import FormulaMethod, Template, golden_template, validate_template

_BEAM_INPUTS: dict[str, tuple[float, str]] = {
    "P": (5000, "N"),
    "L": (2, "m"),
    "E": (200e9, "Pa"),
    "I": (8.33e-6, "m**4"),
}
_BEAM_DEFLECTION_M = 5000 * 2**3 / (48 * 200e9 * 8.33e-6)  # ≈ 5.002e-4 m


def _mini_template(**overrides: Any) -> Template:
    """A tiny valid template to vary per test."""
    fields: dict[str, Any] = {
        "id": "test.case",
        "domain": "testing",
        "inputs": [{"name": "x", "dimension": "length"}],
        "method": {"kind": "formula", "expr": "x"},
        "output": {"name": "y", "dimension": "length"},
        "fixtures": [{"inputs": {"x": [1, "m"]}, "expect": {"y": [1, "m"]}, "tol": 1e-9}],
        "provenance": {"source": "test", "license_tier": "open"},
    }
    fields.update(overrides)
    return validate_template(fields)


def test_golden_beam_evaluates_to_half_a_millimetre() -> None:
    result = execute_template(golden_template(), _BEAM_INPUTS)
    assert result.output == "max_deflection"
    assert result.unit == "meter"
    assert result.value == pytest.approx(_BEAM_DEFLECTION_M, rel=1e-12)
    assert result.value * 1000 == pytest.approx(0.50, abs=0.005)  # 0.50 mm


def test_golden_fixtures_pass() -> None:
    results = run_fixtures(golden_template())
    assert results and all(r.ok for r in results), [r.detail for r in results]


def test_code_execution_attempt_is_rejected_not_run(tmp_path: Path) -> None:
    """Engineering §7: the security fixture — a crafted formula must never execute."""
    canary = tmp_path / "pwned"
    hostile = golden_template().model_copy(  # model_copy skips validation: worst case
        update={
            "method": FormulaMethod.model_construct(
                kind="formula", expr=f"__import__('os').system('touch {canary}')"
            )
        }
    )
    with pytest.raises(UnsafeExpressionError, match="rejected"):
        execute_template(hostile, _BEAM_INPUTS)
    assert not canary.exists()


def test_unparseable_formula_is_rejected_not_a_syntax_error() -> None:
    """Defense in depth: a "formula" that is not even Python must surface as a
    rejection, never a raw SyntaxError — on Windows the hostile-payload test above
    embeds backslashed paths and hits this path in the pre-gate reducer rewrite."""
    for payload in (r"'\U'", "P +", "F(=)"):
        hostile = golden_template().model_copy(
            update={"method": FormulaMethod.model_construct(kind="formula", expr=payload)}
        )
        with pytest.raises(UnsafeExpressionError, match="rejected"):
            execute_template(hostile, _BEAM_INPUTS)


def test_dunder_attribute_access_is_rejected() -> None:
    hostile = golden_template().model_copy(
        update={"method": FormulaMethod.model_construct(kind="formula", expr="P.__class__")}
    )
    with pytest.raises(UnsafeExpressionError):
        execute_template(hostile, _BEAM_INPUTS)


def test_undeclared_symbol_is_rejected() -> None:
    sneaky = golden_template().model_copy(
        update={"method": FormulaMethod.model_construct(kind="formula", expr="P * Q")}
    )
    with pytest.raises(UnsafeExpressionError, match="undeclared inputs: Q"):
        execute_template(sneaky, _BEAM_INPUTS)


def test_e_binds_to_the_input_not_eulers_number() -> None:
    """The locked namespace: ``E`` is the declared symbol, never SymPy's e = 2.718…"""
    template = _mini_template(
        inputs=[{"name": "E", "dimension": "dimensionless"}],
        method={"kind": "formula", "expr": "E"},
        output={"name": "y", "dimension": "dimensionless"},
        fixtures=[{"inputs": {"E": [3, ""]}, "expect": {"y": [3, ""]}, "tol": 1e-9}],
    )
    assert execute_template(template, {"E": (3, "")}).value == 3.0


def test_pi_is_available_and_correct() -> None:
    template = _mini_template(
        inputs=[{"name": "r", "dimension": "length"}],
        method={"kind": "formula", "expr": "2 * pi * r"},
        output={"name": "circumference", "dimension": "length"},
        fixtures=[
            {"inputs": {"r": [1, "m"]}, "expect": {"circumference": [6.283185307, "m"]},
             "tol": 1e-6}
        ],
    )
    assert execute_template(template, {"r": (1, "m")}).value == pytest.approx(2 * math.pi)


def test_sin_requires_dimensionless_argument() -> None:
    template = _mini_template(
        method={"kind": "formula", "expr": "sin(x)"},
        output={"name": "y", "dimension": "dimensionless"},
        fixtures=[{"inputs": {"x": [1, "m"]}, "expect": {"y": [0.84, ""]}, "tol": 1e-2}],
    )
    with pytest.raises(DimensionError, match="dimensionless argument"):
        execute_template(template, {"x": (1, "m")})


def test_sin_evaluates_on_dimensionless_input() -> None:
    template = _mini_template(
        inputs=[{"name": "theta", "dimension": "dimensionless"},
                {"name": "A", "dimension": "length"}],
        method={"kind": "formula", "expr": "A * sin(theta)"},
        output={"name": "displacement", "dimension": "length"},
        fixtures=[
            {"inputs": {"theta": [0.5, "rad"], "A": [2, "m"]},
             "expect": {"displacement": [2 * math.sin(0.5), "m"]}, "tol": 1e-9}
        ],
    )
    assert all(r.ok for r in run_fixtures(template))


def test_missing_required_input_fails_clear() -> None:
    inputs = dict(_BEAM_INPUTS)
    del inputs["I"]
    with pytest.raises(MissingInputError, match="missing required inputs.*: I"):
        execute_template(golden_template(), inputs)


def test_undeclared_input_is_rejected() -> None:
    with pytest.raises(InputError, match="not declared.*: W"):
        execute_template(golden_template(), {**_BEAM_INPUTS, "W": (1, "m")})


def test_input_with_wrong_dimension_is_rejected() -> None:
    with pytest.raises(InputError, match="'P' must have dimension 'force'"):
        execute_template(golden_template(), {**_BEAM_INPUTS, "P": (5000, "m")})


def test_unknown_unit_is_rejected() -> None:
    with pytest.raises(InputError, match="unknown unit 'flibbers'"):
        execute_template(golden_template(), {**_BEAM_INPUTS, "P": (5000, "flibbers")})


def test_unknown_dimension_name_is_rejected() -> None:
    """Since the round-4 §5 ruling this dies at VALIDATE time (the attested-token
    vocabulary), not at first execution — a strictly earlier, clearer refusal."""
    from assay.templates import TemplateValidationError

    with pytest.raises(TemplateValidationError, match="unknown dimension token 'flibber'"):
        _mini_template(inputs=[{"name": "x", "dimension": "flibber"}])


def test_result_dimension_contradicting_declaration_is_refused() -> None:
    golden = golden_template()
    wrong = golden.model_copy(
        update={"output": golden.output.model_copy(update={"dimension": "time"})}
    )
    with pytest.raises(DimensionError, match="refusing to return it"):
        execute_template(wrong, _BEAM_INPUTS)


def test_division_by_zero_fails_clear() -> None:
    template = _mini_template(
        inputs=[{"name": "a", "dimension": "length"}, {"name": "b", "dimension": "length"}],
        method={"kind": "formula", "expr": "a / b"},
        output={"name": "y", "dimension": "dimensionless"},
        fixtures=[{"inputs": {"a": [1, "m"], "b": [2, "m"]}, "expect": {"y": [0.5, ""]},
                   "tol": 1e-9}],
    )
    with pytest.raises(ExecutionError, match="evaluation failed"):
        execute_template(template, {"a": (1, "m"), "b": (0, "m")})


def test_unknown_solver_binding_fails_clear() -> None:
    """E3.5: a binding is a NAME in the curated registry, never an import path — an
    unknown one refuses with the available universe named (so it can't pass the gate)."""
    template = _mini_template(method={"kind": "solver", "binding": "scipy.optimize.brentq"})
    with pytest.raises(ExecutionError, match="curated solvers only"):
        execute_template(template, {}, setup={"expression": "x - 1"})


def test_failing_fixture_reports_the_difference() -> None:
    golden = golden_template()
    bad_fixture = golden.fixtures[0].model_copy(
        update={"expect": {"max_deflection": (999.0, "m")}}
    )
    results = run_fixtures(golden.model_copy(update={"fixtures": [bad_fixture]}))
    assert not results[0].ok
    assert "exceeds relative tol" in results[0].detail


def test_execute_ir_end_to_end() -> None:
    """A-1: a validated IR (inputs + resolved facts) executes against its template."""
    golden = golden_template()
    ir = IR.model_validate(
        {
            "domain": "structural_mechanics",
            "task": golden.id,
            "inputs": {
                "P": {"value": 5000, "unit": "N"},
                "L": {"value": 2, "unit": "m"},
                "I": {"value": 8.33e-6, "unit": "m**4"},
            },
            "resolved": {
                "E": {
                    "value": 200e9,
                    "unit": "Pa",
                    "source": {
                        "library": "assay.materials",
                        "key": "steel.structural.E",
                        "version": "0.1",
                    },
                }
            },
        }
    )
    result = execute_ir(ir, golden)
    assert result.value == pytest.approx(_BEAM_DEFLECTION_M, rel=1e-12)


def test_execute_ir_with_missing_inputs_fails_clear_never_fabricates() -> None:
    golden = golden_template()
    ir = IR.model_validate(
        {
            "domain": "structural_mechanics",
            "task": golden.id,
            "inputs": {"P": {"value": 5000, "unit": "N"}, "L": {"value": 2, "unit": "m"}},
            "missing_inputs": ["E", "I"],
        }
    )
    with pytest.raises(MissingInputError, match="E, I.*fabricated"):
        execute_ir(ir, golden)


def test_execute_ir_task_must_name_the_template() -> None:
    ir = IR.model_validate({"domain": "d", "task": "some.other.task"})
    with pytest.raises(ExecutionError, match="does not name template"):
        execute_ir(ir, golden_template())


def test_execute_ir_rejects_supplied_and_resolved_conflict() -> None:
    golden = golden_template()
    ir = IR.model_validate(
        {
            "domain": "structural_mechanics",
            "task": golden.id,
            "inputs": {
                "P": {"value": 5000, "unit": "N"},
                "L": {"value": 2, "unit": "m"},
                "I": {"value": 8.33e-6, "unit": "m**4"},
                "E": {"value": 1, "unit": "Pa"},
            },
            "resolved": {
                "E": {
                    "value": 200e9,
                    "unit": "Pa",
                    "source": {
                        "library": "assay.materials",
                        "key": "steel.structural.E",
                        "version": "0.1",
                    },
                }
            },
        }
    )
    with pytest.raises(InputError, match="both user-supplied and resolved: E"):
        execute_ir(ir, golden)
