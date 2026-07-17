"""E1.4: the verification stage — dimensional, bounds, cross-method; withhold-with-reason.

The done-criteria: a wrong-units template fails the dimensional check; a cross-method
disagreement withholds the answer.
"""

from __future__ import annotations

from typing import Any

import pytest

from assay.execute import InputError, MissingInputError
from assay.ir import IR
from assay.templates import Bounds, VerificationHooks, golden_template, validate_template
from assay.verify import VerifiedExecution, verify_execution, verify_ir

_BEAM_INPUTS: dict[str, tuple[float, str]] = {
    "P": (5000, "N"),
    "L": (2, "m"),
    "E": (200e9, "Pa"),
    "I": (8.33e-6, "m**4"),
}


def _mini(**overrides: Any) -> Any:
    fields: dict[str, Any] = {
        "id": "test.case",
        "domain": "testing",
        "inputs": [{"name": "x", "dimension": "length"}],
        "method": {"kind": "formula", "expr": "x * 2"},
        "output": {"name": "y", "dimension": "length"},
        "fixtures": [{"inputs": {"x": [1, "m"]}, "expect": {"y": [2, "m"]}, "tol": 1e-9}],
        "provenance": {"source": "test", "license_tier": "open"},
    }
    fields.update(overrides)
    return validate_template(fields)


def _check(verified: VerifiedExecution, name: str) -> Any:
    matches = [c for c in verified.verification.checks if c.name == name]
    assert matches, f"no check named {name!r} in {verified.verification.checks}"
    return matches[0]


def test_golden_beam_verifies_green() -> None:
    verified = verify_execution(golden_template(), _BEAM_INPUTS)
    assert verified.verification.ok
    assert _check(verified, "dimension:length").ok
    assert _check(verified, "bounds").ok
    assert verified.result is not None and verified.result == verified.candidate
    assert verified.result.value * 1000 == pytest.approx(0.50, abs=0.005)


def test_wrong_units_template_fails_the_dimensional_check() -> None:
    wrong = _mini(
        output={"name": "y", "dimension": "time"},
        fixtures=[{"inputs": {"x": [1, "m"]}, "expect": {"y": [2, "s"]}, "tol": 1e-9}],
    )
    verified = verify_execution(wrong, {"x": (1, "m")})
    assert not verified.verification.ok
    check = _check(verified, "dimension:time")
    assert not check.ok and "refusing to return it" in check.detail
    assert verified.result is None
    assert verified.candidate is None  # a wrong-dimension value is never handed out at all


def test_cross_method_disagreement_withholds() -> None:
    lying = _mini(verification={"cross_method": "x * 3"})
    verified = verify_execution(lying, {"x": (1, "m")})
    assert not verified.verification.ok
    check = _check(verified, "cross-method")
    assert not check.ok
    assert "2 meter vs independent method 3 meter" in check.detail
    assert "a template bug, not your input" in check.detail
    assert verified.result is None  # withheld …
    assert verified.candidate is not None  # … but explicitly reachable (--unsafe, UX §5.6)


def test_cross_method_agreement_passes() -> None:
    golden = golden_template()
    hooks = VerificationHooks(
        bounds=golden.verification.bounds,
        cross_method="(P * L * L * L) / (48 * E * I)",  # independently keyed-in form
    )
    verified = verify_execution(golden.model_copy(update={"verification": hooks}), _BEAM_INPUTS)
    assert verified.verification.ok
    assert len(verified.verification.checks) == 3
    assert _check(verified, "cross-method").ok


def test_bounds_violation_withholds_the_absurd() -> None:
    golden = golden_template()
    tight = golden.model_copy(
        update={"verification": VerificationHooks(bounds=Bounds(min=0.0, max=1e-6, unit="m"))}
    )
    verified = verify_execution(tight, _BEAM_INPUTS)
    assert not verified.verification.ok
    check = _check(verified, "bounds")
    assert not check.ok and "outside the plausible range" in check.detail
    assert verified.result is None and verified.candidate is not None


def test_bounds_convert_to_their_declared_unit() -> None:
    golden = golden_template()
    in_mm = golden.model_copy(
        update={"verification": VerificationHooks(bounds=Bounds(min=0.0, max=1000.0, unit="mm"))}
    )
    verified = verify_execution(in_mm, _BEAM_INPUTS)
    assert _check(verified, "bounds").ok
    assert "mm" in _check(verified, "bounds").detail


def test_min_only_bounds() -> None:
    golden = golden_template()
    floor = golden.model_copy(
        update={"verification": VerificationHooks(bounds=Bounds(min=0.0, unit="m"))}
    )
    assert verify_execution(floor, _BEAM_INPUTS).verification.ok


def test_incompatible_bounds_unit_is_a_template_bug() -> None:
    golden = golden_template()
    broken = golden.model_copy(
        update={"verification": VerificationHooks(bounds=Bounds(min=0.0, max=1.0, unit="s"))}
    )
    verified = verify_execution(broken, _BEAM_INPUTS)
    check = _check(verified, "bounds")
    assert not check.ok and "incompatible" in check.detail and "template bug" in check.detail
    assert verified.result is None


def test_cross_method_that_cannot_evaluate_fails_the_check() -> None:
    broken = _mini(
        inputs=[{"name": "x", "dimension": "length"}, {"name": "t", "dimension": "time"}],
        method={"kind": "formula", "expr": "x * 2"},
        verification={"cross_method": "x + t"},  # dimensionally impossible
        fixtures=[
            {"inputs": {"x": [1, "m"], "t": [1, "s"]}, "expect": {"y": [2, "m"]}, "tol": 1e-9}
        ],
    )
    verified = verify_execution(broken, {"x": (1, "m"), "t": (1, "s")})
    check = _check(verified, "cross-method")
    assert not check.ok and "failed to evaluate" in check.detail
    assert verified.result is None


def test_verify_ir_end_to_end() -> None:
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
    verified = verify_ir(ir, golden)
    assert verified.verification.ok and verified.result is not None


def test_execution_errors_are_not_verdicts() -> None:
    """Failures to compute raise (UX §3 error state); they never masquerade as checks."""
    golden = golden_template()
    with pytest.raises(InputError, match="unknown unit"):
        verify_execution(golden, {**_BEAM_INPUTS, "P": (5000, "flibbers")})
    ir = IR.model_validate(
        {"domain": "d", "task": golden.id, "missing_inputs": ["E", "I", "L", "P"]}
    )
    with pytest.raises(MissingInputError, match="fabricated"):
        verify_ir(ir, golden)
