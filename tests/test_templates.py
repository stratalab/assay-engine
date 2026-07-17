"""E1.1: the template schema + ``validate_template()`` — the Chisel seam (A-3, A-15).

The done-criteria: the golden beam template validates; a malformed template is rejected
with a clear reason (every reason, in one error — the seam's answer to Chisel).
"""

from __future__ import annotations

import subprocess
import sys
from typing import Any

import pytest

from assay.templates import (
    FormulaMethod,
    Template,
    TemplateValidationError,
    golden_template,
    validate_template,
)


def _record(**overrides: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "id": "beam_deflection.simply_supported.center_point",
        "domain": "structural_mechanics",
        "description": "Max deflection of a simply supported beam, central point load.",
        "inputs": [
            {"name": "P", "dimension": "force"},
            {"name": "L", "dimension": "length"},
            {"name": "E", "dimension": "pressure"},
            {"name": "I", "dimension": "length**4"},
        ],
        "method": {"kind": "formula", "expr": "P * L**3 / (48 * E * I)"},
        "output": {"name": "max_deflection", "dimension": "length"},
        "assumptions": ["euler_bernoulli", "small_deflection", "linear_elastic"],
        "fixtures": [
            {
                "inputs": {
                    "P": [5000, "N"],
                    "L": [2, "m"],
                    "E": [200e9, "Pa"],
                    "I": [8.33e-6, "m**4"],
                },
                "expect": {"max_deflection": [5.0e-4, "m"]},
                "tol": 1e-6,
            }
        ],
        "provenance": {"source": "exam:mom-101", "license_tier": "lawful"},
    }
    fields.update(overrides)
    return fields


def _reject(record: dict[str, Any], match: str) -> None:
    with pytest.raises(TemplateValidationError, match=match):
        validate_template(record)


def test_golden_template_validates() -> None:
    template = golden_template()
    assert template.id == "beam_deflection.simply_supported.center_point"
    assert template.provenance.status == "candidate"  # promotion is the fixture gate (E2.2)
    assert template.provenance.license_tier == "open"
    hints = {inp.name: inp.resolve for inp in template.inputs}
    assert hints["E"] is not None and hints["E"].key == "{material}.E"  # E1.3: E resolves
    assert hints["I"] is None  # I is user-supplied only


def test_resolve_hint_validates_and_round_trips() -> None:
    record = _record()
    record["inputs"][2]["resolve"] = {"library": "assay.materials", "key": "{material}.E"}
    template = validate_template(record)
    assert Template.model_validate_json(template.model_dump_json()) == template


def test_rejects_malformed_resolve_key_patterns() -> None:
    for bad in ("{material", "{0}.E", "{a.b}.E", "{material!r}.E", "{material:>8}.E"):
        record = _record()
        record["inputs"][2]["resolve"] = {"library": "assay.materials", "key": bad}
        _reject(record, match="key pattern|malformed")


def test_validates_and_round_trips_json() -> None:
    template = validate_template(_record())
    assert Template.model_validate_json(template.model_dump_json()) == template


def test_defaults_fail_closed() -> None:
    """An undeclared license tier defaults to 'unknown' — and the seam refuses
    'unknown' outright (round 2), so a record that never declared its tier cannot
    validate at all. Trust status still defaults to candidate."""
    _reject(
        _record(provenance={"source": "exam:mom-101"}),
        match="license_tier 'unknown' is refused",
    )
    template = validate_template(
        _record(provenance={"source": "exam:mom-101", "license_tier": "lawful"})
    )
    assert template.provenance.status == "candidate"
    assert all(inp.required for inp in template.inputs)


def test_solver_method_validates() -> None:
    template = validate_template(
        _record(method={"kind": "solver", "binding": "scipy.optimize.brentq"})
    )
    assert template.method.kind == "solver"


def test_rejects_unknown_field() -> None:
    _reject(_record(bogus=1), match="bogus")


def test_rejects_missing_method_with_a_clear_reason() -> None:
    record = _record()
    del record["method"]
    _reject(record, match="method")


def test_rejects_unknown_schema_version() -> None:
    _reject(_record(schema_version=3), match="schema_version")


def test_rejects_bad_id_and_domain() -> None:
    _reject(_record(id="Beam.Deflection"), match="id")
    _reject(_record(domain="structural mechanics"), match="domain")


def test_rejects_bad_status_and_license_tier() -> None:
    _reject(_record(provenance={"source": "s", "status": "trusted"}), match="status")
    _reject(_record(provenance={"source": "s", "license_tier": "gpl"}), match="license_tier")


def test_rejects_input_names_shadowing_the_safe_namespace() -> None:
    record = _record()
    record["inputs"].append({"name": "pi", "dimension": "dimensionless"})
    _reject(record, match="shadow the safe expression namespace: pi")


def test_rejects_duplicate_input_names() -> None:
    record = _record()
    record["inputs"].append({"name": "P", "dimension": "force"})
    # the duplicate P also makes fixture coverage ambiguous — the dupe is the named reason
    _reject(record, match="duplicate input names: P")


def test_rejects_expr_referencing_undeclared_inputs() -> None:
    _reject(
        _record(method={"kind": "formula", "expr": "P * L**3 / (48 * E * J)"}),
        match="undeclared names: J",
    )


def _steps_method(*steps: tuple[str, str]) -> dict[str, Any]:
    return {"kind": "formula", "steps": [{"name": n, "expr": e} for n, e in steps]}


def test_multi_step_method_validates_and_round_trips() -> None:
    """E2.9: the DAG of assignments — each step gated, later steps see earlier ones."""
    record = _record(
        method=_steps_method(("stiffness", "48 * E * I / L**3"), ("d", "P / stiffness"))
    )
    template = validate_template(record)
    assert isinstance(template.method, FormulaMethod)
    assert [step.name for step in template.method.steps] == ["stiffness", "d"]
    assert Template.model_validate_json(template.model_dump_json()) == template


def test_steps_are_ordered_no_forward_references() -> None:
    _reject(
        _record(method=_steps_method(("d", "P / stiffness"), ("stiffness", "48 * E * I / L**3"))),
        match=r"steps\[0\] \(d\) references undeclared names: stiffness",
    )


def test_step_names_cannot_shadow_or_repeat() -> None:
    _reject(
        _record(method=_steps_method(("P", "E * I"), ("d", "P"))),
        match="shadows an input",
    )
    _reject(
        _record(method=_steps_method(("pi", "E * I"), ("d", "pi"))),
        match="shadows the safe namespace",
    )
    _reject(
        _record(method=_steps_method(("d", "E * I"), ("d", "P * L"))),
        match="duplicate step name",
    )


def test_exactly_one_of_expr_or_steps() -> None:
    both = _record(
        method={"kind": "formula", "expr": "P * L", "steps": [{"name": "a", "expr": "P"}]}
    )
    _reject(both, match="exactly one of expr or steps")
    _reject(_record(method={"kind": "formula"}), match="exactly one of expr or steps")


def test_step_expressions_pass_the_same_gate() -> None:
    _reject(
        _record(method=_steps_method(("a", "__import__('os').system('true')"))),
        match="safe math functions|disallowed",
    )


def test_fixtures_must_cover_inputs_used_by_any_step() -> None:
    record = _record(method=_steps_method(("a", "E * I"), ("d", "a * P * L")))
    del record["fixtures"][0]["inputs"]["I"]  # used only inside step 'a'
    _reject(record, match=r"fixtures\[0\] is missing inputs: I")


def test_rejects_expr_syntax_error() -> None:
    _reject(_record(method={"kind": "formula", "expr": "P * / L"}), match="not a valid expression")


def test_rejects_code_execution_attempt_in_expr() -> None:
    """Engineering §7: a crafted formula must fail validation, never run."""
    _reject(
        _record(method={"kind": "formula", "expr": "__import__('os').system('true')"}),
        match="disallowed|safe math functions",
    )
    _reject(
        _record(method={"kind": "formula", "expr": "P.__class__"}),
        match="disallowed construct",
    )


def test_rejects_call_to_non_whitelisted_function() -> None:
    _reject(
        _record(method={"kind": "formula", "expr": "eval(P)"}),
        match="safe math functions",
    )


def test_rejects_bad_dimension() -> None:
    record = _record()
    record["inputs"][0]["dimension"] = "force**"
    _reject(record, match="invalid dimension")
    record = _record()
    record["inputs"][0]["dimension"] = "force**2.5"
    _reject(record, match="integer exponents")


def test_rejects_fixture_missing_a_required_input() -> None:
    record = _record()
    del record["fixtures"][0]["inputs"]["I"]
    _reject(record, match=r"fixtures\[0\] is missing inputs: I")


def test_rejects_fixture_with_undeclared_input() -> None:
    record = _record()
    record["fixtures"][0]["inputs"]["W"] = [1, "m"]
    _reject(record, match=r"fixtures\[0\] provides undeclared inputs: W")


def test_rejects_fixture_expect_not_matching_output() -> None:
    record = _record()
    record["fixtures"][0]["expect"] = {"deflection": [5.0e-4, "m"]}
    _reject(record, match="expect must include the declared output")


def test_rejects_nonpositive_tol_and_empty_fixtures() -> None:
    record = _record()
    record["fixtures"][0]["tol"] = 0
    _reject(record, match="tol")
    _reject(_record(fixtures=[]), match="fixtures")


def test_rejects_bad_bounds() -> None:
    _reject(_record(verification={"bounds": {"unit": "m"}}), match="min, max, or both")
    _reject(_record(verification={"bounds": {"min": 2.0, "max": 1.0}}), match="min must be <")


def test_cross_method_expr_is_checked_like_the_formula() -> None:
    _reject(
        _record(verification={"cross_method": "Q * L"}),
        match="verification.cross_method references undeclared names: Q",
    )


def test_error_message_lists_every_reason() -> None:
    record = _record(id="BAD ID", domain="bad domain")
    with pytest.raises(TemplateValidationError) as excinfo:
        validate_template(record)
    message = str(excinfo.value)
    assert "'BAD ID' is invalid" in message
    assert message.count("\n  - ") >= 2  # one line per violation


def _symbolic_record(**overrides: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "id": "solve_equation.univariate",
        "domain": "algebra",
        "inputs": [],
        "method": {"kind": "symbolic", "operation": "solve"},
        "output": {"name": "roots", "dimension": "dimensionless"},
        "fixtures": [
            {"setup": {"expression": "x**2 - 4"}, "expect": {"roots": [-2, 2]}, "tol": 1e-9}
        ],
        "provenance": {"source": "test", "license_tier": "open"},
    }
    fields.update(overrides)
    return fields


def test_symbolic_template_validates_and_round_trips() -> None:
    template = validate_template(_symbolic_record())
    assert Template.model_validate_json(template.model_dump_json()) == template


def test_symbolic_rejects_dimensioned_inputs() -> None:
    _reject(
        _symbolic_record(inputs=[{"name": "x", "dimension": "length"}]),
        match="no dimensioned inputs",
    )


def test_symbolic_output_must_be_dimensionless() -> None:
    _reject(
        _symbolic_record(output={"name": "roots", "dimension": "length"}),
        match="must be dimensionless",
    )


def test_symbolic_rejects_declarative_hooks() -> None:
    _reject(
        _symbolic_record(verification={"bounds": {"min": 0.0, "unit": ""}}),
        match="hooks don't apply to symbolic",
    )


def test_symbolic_fixture_needs_an_expression() -> None:
    _reject(
        _symbolic_record(fixtures=[{"setup": {}, "expect": {"roots": [1]}}]),
        match="needs 'expression'",
    )


def test_symbolic_fixture_expression_is_gated() -> None:
    hostile = {"setup": {"expression": "__import__('os').system('true')"},
               "expect": {"roots": [1]}}
    _reject(_symbolic_record(fixtures=[hostile]), match="safe math functions|disallowed")


def test_symbolic_fixture_rejects_quantity_expectations() -> None:
    pair = {"setup": {"expression": "x - 2"}, "expect": {"roots": [2.0, "m"]}}
    _reject(_symbolic_record(fixtures=[pair]), match="for formula templates")


def test_formula_fixture_rejects_symbolic_expectations() -> None:
    record = _record()
    record["fixtures"][0]["expect"] = {"max_deflection": [1.0, 2.0]}
    _reject(record, match=r"must be a \[value, unit\] pair")


def test_seam_is_dependency_light() -> None:
    """chisel-alignment §10: importing the validator pulls pydantic + stdlib only."""
    code = (
        "import sys, assay.templates; "
        "heavy = sorted(m for m in sys.modules if m.split('.')[0] in "
        "{'sympy', 'scipy', 'numpy', 'pint', 'matplotlib', 'stratadb'}); "
        "assert not heavy, heavy"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
