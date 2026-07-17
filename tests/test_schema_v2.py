"""E2.11: schema v2 — list inputs + reducers, and the `cases` discriminator.

The acceptance set is Chisel's round-5 exhibits verbatim: the five-series 90 Ω row
(one list-input template instead of `equivalent_resistance.five_series`), the
three-parallel and series-capacitor `sum_inverse` rows, and the moment-of-inertia
table as one `cases` object. V1 semantics stay frozen: v2 features on a v1 record are
refused, and every v1 template in the shipped catalog validates unchanged.
"""

from __future__ import annotations

from typing import Any

import pytest

from assay.execute import InputError, execute_ir, execute_template, run_fixtures
from assay.inference import CandidateIRError, validate_candidate
from assay.ir import IR
from assay.templates import TemplateValidationError, validate_template
from assay.verify import verify_execution


def _series_resistors(**overrides: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "schema_version": 2,
        "id": "equivalent_resistance.series",
        "domain": "electromagnetism",
        "description": "Equivalent resistance of N resistors in series.",
        "inputs": [{"name": "R_i", "dimension": "resistance", "many": True}],
        "method": {"kind": "formula", "expr": "sum(R_i)"},
        "output": {"name": "equivalent_resistance", "dimension": "resistance"},
        "fixtures": [
            {  # the ch-10 example: four 20 Ω + one 10 Ω → 90 Ω
                "inputs": {
                    "R_i": [[20, "ohm"], [20, "ohm"], [20, "ohm"], [20, "ohm"], [10, "ohm"]]
                },
                "expect": {"equivalent_resistance": [90.0, "ohm"]},
                "tol": 1e-9,
            }
        ],
        "provenance": {"source": "assay:demo", "license_tier": "open"},
    }
    record.update(overrides)
    return record


_MOI_CASES = {
    "schema_version": 2,
    "id": "moment_of_inertia.standard_bodies",
    "domain": "dynamics",
    "description": "Moment of inertia of standard bodies (the m58330 table).",
    "inputs": [
        {"name": "M", "dimension": "mass"},
        {"name": "L", "dimension": "length", "required": False},
        {"name": "R", "dimension": "length", "required": False},
    ],
    "method": {
        "kind": "cases",
        "discriminator": "geometry",
        "cases": {
            "thin_rod_center": "M * L**2 / 12",
            "thin_rod_end": "M * L**2 / 3",
            "solid_disk": "M * R**2 / 2",
            "solid_sphere": "2 * M * R**2 / 5",
            "point_mass": "M * R**2",
        },
    },
    "output": {"name": "moment_of_inertia", "dimension": "mass*length**2"},
    "fixtures": [
        {
            "setup": {"geometry": "solid_disk"},
            "inputs": {"M": [2.0, "kg"], "R": [3.0, "m"]},
            "expect": {"moment_of_inertia": [9.0, "kg*m**2"]},
            "tol": 1e-9,
        },
        {
            "setup": {"geometry": "thin_rod_end"},
            "inputs": {"M": [3.0, "kg"], "L": [2.0, "m"]},
            "expect": {"moment_of_inertia": [4.0, "kg*m**2"]},
            "tol": 1e-9,
        },
    ],
    "provenance": {"source": "assay:demo", "license_tier": "open"},
}


def test_the_five_series_row_is_one_template() -> None:
    """The E2.11 done-criterion: the arity-suffixed family collapses to one
    list-input template, and the book's 90 Ω row gates green."""
    template = validate_template(_series_resistors())
    assert all(r.ok for r in run_fixtures(template))


def test_sum_inverse_covers_parallel_and_series_capacitors() -> None:
    parallel = validate_template(
        _series_resistors(
            id="equivalent_resistance.parallel",
            description="Equivalent resistance of N resistors in parallel.",
            method={"kind": "formula", "expr": "1 / sum_inverse(R_i)"},
            fixtures=[
                {
                    "inputs": {"R_i": [[1.00, "ohm"], [2.00, "ohm"], [2.00, "ohm"]]},
                    "expect": {"equivalent_resistance": [0.50, "ohm"]},
                    "tol": 1e-9,
                }
            ],
        )
    )
    assert all(r.ok for r in run_fixtures(parallel))
    capacitors = validate_template(
        {
            "schema_version": 2,
            "id": "equivalent_capacitance.series",
            "domain": "electromagnetism",
            "description": "Equivalent capacitance of N capacitors in series.",
            "inputs": [{"name": "C_i", "dimension": "capacitance", "many": True}],
            "method": {"kind": "formula", "expr": "1 / sum_inverse(C_i)"},
            "output": {"name": "equivalent_capacitance", "dimension": "capacitance"},
            "fixtures": [
                {
                    "inputs": {"C_i": [[1.000, "uF"], [5.000, "uF"], [8.000, "uF"]]},
                    "expect": {"equivalent_capacitance": [0.755, "uF"]},
                    "tol": 1e-3,
                }
            ],
            "provenance": {"source": "assay:demo", "license_tier": "open"},
        }
    )
    assert all(r.ok for r in run_fixtures(capacitors))


def test_the_moment_of_inertia_table_is_one_cases_object() -> None:
    """The other done-criterion: five flattened ids collapse to one `cases` object;
    both fixture cases gate green and the discriminator selects at execution."""
    template = validate_template(_MOI_CASES)
    assert all(r.ok for r in run_fixtures(template))
    result = execute_template(
        template, {"M": (1.0, "kg"), "R": (2.0, "m")}, setup={"geometry": "point_mass"}
    )
    assert result.value == pytest.approx(4.0)
    verified = verify_execution(
        template, {"M": (1.0, "kg"), "R": (2.0, "m")}, setup={"geometry": "point_mass"}
    )
    assert verified.verification.ok  # dimension check runs on the selected case


def test_case_selection_fails_clear() -> None:
    template = validate_template(_MOI_CASES)
    with pytest.raises(InputError, match="one of:"):
        execute_template(template, {"M": (1.0, "kg"), "R": (2.0, "m")}, setup={})
    with pytest.raises(InputError, match="one of:"):
        execute_template(
            template, {"M": (1.0, "kg"), "R": (2.0, "m")}, setup={"geometry": "torus"}
        )


def test_v1_semantics_stay_frozen() -> None:
    """The pin discipline: v2 features on a schema_version 1 record are refused."""
    with pytest.raises(TemplateValidationError, match="require schema_version 2"):
        validate_template(_series_resistors(schema_version=1))
    moi_v1 = dict(_MOI_CASES)
    moi_v1["schema_version"] = 1
    with pytest.raises(TemplateValidationError, match="requires schema_version 2"):
        validate_template(moi_v1)


def test_reducer_grammar_is_tight() -> None:
    # a list input may appear only inside a reducer
    with pytest.raises(TemplateValidationError, match="only inside a reducer"):
        validate_template(_series_resistors(method={"kind": "formula", "expr": "R_i + 1"}))
    # reducers apply only to list inputs
    record = _series_resistors(
        inputs=[
            {"name": "R_i", "dimension": "resistance", "many": True},
            {"name": "R0", "dimension": "resistance"},
        ],
        method={"kind": "formula", "expr": "sum(R0) + sum(R_i)"},
    )
    with pytest.raises(TemplateValidationError, match="only to list inputs"):
        validate_template(record)
    # a reducer takes exactly one bare name
    with pytest.raises(TemplateValidationError, match="exactly one input name"):
        validate_template(_series_resistors(method={"kind": "formula", "expr": "sum(R_i + 1)"}))
    # a list input cannot carry a resolve hint
    with pytest.raises(TemplateValidationError, match="cannot carry a resolve hint"):
        validate_template(
            _series_resistors(
                inputs=[
                    {
                        "name": "R_i",
                        "dimension": "resistance",
                        "many": True,
                        "resolve": {"library": "assay.constants", "key": "g"},
                    }
                ]
            )
        )


def test_fixture_shapes_are_enforced() -> None:
    with pytest.raises(TemplateValidationError, match="supply a list"):
        validate_template(
            _series_resistors(
                fixtures=[
                    {
                        "inputs": {"R_i": [20, "ohm"]},  # scalar pair for a list input
                        "expect": {"equivalent_resistance": [20.0, "ohm"]},
                        "tol": 1e-9,
                    }
                ]
            )
        )


def test_list_inputs_flow_through_the_ir(tmp_path: object) -> None:
    """End to end: a list-input IR executes, verifies, and reruns exact."""
    from pathlib import Path

    from assay.artifact import create_artifact, load_artifact, rerun, save_artifact

    template = validate_template(_series_resistors())
    ir = IR.model_validate(
        {
            "domain": "electromagnetism",
            "task": "equivalent_resistance.series",
            "inputs": {
                "R_i": [
                    {"value": 20, "unit": "ohm"},
                    {"value": 20, "unit": "ohm"},
                    {"value": 20, "unit": "ohm"},
                    {"value": 20, "unit": "ohm"},
                    {"value": 10, "unit": "ohm"},
                ]
            },
        }
    )
    validate_candidate(ir, (template,))
    result = execute_ir(ir, template)
    assert isinstance(result.value, float)
    assert result.value == pytest.approx(90.0)
    artifact = create_artifact(ir, template)
    assert artifact.answer.verified.ok
    path = save_artifact(artifact, Path(str(tmp_path)) / "series.result.json")
    assert rerun(load_artifact(path)).status == "exact"
    # shape mismatch is caught at the pre-execution gate
    elements = ir.inputs["R_i"]
    assert isinstance(elements, list)
    bad = ir.model_copy(update={"inputs": {"R_i": elements[0]}})
    with pytest.raises(CandidateIRError, match="list"):
        validate_candidate(bad, (template,))


def test_wrong_dimension_element_in_a_list_is_refused() -> None:
    template = validate_template(_series_resistors())
    with pytest.raises(InputError, match="dimension"):
        execute_template(
            template, {"R_i": [(20.0, "ohm"), (5.0, "m")]}  # a length among resistances
        )


def test_empty_list_fails_clear() -> None:
    template = validate_template(_series_resistors())
    with pytest.raises(InputError, match="at least one element"):
        execute_template(template, {"R_i": []})