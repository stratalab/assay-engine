"""E2.15: the execution trace — step-by-step that IS the computation.

Every trace entry is what the executor actually evaluated, in base units, inside the
same run that produced the answer — never a post-hoc narration. The trace travels in
the answer object (and therefore the artifact), upgrades every steps-DAG template in
the shipped catalog at once, and stays out of withheld answers (A-6).
"""

from __future__ import annotations

import pytest

from assay.artifact import create_artifact, load_artifact, rerun, save_artifact
from assay.execute import execute_template
from assay.ir import IR, Quantity
from assay.templates import golden_template, golden_templates


def _golden(template_id: str):  # type: ignore[no-untyped-def]
    return next(t for t in golden_templates() if t.id == template_id)


def test_a_dag_traces_every_step_in_order() -> None:
    """Mohr's circle (the E2.9 golden): the trace is the DAG's literal walk."""
    mohr = _golden("principal_stress.plane.max")
    inputs = {"sx": (80.0, "MPa"), "sy": (20.0, "MPa"), "txy": (40.0, "MPa")}
    result = execute_template(mohr, inputs)
    from assay.templates import FormulaMethod

    assert isinstance(mohr.method, FormulaMethod)
    assert [s.label for s in result.trace] == [step.name for step in mohr.method.steps]
    assert [s.expr for s in result.trace] == [step.expr for step in mohr.method.steps]
    # each entry is the actual intermediate: sigma_avg = (80+20)/2 MPa = 5e7 Pa
    by_label = {s.label: s for s in result.trace}
    assert by_label["sigma_avg"].value == pytest.approx(5.0e7)
    assert by_label["sigma_avg"].unit == "kilogram / meter / second ** 2"
    # the last step IS the result
    assert result.trace[-1].value == pytest.approx(float(result.values[0].value))


def test_a_single_expression_traces_one_step() -> None:
    template = golden_template()  # the beam-deflection golden (single expr)
    result = execute_template(
        template,
        {
            "P": (1000.0, "N"), "L": (2.0, "m"),
            "E": (200.0, "GPa"), "I": (8.0e-6, "m**4"),
        },
    )
    from assay.templates import FormulaMethod

    assert isinstance(template.method, FormulaMethod)
    assert len(result.trace) == 1
    assert result.trace[0].expr == template.method.expr
    assert result.trace[0].value == pytest.approx(float(result.values[0].value))


def test_the_trace_travels_in_answer_and_artifact_and_reruns() -> None:
    mohr = _golden("principal_stress.plane.max")
    ir = IR(
        domain=mohr.domain,
        task=mohr.id,
        inputs={
            "sx": Quantity(value=80, unit="MPa"),
            "sy": Quantity(value=20, unit="MPa"),
            "txy": Quantity(value=40, unit="MPa"),
        },
    )
    artifact = create_artifact(ir, mohr)
    assert artifact.answer.verified.ok
    assert [s.label for s in artifact.answer.steps] == [
        step.name for step in mohr.method.steps
    ]
    assert rerun(artifact).status == "exact"


def test_the_trace_round_trips_through_the_file(tmp_path) -> None:  # type: ignore[no-untyped-def]
    mohr = _golden("principal_stress.plane.max")
    ir = IR(
        domain=mohr.domain,
        task=mohr.id,
        inputs={
            "sx": Quantity(value=80, unit="MPa"),
            "sy": Quantity(value=20, unit="MPa"),
            "txy": Quantity(value=40, unit="MPa"),
        },
    )
    path = save_artifact(create_artifact(ir, mohr), tmp_path / "mohr.result.json")
    loaded = load_artifact(path)
    assert len(loaded.answer.steps) == len(mohr.method.steps)


def test_a_withheld_answer_carries_no_steps() -> None:
    """A-6: the trace is part of the ANSWER; a withheld answer keeps its candidate
    private, steps included."""
    from assay.templates import validate_template

    bad = validate_template({
        "id": "bounds.selfcheck",
        "domain": "mechanics",
        "description": "always out of bounds",
        "inputs": [{"name": "x", "dimension": "length"}],
        "method": {"kind": "formula", "expr": "x"},
        "output": {"name": "y", "dimension": "length"},
        "verification": {"bounds": {"min": 0.0, "max": 1e-9, "unit": "m"}},
        "fixtures": [{"inputs": {"x": [1e-10, "m"]}, "expect": {"y": [1e-10, "m"]}}],
        "provenance": {"source": "assay:demo", "license_tier": "open"},
    })
    ir = IR(domain="mechanics", task=bad.id, inputs={"x": Quantity(value=5, unit="m")})
    artifact = create_artifact(ir, bad)
    assert not artifact.answer.verified.ok
    assert artifact.answer.result == []
    assert artifact.answer.steps == []


def test_solve_for_traces_its_recovery() -> None:
    from assay.execute.solve_for import solve_for_input

    template = _golden("kinetic_energy.point_mass")
    result = solve_for_input(
        template, "v", {"m": (2.0, "kg")}, (100.0, "J")
    )
    assert result.trace
    assert all("symbolic inversion" in step.note for step in result.trace)
    # both-roots honesty: ±v, sorted ascending — the trace records each recovery
    assert [step.value for step in result.trace] == pytest.approx([-10.0, 10.0])
