"""E1.6: the four-part answer + reproducible artifact.

The done-criteria: an answer's artifact reruns to an identical result (exact,
same-platform); the reproduction report states exact vs within-tolerance.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from assay.artifact import (
    Artifact,
    ArtifactError,
    build_answer,
    capture_environment,
    create_artifact,
    fetch_artifact,
    load_artifact,
    rerun,
    save_artifact,
    store_artifact,
)
from assay.ir import IR
from assay.store import open_store
from assay.templates import Bounds, VerificationHooks, golden_template, golden_templates
from assay.verify import verify_ir

_GOLDENS = {template.id: template for template in golden_templates()}


def _beam_ir() -> IR:
    return IR.model_validate(
        {
            "assay_version": "0.0.1",
            "domain": "structural_mechanics",
            "task": "beam_deflection.simply_supported.center_point",
            "query": "max deflection of a steel beam",
            "setup": {"material": "steel.structural"},
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


def _beam_artifact() -> Artifact:
    return create_artifact(_beam_ir(), golden_template())


def test_answer_has_its_four_parts_plus_provenance() -> None:
    answer = _beam_artifact().answer
    assert answer.result[0].label == "max_deflection"
    assert answer.result[0].unit == "meter"
    assert "simply supported beam" in answer.interpretation
    assert "euler_bernoulli" in answer.interpretation
    assert answer.method == "P * L**3 / (48 * E * I)"
    assert answer.facts[0].name == "E"
    assert answer.facts[0].source.key == "steel.structural.E"  # auditable to the fact
    assert answer.verified.ok
    assert answer.ir_hash == _beam_ir().content_hash()
    assert set(answer.versions) == {"sympy", "pint", "pydantic"}


def test_artifact_reruns_exact_on_the_same_platform() -> None:
    """The E1.6 done-criterion."""
    reproduction = rerun(_beam_artifact())
    assert reproduction.status == "exact"
    assert "identical" in reproduction.detail and "sympy" in reproduction.detail
    assert reproduction.answer.result == _beam_artifact().answer.result


def test_symbolic_artifacts_rerun_exact() -> None:
    """NFR-2: symbolic results are exactly reproducible everywhere."""
    ir = IR.model_validate(
        {
            "domain": "algebra",
            "task": "solve_equation.univariate",
            "setup": {"expression": "x**2 - 5*x + 6", "variable": "x"},
        }
    )
    artifact = create_artifact(ir, _GOLDENS["solve_equation.univariate"])
    assert [v.value for v in artifact.answer.result] == [2.0, 3.0]
    assert rerun(artifact).status == "exact"


def test_reproduction_reports_within_tolerance_for_float_noise() -> None:
    artifact = _beam_artifact()
    nudged = artifact.answer.result[0].model_copy(
        update={"value": float(artifact.answer.result[0].value) * (1 + 1e-12)}
    )
    doctored = artifact.model_copy(
        update={"answer": artifact.answer.model_copy(update={"result": [nudged]})}
    )
    reproduction = rerun(doctored)
    assert reproduction.status == "within-tolerance"
    assert "within tolerance" in reproduction.detail


def test_reproduction_reports_equivalent_symbolic_drift_as_within_tolerance() -> None:
    ir = IR.model_validate(
        {
            "domain": "calculus",
            "task": "integrate.univariate",
            "setup": {"expression": "x**2", "variable": "x"},
        }
    )
    artifact = create_artifact(ir, _GOLDENS["integrate.univariate"])
    rewritten = artifact.answer.result[0].model_copy(update={"value": "x**3 * (1/3)"})
    doctored = artifact.model_copy(
        update={"answer": artifact.answer.model_copy(update={"result": [rewritten]})}
    )
    assert rerun(doctored).status == "within-tolerance"


def test_reproduction_fails_loud_on_a_real_mismatch() -> None:
    artifact = _beam_artifact()
    wrong = artifact.answer.result[0].model_copy(
        update={"value": float(artifact.answer.result[0].value) * 2}
    )
    doctored = artifact.model_copy(
        update={"answer": artifact.answer.model_copy(update={"result": [wrong]})}
    )
    reproduction = rerun(doctored)
    assert reproduction.status == "failed"
    assert "reran as" in reproduction.detail


def test_artifact_file_round_trips_and_is_byte_stable(tmp_path: Path) -> None:
    artifact = _beam_artifact()
    first, second = tmp_path / "a.result.json", tmp_path / "b.result.json"
    save_artifact(artifact, first)
    save_artifact(_beam_artifact(), second)  # the same computation, recorded twice
    assert first.read_bytes() == second.read_bytes()  # no timestamps, stable order
    loaded = load_artifact(first)
    assert loaded == artifact
    assert rerun(loaded).status == "exact"


def test_artifact_persists_to_the_store_with_lineage() -> None:
    store = open_store()
    artifact = _beam_artifact()
    key = store_artifact(store, artifact)
    assert key.startswith(artifact.ir.content_hash())
    assert fetch_artifact(store, key) == artifact
    assert fetch_artifact(store, "missing") is None
    assert store.lineage("artifact") == [key]


def test_withheld_answers_stay_withheld_and_reproduce() -> None:
    golden = golden_template()
    tight = golden.model_copy(
        update={"verification": VerificationHooks(bounds=Bounds(min=0.0, max=1e-6, unit="m"))}
    )
    artifact = create_artifact(_beam_ir(), tight)
    assert artifact.answer.result == []  # withheld (A-6)
    assert not artifact.answer.verified.ok
    reproduction = rerun(artifact)
    assert reproduction.status == "exact"
    assert "withheld verdict reproduced" in reproduction.detail


def test_future_schemas_are_refused_not_misexecuted() -> None:
    artifact = _beam_artifact()
    record: dict[str, Any] = artifact.model_dump(mode="json")
    record["artifact_version"] = 2
    with pytest.raises(Exception, match="artifact_version"):
        Artifact.model_validate(record)
    future_ir = artifact.ir.model_copy(update={"ir_version": 2})
    with pytest.raises(ArtifactError, match="cannot honor ir_version 2"):
        rerun(artifact.model_copy(update={"ir": future_ir}))


def test_environment_capture_is_complete() -> None:
    environment = capture_environment()
    assert environment.assay_version and environment.python and environment.platform
    assert all(environment.versions[name] for name in ("sympy", "pint", "pydantic"))


def test_build_answer_interpretation_covers_symbolic_setup() -> None:
    ir = IR.model_validate(
        {
            "domain": "algebra",
            "task": "solve_equation.univariate",
            "setup": {"expression": "x**2 - 4", "variable": "x"},
        }
    )
    template = _GOLDENS["solve_equation.univariate"]
    artifact = create_artifact(ir, template)
    assert "expression = x**2 - 4" in artifact.answer.interpretation
    assert artifact.answer.method == "solve (SymPy)"
    answer = build_answer(ir, template, verify_ir(ir, template), capture_environment())
    assert answer.verified.ok and answer == artifact.answer
