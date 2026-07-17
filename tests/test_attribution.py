"""E2.7: IR-model attribution — the artifact names the NL→IR producer.

The done-criterion: an answer's artifact names the NL→IR model. Plus the two
disciplines around it: attribution is provenance, not content (the content hash — and
therefore the cache key — ignores it: the same IR is the same computation whoever
wrote it), and the reproducibility caveat surfaces only for model-produced IRs
(re-asking may read differently; the artifact reruns the exact IR either way).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from assay.artifact import (
    Artifact,
    artifact_key,
    create_artifact,
    load_artifact,
    rerun,
    save_artifact,
)
from assay.cli import main
from assay.inference import DeterministicBackend, ProposedIR, validate_candidate
from assay.inference.llama import LlamaBackend
from assay.ir import IR, Producer
from assay.resolver import Resolver
from assay.templates import golden_templates

_CATALOG = golden_templates()
_BEAM = "beam_deflection.simply_supported.center_point"
_BEAM_QUESTION = (
    "max deflection of a simply supported steel beam,"
    " 5 kN center load, 2 m span, I = 8.33e-6 m^4"
)


def _artifact_from(backend: DeterministicBackend | LlamaBackend) -> Artifact:
    proposal = backend.propose(_BEAM_QUESTION, _CATALOG)
    assert isinstance(proposal, ProposedIR)
    template = validate_candidate(proposal.ir, _CATALOG)
    return create_artifact(Resolver().resolve_missing(proposal.ir, template).ir, template)


def test_the_artifact_names_the_deterministic_producer(tmp_path: Path) -> None:
    artifact = _artifact_from(DeterministicBackend())
    assert artifact.ir.produced_by == Producer(provider="assay", model="deterministic-rules/v0")
    # …and it survives the save/load round trip: the file names the model
    path = save_artifact(artifact, tmp_path / "beam.result.json")
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["ir"]["produced_by"] == {"provider": "assay", "model": "deterministic-rules/v0"}
    assert rerun(load_artifact(path)).status == "exact"


def test_the_artifact_names_the_llama_producer() -> None:
    reply = json.dumps(
        {
            "outcome": "candidate",
            "task": _BEAM,
            "setup": {"material": "steel.structural"},
            "inputs": {
                "P": {"value": 5000, "unit": "N"},
                "L": {"value": 2, "unit": "m"},
                "I": {"value": 8.33e-6, "unit": "m**4"},
            },
            "missing_inputs": ["E"],
        }
    )
    artifact = _artifact_from(LlamaBackend(complete=lambda prompt: reply))
    producer = artifact.ir.produced_by
    assert producer is not None
    assert (producer.provider, producer.model) == ("llama.cpp", "injected")


def test_attribution_is_provenance_not_content() -> None:
    """Two identical computations from different producers hash — and cache — the
    same: a hand-built IR, a rules-produced IR, and a model-produced IR."""
    base = {
        "domain": "algebra",
        "task": "solve_equation.univariate",
        "setup": {"expression": "x**2 - 5*x + 6 = 0"},
    }
    hand_built = IR.model_validate(base)
    by_rules = IR.model_validate(
        {**base, "produced_by": {"provider": "assay", "model": "deterministic-rules/v0"}}
    )
    by_model = IR.model_validate(
        {**base, "produced_by": {"provider": "llama.cpp", "model": "some.gguf"}}
    )
    assert hand_built.content_hash() == by_rules.content_hash() == by_model.content_hash()
    template = validate_candidate(hand_built, _CATALOG)
    keys = {
        artifact_key(create_artifact(ir, template)) for ir in (hand_built, by_rules, by_model)
    }
    assert len(keys) == 1  # one cache entry, not three


def test_cli_show_provenance_names_the_producer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    assert main(["ask", _BEAM_QUESTION, "--out", "beam.result.json"]) == 0
    out = capsys.readouterr().out
    assert "re-asking" not in out  # deterministic rules: no caveat — same question, same IR
    assert main(["show", "beam.result.json", "--provenance"]) == 0
    assert "NL→IR: assay deterministic-rules/v0" in capsys.readouterr().out


def test_model_produced_artifacts_carry_the_reask_caveat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    artifact = _artifact_from(
        LlamaBackend(
            complete=lambda prompt: json.dumps(
                {
                    "outcome": "candidate",
                    "task": "solve_equation.univariate",
                    "setup": {"expression": "x**2 - 5*x + 6 = 0"},
                }
            )
        )
    )
    save_artifact(artifact, "solved.result.json")
    assert main(["show", "solved.result.json", "--provenance"]) == 0
    out = capsys.readouterr().out
    assert "NL→IR: llama.cpp injected" in out
    assert "re-asking may read differently" in out
    assert "reruns this exact IR" in out


def test_hand_built_irs_have_no_producer(tmp_path: Path) -> None:
    """`assay run my.ir.json` and API callers build IRs by hand: no attribution, no
    caveat — there was no model to name."""
    ir = IR.model_validate(
        {
            "domain": "algebra",
            "task": "solve_equation.univariate",
            "setup": {"expression": "x - 1 = 0"},
        }
    )
    assert ir.produced_by is None
    template = validate_candidate(ir, _CATALOG)
    artifact = create_artifact(ir, template)
    assert artifact.ir.produced_by is None