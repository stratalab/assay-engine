"""E1.7: the deterministic CLI — the no-model surface (A-4).

The done-criteria: the UX §5.1 solve flow and §5.8 inspect flows work end to end.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from assay.artifact import create_artifact, load_artifact, save_artifact
from assay.cli import main
from assay.ir import IR
from assay.templates import golden_template


@pytest.fixture(autouse=True)
def _workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_ux_5_1_solve_flow_end_to_end(capsys: pytest.CaptureFixture[str]) -> None:
    """The done-criterion: caret + implicit multiplication in, verified roots out."""
    assert main(["solve", "x^2 + 3x - 4 = 0"]) == 0
    out = capsys.readouterr().out
    assert "x = -4" in out and "x = 1" in out
    assert "solve (SymPy)" in out
    assert "✓ substitution" in out
    assert "rerun: assay run" in out
    assert Path("x_2_3x_4.result.json").exists()


def test_ux_5_8_reproduce_flow(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["solve", "x^2 - 4 = 0", "--out", "roots.result.json"]) == 0
    capsys.readouterr()
    assert main(["run", "roots.result.json"]) == 0
    out = capsys.readouterr().out
    assert "reproduced ✓" in out and "identical" in out and "sympy" in out


def test_show_progressive_disclosure(capsys: pytest.CaptureFixture[str]) -> None:
    """UX §2: the full rendering, then --method / --provenance / --ir on a beam
    artifact (it has a resolved fact to disclose)."""
    ir = IR.model_validate(
        {
            "domain": "structural_mechanics",
            "task": "beam_deflection.simply_supported.center_point",
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
    save_artifact(create_artifact(ir, golden_template()), "beam.result.json")

    assert main(["show", "beam.result.json"]) == 0
    out = capsys.readouterr().out
    assert "max_deflection: 0.0005002" in out
    assert "[resolved, not assumed]" in out
    assert "✓ dimension:length" in out and "✓ bounds" in out

    assert main(["show", "beam.result.json", "--method"]) == 0
    out = capsys.readouterr().out
    assert "P * L**3 / (48 * E * I)" in out and "euler_bernoulli" in out

    assert main(["show", "beam.result.json", "--provenance"]) == 0
    out = capsys.readouterr().out
    assert "assay.materials steel.structural.E v0.1" in out and "pinned:" in out

    assert main(["show", "beam.result.json", "--ir"]) == 0
    assert '"task"' in capsys.readouterr().out


def test_integrate_flow(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["integrate", "x^2"]) == 0
    out = capsys.readouterr().out
    assert "antiderivative = x**3/3" in out and "✓ derivative" in out


def test_units_conversion(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["units", "30 psi to kPa"]) == 0
    out = capsys.readouterr().out
    assert "206.84" in out and "kPa" in out and "✓ dimension" in out


def test_units_incompatible_dimensions_fail_clear(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["units", "30 psi to m"]) == 2
    assert "error:" in capsys.readouterr().err


def test_units_malformed_query_fails_clear(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["units", "psi into kPa please"]) == 2
    assert "expected '<value> <unit> to <unit>'" in capsys.readouterr().err


def test_normalization_preserves_scientific_notation(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["solve", "x - 5e9 = 0", "--out", "big.result.json"]) == 0
    assert "x = 5e+09" in capsys.readouterr().out


def test_normalization_handles_implicit_parens(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["solve", "2(x+1) = 0", "--out", "p.result.json"]) == 0
    assert "x = -1" in capsys.readouterr().out


def test_ambiguous_variable_fails_clear(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["solve", "x*y + 1"]) == 2
    assert "specify setup 'variable'" in capsys.readouterr().err


def test_explicit_variable_flag(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["solve", "a^2 - 4", "--variable", "a", "--out", "a.result.json"]) == 0
    out = capsys.readouterr().out
    assert "a = -2" in out and "a = 2" in out


def test_json_emits_the_answer_object(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["solve", "x^2 - 9 = 0", "--json", "--out", "j.result.json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["verified"]["ok"] is True
    assert [v["value"] for v in payload["result"]] == [-3.0, 3.0]
    assert payload["ir_hash"]


def test_run_reports_failure_loud(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["solve", "x^2 - 4 = 0", "--out", "d.result.json"]) == 0
    capsys.readouterr()
    artifact = load_artifact("d.result.json")
    wrong = artifact.answer.result[0].model_copy(update={"value": 7.0})
    doctored = artifact.model_copy(
        update={
            "answer": artifact.answer.model_copy(
                update={"result": [wrong, artifact.answer.result[1]]}
            )
        }
    )
    save_artifact(doctored, "d.result.json")
    assert main(["run", "d.result.json"]) == 1
    assert "FAILED" in capsys.readouterr().out


def test_missing_artifact_fails_clear(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["run", "nope.result.json"]) == 2
    assert "error:" in capsys.readouterr().err


def test_no_command_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0
    assert "solve" in capsys.readouterr().out
