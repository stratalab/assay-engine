"""M1 gate: the v0 acceptance test (implementation plan §2; PRD §16 v0).

Every criterion of the gate, encoded — the deterministic spine, end to end:
each golden answers a worked example correctly, reproducibly, with per-fact
provenance; the beam declares E and I missing, resolves E for steel from the curated
table (never fabricated), and reproduces; figures render byte-stably from verified
data; all fixtures are green; and **no model is anywhere in the loop**.
(The remaining criterion — ruff + mypy + pytest green on the Linux/macOS/Windows
matrix — is CI's, on every push.)
"""

from __future__ import annotations

from pathlib import Path

import pytest

import assay
from assay.artifact import create_artifact, load_artifact, rerun, save_artifact
from assay.execute import MissingInputError, execute_ir, run_fixtures
from assay.ir import IR, Quantity
from assay.render import function_plot_data, render_svg
from assay.resolver import Resolver
from assay.templates import golden_templates
from assay.verify import verify_execution

_GOLDENS = {template.id: template for template in golden_templates()}


def test_gate_solve_answers_reproducibly() -> None:
    ir = IR.model_validate(
        {
            "domain": "algebra",
            "task": "solve_equation.univariate",
            "setup": {"expression": "x**2 - 5*x + 6 = 0"},
        }
    )
    artifact = create_artifact(ir, _GOLDENS["solve_equation.univariate"])
    assert [v.value for v in artifact.answer.result] == [2.0, 3.0]  # correct
    assert artifact.answer.verified.ok  # verified (substitution)
    assert rerun(artifact).status == "exact"  # reproducible


def test_gate_integrate_answers_reproducibly() -> None:
    ir = IR.model_validate(
        {
            "domain": "calculus",
            "task": "integrate.univariate",
            "setup": {"expression": "sin(x)**2", "variable": "x"},
        }
    )
    artifact = create_artifact(ir, _GOLDENS["integrate.univariate"])
    assert artifact.answer.verified.ok  # verified (derivative)
    assert rerun(artifact).status == "exact"


def test_gate_beam_missing_input_flow(tmp_path: Path) -> None:
    """THE flagship criterion: E and I declared missing; E resolves for steel from the
    curated table with its source recorded; I is never fabricated; the answer carries
    per-fact provenance and reproduces exactly."""
    golden = _GOLDENS["beam_deflection.simply_supported.center_point"]
    ir = IR.model_validate(
        {
            "domain": "structural_mechanics",
            "task": golden.id,
            "setup": {"material": "steel.structural"},
            "inputs": {"P": {"value": 5000, "unit": "N"}, "L": {"value": 2, "unit": "m"}},
            "missing_inputs": ["E", "I"],
        }
    )
    resolution = Resolver().resolve_missing(ir, golden)
    fact = resolution.ir.resolved["E"]  # resolved, never fabricated …
    assert (fact.value, fact.unit) == (200e9, "Pa")
    assert (fact.source.library, fact.source.key) == ("assay.materials", "steel.structural.E")
    assert resolution.ir.missing_inputs == ["I"]  # … and I stays missing,
    assert "will not be fabricated" in resolution.unresolved["I"]  # with the reason
    with pytest.raises(MissingInputError, match="fabricated"):  # batch fails clear (A-8)
        execute_ir(resolution.ir, golden)

    supplied = resolution.ir.model_copy(
        update={
            "inputs": {**resolution.ir.inputs, "I": Quantity(value=8.33e-6, unit="m**4")},
            "missing_inputs": [],
        }
    )
    artifact = create_artifact(supplied, golden)
    value = artifact.answer.result[0]
    assert float(value.value) * 1000 == pytest.approx(0.50, abs=0.005)  # 0.50 mm
    assert artifact.answer.verified.ok
    assert artifact.answer.facts[0].source.version  # auditable to the fact (A-11)

    path = save_artifact(artifact, tmp_path / "beam.result.json")
    reproduction = rerun(load_artifact(path))
    assert reproduction.status == "exact"  # bit-identical, same platform (NFR-2)


def test_gate_all_golden_fixtures_green() -> None:
    for template in _GOLDENS.values():
        results = run_fixtures(template)
        assert results and all(r.ok for r in results), (
            template.id,
            [r.detail for r in results if not r.ok],
        )


def test_gate_figures_are_byte_stable_views_of_verified_data(tmp_path: Path) -> None:
    verified = verify_execution(
        _GOLDENS["solve_equation.univariate"], {}, setup={"expression": "x**2 - 5*x + 6"}
    )
    assert verified.verification.ok and verified.result is not None
    marks = [(f"x = {v.value}", float(v.value), 0.0) for v in verified.result.values]
    data = function_plot_data("x**2 - 5*x + 6", marks=marks, mark_extrema=True)
    assert {(m.x, m.y) for m in data.marks} >= {(2.0, 0.0), (3.0, 0.0)}  # verified roots
    first = render_svg(data, tmp_path / "a.svg").read_bytes()
    second = render_svg(data, tmp_path / "b.svg").read_bytes()
    assert first == second and b"dc:date" not in first


def test_gate_no_model_anywhere_in_the_loop() -> None:
    """A-4/§16: the spine is deterministic — a model is touched ONLY inside the
    inference seam (engineering §4; since E2.1 the seam exists, opt-in and lazy), and
    nothing in the package imports provider or network machinery at all."""
    import assay.inference as inference

    # The seam's entry module (what the CLI imports) stays model-free: the default
    # backend is rule-based, and the llama binding lives in its own lazily-imported
    # module (assay/inference/llama.py) — `assay ask` without --llm never loads a model.
    assert inference.DeterministicBackend.attribution.provider == "assay"

    providers = ("openai", "anthropic", "requests", "httpx", "socket")
    package_dir = Path(assay.__file__).parent
    binding = package_dir / "inference" / "llama.py"
    fetcher = package_dir / "models.py"
    engine = ("execute", "verify", "resolver", "render", "templates", "ir")
    for source in sorted(package_dir.rglob("*.py")):
        for line in source.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith(("import ", "from ")):
                assert not any(token in stripped for token in providers), (source, stripped)
                if source != fetcher:
                    # the network (urllib): only inside the explicit, checksum-gated
                    # model fetcher (E3.3) — never anywhere near the compute path.
                    assert "urllib" not in stripped, (source, stripped)
                if source != binding and "assay.inference" not in stripped:
                    # the model itself (llama_cpp): only inside its own binding module;
                    # everything else reaches it through the seam's interface.
                    assert "llama" not in stripped, (source, stripped)
                in_engine = any(
                    source.parent == package_dir / part or source == package_dir / f"{part}.py"
                    for part in engine
                )
                if in_engine:
                    # the engine never imports the fetcher: no network reachable from
                    # execute/verify/resolve/render, even indirectly.
                    assert "assay.models" not in stripped, (source, stripped)
