"""E2.1: the inference seam — NL → candidate IR, propose → validate → execute.

The done-criteria: a natural-language beam question yields the correct verified answer;
an under-specified one asks (interactive) or fails clear (batch); an out-of-scope one
refuses. Plus the seam's guarantees: every candidate IR passes ``validate_candidate``
before anything runs (hallucinations are rejected, with every reason stated), and both
shipped backends — the deterministic rules and the llama.cpp binding (exercised via an
injected completer: no native dependency, no model file) — stand behind one interface.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from assay.artifact import create_artifact
from assay.cli import main
from assay.inference import (
    Ambiguity,
    CandidateIRError,
    DeterministicBackend,
    OutOfScope,
    ProposalError,
    ProposedIR,
    validate_candidate,
)
from assay.inference.llama import LlamaBackend
from assay.ir import IR, Quantity
from assay.resolver import Resolver
from assay.templates import Template, golden_templates, validate_template

_CATALOG = golden_templates()
_BEAM = "beam_deflection.simply_supported.center_point"
_BEAM_QUESTION = (
    "max deflection of a simply supported steel beam,"
    " 5 kN center load, 2 m span, I = 8.33e-6 m^4"
)


def _cantilever() -> Template:
    """A second beam variant (hand-authored, test-only) to make ambiguity real."""
    return validate_template(
        {
            "schema_version": 1,
            "id": "beam_deflection.cantilever.end_point",
            "domain": "structural_mechanics",
            "description": "End deflection of a cantilever beam under an end point load.",
            "inputs": [
                {"name": "P", "dimension": "force"},
                {"name": "L", "dimension": "length"},
                {"name": "E", "dimension": "pressure",
                 "resolve": {"library": "assay.materials", "key": "{material}.E"}},
                {"name": "I", "dimension": "length**4"},
            ],
            "method": {"kind": "formula", "expr": "P * L**3 / (3 * E * I)"},
            "output": {"name": "max_deflection", "dimension": "length"},
            "fixtures": [
                {
                    "inputs": {
                        "P": [1000, "N"],
                        "L": [1, "m"],
                        "E": [200e9, "Pa"],
                        "I": [1e-6, "m**4"],
                    },
                    "expect": {"max_deflection": [1.6666667e-3, "m"]},
                    "tol": 1e-6,
                }
            ],
            "provenance": {"source": "test:hand-authored", "license_tier": "open"},
        }
    )


# --- the deterministic backend: extraction, honest states --------------------------


def test_fake_extracts_the_beam_ir_by_dimension() -> None:
    proposal = DeterministicBackend().propose(_BEAM_QUESTION, _CATALOG)
    assert isinstance(proposal, ProposedIR)
    ir = proposal.ir
    assert ir.task == _BEAM and ir.query == _BEAM_QUESTION
    scalars = {
        name: value for name, value in ir.inputs.items() if isinstance(value, Quantity)
    }
    assert (scalars["P"].value, scalars["P"].unit) == (5.0, "kN")
    assert (scalars["L"].value, scalars["L"].unit) == (2.0, "m")
    assert (scalars["I"].value, scalars["I"].unit) == (8.33e-6, "m**4")
    assert ir.setup == {"material": "steel.structural"}  # matched to the curated table
    assert ir.missing_inputs == ["E"]  # not stated, not invented — flagged


def test_nl_beam_question_yields_the_correct_verified_answer() -> None:
    """THE done-criterion: propose → validate → resolve → execute, end to end."""
    proposal = DeterministicBackend().propose(_BEAM_QUESTION, _CATALOG)
    assert isinstance(proposal, ProposedIR)
    template = validate_candidate(proposal.ir, _CATALOG)
    resolution = Resolver().resolve_missing(proposal.ir, template)
    assert resolution.ir.resolved["E"].source.library == "assay.materials"
    assert not resolution.ir.missing_inputs
    artifact = create_artifact(resolution.ir, template)
    assert artifact.answer.verified.ok
    value = artifact.answer.result[0]
    assert float(value.value) * 1000 == pytest.approx(0.50, abs=0.005)  # 0.50 mm


def test_under_specified_question_leaves_the_input_missing() -> None:
    question = "max deflection of a simply supported steel beam, 5 kN center load, 2 m span"
    proposal = DeterministicBackend().propose(question, _CATALOG)
    assert isinstance(proposal, ProposedIR)
    assert set(proposal.ir.missing_inputs) == {"E", "I"}
    template = validate_candidate(proposal.ir, _CATALOG)
    resolution = Resolver().resolve_missing(proposal.ir, template)
    assert resolution.ir.missing_inputs == ["I"]  # E resolves; I is asked, never guessed
    assert "will not be fabricated" in resolution.unresolved["I"]


def test_a_single_stray_keyword_is_not_a_reading() -> None:
    """One family-token hit + nothing extractable = noise for a formula template
    (refuse; the catalog-growth guard). Symbolic/solver stay exempt — one verb plus
    the problem in prose is a legitimate reading ('minimize (x-2)^2 + 1 …')."""
    stray = DeterministicBackend().propose("tell me about heat", _CATALOG)
    assert isinstance(stray, OutOfScope)
    assert "stray keyword" in stray.reason
    verb = DeterministicBackend().propose("minimize (x - 2)**2 + 1 over some interval", _CATALOG)
    assert isinstance(verb, ProposedIR)
    assert verb.ir.task == "minimize.univariate.numeric"


def test_specificity_and_single_counting(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two volume-3 lessons, encoded: (a) an unqualified question prefers the
    unqualified reading — 'kinetic energy of a 2 kg mass…' is classical, not
    relativistic (whose extra family token the question never said); (b) a question
    token counts once — 'mach_angle.from_mach_number' cannot score 'mach' twice and
    dodge the stray-keyword floor."""
    from assay.templates.plugins import full_catalog

    classical = DeterministicBackend().propose(
        "kinetic energy of a 2 kg mass moving at 3 m/s", full_catalog()
    )
    assert isinstance(classical, ProposedIR)
    assert classical.ir.task == "kinetic_energy.point_mass"
    explicit = DeterministicBackend().propose(
        "relativistic kinetic energy of an electron moving at 2.9e8 m/s", full_catalog()
    )
    assert isinstance(explicit, ProposedIR)
    assert explicit.ir.task == "relativistic_kinetic_energy.point_mass"
    stray = DeterministicBackend().propose(
        "simulate turbulent flow over an airfoil at Mach 0.8", full_catalog()
    )
    assert isinstance(stray, OutOfScope)


def test_out_of_scope_refuses_with_whats_covered() -> None:
    proposal = DeterministicBackend().propose(
        "simulate turbulent flow over an airfoil at Mach 0.8", _CATALOG
    )
    assert isinstance(proposal, OutOfScope)
    assert "no task template matches" in proposal.reason
    assert "algebra" in proposal.covered and "structural_mechanics" in proposal.covered


def test_ambiguous_question_surfaces_the_fork() -> None:
    catalog = [*_CATALOG, _cantilever()]
    proposal = DeterministicBackend().propose(
        "deflection of a steel beam, 5 kN load, 2 m", catalog
    )
    assert isinstance(proposal, Ambiguity)
    assert {option.ir.task for option in proposal.options} == {
        _BEAM,
        "beam_deflection.cantilever.end_point",
    }


def test_qualifiers_disambiguate() -> None:
    catalog = [*_CATALOG, _cantilever()]
    proposal = DeterministicBackend().propose(
        "deflection of a simply supported steel beam, 5 kN load, 2 m", catalog
    )
    assert isinstance(proposal, ProposedIR) and proposal.ir.task == _BEAM


def test_symbolic_extraction_strips_the_verb_and_normalizes() -> None:
    proposal = DeterministicBackend().propose("solve x^2 - 5x + 6 = 0", _CATALOG)
    assert isinstance(proposal, ProposedIR)
    assert proposal.ir.task == "solve_equation.univariate"
    assert proposal.ir.setup["expression"] == "x**2 - 5*x + 6 = 0"
    integral = DeterministicBackend().propose("what is the integral of sin(x)^2?", _CATALOG)
    assert isinstance(integral, ProposedIR)
    assert integral.ir.task == "integrate.univariate"
    assert integral.ir.setup["expression"] == "sin(x)**2"


# --- validate_candidate: the pre-execution gate (A-5) -------------------------------


def test_hallucinated_task_is_rejected_not_executed() -> None:
    ir = IR(domain="alchemy", task="transmute.lead_to_gold", setup={})
    with pytest.raises(CandidateIRError, match="refusing to execute a guess"):
        validate_candidate(ir, _CATALOG)


def test_every_contract_violation_is_reported_at_once() -> None:
    ir = IR.model_validate(
        {
            "domain": "astrology",  # wrong domain
            "task": _BEAM,
            "inputs": {
                "P": {"value": 5, "unit": "m"},  # wrong dimension (length, not force)
                "Q": {"value": 1, "unit": "m"},  # undeclared input
            },
            "missing_inputs": ["E"],  # L and I unaccounted for
        }
    )
    with pytest.raises(CandidateIRError) as excinfo:
        validate_candidate(ir, _CATALOG)
    reasons = "\n".join(excinfo.value.reasons)
    assert len(excinfo.value.reasons) >= 4
    assert "domain" in reasons and "Q" in reasons
    assert "must have dimension 'force'" in reasons
    assert "neither supplied nor flagged missing" in reasons


def test_hostile_setup_expression_is_gated_before_execution() -> None:
    ir = IR.model_validate(
        {
            "domain": "algebra",
            "task": "solve_equation.univariate",
            "setup": {"expression": "__import__('os').system('true')"},
        }
    )
    with pytest.raises(CandidateIRError, match="expression rejected"):
        validate_candidate(ir, _CATALOG)


# --- the llama.cpp binding, via an injected completer (no model needed) -------------


def _llama(reply: str) -> LlamaBackend:
    return LlamaBackend(complete=lambda prompt: reply)


def test_llama_prompt_carries_the_catalog_and_trusted_vocabulary() -> None:
    prompts: list[str] = []

    def capture(prompt: str) -> str:
        prompts.append(prompt)
        return json.dumps({"outcome": "out_of_scope", "reason": "n/a"})

    LlamaBackend(complete=capture).propose("anything", _CATALOG)
    assert all(template.id in prompts[0] for template in _CATALOG)
    assert "steel.structural" in prompts[0]  # the curated vocabulary, offered not invented
    assert "NEVER invent" in prompts[0]


def test_llama_candidate_reply_becomes_a_validated_verified_answer() -> None:
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
    backend = _llama(f"```json\n{reply}\n```")  # fenced output is tolerated
    proposal = backend.propose(_BEAM_QUESTION, _CATALOG)
    assert isinstance(proposal, ProposedIR)
    assert proposal.attribution.provider == "llama.cpp"
    template = validate_candidate(proposal.ir, _CATALOG)  # the same gate as every backend
    resolution = Resolver().resolve_missing(proposal.ir, template)
    artifact = create_artifact(resolution.ir, template)
    assert artifact.answer.verified.ok
    assert float(artifact.answer.result[0].value) * 1000 == pytest.approx(0.50, abs=0.005)


def test_llama_hallucinated_inputs_die_at_the_gate() -> None:
    reply = json.dumps(
        {
            "outcome": "candidate",
            "task": _BEAM,
            "inputs": {"flux_capacitance": {"value": 1.21, "unit": "GW"}},
            "missing_inputs": ["P", "L", "E", "I"],
        }
    )
    proposal = _llama(reply).propose("beam question", _CATALOG)
    assert isinstance(proposal, ProposedIR)  # the backend passes it through…
    with pytest.raises(CandidateIRError, match="flux_capacitance"):
        validate_candidate(proposal.ir, _CATALOG)  # …and the gate rejects it


def test_llama_failure_modes_are_stated_not_guessed() -> None:
    with pytest.raises(ProposalError, match="did not emit a JSON proposal"):
        _llama("I think the answer is 42.").propose("q", _CATALOG)
    with pytest.raises(ProposalError, match="unknown task"):
        _llama(json.dumps({"outcome": "candidate", "task": "made.up"})).propose("q", _CATALOG)
    with pytest.raises(ProposalError, match="unknown outcome"):
        _llama(json.dumps({"outcome": "vibes"})).propose("q", _CATALOG)
    with pytest.raises(ProposalError, match="two or more options"):
        _llama(json.dumps({"outcome": "ambiguous", "options": []})).propose("q", _CATALOG)


def test_llama_out_of_scope_and_ambiguous_shapes() -> None:
    refusal = _llama(
        json.dumps({"outcome": "out_of_scope", "reason": "no CFD templates"})
    ).propose("q", _CATALOG)
    assert isinstance(refusal, OutOfScope) and refusal.reason == "no CFD templates"
    option = {
        "outcome": "candidate",
        "task": "solve_equation.univariate",
        "setup": {"expression": "x**2 - 1 = 0"},
        "reading": "solve it",
    }
    fork = _llama(json.dumps({"outcome": "ambiguous", "options": [option, option]})).propose(
        "q", _CATALOG
    )
    assert isinstance(fork, Ambiguity) and len(fork.options) == 2
    assert fork.options[0].reading == "solve it"


# --- the CLI: `assay ask` and the honest states end to end --------------------------


@pytest.fixture(autouse=True)
def _workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_cli_ask_answers_a_full_beam_question(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["ask", _BEAM_QUESTION]) == 0
    out = capsys.readouterr().out
    assert "max_deflection" in out
    assert "assay.materials steel.structural.E" in out  # per-fact provenance, shown
    assert "[resolved, not assumed]" in out
    assert "rerun: assay run" in out


def test_cli_ask_solves_in_natural_language(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["ask", "solve x^2 - 5x + 6 = 0"]) == 0
    out = capsys.readouterr().out
    assert "x = 2" in out and "x = 3" in out and "✓ substitution" in out


def test_cli_ask_batch_fails_clear_on_missing_input(
    capsys: pytest.CaptureFixture[str],
) -> None:
    question = "max deflection of a simply supported steel beam, 5 kN center load, 2 m span"
    assert main(["ask", question, "--batch"]) == 2
    err = capsys.readouterr().err
    assert "missing required input 'I'" in err
    assert "resolved: E = 2e+11 Pa" in err  # what WAS resolved is shown
    assert "provided: L = 2 m,  P = 5 kN" in err
    assert "fabricated" in err


def test_cli_ask_interactive_prompts_for_exactly_whats_missing(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    question = "max deflection of a simply supported steel beam, 5 kN center load, 2 m span"
    monkeypatch.setattr("sys.stdin", types.SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr("builtins.input", lambda _prompt: "8.33e-6 m^4")
    assert main(["ask", question]) == 0
    out = capsys.readouterr().out
    assert "I need one input to answer this:" in out
    assert "• I — dimension length**4" in out
    assert "Resolved   E = 2e+11 Pa" in out
    assert "max_deflection" in out


def test_cli_ask_refuses_out_of_scope(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["ask", "simulate turbulent flow over an airfoil at Mach 0.8"]) == 2
    out = capsys.readouterr().out
    assert "outside what I cover" in out and "I won't guess." in out


def test_cli_ask_surfaces_ambiguity_and_honors_pick(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    extended = (*golden_templates(), _cantilever())
    monkeypatch.setattr("assay.templates.plugins.golden_templates", lambda: extended)
    question = "deflection of a steel beam, 5 kN load, 2 m"
    assert main(["ask", question, "--batch"]) == 2
    out = capsys.readouterr().out
    assert "This is ambiguous" in out and "--pick" in out
    assert _BEAM in out and "beam_deflection.cantilever.end_point" in out
    # picking a reading proceeds into the normal flow (here: fail-clear on missing I)
    assert main(["ask", question, "--batch", "--pick", "1"]) == 2
    assert "missing required input 'I'" in capsys.readouterr().err


def test_cli_ask_emit_ir_then_run_closes_the_editing_loop(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """UX §5.4: the IR is the same object whether Assay wrote it or you did."""
    assert main(["ask", _BEAM_QUESTION, "--emit-ir", "beam.ir.json"]) == 0
    emitted = json.loads(Path("beam.ir.json").read_text(encoding="utf-8"))
    assert emitted["resolved"]["E"]["source"]["library"] == "assay.materials"
    capsys.readouterr()
    assert main(["run", "beam.ir.json"]) == 0
    out = capsys.readouterr().out
    assert "max_deflection" in out and "Artifact" in out


def test_cli_run_ir_with_missing_input_fails_clear(
    capsys: pytest.CaptureFixture[str],
) -> None:
    ir = IR(
        domain="structural_mechanics",
        task=_BEAM,
        setup={"material": "steel.structural"},
        inputs={},
        missing_inputs=["P", "L", "E", "I"],
    )
    Path("incomplete.ir.json").write_text(ir.model_dump_json(), encoding="utf-8")
    assert main(["run", "incomplete.ir.json"]) == 2
    assert "fabricated" in capsys.readouterr().err


def test_cli_run_still_reproduces_artifacts(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["ask", _BEAM_QUESTION, "--out", "beam.result.json"]) == 0
    capsys.readouterr()
    assert main(["run", "beam.result.json"]) == 0
    assert "reproduced ✓" in capsys.readouterr().out


def test_cli_ask_uses_no_model_by_default() -> None:
    assert "llama_cpp" not in sys.modules  # the default path never touches the binding
