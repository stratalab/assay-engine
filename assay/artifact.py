"""The reproducible artifact (E1.6, PRD §10): the answer, assembled and rerunnable (A-7).

``build_answer`` assembles the four-part answer object (E0.3) from the pipeline's
stages: the verified execution becomes ``result`` (withheld → empty, the verdicts say
why), the IR's resolved facts become ``facts`` with their sources (A-11), and the
environment's pinned versions travel in the answer.

An ``Artifact`` is the executable record: the IR, the **embedded template** (pure data —
so the artifact reruns anywhere, offline, forever, independent of registry drift, UX
§5.8), the answer, and the environment (platform + pinned versions, engineering NFR-2).
No timestamps — the same computation writes byte-identical artifacts.

``rerun`` re-executes and reports **honestly** (NFR-2): ``exact`` (identical values —
the same-platform guarantee; symbolic results are exact everywhere), ``within-tolerance``
(numerically or algebraically equivalent — the cross-platform/version-drift case, with
the drift named), or ``failed`` (with the mismatch). Version fields are ``Literal`` so an
artifact from a future schema is *refused at validation*, never mis-executed
(engineering §10).
"""

from __future__ import annotations

import math
import platform as platform_module
from importlib import metadata
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from assay import __version__
from assay.answer import Answer, Fact, ResultValue
from assay.execute import symbolically_equal
from assay.hashing import content_sha256
from assay.ir import IR
from assay.store import Store
from assay.templates import CasesMethod, FormulaMethod, SolverMethod, Template
from assay.verify import VerifiedExecution, verify_ir

__all__ = [
    "Artifact",
    "ArtifactError",
    "Environment",
    "Reproduction",
    "build_answer",
    "capture_environment",
    "create_artifact",
    "fetch_artifact",
    "load_artifact",
    "rerun",
    "save_artifact",
    "store_artifact",
]

# The libraries whose versions pin a numeric result (engineering §6): they travel in
# every answer and artifact, and the reproduction report names any drift.
_PINNED = ("sympy", "pint", "pydantic")

# Rerun comparison tolerances (NFR-2): same platform + versions reproduces float-
# identically; these bound the honest "within-tolerance" verdict for everything else.
_RERUN_RTOL = 1e-9
_RERUN_ATOL = 1e-12


class ArtifactError(Exception):
    """The artifact cannot be honored — refused with the reason, never mis-executed."""


class Environment(BaseModel):
    """Where an answer was computed: the reproducibility record (NFR-2, A-11)."""

    model_config = ConfigDict(extra="forbid")
    assay_version: str
    python: str
    platform: str
    versions: dict[str, str]


class Artifact(BaseModel):
    """The executable artifact — the fourth part of every answer (PRD §10)."""

    model_config = ConfigDict(extra="forbid")
    artifact_version: Literal[1] = 1
    ir: IR
    template: Template
    answer: Answer
    environment: Environment


class Reproduction(BaseModel):
    """The report of a rerun — exact vs within-tolerance stated, never blurred (NFR-2)."""

    model_config = ConfigDict(extra="forbid")
    status: Literal["exact", "within-tolerance", "failed"]
    detail: str
    answer: Answer  # the recomputed answer, under the current environment


def capture_environment() -> Environment:
    return Environment(
        assay_version=__version__,
        python=platform_module.python_version(),
        platform=platform_module.platform(),
        versions={name: metadata.version(name) for name in _PINNED},
    )


def _method_line(template: Template) -> str:
    if isinstance(template.method, FormulaMethod):
        if template.method.steps:  # the DAG (E2.9), rendered as its assignments
            return " ; ".join(f"{step.name} = {step.expr}" for step in template.method.steps)
        return template.method.expr or ""
    if isinstance(template.method, SolverMethod):
        return f"solver: {template.method.binding}"
    if isinstance(template.method, CasesMethod):
        cases = ", ".join(sorted(template.method.cases))
        return f"cases[{template.method.discriminator}]: {cases}"
    return f"{template.method.operation} (SymPy)"


def build_answer(
    ir: IR, template: Template, verified: VerifiedExecution, environment: Environment
) -> Answer:
    """Assemble the four-part answer (PRD §10) from the pipeline's stages.

    A withheld result stays withheld: ``result`` is empty and the verification verdicts
    carry the reason (A-6). Facts come from the IR's ``resolved`` entries with their
    exact sources (A-11).
    """
    result = []
    if verified.result is not None:
        result = [
            ResultValue(label=value.label, value=value.value, unit=value.unit)
            for value in verified.result.values
        ]
    facts = [
        Fact(name=name, value=fact.value, unit=fact.unit, source=fact.source)
        for name, fact in sorted(ir.resolved.items())
    ]
    parts = [template.description or template.id]
    if ir.solve_for is not None and ir.given_output is not None:
        parts.append(
            f"solved for {ir.solve_for} given {template.output.name} ="
            f" {ir.given_output.value:g} {ir.given_output.unit}".rstrip()
        )
    parts += [
        f"{key} = {value}"
        for key, value in sorted(ir.setup.items())
        if isinstance(value, str)
    ]
    parts += template.assumptions
    return Answer(
        result=result,
        interpretation=" · ".join(parts),
        method=_method_line(template),
        # the trace travels only with a VERIFIED result — a withheld answer keeps
        # its steps in the candidate, never in the answer (A-6)
        steps=(
            list(verified.result.trace)
            if verified.result is not None and verified.verification.ok
            else []
        ),
        facts=facts,
        verified=verified.verification,
        ir_hash=ir.content_hash(),
        assay_version=environment.assay_version,
        versions=environment.versions,
    )


def create_artifact(ir: IR, template: Template) -> Artifact:
    """Run the pipeline on a validated IR (A-1) and record it: execute + verify (E1.2,
    E1.4), assemble the answer, and capture the environment."""
    verified = verify_ir(ir, template)
    environment = capture_environment()
    answer = build_answer(ir, template, verified, environment)
    return Artifact(ir=ir, template=template, answer=answer, environment=environment)


def save_artifact(artifact: Artifact, path: str | Path) -> Path:
    """Write the artifact as JSON. Deterministic: no timestamps, stable field order —
    the same computation writes byte-identical files."""
    destination = Path(path)
    destination.write_text(artifact.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return destination


def load_artifact(path: str | Path) -> Artifact:
    """Read + validate an artifact file; a schema this version can't honor is refused
    by validation (engineering §10), never mis-executed."""
    return Artifact.model_validate_json(Path(path).read_text(encoding="utf-8"))


def artifact_key(artifact: Artifact) -> str:
    """The store key: IR content hash + an environment digest — the cache doctrine
    (PRD §6): the hash names the computation, the environment names what ran it."""
    environment_digest = content_sha256(artifact.environment.model_dump(mode="json"))[:12]
    return f"{artifact.ir.content_hash()}.{environment_digest}"


def store_artifact(store: Store, artifact: Artifact) -> str:
    """Persist to the embedded store's ``artifacts`` namespace (E0.2) and append the
    lineage record; returns the key."""
    key = artifact_key(artifact)
    store.put("artifacts", key, artifact.model_dump_json())
    store.append_lineage("artifact", key)
    return key


def fetch_artifact(store: Store, key: str) -> Artifact | None:
    text = store.get("artifacts", key)
    return None if text is None else Artifact.model_validate_json(text)


def _compare_values(
    recorded: list[ResultValue], current: list[ResultValue]
) -> tuple[str, str]:
    """Compare result values: ``exact`` / ``within-tolerance`` / ``failed`` + detail."""
    if [(v.label, v.unit) for v in recorded] != [(v.label, v.unit) for v in current]:
        return "failed", "the result's shape (labels/units) changed on rerun"
    verdict = "exact"
    for old, new in zip(recorded, current, strict=True):
        if isinstance(old.value, float) and isinstance(new.value, float):
            if old.value == new.value:
                continue
            if math.isclose(old.value, new.value, rel_tol=_RERUN_RTOL, abs_tol=_RERUN_ATOL):
                verdict = "within-tolerance"
                continue
            return "failed", f"{old.label}: {old.value!r} reran as {new.value!r}"
        if isinstance(old.value, str) and isinstance(new.value, str):
            if old.value == new.value:
                continue
            if symbolically_equal(old.value, new.value):
                verdict = "within-tolerance"
                continue
            return "failed", f"{old.label}: {old.value!r} reran as {new.value!r}"
        return "failed", f"{old.label}: the value's kind changed on rerun"
    return verdict, ""


def _environment_drift(recorded: Environment, current: Environment) -> list[str]:
    """What changed between the artifact's environment and this one — the diagnosis
    that must accompany any non-exact verdict (E3.4, NFR-2)."""
    drift: list[str] = []
    for name in sorted(set(recorded.versions) | set(current.versions)):
        old, new = recorded.versions.get(name, "?"), current.versions.get(name, "?")
        if old != new:
            drift.append(f"{name} {old}→{new}")
    if recorded.assay_version != current.assay_version:
        drift.append(f"assay {recorded.assay_version}→{current.assay_version}")
    if recorded.python != current.python:
        drift.append(f"python {recorded.python}→{current.python}")
    if recorded.platform != current.platform:
        drift.append(f"platform {recorded.platform} → {current.platform}")
    return drift


def _drift_clause(recorded: Environment, current: Environment) -> str:
    drift = _environment_drift(recorded, current)
    if drift:
        return "environment drift: " + ", ".join(drift)
    return (
        "NO environment drift — same platform and pinned versions should reproduce"
        " exactly (NFR-2); this may be a reproducibility bug worth reporting"
    )


def rerun(artifact: Artifact) -> Reproduction:
    """Re-execute the artifact and report honestly (A-7, NFR-2): exact, within-tolerance
    (naming the environment drift), or failed (naming the mismatch)."""
    if artifact.ir.ir_version != 1:
        raise ArtifactError(
            f"cannot honor ir_version {artifact.ir.ir_version} — refusing rather than"
            " mis-executing (engineering §10)"
        )
    verified = verify_ir(artifact.ir, artifact.template)
    environment = capture_environment()
    answer = build_answer(artifact.ir, artifact.template, verified, environment)
    versions = ", ".join(f"{k} {v}" for k, v in sorted(environment.versions.items()))
    if not verified.verification.ok:
        if artifact.answer.verified == answer.verified:
            return Reproduction(
                status="exact",
                detail=f"the withheld verdict reproduced identically; {versions}",
                answer=answer,
            )
        return Reproduction(
            status="failed",
            detail="verification failed on rerun: "
            + "; ".join(c.detail for c in answer.verified.checks if not c.ok)
            + f" ({_drift_clause(artifact.environment, environment)})",
            answer=answer,
        )
    status, mismatch = _compare_values(artifact.answer.result, answer.result)
    if status == "failed":
        return Reproduction(
            status="failed",
            detail=f"{mismatch} ({_drift_clause(artifact.environment, environment)})",
            answer=answer,
        )
    if status == "exact":
        return Reproduction(status="exact", detail=f"identical; {versions}", answer=answer)
    return Reproduction(
        status="within-tolerance",
        detail=f"within tolerance; {_drift_clause(artifact.environment, environment)}",
        answer=answer,
    )
