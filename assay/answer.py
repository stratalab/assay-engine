"""The four-part answer object (PRD §10) — one shape for terminal + JSON.

result / interpretation / method / facts / verified / artifact, plus an optional figure
(PRD §10.1 — a rendering of verified data, never a result). Built by M1; E0.3 defines the
type. The same object is rendered concisely in a terminal and serialized to JSON for an
agent (UX §2, §6).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from assay.ir import FactSource

__all__ = [
    "Answer",
    "Fact",
    "Figure",
    "ResultValue",
    "TraceStep",
    "Verification",
    "VerificationCheck",
]


class ResultValue(BaseModel):
    """One value in the answer's result — numeric (with unit) or symbolic (a string)."""

    model_config = ConfigDict(extra="forbid")
    label: str
    value: float | str
    unit: str = ""


class Fact(BaseModel):
    """A resolved fact shown in the answer, with its source (no fabricated facts, PRD §8)."""

    model_config = ConfigDict(extra="forbid")
    name: str
    value: float
    unit: str = ""
    source: FactSource


class TraceStep(BaseModel):
    """One step of the execution trace (E2.15) — the literal computation record, not
    a narration: each entry is what the executor actually evaluated, in base units,
    inside the same run that produced the answer. Step-by-step that reruns."""

    model_config = ConfigDict(extra="forbid")
    label: str  # the step (or output/target) name
    expr: str | None = None  # the expression as authored in the template
    value: float
    unit: str = ""
    note: str = ""  # e.g. the selected case, or the solve-for provenance


class VerificationCheck(BaseModel):
    """One verification check and its verdict (PRD §9)."""

    model_config = ConfigDict(extra="forbid")
    name: str  # e.g. "dimension:length", "bounds", "cross-method"
    ok: bool
    detail: str = ""


class Verification(BaseModel):
    """The answer's verification status — honest by construction (UX §3)."""

    model_config = ConfigDict(extra="forbid")
    ok: bool
    checks: list[VerificationCheck] = []


class Figure(BaseModel):
    """A rendering of verified data (PRD §10.1) — attached, labelled a rendering."""

    model_config = ConfigDict(extra="forbid")
    path: str
    kind: str = ""


class Answer(BaseModel):
    """The answer object. ``ir_hash`` links to the IR that produced it; ``assay_version``
    + ``versions`` are the reproducibility record (PRD §10)."""

    model_config = ConfigDict(extra="forbid")

    result: list[ResultValue] = []
    interpretation: str = ""
    method: str = ""
    # The execution trace (E2.15): the record of what was actually evaluated, step
    # by step — empty when the answer is withheld or the method has no numeric steps.
    steps: list[TraceStep] = []
    facts: list[Fact] = []
    verified: Verification
    figure: Figure | None = None
    ir_hash: str = ""
    assay_version: str = ""
    versions: dict[str, str] = {}  # pinned library versions (reproducibility, PRD §10)
