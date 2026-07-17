"""The intermediate representation (PRD §6): the single execution contract.

An ``IR`` is the structured, inspectable, executable form of a question — validated by
pydantic (A-1), content-addressed by a stable hash (A-11), and JSON round-trippable.
Producers (a model, the CLI, a hand-authored file) create one; the executor consumes it
(E1.2). The compute/render split lives here too: an optional ``render`` directive is a
*view* of the computed result, never a result (PRD §10.1).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from assay.hashing import content_sha256

__all__ = ["IR", "FactSource", "Producer", "Quantity", "RenderDirective", "ResolvedFact"]


class Producer(BaseModel):
    """Who produced this IR from natural language (E2.7, engineering §11 Q1).

    Provenance, not content: a different NL→IR model can yield a different
    (still-validated) IR, so the producer is recorded for audit — but two identical
    IRs are the *same computation* whoever wrote them, so this field is excluded from
    the content hash (like ``query``).
    """

    model_config = ConfigDict(extra="forbid")
    provider: str
    model: str


class Quantity(BaseModel):
    """A numeric value with a unit — an IR input (PRD §6)."""

    model_config = ConfigDict(extra="forbid")
    value: float
    unit: str = ""


class FactSource(BaseModel):
    """Where a resolved fact came from — the no-fabricated-facts audit trail (PRD §8)."""

    model_config = ConfigDict(extra="forbid")
    library: str
    key: str
    version: str


class ResolvedFact(BaseModel):
    """A fact the resolver filled from a trusted source — never the model (PRD §8)."""

    model_config = ConfigDict(extra="forbid")
    value: float
    unit: str = ""
    source: FactSource


class RenderDirective(BaseModel):
    """An optional view spec (PRD §10.1): render verified data, never an answer."""

    model_config = ConfigDict(extra="forbid")
    kind: str  # function_plot | scatter | geometry_diagram | ...
    spec: dict[str, Any] = {}


class IR(BaseModel):
    """The intermediate representation — the single execution contract (A-1)."""

    model_config = ConfigDict(extra="forbid")

    assay_version: str = ""
    ir_version: int = 1
    query: str = ""
    produced_by: Producer | None = None  # NL→IR attribution (E2.7); None = hand-built
    domain: str
    task: str
    setup: dict[str, Any] = {}
    # scalar inputs: one Quantity; list inputs (many, schema_version 2): a list
    inputs: dict[str, Quantity | list[Quantity]] = {}
    missing_inputs: list[str] = []
    # Solve-for (E2.10): recover `solve_for` (a declared input) given `given_output`
    # (the stated value of the template's output). Computational fields — both are IN
    # the content hash. None/None for ordinary forward execution.
    solve_for: str | None = None
    given_output: Quantity | None = None
    resolved: dict[str, ResolvedFact] = {}
    assumptions: list[str] = []
    execution_plan: list[str] = []
    outputs: list[str] = []
    render: RenderDirective | None = None

    def content_hash(self) -> str:
        """Stable content hash (A-11): canonical JSON over the *computational* fields —
        excluding ``assay_version`` (software version), ``query`` and ``produced_by``
        (NL provenance: the same IR is the same computation whoever wrote it) — then
        sha256. Order-independent over dict keys; identical content, identical hash.
        """
        return content_sha256(
            self.model_dump(mode="json", exclude={"assay_version", "query", "produced_by"})
        )
