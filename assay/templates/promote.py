"""Candidate → verified: the fixture gate (E2.2, A-14, PRD §7.2).

Extraction never becomes authority silently: every template arrives ``status:
candidate`` (Chisel stamps it on emit; hand-authored ones default to it) and is served
only after **its own fixtures pass under the generic executor** — the correctness proof
it carries with it. ``promote`` is the one place ``verified`` is minted, and the label
never bypasses the gate: a template *claiming* ``verified`` whose fixtures fail is
refused all the same.

The two tier axes stay distinct (chisel-alignment §7): promotion judges only the trust
axis (``status``) and never reads or writes the license axis
(``provenance.license_tier``) — a correctly-computing template from an unknown-license
source promotes, and what the license permits is a separate question answered at
distribution time (E2.3).

This is a submodule so the seam stays dependency-light (A-15): ``import
assay.templates`` still pulls pydantic + stdlib only; the executor is imported here,
where promotion actually runs fixtures.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict

from assay.execute import FixtureResult, run_fixtures
from assay.templates import Template, TemplateRegistry, validate_template

__all__ = [
    "PIPELINE_FIXTURE_FLOOR",
    "PromotionError",
    "PromotionReport",
    "fixture_gate",
    "ingest",
    "promote",
]

# The fixture-floor ruling (chisel-alignment round 2, §14.3): a PIPELINE-emitted
# template promotes only with at least this many independent passing fixtures — an
# emitter reports true counts and emits below-floor templates as visible candidates,
# never manufactures fixtures to clear it. Hand-authored templates default to floor 1
# (they carry maintainer review instead); the E2.3 ingest passes this constant.
PIPELINE_FIXTURE_FLOOR = 3


class PromotionError(Exception):
    """The template stays candidate: its fixtures failed under the generic executor
    (or fell below the promotion floor). The message names every reason (A-12)."""

    def __init__(
        self, template_id: str, failures: list[FixtureResult], reason: str | None = None
    ) -> None:
        self.template_id = template_id
        self.failures = failures
        detail = reason or (
            "fixtures failed under the generic executor: "
            + "; ".join(f"fixture {result.index}: {result.detail}" for result in failures)
        )
        super().__init__(f"template {template_id!r} stays candidate — {detail}")


class PromotionReport(BaseModel):
    """One gate run: the verdict and every fixture's result — the audit trail a
    promotion decision leaves behind."""

    model_config = ConfigDict(extra="forbid")
    template_id: str
    promoted: bool
    fixtures: list[FixtureResult]

    @property
    def failures(self) -> list[FixtureResult]:
        return [result for result in self.fixtures if not result.ok]


def fixture_gate(template: Template, *, floor: int = 1) -> PromotionReport:
    """Run the template's fixtures under the generic executor and report the verdict —
    the gate itself, with no side effects (Chisel's local test-promote,
    chisel-alignment §10). ``floor`` is the promotion floor: fewer than ``floor``
    fixtures (however green) is not enough proof to promote."""
    results = run_fixtures(template)  # the schema guarantees at least one fixture
    return PromotionReport(
        template_id=template.id,
        promoted=all(result.ok for result in results) and len(results) >= floor,
        fixtures=results,
    )


def promote(template: Template, *, floor: int = 1) -> Template:
    """Return a ``verified`` copy of the template iff every fixture passes — and at
    least ``floor`` of them exist (A-14; the round-2 fixture-floor ruling).

    Pure: the input template is untouched, and the returned copy differs only in
    ``provenance.status`` — never the license axis. Raises ``PromotionError`` (naming
    every reason) when the gate refuses; a pre-existing ``verified`` label does not
    skip the gate.
    """
    report = fixture_gate(template, floor=floor)
    if not report.promoted:
        if report.failures:
            raise PromotionError(template.id, report.failures)
        raise PromotionError(
            template.id,
            [],
            reason=(
                f"{len(report.fixtures)} passing fixture(s) is below the promotion floor"
                f" of {floor} — more independent proof is needed, never manufactured"
            ),
        )
    if template.provenance.status == "verified":
        return template
    return template.model_copy(
        update={"provenance": template.provenance.model_copy(update={"status": "verified"})}
    )


def ingest(
    record: Template | Mapping[str, Any], registry: TemplateRegistry, *, floor: int = 1
) -> PromotionReport:
    """The ingest path (PRD §7.2): validate (the A-15 seam) → gate → register.

    A passing candidate registers **promoted** and serves; a failing one registers
    **quarantined** — a candidate (even if the record *claimed* ``verified``: the label
    is re-earned here, never imported), visible in ``registry.ids(status="candidate")``
    and inspectable via ``get(..., allow_candidate=True)`` (the explicit UX §5.7 opt-in),
    but refused by default. Either way the report says exactly what happened.
    """
    template = record if isinstance(record, Template) else validate_template(record)
    report = fixture_gate(template, floor=floor)
    if report.promoted:
        registry.register(promote(template, floor=floor))
    else:
        registry.register(_demoted(template))
    return report


def _demoted(template: Template) -> Template:
    if template.provenance.status == "candidate":
        return template
    return template.model_copy(
        update={"provenance": template.provenance.model_copy(update={"status": "candidate"})}
    )
