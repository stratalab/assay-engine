"""E2.2: the candidate → verified tier gate (A-14, PRD §7.2, chisel-alignment §7).

The done-criteria: a candidate with failing fixtures stays quarantined; one with
passing fixtures promotes and serves. Plus the axis discipline: ``verified`` is minted
only by the gate (a claimed label is re-earned, never imported), and promotion never
reads or writes the license axis.
"""

from __future__ import annotations

from typing import Any

import pytest

from assay.templates import (
    CandidateTemplateError,
    Template,
    TemplateRegistry,
    golden_templates,
    validate_template,
)
from assay.templates.promote import (
    PIPELINE_FIXTURE_FLOOR,
    PromotionError,
    fixture_gate,
    ingest,
    promote,
)


def _candidate(**overrides: Any) -> Template:
    """A minimal formula candidate whose one fixture passes (unless overridden)."""
    record: dict[str, Any] = {
        "schema_version": 1,
        "id": "test.hookes_law",
        "domain": "mechanics",
        "description": "Spring force at a given extension.",
        "inputs": [
            {"name": "k", "dimension": "force/length"},
            {"name": "x", "dimension": "length"},
        ],
        "method": {"kind": "formula", "expr": "k * x"},
        "output": {"name": "force", "dimension": "force"},
        "fixtures": [
            {
                "inputs": {"k": [10, "N/m"], "x": [2, "m"]},
                "expect": {"force": [20.0, "N"]},
                "tol": 1e-9,
            }
        ],
        "provenance": {"source": "test:hand-authored", "license_tier": "open"},
    }
    record.update(overrides)
    return validate_template(record)


def _failing_candidate() -> Template:
    return _candidate(
        fixtures=[
            {
                "inputs": {"k": [10, "N/m"], "x": [2, "m"]},
                "expect": {"force": [99.0, "N"]},  # wrong on purpose
                "tol": 1e-9,
            }
        ]
    )


def test_passing_candidate_promotes_and_serves() -> None:
    template = _candidate()
    promoted = promote(template)
    assert promoted.provenance.status == "verified"
    registry = TemplateRegistry()
    registry.register(promoted)
    assert registry.get("test.hookes_law").provenance.status == "verified"  # serves


def test_failing_candidate_stays_quarantined() -> None:
    template = _failing_candidate()
    with pytest.raises(PromotionError, match="stays candidate") as excinfo:
        promote(template)
    assert "99.0" in str(excinfo.value)  # the failing fixture's detail, stated
    registry = TemplateRegistry()
    report = ingest(template, registry)
    assert not report.promoted and len(report.failures) == 1
    with pytest.raises(CandidateTemplateError):  # quarantined: refused by default
        registry.get("test.hookes_law")
    assert registry.ids(status="candidate") == ["test.hookes_law"]  # …but visible
    opted_in = registry.get("test.hookes_law", allow_candidate=True)  # UX §5.7 opt-in
    assert opted_in.provenance.status == "candidate"


def test_promote_is_pure_and_only_touches_the_trust_axis() -> None:
    template = _candidate(
        provenance={"source": "test:restricted-source", "license_tier": "restricted"}
    )
    promoted = promote(template)
    assert template.provenance.status == "candidate"  # the input is untouched
    assert promoted.provenance.status == "verified"
    # the license axis is not promotion's question (chisel-alignment §7): a
    # restricted-license template still promotes on correctness, and stays restricted —
    # correctness never launders licensing. ("unknown" never gets this far: the seam
    # refuses it at validation, round 2.)
    assert promoted.provenance.license_tier == "restricted"
    assert promoted.provenance.source == "test:restricted-source"


def test_a_claimed_verified_label_is_not_imported() -> None:
    """The label is re-earned at the gate, in both directions."""
    liar = _failing_candidate().model_copy(deep=True)
    liar.provenance.status = "verified"  # claims trust it hasn't earned
    with pytest.raises(PromotionError):
        promote(liar)  # the label does not bypass the gate
    registry = TemplateRegistry()
    report = ingest(liar, registry)
    assert not report.promoted
    with pytest.raises(CandidateTemplateError):  # ingested as candidate, not as claimed
        registry.get("test.hookes_law")


def test_ingest_validates_raw_records_through_the_seam() -> None:
    registry = TemplateRegistry()
    report = ingest(
        {
            "schema_version": 1,
            "id": "test.identity",
            "domain": "algebra",
            "inputs": [{"name": "x", "dimension": "dimensionless"}],
            "method": {"kind": "formula", "expr": "x"},
            "output": {"name": "y", "dimension": "dimensionless"},
            "fixtures": [
                {"inputs": {"x": [3, ""]}, "expect": {"y": [3.0, ""]}, "tol": 1e-12}
            ],
            "provenance": {"source": "test:record", "license_tier": "open"},
        },
        registry,
    )
    assert report.promoted
    assert registry.get("test.identity").provenance.status == "verified"


def test_fixture_gate_reports_without_side_effects() -> None:
    report = fixture_gate(_failing_candidate())
    assert not report.promoted
    assert report.template_id == "test.hookes_law"
    assert report.failures[0].detail  # each failure carries its reason


def test_every_shipped_golden_passes_the_gate() -> None:
    for template in golden_templates():
        assert template.provenance.status == "candidate"  # shipped honest
        assert promote(template).provenance.status == "verified"


def test_the_fixture_floor_holds(  # the round-2 ruling (chisel-alignment §14.3)
) -> None:
    template = _candidate()  # one green fixture
    assert promote(template).provenance.status == "verified"  # hand-authored floor: 1
    report = fixture_gate(template, floor=PIPELINE_FIXTURE_FLOOR)
    assert not report.promoted and not report.failures  # green, but not enough proof
    with pytest.raises(PromotionError, match="below the promotion floor of 3"):
        promote(template, floor=PIPELINE_FIXTURE_FLOOR)
    registry = TemplateRegistry()
    ingested = ingest(template, registry, floor=PIPELINE_FIXTURE_FLOOR)
    assert not ingested.promoted  # registers quarantined: visible, count on record
    with pytest.raises(CandidateTemplateError):
        registry.get("test.hookes_law")
