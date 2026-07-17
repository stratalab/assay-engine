"""E2.14: the coverage map — targeted growth, not bulk ingestion.

The map (subject → field → topic, each with pending / in-progress / complete) is the
planning counterpart of the taxonomy. The lockstep rules keep it honest against the
shipped catalog: statuses cannot claim coverage that does not exist, and shipped
content cannot go unclaimed — so the map is always a true statement about where
Assay stands.
"""

from __future__ import annotations

from typing import Any

from assay.templates import coverage
from assay.templates.plugins import full_catalog

_STATUSES = {"pending", "in-progress", "complete"}


def _topics() -> list[tuple[str, str, str, dict[str, Any]]]:
    flattened = []
    for subject, subject_node in coverage()["subjects"].items():
        for field, field_node in subject_node["fields"].items():
            for topic, node in field_node["topics"].items():
                flattened.append((subject, field, topic, node))
    return flattened


def test_the_map_is_well_formed() -> None:
    topics = _topics()
    assert len(topics) >= 5  # the demo catalog map (the full map is private)
    for subject, field, topic, node in topics:
        where = f"{subject}/{field}/{topic}"
        assert node["status"] in _STATUSES, where
        assert isinstance(node.get("domains"), list), where
        assert node.get("source"), f"{where}: every topic names its intended source"


def test_subjects_and_fields_are_cip_anchored() -> None:
    """The subject/field vocabulary is the published federal taxonomy (CIP 2020,
    NCES — public domain), not an invented one; topics stay Assay-operational."""
    import re

    # a subject anchors to a CIP series (14) or sub-series (40.08); a field to
    # 4-digit program codes (14.0801)
    series = re.compile(r"^\d{2}(\.\d{2})?$")
    program = re.compile(r"^\d{2}\.\d{4}$")
    for subject, subject_node in coverage()["subjects"].items():
        assert series.match(subject_node.get("cip", "")), f"{subject}: needs a CIP series"
        for field, field_node in subject_node["fields"].items():
            codes = field_node.get("cip", [])
            assert codes and all(program.match(c) for c in codes), (
                f"{subject}/{field}: needs CIP program codes (NN.NNNN)"
            )


def test_statuses_are_honest_against_the_catalog() -> None:
    live = {t.domain for t in full_catalog()}
    for subject, field, topic, node in _topics():
        where = f"{subject}/{field}/{topic}"
        claimed = set(node["domains"])
        # nothing may claim a domain that does not ship
        assert claimed <= live, f"{where} claims unshipped domains: {claimed - live}"
        if node["status"] == "pending":
            assert not claimed, f"{where} is pending but claims shipped domains"
        if node["status"] == "complete":
            assert claimed, f"{where} is complete but claims no domains"


def test_every_shipped_domain_is_claimed() -> None:
    """The reverse lockstep: content cannot ship outside the map — a new domain
    forces a same-day coverage entry, exactly like the taxonomy placement rule."""
    live = {t.domain for t in full_catalog()}
    claimed: set[str] = set()
    for _subject, _field, _topic, node in _topics():
        if node["status"] != "pending":
            claimed |= set(node["domains"])
    assert live <= claimed, f"unclaimed shipped domains: {sorted(live - claimed)}"


def test_pending_topics_name_their_blockers_or_sources() -> None:
    """A pending topic is a plan, not a wish: it names the source to extract and,
    when blocked on engine work, the gate."""
    for subject, field, topic, node in _topics():
        if node["status"] == "pending":
            assert node.get("source") or node.get("gate"), (
                f"{subject}/{field}/{topic}: pending without a source or gate"
            )
