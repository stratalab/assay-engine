"""The catalog taxonomy (subject → topic → domain) — curated order over 400+ templates.

The governing rule: **every domain in the catalog is placed exactly once, and every
placement points at a real domain** — the lockstep test makes an unplaced domain a red
CI, so the catalog cannot silently sprawl. The `domain` strings themselves (the Chisel
contract) are untouched; the taxonomy is Assay-side curation, like the resolver tables.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from assay.api import create_app
from assay.cli import main
from assay.templates import domain_placement, golden_templates, taxonomy
from assay.templates.plugins import full_catalog


def test_every_domain_is_placed_exactly_once_lockstep() -> None:
    placement = domain_placement()  # raises on a double placement
    catalog_domains = {template.domain for template in full_catalog()}
    unplaced = catalog_domains - set(placement)
    assert not unplaced, f"domains shipping without a taxonomy home: {sorted(unplaced)}"
    dead = set(placement) - catalog_domains
    assert not dead, f"taxonomy places domains that no longer exist: {sorted(dead)}"


def test_the_near_duplicate_domains_are_gone() -> None:
    """The mess this taxonomy was built to stop: our goldens' stray singletons
    (oscillation/electricity/astronomy) are aligned to the shared vocabulary."""
    domains = {template.domain for template in golden_templates()}
    assert "oscillation" not in domains and "oscillations" in domains
    assert "electricity" not in domains and "electromagnetism" in domains
    assert "astronomy" not in domains and "gravitation" in domains


def test_the_tree_has_the_expected_shape() -> None:
    subjects = taxonomy()["subjects"]
    assert set(subjects) == {"mathematics", "physics", "chemistry", "engineering"}
    assert "mechanics" in subjects["physics"]["topics"]
    assert set(subjects["physics"]["topics"]["mechanics"]["domains"]) >= {
        "kinematics", "dynamics", "oscillations", "gravitation",
    }


def test_cli_domains_renders_the_hierarchy(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["domains"]) == 0
    out = capsys.readouterr().out
    for heading in ("Mathematics", "Physics", "Chemistry", "Engineering"):
        assert heading in out
    assert "Mechanics" in out and "Waves" in out
    # the tier labels and template lines survive under the new grouping
    assert "[shipped]" in out  # the corpus tier renders when content batches are installed
    assert out.index("Physics") < out.index("Mechanics")  # topic nests under subject


def test_api_carries_the_placement() -> None:
    client = TestClient(create_app())
    entries = client.get("/v1/domains").json()
    by_id = {entry["id"]: entry for entry in entries}
    beam = by_id["beam_deflection.simply_supported.center_point"]
    assert (beam["subject"], beam["topic"]) == ("engineering", "structures")
    pendulum = by_id["pendulum.period.simple"]
    assert (pendulum["subject"], pendulum["topic"]) == ("physics", "mechanics")
    assert all(entry["subject"] for entry in entries)  # nothing unplaced over the wire
    tree = client.get("/v1/taxonomy").json()
    assert set(tree["subjects"]) == {"mathematics", "physics", "chemistry", "engineering"}