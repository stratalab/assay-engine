"""E3.2: the web glass box — the answer object, rendered (UX §4, §9.5).

A minimal single-file UI served by the API app at ``/``: concise answer first, the
IR / method / provenance drill-down one click deep (progressive disclosure, never an
overwhelming form), every honest state rendered as itself. Self-contained by doctrine:
inline CSS/JS, no external assets, no build step — the page works offline against its
own server, like everything else Assay ships.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from assay.api import create_app


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(create_app())


def test_the_glass_box_serves(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    html = response.text
    assert "the glass box" in html
    assert "/v1/ask" in html and "/v1/run" in html and "/v1/domains" in html


def test_the_page_is_self_contained(client: TestClient) -> None:
    """No CDN, no external fonts, no remote scripts — the page carries everything
    (the same self-contained doctrine as the engine)."""
    html = client.get("/").text
    assert 'src="http' not in html and "src='http" not in html
    assert 'href="http' not in html and "href='http" not in html
    assert "@import" not in html and "cdn." not in html


def test_the_four_bands_and_honest_states_are_rendered(client: TestClient) -> None:
    html = client.get("/").text
    for band in ("Interpretation", "Method", "Facts", "Verified"):
        assert band in html
    for state in ("missing_inputs", "ambiguous", "out_of_scope"):
        assert state in html  # each honest state has its own rendering, not an alert
    assert "I won't guess." in html
    assert "nothing will be fabricated" in html
    assert "withheld" in html  # the verification-failed posture, present in the UI


def test_the_drilldown_is_data_not_a_form(client: TestClient) -> None:
    """UX §9.5: the glass box shows the IR as data (collapsible JSON), and the only
    inputs ever rendered are the missing ones."""
    html = client.get("/").text
    assert "IR — what was understood, as data" in html
    assert "Artifact — rerun this exact computation" in html
    assert "[resolved, not assumed]" in html