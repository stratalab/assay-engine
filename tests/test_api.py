"""E2.6: the HTTP API — the answer object over the wire (PRD §12, UX §6).

The done-criterion: an agent gets a stable, citable, reproducible JSON answer. Citable:
``facts[].source`` names the library/key/version. Reproducible: the response carries
the full artifact, and POSTing it back to /v1/rerun reproduces exactly. Stable: the
response IS the ``Answer`` object — one shape forever — and every honest state is a
first-class ``outcome``.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from assay.api import create_app

_BEAM_QUESTION = (
    "max deflection of a simply supported steel beam,"
    " 5 kN center load, 2 m span, I = 8.33e-6 m^4"
)


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(create_app())


def test_answer_is_citable_and_verified(client: TestClient) -> None:
    response = client.post("/v1/ask", json={"question": _BEAM_QUESTION})
    assert response.status_code == 200
    body = response.json()
    assert body["outcome"] == "answer"
    answer = body["answer"]
    assert answer["verified"]["ok"] is True
    assert answer["verified"]["checks"]  # per-check verdicts, not a bare boolean
    (value,) = answer["result"]
    assert value["label"] == "max_deflection"
    assert value["value"] * 1000 == pytest.approx(0.50, abs=0.005)
    (fact,) = answer["facts"]  # citable: the fact names its source
    assert fact["source"] == {
        "library": "assay.materials",
        "key": "steel.structural.E",
        "version": "0.1",
    }
    assert answer["ir_hash"] and answer["versions"]  # the reproducibility record


def test_artifact_round_trips_through_rerun(client: TestClient) -> None:
    """Reproducible: the artifact from one call reruns exactly via another —
    stateless on the server, bit-for-bit for the caller."""
    asked = client.post("/v1/ask", json={"question": "solve x^2 - 5x + 6 = 0"}).json()
    assert asked["outcome"] == "answer"
    rerun = client.post("/v1/rerun", json={"artifact": asked["artifact"]})
    assert rerun.status_code == 200
    assert rerun.json()["status"] == "exact"


def test_missing_inputs_fail_clear_and_resume_via_run(client: TestClient) -> None:
    """The API never prompts and never fabricates: it says exactly what's missing and
    returns the understood-so-far IR; the caller completes it and POSTs /v1/run."""
    question = "max deflection of a simply supported steel beam, 5 kN center load, 2 m span"
    body = client.post("/v1/ask", json={"question": question}).json()
    assert body["outcome"] == "missing_inputs"
    (needed,) = body["needed"]
    assert needed["name"] == "I" and needed["dimension"] == "length**4"
    assert "will not be fabricated" in needed["reason"]
    assert body["ir"]["resolved"]["E"]["source"]["library"] == "assay.materials"

    completed = dict(body["ir"])
    completed["inputs"] = {**completed["inputs"], "I": {"value": 8.33e-6, "unit": "m**4"}}
    completed["missing_inputs"] = []
    ran = client.post("/v1/run", json={"ir": completed}).json()
    assert ran["outcome"] == "answer" and ran["answer"]["verified"]["ok"] is True


def test_out_of_scope_refuses(client: TestClient) -> None:
    body = client.post(
        "/v1/ask", json={"question": "simulate turbulent flow over an airfoil at Mach 0.8"}
    ).json()
    assert body["outcome"] == "out_of_scope"
    assert "no task template matches" in body["reason"]
    assert "algebra" in body["covered"]


def test_hostile_ir_is_refused_with_the_reason(client: TestClient) -> None:
    ir = {
        "domain": "algebra",
        "task": "solve_equation.univariate",
        "setup": {"expression": "__import__('os').system('true')"},
    }
    response = client.post("/v1/run", json={"ir": ir})
    assert response.status_code == 400
    assert "rejected" in response.json()["error"]


def test_unknown_task_is_404(client: TestClient) -> None:
    ir = {"domain": "alchemy", "task": "transmute.lead_to_gold"}
    response = client.post("/v1/run", json={"ir": ir})
    assert response.status_code == 400  # the pre-execution gate names the guess
    assert "refusing to execute a guess" in response.json()["error"]


def test_domains_lists_the_catalog(client: TestClient) -> None:
    body = client.get("/v1/domains").json()
    ids = {entry["id"] for entry in body}
    assert "solve_equation.univariate" in ids and "pendulum.period.simple" in ids
    assert all(entry["status"] == "candidate" for entry in body)  # honest until gated


def test_operations_endpoint_maps_ops_to_runners(client: TestClient) -> None:
    """The web Operations panel needs op → template; each entry runs via /v1/run."""
    ops = {o["operation"]: o for o in client.get("/v1/operations").json()}
    assert {"gradient", "ode_solve", "divergence", "curl", "integrate_multiple"} <= set(ops)
    assert ops["gradient"]["task"] and ops["gradient"]["domain"]


def test_structured_operation_runs_through_run(client: TestClient) -> None:
    """The panel's path: a structured IR for ops the NL box can't reach from prose —
    ode_solve (setup 'equation', not 'expression') and a vector-valued gradient. The
    task ids come from /v1/operations, so this holds whatever catalog is installed."""
    ops = {o["operation"]: o for o in client.get("/v1/operations").json()}
    ode_task = ops["ode_solve"]
    ode = client.post("/v1/run", json={"ir": {
        "domain": ode_task["domain"], "task": ode_task["task"],
        "setup": {"equation": "y'' + 3*y' - 4*y = 0", "ivp": {"y": [0, 1], "y'": [0, -9]}},
    }}).json()
    assert ode["outcome"] == "answer" and ode["answer"]["verified"]["ok"]
    grad_task = ops["gradient"]
    grad = client.post("/v1/run", json={"ir": {
        "domain": grad_task["domain"], "task": grad_task["task"],
        "setup": {"expression": "x**2 - x*y + 3*y**2", "variables": ["x", "y"]},
    }}).json()
    assert [v["value"] for v in grad["answer"]["result"]] == ["2*x - y", "-x + 6*y"]


def test_health_names_the_pinned_versions(client: TestClient) -> None:
    body = client.get("/v1/health").json()
    assert body["status"] == "ok"
    assert "sympy" in body["versions"] and "pint" in body["versions"]


def test_malformed_request_is_422(client: TestClient) -> None:
    response = client.post("/v1/ask", json={"quest": "typo"})
    assert response.status_code == 422