"""M0 gate: a trivial IR round-trips end to end through the store, keyed by its hash.

Ties the foundation together — E0.3 (IR + content hash + JSON) and E0.2 (the store) —
the thin plumbing slice that closes Milestone 0 (real compute is M1).
"""

from __future__ import annotations

from assay.ir import IR
from assay.store import open_store


def test_ir_round_trips_through_the_store() -> None:
    ir = IR.model_validate(
        {"domain": "algebra", "task": "solve_equation", "setup": {"equation": "x**2 - 5*x + 6"}}
    )
    store = open_store()  # in-memory (the default backend)
    key = ir.content_hash()

    store.put("artifacts", key, ir.model_dump_json())
    loaded = IR.model_validate_json(store.get("artifacts", key) or "")

    assert loaded == ir
    assert loaded.content_hash() == key
    assert store.keys("artifacts") == [key]
