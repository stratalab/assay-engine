"""E0.2: the store seam — round-trip through the in-memory backend (and StrataDB if present).

The StrataDB backend runs the same round-trips *iff* ``stratadb`` is installed. CI installs
the ``assay[stratadb]`` extra on all three platforms, so the adapter is exercised there; a
bare local env (no extra) skips those params rather than failing.
"""

from __future__ import annotations

import importlib.util

import pytest

from assay.store import InMemoryStore, Store, open_store

_HAS_STRATADB = importlib.util.find_spec("stratadb") is not None
BACKENDS = ["memory", *(["stratadb"] if _HAS_STRATADB else [])]


def test_default_backend_is_in_memory() -> None:
    assert isinstance(open_store(), InMemoryStore)


def test_in_memory_conforms_to_store() -> None:
    assert isinstance(InMemoryStore(), Store)


def test_unknown_backend_raises() -> None:
    with pytest.raises(ValueError, match="unknown store backend"):
        open_store("postgres")


@pytest.mark.parametrize("backend", BACKENDS)
def test_kv_roundtrip(backend: str) -> None:
    store = open_store(backend)
    assert store.get("artifacts", "a") is None
    assert not store.has("artifacts", "a")
    store.put("artifacts", "a", "hello")
    assert store.get("artifacts", "a") == "hello"
    assert store.has("artifacts", "a")
    assert store.delete("artifacts", "a") is True
    assert store.delete("artifacts", "a") is False
    assert store.get("artifacts", "a") is None


@pytest.mark.parametrize("backend", BACKENDS)
def test_namespaces_are_isolated(backend: str) -> None:
    store = open_store(backend)
    store.put("artifacts", "k", "1")
    store.put("cache", "k", "2")
    assert store.get("artifacts", "k") == "1"
    assert store.get("cache", "k") == "2"
    assert store.keys("artifacts") == ["k"]


@pytest.mark.parametrize("backend", BACKENDS)
def test_lineage_is_append_only(backend: str) -> None:
    store = open_store(backend)
    assert store.lineage("run") == []
    store.append_lineage("run", "r1")
    store.append_lineage("run", "r2")
    assert store.lineage("run") == ["r1", "r2"]
