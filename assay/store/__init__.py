"""The embedded store seam (PRD §19, engineering §4).

One interface, two backends: a pure-Python **in-memory** store (the default — dev/test, so
nothing blocks on StrataDB) and **embedded StrataDB** (the production backend, via the
``stratadb`` PyO3 SDK — an optional dependency until StrataDB is production-ready, per the
"thin seam, adopt later" doctrine).

The store holds Assay's content-addressed artifacts + compute cache (a namespaced
key-value store) and an append-only **lineage** log. Values are UTF-8 text (JSON), matching
StrataDB's KV. The default backend will flip from ``memory`` to ``stratadb`` when the
StrataDB embedded surface is adopted (the hard requirement, A-10).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from assay.store.stratadb import StrataDBStore

__all__ = ["InMemoryStore", "Store", "StrataDBStore", "open_store"]


@runtime_checkable
class Store(Protocol):
    """The store interface. Backends implement it structurally.

    Namespaces separate the store's uses (``artifacts`` / ``cache`` / ...); ``lineage`` is
    an append-only log keyed by a ``kind``.
    """

    def put(self, namespace: str, key: str, value: str) -> None: ...
    def get(self, namespace: str, key: str) -> str | None: ...
    def has(self, namespace: str, key: str) -> bool: ...
    def delete(self, namespace: str, key: str) -> bool: ...
    def keys(self, namespace: str) -> list[str]: ...
    def append_lineage(self, kind: str, record: str) -> None: ...
    def lineage(self, kind: str) -> list[str]: ...


class InMemoryStore:
    """Dict-backed store — the default backend (dev/test); non-persistent."""

    def __init__(self) -> None:
        self._kv: dict[tuple[str, str], str] = {}
        self._lineage: dict[str, list[str]] = {}

    def put(self, namespace: str, key: str, value: str) -> None:
        self._kv[(namespace, key)] = value

    def get(self, namespace: str, key: str) -> str | None:
        return self._kv.get((namespace, key))

    def has(self, namespace: str, key: str) -> bool:
        return (namespace, key) in self._kv

    def delete(self, namespace: str, key: str) -> bool:
        return self._kv.pop((namespace, key), None) is not None

    def keys(self, namespace: str) -> list[str]:
        return sorted(key for (ns, key) in self._kv if ns == namespace)

    def append_lineage(self, kind: str, record: str) -> None:
        self._lineage.setdefault(kind, []).append(record)

    def lineage(self, kind: str) -> list[str]:
        return list(self._lineage.get(kind, []))


def open_store(backend: str = "memory", *, path: str | None = None) -> Store:
    """Open a store.

    ``memory`` (default) is the dev/test backend; ``stratadb`` is the production backend
    (embedded StrataDB, ``path=None`` → in-memory ``Strata.cache()``, a path → persistent).
    """
    if backend == "memory":
        return InMemoryStore()
    if backend == "stratadb":
        return StrataDBStore(path=path)
    raise ValueError(f"unknown store backend {backend!r}; expected 'memory' or 'stratadb'")
