"""Embedded StrataDB backend (PRD §19), via the ``stratadb`` PyO3 SDK.

Mapped to StrataDB's **namespace API** (validated against ``stratadb`` 0.14.5): KV
(``db.kv.put`` / ``get`` / ``delete`` / ``list``) for artifacts + cache, and the Event Log
(``db.events.append`` / ``list``) for lineage. ``stratadb`` is an *optional* dependency
until StrataDB's embedded surface is production-ready, so the import is lazy; the store
tests exercise this adapter whenever ``stratadb`` is installed. (The SDK's Python wrapper
is still evolving — re-run those tests after an SDK bump.)
"""

from __future__ import annotations


class StrataDBStore:
    """A ``Store`` backed by embedded StrataDB.

    Namespaced KV keys are ``"{namespace}:{key}"``; lineage uses the Event Log keyed by
    ``kind`` (the appended record is the event payload).
    """

    def __init__(self, *, path: str | None = None) -> None:
        try:
            from stratadb import Strata
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "the StrataDB backend needs the `stratadb` package (the embedded-store "
                "dependency, gated on StrataDB maturity); use backend='memory' for now"
            ) from exc
        self._db = Strata.cache() if path is None else Strata.open(path)

    @staticmethod
    def _k(namespace: str, key: str) -> str:
        return f"{namespace}:{key}"

    def put(self, namespace: str, key: str, value: str) -> None:
        self._db.kv.put(self._k(namespace, key), value)

    def get(self, namespace: str, key: str) -> str | None:
        value = self._db.kv.get(self._k(namespace, key))
        return None if value is None else str(value)

    def has(self, namespace: str, key: str) -> bool:
        return self._db.kv.get(self._k(namespace, key)) is not None

    def delete(self, namespace: str, key: str) -> bool:
        return bool(self._db.kv.delete(self._k(namespace, key)))

    def keys(self, namespace: str) -> list[str]:
        prefix = f"{namespace}:"
        return [str(k)[len(prefix):] for k in self._db.kv.list(prefix=prefix)]

    def append_lineage(self, kind: str, record: str) -> None:
        # StrataDB's Event Log requires a JSON-object payload; wrap the record string.
        self._db.events.append(kind, {"record": record})

    def lineage(self, kind: str) -> list[str]:
        records: list[str] = []
        for event in self._db.events.list(kind):
            value = event.get("value", {})  # StrataDB wraps the payload under "value"
            record = value.get("record", "") if isinstance(value, dict) else value
            records.append(str(record))
        return records
