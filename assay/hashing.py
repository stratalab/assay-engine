"""Stable content hashing (A-11).

Canonical JSON (sorted keys, tight separators) -> sha256. Order-independent over dict
keys and deterministic across runs/platforms, so identical content always yields the same
hash. Used for IR identity, cache keys, and artifact addressing.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def content_sha256(obj: Any) -> str:
    """Return the sha256 of ``obj``'s canonical JSON serialization."""
    canonical = json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
