"""Local GGUF model management (E3.3, engineering §8): explicit fetch, checksum-gated.

The model is **not in the wheel** — a no-model install is a complete deterministic
engine (A-4). Fetching is an **explicit user action** (``assay model fetch``), never
triggered by the compute path (NFR-1: no network in compute; this module is the one
place in the package allowed to touch the network, and nothing in the engine imports
it). Every fetch requires a **sha256** and fails closed on mismatch — a model is a
trust decision, and the checksum is its provenance.

Models land under the user data dir (``ASSAY_HOME`` overrides). ``assay ask --llm``
resolves a model as: explicit path → ``$ASSAY_LLM_MODEL`` → the single cached model
(two cached models is an ambiguity, and ambiguity asks — it never picks silently).
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
import urllib.request  # gate-scoped: the ONLY permitted network import in the package
from pathlib import Path

__all__ = ["ModelFetchError", "cached_models", "fetch_model", "models_dir", "resolve_model"]

_CHUNK = 1 << 20


class ModelFetchError(Exception):
    """The fetch failed or the checksum disagreed — stated, and nothing kept (A-12)."""


def models_dir() -> Path:
    """Where fetched models live: ``$ASSAY_HOME/models`` if set, else the platform's
    user data dir."""
    home = os.environ.get("ASSAY_HOME")
    if home:
        return Path(home) / "models"
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    elif sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / "assay" / "models"


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_model(
    url: str, sha256: str, *, name: str | None = None, directory: Path | None = None
) -> Path:
    """Download a model, verify its sha256, and cache it. Idempotent: an existing file
    with the right digest is kept (and never re-downloaded); a wrong digest — cached
    or downloaded — fails closed and leaves nothing behind."""
    target_dir = directory or models_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    filename = name or url.rstrip("/").rsplit("/", 1)[-1]
    if not filename:
        raise ModelFetchError(f"cannot derive a filename from {url!r} — pass --name")
    target = target_dir / filename
    expected = sha256.lower()
    if target.exists():
        if _digest(target) == expected:
            return target  # already fetched and verified
        raise ModelFetchError(
            f"{target} exists but its sha256 does not match — remove it to re-fetch"
        )
    partial = target.with_suffix(target.suffix + ".part")
    digest = hashlib.sha256()
    try:
        with urllib.request.urlopen(url) as source, partial.open("wb") as sink:  # noqa: S310
            while chunk := source.read(_CHUNK):
                digest.update(chunk)
                sink.write(chunk)
    except OSError as exc:
        partial.unlink(missing_ok=True)
        raise ModelFetchError(f"fetch failed: {exc}") from exc
    if digest.hexdigest() != expected:
        partial.unlink(missing_ok=True)
        raise ModelFetchError(
            f"sha256 mismatch for {url!r}: got {digest.hexdigest()}, expected {expected}"
            " — nothing was kept"
        )
    shutil.move(partial, target)
    return target


def cached_models(directory: Path | None = None) -> list[Path]:
    target_dir = directory or models_dir()
    if not target_dir.is_dir():
        return []
    return sorted(path for path in target_dir.glob("*.gguf") if path.is_file())


def resolve_model(explicit: str | None, directory: Path | None = None) -> Path | None:
    """The ``--llm`` resolution order: explicit path → $ASSAY_LLM_MODEL → the single
    cached model. ``None`` when nothing resolves — the caller asks, never guesses."""
    if explicit:
        return Path(explicit)
    from_env = os.environ.get("ASSAY_LLM_MODEL")
    if from_env:
        return Path(from_env)
    cached = cached_models(directory)
    if len(cached) == 1:
        return cached[0]
    return None
