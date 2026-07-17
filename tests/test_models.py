"""E3.3: the model-fetch UX — explicit, checksum-gated, never in the compute path.

Fetches in these tests use ``file://`` URLs: the mechanism is real (urllib + streaming
sha256 + atomic rename), the network is not required.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from assay.cli import main
from assay.models import ModelFetchError, cached_models, fetch_model, models_dir, resolve_model


@pytest.fixture()
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("ASSAY_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("ASSAY_LLM_MODEL", raising=False)
    return tmp_path


def _source_model(tmp_path: Path, content: bytes = b"not-really-weights") -> tuple[str, str]:
    source = tmp_path / "tiny.gguf"
    source.write_bytes(content)
    return source.as_uri(), hashlib.sha256(content).hexdigest()


def test_fetch_verifies_and_caches(home: Path) -> None:
    url, digest = _source_model(home)
    path = fetch_model(url, digest)
    assert path == models_dir() / "tiny.gguf" and path.is_file()
    assert fetch_model(url, digest) == path  # idempotent: verified, not re-downloaded
    assert cached_models() == [path]


def test_wrong_checksum_fails_closed_and_keeps_nothing(home: Path) -> None:
    url, _ = _source_model(home)
    with pytest.raises(ModelFetchError, match="sha256 mismatch"):
        fetch_model(url, "0" * 64)
    assert cached_models() == []
    assert not list(models_dir().glob("*.part"))  # no partial left behind


def test_tampered_cache_is_refused(home: Path) -> None:
    url, digest = _source_model(home)
    path = fetch_model(url, digest)
    path.write_bytes(b"tampered")
    with pytest.raises(ModelFetchError, match="does not match"):
        fetch_model(url, digest)


def test_resolution_order(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert resolve_model(None) is None  # nothing anywhere: the caller must ask
    url, digest = _source_model(home)
    cached = fetch_model(url, digest)
    assert resolve_model(None) == cached  # exactly one cached model
    monkeypatch.setenv("ASSAY_LLM_MODEL", "/env/model.gguf")
    assert resolve_model(None) == Path("/env/model.gguf")  # env beats cache
    assert resolve_model("/explicit.gguf") == Path("/explicit.gguf")  # explicit beats all
    monkeypatch.delenv("ASSAY_LLM_MODEL")
    fetch_model(url, digest, name="second.gguf")
    assert resolve_model(None) is None  # two models: ambiguous, never picked silently


def test_cli_fetch_and_list(home: Path, capsys: pytest.CaptureFixture[str]) -> None:
    url, digest = _source_model(home)
    assert main(["model", "fetch", url, "--sha256", digest]) == 0
    assert "fetched and verified" in capsys.readouterr().out
    assert main(["model", "list"]) == 0
    assert "tiny.gguf" in capsys.readouterr().out
    assert main(["model", "fetch", url, "--sha256", "0" * 64]) == 2  # fail-clear
    assert "does not match" in capsys.readouterr().err


def test_bare_llm_flag_asks_when_nothing_resolves(
    home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["ask", "solve x^2 - 1 = 0", "--llm"]) == 2
    err = capsys.readouterr().err
    assert "no model to serve" in err and "assay model fetch" in err