"""E2.4: the plugin SDK — new domains arrive as installed packages, not core edits (A-9).

The done-criterion: a third-party template package installs and serves with zero core
edits. Proven with a real distribution: each test writes an actual package + dist-info
(entry point and all) onto ``sys.path`` and lets ``importlib.metadata`` discover it —
the same machinery a pip-installed wheel uses. Trust discipline throughout: discovery
grants presence; serving still requires the E2.2 fixture gate.
"""

from __future__ import annotations

import importlib
import sys
from importlib import metadata
from pathlib import Path

import pytest

import assay.cli
from assay.cli import main
from assay.templates import CandidateTemplateError, TemplateRegistry
from assay.templates.plugins import DiscoveredPlugin, discover_plugins, ingest_plugins

_BEAMKIT = '''\
"""beamkit: a third-party Assay template package (the E2.4 proof)."""

def templates():
    return [
        {
            "schema_version": 1,
            "id": "beam_deflection.cantilever.end_point",
            "domain": "structural_mechanics",
            "description": "End deflection of a cantilever beam under an end point load.",
            "inputs": [
                {"name": "P", "dimension": "force"},
                {"name": "L", "dimension": "length"},
                {"name": "E", "dimension": "pressure",
                 "resolve": {"library": "assay.materials", "key": "{material}.E"}},
                {"name": "I", "dimension": "length**4"},
            ],
            "method": {"kind": "formula", "expr": "P * L**3 / (3 * E * I)"},
            "output": {"name": "max_deflection", "dimension": "length"},
            "assumptions": ["euler_bernoulli"],
            "fixtures": [
                {
                    "inputs": {"P": [1000, "N"], "L": [1, "m"],
                               "E": [200e9, "Pa"], "I": [1e-6, "m**4"]},
                    "expect": {"max_deflection": [1.6666666666666667e-3, "m"]},
                    "tol": 1e-9,
                }
            ],
            "provenance": {"source": "beamkit:hand-authored", "license_tier": "open"},
        }
    ]
'''

_BADKIT = '''\
def templates():
    raise RuntimeError("boom")
'''

_JUNKKIT = '''\
def templates():
    return [{"schema_version": 1, "id": "junk"}]  # nowhere near the contract
'''

_WRONGKIT = '''\
def templates():
    return [
        {
            "schema_version": 1,
            "id": "test.wrong_spring",
            "domain": "mechanics",
            "inputs": [
                {"name": "k", "dimension": "force/length"},
                {"name": "x", "dimension": "length"},
            ],
            "method": {"kind": "formula", "expr": "k * x"},
            "output": {"name": "force", "dimension": "force"},
            "fixtures": [
                {
                    "inputs": {"k": [10, "N/m"], "x": [2, "m"]},
                    "expect": {"force": [99.0, "N"]},
                    "tol": 1e-9,
                }
            ],
            "provenance": {"source": "wrongkit:hand-authored", "license_tier": "open"},
        }
    ]
'''


@pytest.fixture(autouse=True)
def _workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    assay.cli._promoted.cache_clear()  # the CLI cache must see this test's installs
    return tmp_path


def _install(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    version: str,
    source: str,
) -> None:
    """A real install, minus pip: package module + dist-info (METADATA, entry point)
    on sys.path — exactly what importlib.metadata discovers for a wheel."""
    (tmp_path / f"{name}.py").write_text(source, encoding="utf-8")
    dist_info = tmp_path / f"{name}-{version}.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n", encoding="utf-8"
    )
    (dist_info / "entry_points.txt").write_text(
        f"[assay.templates]\n{name} = {name}:templates\n", encoding="utf-8"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, name, raising=False)
    importlib.invalidate_caches()
    metadata.MetadataPathFinder.invalidate_caches()


def _plugin(name: str) -> DiscoveredPlugin:
    matches = [plugin for plugin in discover_plugins() if plugin.name == name]
    assert len(matches) == 1, f"expected exactly one {name!r} plugin"
    return matches[0]


def test_discovery_validates_and_carries_the_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install(tmp_path, monkeypatch, "beamkit", "0.1.0", _BEAMKIT)
    plugin = _plugin("beamkit")
    assert plugin.distribution == "beamkit 0.1.0"  # versioned provenance
    assert not plugin.errors
    (template,) = plugin.templates
    assert template.id == "beam_deflection.cantilever.end_point"
    assert template.provenance.status == "candidate"  # presence, not trust


def test_third_party_template_serves_with_zero_core_edits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """THE done-criterion — install a package, ask in natural language, get a
    verified answer through a template Assay's core has never heard of."""
    _install(tmp_path, monkeypatch, "beamkit", "0.1.0", _BEAMKIT)
    question = "deflection of a cantilever steel beam, 1 kN end load, 1 m span, I = 1e-6 m^4"
    assert main(["ask", question]) == 0
    out = capsys.readouterr().out
    assert "max_deflection: 0.00166667 m" in out
    assert "assay.materials steel.structural.E" in out  # resolver still owns the facts
    assert "✓ dimension:length" in out


def test_plugin_participates_in_disambiguation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _install(tmp_path, monkeypatch, "beamkit", "0.1.0", _BEAMKIT)
    assert main(["ask", "deflection of a steel beam, 5 kN load, 2 m", "--batch"]) == 2
    out = capsys.readouterr().out
    assert "This is ambiguous" in out
    assert "beam_deflection.cantilever.end_point" in out


def test_broken_and_invalid_plugins_are_contained(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _install(tmp_path, monkeypatch, "badkit", "0.0.1", _BADKIT)
    _install(tmp_path, monkeypatch, "junkkit", "0.0.2", _JUNKKIT)
    _install(tmp_path, monkeypatch, "beamkit", "0.1.0", _BEAMKIT)
    assert "boom" in _plugin("badkit").errors[0]
    assert "junkkit 0.0.2 record 0" in _plugin("junkkit").errors[0]
    # the healthy plugin — and the engine — are unaffected
    question = "deflection of a cantilever steel beam, 1 kN end load, 1 m span, I = 1e-6 m^4"
    assert main(["ask", question]) == 0
    capsys.readouterr()
    assert main(["domains"]) == 0
    out = capsys.readouterr().out
    assert "! plugin 'badkit'" in out and "boom" in out


def test_gated_ingest_verifies_or_quarantines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install(tmp_path, monkeypatch, "beamkit", "0.1.0", _BEAMKIT)
    _install(tmp_path, monkeypatch, "wrongkit", "0.0.3", _WRONGKIT)
    registry = TemplateRegistry()
    result = ingest_plugins(registry)
    assert result.verified == ["beam_deflection.cantilever.end_point"]
    assert result.quarantined == ["test.wrong_spring"]
    assert any("fixtures failed" in error for error in result.errors)
    served = registry.get("beam_deflection.cantilever.end_point")
    assert served.provenance.status == "verified"
    with pytest.raises(CandidateTemplateError):
        registry.get("test.wrong_spring")  # quarantined: present, refused


def test_domains_lists_shipped_and_plugins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _install(tmp_path, monkeypatch, "beamkit", "0.1.0", _BEAMKIT)
    assert main(["domains"]) == 0
    out = capsys.readouterr().out
    assert "structural_mechanics" in out and "algebra" in out
    assert "[shipped]" in out
    assert "beam_deflection.cantilever.end_point" in out and "[beamkit 0.1.0]" in out


def test_a_plugin_cannot_shadow_an_installed_template(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    impostor = _BEAMKIT.replace(
        "beam_deflection.cantilever.end_point",
        "beam_deflection.simply_supported.center_point",  # a shipped golden's id
    )
    _install(tmp_path, monkeypatch, "copykit", "6.6.6", impostor)
    assert main(["domains"]) == 0
    out = capsys.readouterr().out
    assert "collides with an installed template — ignored" in out