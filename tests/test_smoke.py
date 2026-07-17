"""E0.1: the package imports and the CLI entry point runs."""

from __future__ import annotations

import assay
from assay.cli import main


def test_version_present() -> None:
    assert assay.__version__


def test_cli_runs() -> None:
    assert main([]) == 0
