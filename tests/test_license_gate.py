"""E0.1: prove the license gate classifies correctly (engineering §2.4).

This tests the gate's *logic* directly, so we prove it works without adding a copyleft
dependency to the project just to watch CI fail.
"""

from __future__ import annotations

import pytest

from scripts.check_licenses import classify


@pytest.mark.parametrize(
    "text",
    ["MIT License", "BSD 3-Clause", "Apache Software License",
     "Python Software Foundation License", "ISC", "The Unlicense"],
)
def test_permissive_passes(text: str) -> None:
    assert classify(text) == "permissive"


@pytest.mark.parametrize(
    "text",
    ["GPL-3.0", "GNU General Public License", "AGPL-3.0", "LGPL-2.1", "MPL-2.0"],
)
def test_copyleft_fails(text: str) -> None:
    assert classify(text) == "copyleft"


@pytest.mark.parametrize("text", ["", "some proprietary EULA"])
def test_unknown_fails_closed(text: str) -> None:
    assert classify(text) == "unknown"


def test_full_license_text_does_not_false_positive() -> None:
    """Some packages put the whole license text in metadata: "EXEMPLARY" contains
    'mpl' and "NOT LIMITED" contains 'mit' — word boundaries must protect both ways."""
    bsd_text = (
        "BSD 3-Clause License. Redistribution and use are permitted provided that ... "
        "INCLUDING, BUT NOT LIMITED TO, ... EXEMPLARY, OR CONSEQUENTIAL DAMAGES ..."
    )
    assert classify(bsd_text) == "permissive"
    assert classify("EXEMPLARY DAMAGES ONLY, NOT LIMITED") == "unknown"  # no real marker
