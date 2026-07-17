#!/usr/bin/env python3
"""Runtime-dependency license gate (engineering §2.4).

Assay is MIT and ships embedded/redistributed, so **every runtime dependency must be
permissively licensed** (MIT/BSD/Apache/PSF/ISC/...). This script inspects the installed
distributions and exits non-zero if any is not clearly permissive.

Run it in a **runtime-only** environment (``uv sync --no-dev``) so the installed set is
exactly the runtime closure. Classification prefers the Trove ``License ::`` classifiers,
falls back to the free-text ``License`` field, consults an explicit override map for
ambiguous metadata, and **fails closed** on anything it cannot classify as permissive.
"""

from __future__ import annotations

import re
from importlib import metadata

# Markers match on WORD BOUNDARIES: some packages put their full license text in the
# metadata, where raw substrings misfire ("EXEMPLARY" contains mpl; "NOT LIMITED"
# contains mit). Spelled-out phrases cover "GNU General Public License" (no "gpl").
PERMISSIVE = re.compile(
    r"\b(mit|bsd|apache|psf|isc|0bsd|unlicense|zlib)\b|python software foundation"
)
# Copyleft is checked first — MPL/LGPL count as copyleft for our embedded distribution.
COPYLEFT = re.compile(
    r"\b(a?gpl|lgpl|mpl|eupl|cecill|osl|epl)\b"
    r"|general public license|mozilla public license"
)

# Build/packaging tooling and the project itself — not shipped at runtime.
IGNORE: frozenset[str] = frozenset({"assay", "pip", "setuptools", "wheel", "uv"})

# Explicit overrides (dist name -> SPDX id) where upstream metadata is missing/ambiguous.
OVERRIDES: dict[str, str] = {}


def classify(text: str) -> str:
    """Classify a license string as ``permissive`` / ``copyleft`` / ``unknown``."""
    t = text.lower()
    if COPYLEFT.search(t):
        return "copyleft"
    if PERMISSIVE.search(t):
        return "permissive"
    return "unknown"


def classify_dist(dist: metadata.Distribution) -> tuple[str, str]:
    """Classify a distribution by metadata **precedence**: an explicit override, the
    PEP 639 SPDX ``License-Expression``, the Trove ``License ::`` classifiers, and only
    then the free-text ``License`` field — which may embed the *full* license text (or,
    like matplotlib's, a whole bundled-components inventory) and is too noisy to scan
    when structured metadata exists. Returns ``(verdict, evidence)``."""
    md = dist.metadata
    name = str(md.get("Name") or "").lower()
    if name in OVERRIDES:
        return classify(OVERRIDES[name]), OVERRIDES[name]
    expr = md.get("License-Expression")  # authoritative when present (e.g. pydantic)
    if expr:
        return classify(str(expr)), str(expr)
    classifiers = [
        str(c) for c in (md.get_all("Classifier") or []) if str(c).startswith("License ::")
    ]
    if classifiers:
        joined = " ".join(classifiers)
        verdict = classify(joined)
        if verdict != "unknown":  # e.g. bare "License :: OSI Approved" says nothing
            return verdict, joined
    field = md.get("License")
    if field:
        summary = " ".join(str(field).split())
        return classify(summary), summary[:120]
    return "unknown", "no license metadata"


def main() -> int:
    problems: list[str] = []
    for dist in metadata.distributions():
        name = str(dist.metadata.get("Name") or "").lower()
        if not name or name in IGNORE:
            continue
        verdict, evidence = classify_dist(dist)
        if verdict != "permissive":
            problems.append(f"{name} {dist.version}: {verdict} ({evidence})")
    if problems:
        print("LICENSE GATE FAILED — non-permissive runtime dependencies:")
        for line in sorted(set(problems)):
            print(f"  x {line}")
        print("\nRuntime deps must be permissive (MIT/BSD/Apache/PSF/ISC). See engineering §2.")
        return 1
    print("license gate: all runtime dependencies are permissive (ok)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
