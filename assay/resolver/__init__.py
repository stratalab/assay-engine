"""The resolver (E1.3, PRD §8): facts from trusted sources, never the model (A-2).

Given the inputs an IR left in ``missing_inputs``, the resolver fills what it can from
the **curated tables** (``assay.constants``, ``assay.materials`` — shipped data, one
``source`` + ``license`` per value, engineering §2.5) and leaves the rest missing with a
stated reason — resolve / ask / fail-closed (A-8, UX §5.2). It never lets a model fill a
value and never silently defaults one.

*What* to resolve is the template's declaration, not a guess: an input's ``resolve``
hint (``FactRef``) names the trusted library and a key pattern whose placeholders fill
from the IR's ``setup`` — template-owns-resolution, the same doctrine as
template-owns-method. Every resolved fact records its exact ``source``
(library + key + table version), so the answer is auditable to the fact (A-11).

Pure and deterministic (engineering §4): in-memory lookups over shipped data — no
model, no network.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from functools import lru_cache
from importlib import resources
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from assay.ir import IR, FactSource, ResolvedFact
from assay.templates import Template

__all__ = [
    "FactRecord",
    "FactTable",
    "Resolution",
    "Resolver",
    "builtin_tables",
    "key_vocabulary",
]


class FactRecord(BaseModel):
    """One curated value. ``source`` and ``license`` are per-value and required —
    resolver data is content, and content has licenses (engineering §2.5)."""

    model_config = ConfigDict(extra="forbid")
    value: float
    unit: str = ""
    source: str = Field(min_length=1)
    license: str = Field(min_length=1)


class FactTable(BaseModel):
    """A curated table: a library name (``assay.materials``), its version — which
    travels into every resolved fact's provenance — and the keyed values."""

    model_config = ConfigDict(extra="forbid")
    library: str = Field(min_length=1)
    version: str = Field(min_length=1)
    facts: dict[str, FactRecord]


@lru_cache(maxsize=1)
def builtin_tables() -> tuple[FactTable, ...]:
    """The curated tables shipped with Assay — open/public-domain values only, each
    carrying its own source + license (never a restricted table, engineering §2.5)."""
    directory = resources.files("assay.resolver") / "tables"
    return tuple(
        FactTable.model_validate_json(entry.read_text(encoding="utf-8"))
        for entry in sorted(directory.iterdir(), key=lambda entry: entry.name)
        if entry.name.endswith(".json")
    )


def key_vocabulary() -> dict[str, dict[str, Any]]:
    """The curated key list, machine-readable — the emit-time contract artifact
    (chisel-alignment round 2, §14.2): the moment an extraction reads "steel beam" it
    must emit a resolver *reference* (``steel.structural.E``), never a literal from the
    problem text, and this is the vocabulary it may reference. Per library: the table
    version (which travels into every resolved fact's provenance) and every key with
    its unit and source. Also served as ``assay facts --json``."""
    return {
        table.library: {
            "version": table.version,
            "keys": {
                key: {"unit": record.unit, "source": record.source}
                for key, record in sorted(table.facts.items())
            },
        }
        for table in builtin_tables()
    }


class Resolution(BaseModel):
    """One resolver pass: the updated IR, what was filled, and — for everything that
    was not — the reason, stated (A-12): the engine asks or fails clear, never guesses."""

    model_config = ConfigDict(extra="forbid")
    ir: IR
    resolved: dict[str, ResolvedFact] = {}
    unresolved: dict[str, str] = {}


class _SetupStrings:
    """A ``format_map`` view of ``IR.setup`` exposing only string values, so a key
    pattern can never silently interpolate a structured setup entry."""

    def __init__(self, setup: Mapping[str, Any]) -> None:
        self._setup = setup

    def __getitem__(self, key: str) -> str:
        value = self._setup[key]
        if not isinstance(value, str):
            raise KeyError(key)
        return value


class Resolver:
    """Facts from trusted sources (A-2). Defaults to the shipped curated tables;
    additional tables (later: plugin/domain data) are passed in explicitly."""

    def __init__(self, tables: Iterable[FactTable] | None = None) -> None:
        self._tables: dict[str, FactTable] = {}
        for table in builtin_tables() if tables is None else tables:
            if table.library in self._tables:
                raise ValueError(f"duplicate fact table library {table.library!r}")
            self._tables[table.library] = table

    def lookup(self, library: str, key: str) -> ResolvedFact | None:
        """Look one fact up. ``None`` means *not found* — the caller declares it
        missing; nothing is ever guessed or defaulted in its place."""
        table = self._tables.get(library)
        record = table.facts.get(key) if table is not None else None
        if table is None or record is None:
            return None
        return ResolvedFact(
            value=record.value,
            unit=record.unit,
            source=FactSource(library=library, key=key, version=table.version),
        )

    def resolve_missing(self, ir: IR, template: Template) -> Resolution:
        """Resolve what the IR's ``missing_inputs`` allow; leave the rest missing.

        Returns a new IR (the input IR is untouched): resolved facts move into
        ``resolved`` with full provenance, everything else stays in ``missing_inputs``
        with its reason in ``unresolved`` — the material for ask (interactive) or
        fail-clear (batch), UX §5.2.
        """
        declared = {inp.name: inp for inp in template.inputs}
        resolved: dict[str, ResolvedFact] = {}
        unresolved: dict[str, str] = {}
        for name in ir.missing_inputs:
            declared_input = declared.get(name)
            if declared_input is None:
                unresolved[name] = f"{name!r} is not an input of template {template.id!r}"
                continue
            hint = declared_input.resolve
            if hint is None:
                unresolved[name] = (
                    f"no trusted source is declared for {name!r} — supply it;"
                    " it will not be fabricated"
                )
                continue
            try:
                key = hint.key.format_map(_SetupStrings(ir.setup))
            except KeyError as exc:
                unresolved[name] = (
                    f"resolving {name!r} from {hint.library} needs setup key"
                    f" {exc.args[0]!r} (a string) to fill {hint.key!r}"
                )
                continue
            fact = self.lookup(hint.library, key)
            if fact is None:
                unresolved[name] = (
                    f"couldn't resolve {name!r}: {key!r} is not in {hint.library} —"
                    " supply it, or add a source"
                )
                continue
            resolved[name] = fact
        updated = ir.model_copy(
            update={
                "resolved": {**ir.resolved, **resolved},
                "missing_inputs": [n for n in ir.missing_inputs if n not in resolved],
            }
        )
        return Resolution(ir=updated, resolved=resolved, unresolved=unresolved)
