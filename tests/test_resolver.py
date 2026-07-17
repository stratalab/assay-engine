"""E1.3: the resolver + curated tables — facts from trusted sources, never the model.

The done-criteria: ``E`` for steel resolves from the curated table with its source
recorded; an unresolvable input is declared missing, never fabricated.
"""

from __future__ import annotations

from typing import Any

import pytest

from assay.execute import execute_ir
from assay.ir import IR
from assay.resolver import FactRecord, FactTable, Resolver, builtin_tables
from assay.templates import golden_template


def _beam_ir(**overrides: Any) -> IR:
    fields: dict[str, Any] = {
        "domain": "structural_mechanics",
        "task": "beam_deflection.simply_supported.center_point",
        "setup": {"material": "steel.structural"},
        "inputs": {"P": {"value": 5000, "unit": "N"}, "L": {"value": 2, "unit": "m"}},
        "missing_inputs": ["E", "I"],
    }
    fields.update(overrides)
    return IR.model_validate(fields)


def test_e_for_steel_resolves_with_its_source_recorded() -> None:
    resolution = Resolver().resolve_missing(_beam_ir(), golden_template())
    fact = resolution.ir.resolved["E"]
    assert fact.value == 200e9
    assert fact.unit == "Pa"
    assert fact.source.library == "assay.materials"
    assert fact.source.key == "steel.structural.E"
    assert fact.source.version == "0.1"
    assert resolution.ir.missing_inputs == ["I"]  # I has no trusted source: stays missing


def test_input_without_a_source_stays_missing_with_the_reason() -> None:
    resolution = Resolver().resolve_missing(_beam_ir(), golden_template())
    assert "no trusted source is declared for 'I'" in resolution.unresolved["I"]
    assert "will not be fabricated" in resolution.unresolved["I"]


def test_unknown_material_is_never_fabricated() -> None:
    ir = _beam_ir(setup={"material": "unobtainium"})
    resolution = Resolver().resolve_missing(ir, golden_template())
    assert resolution.resolved == {}
    assert "E" in resolution.ir.missing_inputs
    assert "'unobtainium.E' is not in assay.materials" in resolution.unresolved["E"]
    assert "supply it, or add a source" in resolution.unresolved["E"]


def test_missing_setup_key_is_named() -> None:
    resolution = Resolver().resolve_missing(_beam_ir(setup={}), golden_template())
    assert "needs setup key 'material'" in resolution.unresolved["E"]


def test_non_string_setup_value_is_refused() -> None:
    ir = _beam_ir(setup={"material": {"name": "steel"}})
    resolution = Resolver().resolve_missing(ir, golden_template())
    assert "needs setup key 'material'" in resolution.unresolved["E"]


def test_the_input_ir_is_untouched() -> None:
    ir = _beam_ir()
    Resolver().resolve_missing(ir, golden_template())
    assert ir.missing_inputs == ["E", "I"]
    assert ir.resolved == {}


def test_missing_input_not_declared_by_the_template() -> None:
    resolution = Resolver().resolve_missing(
        _beam_ir(missing_inputs=["E", "Z"]), golden_template()
    )
    assert "'Z' is not an input of template" in resolution.unresolved["Z"]


def test_nothing_to_resolve_is_a_no_op() -> None:
    ir = _beam_ir(missing_inputs=[], inputs={})
    resolution = Resolver().resolve_missing(ir, golden_template())
    assert resolution.ir == ir
    assert resolution.resolved == {} and resolution.unresolved == {}


def test_resolved_ir_executes_the_beam_end_to_end() -> None:
    """The v0 spine slice: declare missing → resolve from the curated table → execute
    with per-fact provenance (the M1-gate beam flow, minus the CLI)."""
    golden = golden_template()
    ir = _beam_ir(
        inputs={
            "P": {"value": 5000, "unit": "N"},
            "L": {"value": 2, "unit": "m"},
            "I": {"value": 8.33e-6, "unit": "m**4"},
        },
        missing_inputs=["E"],
    )
    resolution = Resolver().resolve_missing(ir, golden)
    assert resolution.ir.missing_inputs == []
    result = execute_ir(resolution.ir, golden)
    assert result.value == pytest.approx(5000 * 2**3 / (48 * 200e9 * 8.33e-6), rel=1e-12)
    assert result.value * 1000 == pytest.approx(0.50, abs=0.005)  # 0.50 mm


def test_lookup_a_constant() -> None:
    fact = Resolver().lookup("assay.constants", "g")
    assert fact is not None
    assert fact.value == 9.80665
    assert fact.unit == "m/s**2"
    assert fact.source.library == "assay.constants"


def test_lookup_unknown_library_or_key_returns_none() -> None:
    resolver = Resolver()
    assert resolver.lookup("assay.nope", "g") is None
    assert resolver.lookup("assay.constants", "nope") is None


def test_every_builtin_fact_records_source_and_license() -> None:
    """Engineering §2.5: resolver data is content; provenance + license per value."""
    tables = builtin_tables()
    assert {t.library for t in tables} == {"assay.constants", "assay.materials"}
    for table in tables:
        assert table.version
        for key, record in table.facts.items():
            assert record.source, f"{table.library}:{key} has no source"
            assert record.license, f"{table.library}:{key} has no license"


def test_custom_tables_and_duplicate_libraries() -> None:
    table = FactTable(
        library="assay.test",
        version="0.0",
        facts={"x": FactRecord(value=1.0, source="test", license="public-domain")},
    )
    resolver = Resolver(tables=[table])
    assert resolver.lookup("assay.test", "x") is not None
    assert resolver.lookup("assay.materials", "steel.structural.E") is None  # not loaded
    with pytest.raises(ValueError, match="duplicate fact table"):
        Resolver(tables=[table, table])
