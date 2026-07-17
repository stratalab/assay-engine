"""E0.3: the IR — validation, stable content hashing, JSON round-trip (A-1, A-11)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from assay.ir import IR

_STEEL_E = {
    "value": 200e9,
    "unit": "Pa",
    "source": {"library": "assay.materials", "key": "steel.structural.E", "version": "0.1"},
}


def _beam(**overrides: object) -> IR:
    fields: dict[str, object] = {
        "assay_version": "0.0.1",
        "query": "max deflection of a steel beam",
        "domain": "structural_mechanics",
        "task": "beam_deflection",
        "inputs": {"P": {"value": 5000, "unit": "N"}, "L": {"value": 2, "unit": "m"}},
        "missing_inputs": ["E", "I"],
        "resolved": {"E": _STEEL_E},
        "assumptions": ["euler_bernoulli"],
        "execution_plan": ["evaluate_formula"],
        "outputs": ["max_deflection"],
    }
    fields.update(overrides)
    return IR.model_validate(fields)


def test_validates_and_round_trips_json() -> None:
    ir = _beam()
    assert IR.model_validate_json(ir.model_dump_json()) == ir


def test_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        IR.model_validate({"domain": "d", "task": "t", "bogus": 1})


def test_requires_domain_and_task() -> None:
    with pytest.raises(ValidationError):
        IR.model_validate({"domain": "d"})  # missing task


def test_content_hash_is_order_independent() -> None:
    a = _beam()
    b = _beam(inputs={"L": {"value": 2, "unit": "m"}, "P": {"value": 5000, "unit": "N"}})
    assert a.content_hash() == b.content_hash()


def test_content_hash_ignores_software_version_and_query() -> None:
    a = _beam()
    b = a.model_copy(update={"assay_version": "9.9.9", "query": "different phrasing entirely"})
    assert a.content_hash() == b.content_hash()


def test_content_hash_changes_with_content() -> None:
    a = _beam()
    assert a.content_hash() != _beam(task="cantilever_deflection").content_hash()
    assert a.content_hash() != _beam(assumptions=["timoshenko"]).content_hash()
