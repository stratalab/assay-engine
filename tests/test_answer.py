"""E0.3: the answer object — validation + JSON round-trip (PRD §10)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from assay.answer import Answer


def _answer() -> Answer:
    return Answer.model_validate(
        {
            "result": [{"label": "max_deflection", "value": 5.0e-4, "unit": "m"}],
            "interpretation": "simply supported beam, central point load",
            "method": "delta = P*L**3/(48*E*I)",
            "facts": [
                {
                    "name": "E",
                    "value": 200e9,
                    "unit": "Pa",
                    "source": {
                        "library": "assay.materials",
                        "key": "steel.structural.E",
                        "version": "0.1",
                    },
                }
            ],
            "verified": {"ok": True, "checks": [{"name": "dimension:length", "ok": True}]},
            "ir_hash": "abc123",
            "assay_version": "0.0.1",
        }
    )


def test_answer_round_trips_json() -> None:
    ans = _answer()
    again = Answer.model_validate_json(ans.model_dump_json())
    assert again == ans
    assert again.verified.ok is True
    assert again.result[0].unit == "m"


def test_verification_is_required() -> None:
    with pytest.raises(ValidationError):
        Answer.model_validate({"interpretation": "no verified field"})


def test_symbolic_result_value() -> None:
    ans = Answer.model_validate(
        {"result": [{"label": "integral", "value": "x**3/3"}], "verified": {"ok": True}}
    )
    assert ans.result[0].value == "x**3/3"
