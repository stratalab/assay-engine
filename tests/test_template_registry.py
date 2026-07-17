"""E1.1: the template registry — serve only ``verified`` by default (A-9, A-14 posture)."""

from __future__ import annotations

import pytest

from assay.templates import (
    CandidateTemplateError,
    Template,
    TemplateRegistry,
    golden_template,
)


def _with_status(template: Template, status: str) -> Template:
    return template.model_copy(
        update={"provenance": template.provenance.model_copy(update={"status": status})}
    )


def test_register_and_get_verified() -> None:
    registry = TemplateRegistry()
    verified = _with_status(golden_template(), "verified")
    registry.register(verified)
    assert registry.get(verified.id) == verified
    assert verified.id in registry
    assert len(registry) == 1


def test_candidate_is_refused_by_default() -> None:
    registry = TemplateRegistry()
    candidate = golden_template()  # ships as candidate
    registry.register(candidate)
    with pytest.raises(CandidateTemplateError, match="does not serve by default"):
        registry.get(candidate.id)
    assert registry.get(candidate.id, allow_candidate=True) == candidate


def test_unknown_id_raises_key_error() -> None:
    with pytest.raises(KeyError, match="unknown template"):
        TemplateRegistry().get("no.such.template")


def test_duplicate_registration_rejected() -> None:
    registry = TemplateRegistry()
    registry.register(golden_template())
    with pytest.raises(ValueError, match="already registered"):
        registry.register(golden_template())


def test_ids_sorted_and_filtered_by_status() -> None:
    registry = TemplateRegistry()
    golden = golden_template()
    other = _with_status(
        golden.model_copy(update={"id": "beam_deflection.cantilever.end_point"}), "verified"
    )
    registry.register(golden)
    registry.register(other)
    assert registry.ids() == [
        "beam_deflection.cantilever.end_point",
        "beam_deflection.simply_supported.center_point",
    ]
    assert registry.ids(status="candidate") == ["beam_deflection.simply_supported.center_point"]
    assert registry.ids(status="verified") == ["beam_deflection.cantilever.end_point"]
