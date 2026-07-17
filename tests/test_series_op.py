"""E2.16: the series operations — built against Chisel's round-8 exhibits verbatim.

Exhibit set A: Taylor/Maclaurin polynomial to order n (verified by the derivative
table at the center). Exhibit set B: radius + interval of convergence by the ratio
test (verified by numeric term behavior; endpoint inclusion by SymPy's convergence
machinery, undecidable endpoints staying open — the agreed honest boundary). The
term grammar's ``factorial(...)`` is a SCOPED extension: symbolic-only, recorded in
the sandboxing ledger.
"""

from __future__ import annotations

from typing import Any

import pytest

from assay.execute import run_fixtures
from assay.templates import TemplateValidationError, validate_template
from assay.verify import verify_execution

_PROV = {"source": "assay:demo", "license_tier": "open"}


def _taylor(fixtures: list[dict[str, Any]]) -> Any:
    return validate_template({
        "id": "taylor_polynomial.univariate",
        "domain": "calculus",
        "description": "The nth Taylor (Maclaurin at center 0) polynomial of f.",
        "method": {"kind": "symbolic", "operation": "taylor_polynomial"},
        "output": {"name": "polynomial", "dimension": "dimensionless"},
        "fixtures": fixtures,
        "provenance": _PROV,
    })


def _convergence(fixtures: list[dict[str, Any]]) -> Any:
    return validate_template({
        "id": "series.radius_interval_of_convergence",
        "domain": "calculus",
        "description": "Radius and interval of convergence by the ratio test.",
        "method": {"kind": "symbolic", "operation": "series_convergence"},
        "output": {"name": "interval", "dimension": "dimensionless"},
        "fixtures": fixtures,
        "provenance": _PROV,
    })


def test_exhibit_a_taylor_polynomials() -> None:
    """Exhibits 1-3 (m53817): e^x order 3, ln x at 1 order 3, cos x order 4 —
    printed answers verbatim, compared by algebraic equivalence."""
    template = _taylor([
        {"setup": {"expression": "exp(x)", "order": 3},
         "expect": {"polynomial": "1 + x + x**2/2 + x**3/6"}},
        {"setup": {"expression": "log(x)", "center": 1, "order": 3},
         "expect": {"polynomial": "(x - 1) - (x - 1)**2/2 + (x - 1)**3/3"}},
        {"setup": {"expression": "cos(x)", "order": 4},
         "expect": {"polynomial": "1 - x**2/2 + x**4/24"}},
    ])
    assert all(r.ok for r in run_fixtures(template))
    verified = verify_execution(
        template, {}, setup={"expression": "log(x)", "center": 1, "order": 3}
    )
    assert verified.verification.ok
    assert verified.verification.checks[0].name == "taylor-coefficients"


def test_exhibit_b_radius_and_interval() -> None:
    """Exhibits 4-5 (m53761): the four ratio-test archetypes — R = oo, R = 1 open,
    R = 0, and the center-2 half-open case [1, 3)."""
    template = _convergence([
        {"setup": {"term": "x**n / factorial(n)"},
         "expect": {"interval": "(-oo, oo)", "radius": "oo"}},
        {"setup": {"term": "x**n"},
         "expect": {"interval": "(-1, 1)", "radius": [1.0]}},
        {"setup": {"term": "factorial(n) * x**n"},
         "expect": {"interval": "{0}", "radius": [0.0]}},
        {"setup": {"term": "(x - 2)**n / n", "center": 2},
         "expect": {"interval": "[1, 3)", "radius": [1.0]}},
    ])
    assert all(r.ok for r in run_fixtures(template))
    verified = verify_execution(template, {}, setup={"term": "(x - 2)**n / n", "center": 2})
    assert verified.verification.ok
    assert verified.verification.checks[0].name == "term-behavior"


def test_radius_alone_or_interval_alone_also_work() -> None:
    """The expect keys are per-label: a fixture may pin either result or both."""
    template = _convergence([
        {"setup": {"term": "x**n"}, "expect": {"interval": "(-1, 1)"}},
        {"setup": {"term": "x**n / n**2"}, "expect": {"interval": "[-1, 1]"}},
    ])
    assert all(r.ok for r in run_fixtures(template))


def test_factorial_stays_scoped_to_series_terms() -> None:
    """The scoped grammar does NOT leak: factorial is still rejected in formulas
    and ordinary symbolic expressions (the sandboxing ledger's boundary)."""
    with pytest.raises(TemplateValidationError, match="safe math functions"):
        validate_template({
            "id": "leak.check", "domain": "statistics", "description": "leak check",
            "inputs": [{"name": "k", "dimension": "dimensionless"}],
            "method": {"kind": "formula", "expr": "factorial(k)"},
            "output": {"name": "out", "dimension": "dimensionless"},
            "fixtures": [{"inputs": {"k": [3, ""]}, "expect": {"out": [6.0, ""]}}],
            "provenance": _PROV,
        })


def test_series_setup_is_gated() -> None:
    # a hostile term fails at validation
    with pytest.raises(TemplateValidationError, match="safe math functions"):
        _convergence([
            {"setup": {"term": "__import__('os').system('true')"},
             "expect": {"interval": "(-1, 1)"}},
        ])
    # the interval expectation must be canonical notation
    with pytest.raises(TemplateValidationError, match="canonical"):
        _convergence([
            {"setup": {"term": "x**n"}, "expect": {"interval": "radius one"}},
        ])
    # taylor needs an order
    with pytest.raises(TemplateValidationError, match="'order'"):
        _taylor([
            {"setup": {"expression": "exp(x)"}, "expect": {"polynomial": "1 + x"}},
        ])
