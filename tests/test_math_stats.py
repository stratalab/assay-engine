"""E2.13: the mathematics & statistics schema extension.

What the three-book retrieval (College Algebra, Calculus, Introductory Statistics)
needs that v2 lacked: the statistical reducer vocabulary (count/mean/sum_sq +
order statistics + paired sum_product), multi-output steps templates (slope AND
intercept; mean AND std), the ``limit`` and ``solve_inequality`` operations,
definite/improper symbolic integration, and ``erf`` (the normal CDF) — every new
result shape carrying its own independent verification.
"""

from __future__ import annotations

from typing import Any

import pytest

from assay.execute import InputError, execute_template, run_fixtures
from assay.templates import TemplateValidationError, validate_template
from assay.verify import verify_execution

_PROV = {"source": "assay:demo", "license_tier": "open"}


def _symbolic(operation: str, fixtures: list[dict[str, Any]], **overrides: Any) -> Any:
    record: dict[str, Any] = {
        "id": f"{operation}.exhibit",
        "domain": "calculus",
        "description": f"the {operation} exhibit",
        "method": {"kind": "symbolic", "operation": operation},
        "output": {"name": "result", "dimension": "dimensionless"},
        "fixtures": fixtures,
        "provenance": _PROV,
    }
    record.update(overrides)
    return validate_template(record)


# --- the statistics reducers -------------------------------------------------------


def _std_template() -> Any:
    """Population σ AND mean as ONE template — the multi-output exhibit."""
    return validate_template({
        "schema_version": 2,
        "id": "population_std.dataset",
        "domain": "statistics",
        "description": "Population standard deviation (and mean) of a dataset.",
        "inputs": [{"name": "x", "dimension": "dimensionless", "many": True}],
        "method": {"kind": "formula", "steps": [
            {"name": "x_mean", "expr": "mean(x)"},
            {"name": "sigma", "expr": "sqrt(sum_sq(x)/count(x) - x_mean**2)"},
        ]},
        "output": {"name": "sigma", "dimension": "dimensionless"},
        "extra_outputs": [{"name": "x_mean", "dimension": "dimensionless"}],
        "fixtures": [{
            "inputs": {"x": [[2, ""], [4, ""], [4, ""], [4, ""],
                             [5, ""], [5, ""], [7, ""], [9, ""]]},
            "expect": {"sigma": [2.0, ""], "x_mean": [5.0, ""]},
            "tol": 1e-9,
        }],
        "provenance": _PROV,
    })


def test_mean_and_std_are_one_template() -> None:
    template = _std_template()
    assert all(r.ok for r in run_fixtures(template))
    result = execute_template(template, {"x": [(1.0, ""), (2.0, ""), (3.0, "")]})
    by_label = {v.label: v.value for v in result.values}
    assert by_label["x_mean"] == pytest.approx(2.0)
    assert by_label["sigma"] == pytest.approx((2 / 3) ** 0.5)
    verified = verify_execution(template, {"x": [(1.0, ""), (2.0, ""), (3.0, "")]})
    assert verified.verification.ok


def test_regression_slope_and_intercept_via_sum_product() -> None:
    template = validate_template({
        "schema_version": 2,
        "id": "linear_regression.least_squares",
        "domain": "statistics",
        "description": "Least-squares slope and intercept over paired data.",
        "inputs": [
            {"name": "x", "dimension": "dimensionless", "many": True},
            {"name": "y", "dimension": "dimensionless", "many": True},
        ],
        "method": {"kind": "formula", "steps": [
            {"name": "slope", "expr": "(count(x)*sum_product(x, y) - sum(x)*sum(y))"
                                      " / (count(x)*sum_sq(x) - sum(x)**2)"},
            {"name": "intercept", "expr": "mean(y) - slope*mean(x)"},
        ]},
        "output": {"name": "intercept", "dimension": "dimensionless"},
        "extra_outputs": [{"name": "slope", "dimension": "dimensionless"}],
        "fixtures": [{
            "inputs": {"x": [[1, ""], [2, ""], [3, ""]],
                       "y": [[2.1, ""], [3.9, ""], [6.0, ""]]},
            "expect": {"intercept": [0.1, ""], "slope": [1.95, ""]},
            "tol": 1e-6,
        }],
        "provenance": _PROV,
    })
    assert all(r.ok for r in run_fixtures(template))


def test_paired_lists_must_match_in_length() -> None:
    template = validate_template({
        "schema_version": 2,
        "id": "sxy.paired",
        "domain": "statistics",
        "description": "Sum of products of paired data.",
        "inputs": [
            {"name": "x", "dimension": "dimensionless", "many": True},
            {"name": "y", "dimension": "dimensionless", "many": True},
        ],
        "method": {"kind": "formula", "expr": "sum_product(x, y)"},
        "output": {"name": "sxy", "dimension": "dimensionless"},
        "fixtures": [{
            "inputs": {"x": [[1, ""], [2, ""]], "y": [[3, ""], [4, ""]]},
            "expect": {"sxy": [11.0, ""]}, "tol": 1e-9,
        }],
        "provenance": _PROV,
    })
    assert all(r.ok for r in run_fixtures(template))
    with pytest.raises(InputError, match="same length"):
        execute_template(template, {"x": [(1.0, ""), (2.0, "")], "y": [(3.0, "")]})


def test_order_statistics_bind_deterministically() -> None:
    """min/max/median have no arithmetic expansion — they bind at input time, on
    base-unit magnitudes, so mixed units of one dimension order correctly."""
    template = validate_template({
        "schema_version": 2,
        "id": "resistance_spread.dataset",
        "domain": "electromagnetism",
        "description": "Spread (max - min) of measured resistances.",
        "inputs": [{"name": "R", "dimension": "resistance", "many": True}],
        "method": {"kind": "formula", "expr": "max(R) - min(R)"},
        "output": {"name": "spread", "dimension": "resistance"},
        "fixtures": [{  # 0.5 kΩ is the max even though 20 > 0.5 numerically
            "inputs": {"R": [[20, "ohm"], [0.5, "kiloohm"], [80, "ohm"]]},
            "expect": {"spread": [480.0, "ohm"]},
            "tol": 1e-9,
        }],
        "provenance": _PROV,
    })
    assert all(r.ok for r in run_fixtures(template))
    median = validate_template({
        "schema_version": 2,
        "id": "median.dataset",
        "domain": "statistics",
        "description": "Median of a dataset.",
        "inputs": [{"name": "x", "dimension": "dimensionless", "many": True}],
        "method": {"kind": "formula", "expr": "median(x)"},
        "output": {"name": "median_value", "dimension": "dimensionless"},
        "fixtures": [
            {"inputs": {"x": [[7, ""], [1, ""], [9, ""]]},
             "expect": {"median_value": [7.0, ""]}, "tol": 1e-9},
            {"inputs": {"x": [[7, ""], [1, ""], [9, ""], [3, ""]]},  # even: mid-average
             "expect": {"median_value": [5.0, ""]}, "tol": 1e-9},
        ],
        "provenance": _PROV,
    })
    assert all(r.ok for r in run_fixtures(median))


def test_normal_cdf_via_erf() -> None:
    template = validate_template({
        "id": "normal_cdf.standard",
        "domain": "statistics",
        "description": "Standard normal CDF via erf.",
        "inputs": [{"name": "z", "dimension": "dimensionless"}],
        "method": {"kind": "formula", "expr": "(1 + erf(z / sqrt(2))) / 2"},
        "output": {"name": "probability", "dimension": "dimensionless"},
        "fixtures": [
            {"inputs": {"z": [1.0, ""]},
             "expect": {"probability": [0.8413447461, ""]}, "tol": 1e-9},
            {"inputs": {"z": [-1.96, ""]},
             "expect": {"probability": [0.0249978952, ""]}, "tol": 1e-7},
        ],
        "provenance": _PROV,
    })
    assert all(r.ok for r in run_fixtures(template))
    assert verify_execution(template, {"z": (1.0, "")}).verification.ok


# --- multi-output contract ----------------------------------------------------------


def test_extra_outputs_are_gated() -> None:
    base: dict[str, Any] = {
        "schema_version": 2,
        "id": "gate.exhibit",
        "domain": "statistics",
        "description": "gate exhibits",
        "inputs": [{"name": "x", "dimension": "dimensionless", "many": True}],
        "method": {"kind": "formula", "expr": "mean(x)"},
        "output": {"name": "m", "dimension": "dimensionless"},
        "fixtures": [{"inputs": {"x": [[1, ""]]}, "expect": {"m": [1.0, ""]}}],
        "provenance": _PROV,
    }
    # extras need steps
    with pytest.raises(TemplateValidationError, match="formula templates with steps"):
        validate_template(base | {
            "extra_outputs": [{"name": "n", "dimension": "dimensionless"}],
        })
    # extras must name an EARLIER step
    with pytest.raises(TemplateValidationError, match="earlier step"):
        validate_template(base | {
            "method": {"kind": "formula", "steps": [
                {"name": "n", "expr": "count(x)"}, {"name": "m", "expr": "mean(x)"},
            ]},
            "extra_outputs": [{"name": "m2", "dimension": "dimensionless"}],
        })
    # v1 freeze
    with pytest.raises(TemplateValidationError, match="require schema_version 2"):
        validate_template(base | {
            "schema_version": 1,
            "inputs": [{"name": "x", "dimension": "dimensionless"}],
            "method": {"kind": "formula", "steps": [
                {"name": "n", "expr": "x"}, {"name": "m", "expr": "n + 1"},
            ]},
            "extra_outputs": [{"name": "n", "dimension": "dimensionless"}],
            "fixtures": [{"inputs": {"x": [1, ""]}, "expect": {"m": [2.0, ""]}}],
        })
    # expect keys must be declared outputs and include the primary
    with pytest.raises(TemplateValidationError, match="only declared outputs"):
        validate_template(base | {"fixtures": [
            {"inputs": {"x": [[1, ""]]}, "expect": {"m": [1.0, ""], "n": [1.0, ""]}},
        ]})


# --- the calculus operations --------------------------------------------------------


def test_limits_evaluate_and_verify() -> None:
    template = _symbolic("limit", [
        {"setup": {"expression": "sin(x)/x", "point": 0}, "expect": {"result": [1.0]}},
        {"setup": {"expression": "(x**2 - 1)/(x - 1)", "point": 1},
         "expect": {"result": [2.0]}},
        {"setup": {"expression": "1/x", "point": "oo"}, "expect": {"result": [0.0]}},
        {"setup": {"expression": "1/x", "point": 0, "direction": "+"},
         "expect": {"result": "oo"}},
        {"setup": {"expression": "1/x", "point": 0, "direction": "-"},
         "expect": {"result": "-oo"}},
    ])
    assert all(r.ok for r in run_fixtures(template))
    verified = verify_execution(template, {}, setup={"expression": "sin(x)/x", "point": 0})
    assert verified.verification.ok
    assert verified.verification.checks[0].name == "limit-approach"


def test_limit_needs_a_point() -> None:
    with pytest.raises(TemplateValidationError, match="needs 'point'"):
        _symbolic("limit", [{"setup": {"expression": "sin(x)/x"},
                             "expect": {"result": [1.0]}}])


def test_definite_and_improper_integrals() -> None:
    template = _symbolic("integrate", [
        {"setup": {"expression": "exp(-x)", "limits": [0, "oo"]},
         "expect": {"result": [1.0]}},
        {"setup": {"expression": "1/x**2", "limits": [1, "oo"]},
         "expect": {"result": [1.0]}},
        {"setup": {"expression": "1/(1 + x**2)", "limits": ["-oo", "oo"]},
         "expect": {"result": "pi"}},  # exactness is the brand
        {"setup": {"expression": "1/x", "limits": [1, "oo"]},
         "expect": {"result": "oo"}},  # divergence is stated, not approximated
    ], id="integrate.definite.exhibit")
    assert all(r.ok for r in run_fixtures(template))
    verified = verify_execution(
        template, {}, setup={"expression": "1/(1 + x**2)", "limits": ["-oo", "oo"]}
    )
    assert verified.verification.ok
    assert verified.verification.checks[0].name == "cross-method:quadrature"
    divergent = verify_execution(
        template, {}, setup={"expression": "1/x", "limits": [1, "oo"]}
    )
    assert divergent.verification.ok  # partial integrals shown to grow without bound


def test_indefinite_integration_is_unchanged() -> None:
    template = _symbolic("integrate", [
        {"setup": {"expression": "2*x"}, "expect": {"result": "x**2"}},
    ], id="integrate.indefinite.exhibit")
    assert all(r.ok for r in run_fixtures(template))


# --- inequalities ---------------------------------------------------------------


def test_inequalities_solve_to_interval_notation() -> None:
    template = _symbolic("solve_inequality", [
        {"setup": {"expression": "2*x - 3 < 7"}, "expect": {"result": "(-oo, 5)"}},
        {"setup": {"expression": "x**2 <= 4"}, "expect": {"result": "[-2, 2]"}},
        {"setup": {"expression": "x**2 >= 9"},
         "expect": {"result": "(-oo, -3] U [3, oo)"}},
        {"setup": {"expression": "x**2 <= 0"}, "expect": {"result": "{0}"}},
        {"setup": {"expression": "x**2 < 0"}, "expect": {"result": "empty"}},
    ], domain="algebra")
    assert all(r.ok for r in run_fixtures(template))
    verified = verify_execution(template, {}, setup={"expression": "x**2 >= 9"})
    assert verified.verification.ok
    assert verified.verification.checks[0].name == "interval-testpoints"


def test_inequality_grammar_is_tight() -> None:
    # exactly one relational operator
    with pytest.raises(TemplateValidationError, match="exactly one relational"):
        _symbolic("solve_inequality", [
            {"setup": {"expression": "1 < x < 5"}, "expect": {"result": "(1, 5)"}},
        ], domain="algebra")
    # the expect string must be canonical interval notation
    with pytest.raises(TemplateValidationError, match="interval notation"):
        _symbolic("solve_inequality", [
            {"setup": {"expression": "x < 5"}, "expect": {"result": "x less than 5"}},
        ], domain="algebra")
    # both sides pass the ordinary safe gate
    with pytest.raises(TemplateValidationError, match="safe math functions"):
        _symbolic("solve_inequality", [
            {"setup": {"expression": "__import__('os') < 5"},
             "expect": {"result": "empty"}},
        ], domain="algebra")


# --- reducer grammar ----------------------------------------------------------------


def test_new_reducers_keep_the_v2_rules() -> None:
    base: dict[str, Any] = {
        "schema_version": 2,
        "id": "reducer.gate",
        "domain": "statistics",
        "description": "reducer gate exhibits",
        "inputs": [
            {"name": "x", "dimension": "dimensionless", "many": True},
            {"name": "c", "dimension": "dimensionless"},
        ],
        "output": {"name": "out", "dimension": "dimensionless"},
        "fixtures": [{
            "inputs": {"x": [[1, ""], [2, ""]], "c": [1, ""]},
            "expect": {"out": [1.5, ""]}, "tol": 1e-9,
        }],
        "provenance": _PROV,
    }
    assert validate_template(base | {"method": {"kind": "formula", "expr": "mean(x)"}})
    # reducers still apply only to list inputs
    with pytest.raises(TemplateValidationError, match="only to list inputs"):
        validate_template(base | {"method": {"kind": "formula", "expr": "median(c)"}})
    # a list input still appears only inside a reducer
    with pytest.raises(TemplateValidationError, match="only inside a reducer"):
        validate_template(base | {"method": {"kind": "formula", "expr": "min(x) + x"}})
    # sum_product needs two DISTINCT list inputs
    with pytest.raises(TemplateValidationError, match="two distinct input names"):
        validate_template(
            base | {"method": {"kind": "formula", "expr": "sum_product(x, x)"}}
        )
