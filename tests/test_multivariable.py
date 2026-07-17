"""E2.17: the multivariable operation family (Calculus Volume 3) — Chisel's round-9
exhibits verbatim.

Four operations on a multivariable-aware parse: the differentiation group
(partial_derivative / gradient / directional_derivative), iterated integration
(integrate_multiple), the vector-calculus pair (divergence / curl), and the standalone
symbolic second-order ODE solve (ode_solve, verified by substitution). Every exhibit is
an OpenStax worked example with its printed answer; each new result shape carries its
own independent check (finite difference / cubature / residual).
"""

from __future__ import annotations

from typing import Any

import pytest

from assay.execute import ExecutionError, InputError, execute_template, run_fixtures
from assay.templates import TemplateValidationError, validate_template
from assay.verify import verify_execution

_PROV = {"source": "assay:demo", "license_tier": "open"}


def _op(operation: str, output: str, fixtures: list[dict[str, Any]]) -> Any:
    return validate_template({
        "id": f"{operation}.exhibit",
        "domain": "calculus",
        "description": f"the {operation} exhibit",
        "method": {"kind": "symbolic", "operation": operation},
        "output": {"name": output, "dimension": "dimensionless"},
        "fixtures": fixtures,
        "provenance": _PROV,
    })


# --- Op 1: the differentiation group (m53934, m53940) ------------------------------


def test_partial_derivatives() -> None:
    template = _op("partial_derivative", "partial_derivative", [
        {"setup": {"expression": "x**2 - 3*x*y + 2*y**2 - 4*x + 5*y - 12", "wrt": ["x"]},
         "expect": {"partial_derivative": "2*x - 3*y - 4"}},
        {"setup": {"expression": "x**2 - 3*x*y + 2*y**2 - 4*x + 5*y - 12", "wrt": ["y"]},
         "expect": {"partial_derivative": "-3*x + 4*y + 5"}},
        {"setup": {"expression": "sin(x**2*y - 2*x + 4)", "wrt": ["x"]},
         "expect": {"partial_derivative": "(2*x*y - 2)*cos(x**2*y - 2*x + 4)"}},
        {"setup": {"expression": "sqrt(9 - x**2 - y**2)", "wrt": ["x"]},
         "expect": {"partial_derivative": "-x/sqrt(9 - x**2 - y**2)"}},
        {"setup": {"expression": "sqrt(9 - x**2 - y**2)", "wrt": ["x"],
                   "point": {"x": "sqrt(5)", "y": 0}},
         "expect": {"partial_derivative": [-1.1180339887]}},
        {"setup": {"expression": "exp(x*y)", "wrt": ["x"]},
         "expect": {"partial_derivative": "y*exp(x*y)"}},
        {"setup": {"expression": "log(x**6 + y**4)", "wrt": ["y"]},
         "expect": {"partial_derivative": "4*y**3/(x**6 + y**4)"}},
    ])
    assert all(r.ok for r in run_fixtures(template))
    verified = verify_execution(
        template, {}, setup={"expression": "exp(x*y)", "wrt": ["x"]}
    )
    assert verified.verification.ok
    assert verified.verification.checks[0].name == "partial-difference"


def test_higher_and_mixed_partials_clairaut() -> None:
    """f = x*exp(-3*y) + sin(2*x - 5*y): the second partials, including the mixed pair
    that Clairaut's theorem makes equal (m53934 §Second Partials)."""
    f = "x*exp(-3*y) + sin(2*x - 5*y)"
    template = _op("partial_derivative", "partial_derivative", [
        {"setup": {"expression": f, "wrt": ["x", "x"]},
         "expect": {"partial_derivative": "-4*sin(2*x - 5*y)"}},
        {"setup": {"expression": f, "wrt": ["y", "y"]},
         "expect": {"partial_derivative": "9*x*exp(-3*y) - 25*sin(2*x - 5*y)"}},
        {"setup": {"expression": f, "wrt": ["x", "y"]},
         "expect": {"partial_derivative": "-3*exp(-3*y) + 10*sin(2*x - 5*y)"}},
        {"setup": {"expression": f, "wrt": ["y", "x"]},  # Clairaut — equals wrt x,y
         "expect": {"partial_derivative": "-3*exp(-3*y) + 10*sin(2*x - 5*y)"}},
        # a third-order mixed derivative through the wrt chain
        {"setup": {"expression": "x**2*y**3*z - 3*x*y**2*z**3 + 5*x**2*z - y**3*z",
                   "wrt": ["x", "y", "z"]},
         "expect": {"partial_derivative": "6*x*y**2 - 18*y*z**2"}},
    ])
    assert all(r.ok for r in run_fixtures(template))
    assert verify_execution(
        template, {}, setup={"expression": f, "wrt": ["x", "y"]}
    ).verification.ok


def test_gradient() -> None:
    template = _op("gradient", "gradient", [
        {"setup": {"expression": "x**2 - x*y + 3*y**2", "variables": ["x", "y"]},
         "expect": {"gradient": ["2*x - y", "-x + 6*y"]}},
        {"setup": {"expression": "sin(3*x)*cos(3*y)", "variables": ["x", "y"]},
         "expect": {"gradient": ["3*cos(3*x)*cos(3*y)", "-3*sin(3*x)*sin(3*y)"]}},
        {"setup": {"expression": "3*x**2 - 4*x*y + 2*y**2", "variables": ["x", "y"],
                   "point": {"x": -2, "y": 3}},
         "expect": {"gradient": [-24, 20]}},
        {"setup": {"expression": "5*x**2 - 2*x*y + y**2 - 4*y*z + z**2 + 3*x*z",
                   "variables": ["x", "y", "z"]},
         "expect": {"gradient":
                    ["10*x - 2*y + 3*z", "-2*x + 2*y - 4*z", "3*x - 4*y + 2*z"]}},
    ])
    assert all(r.ok for r in run_fixtures(template))
    verified = verify_execution(
        template, {}, setup={"expression": "x**2 - x*y + 3*y**2", "variables": ["x", "y"]}
    )
    assert verified.verification.ok
    assert verified.verification.checks[0].name == "gradient-difference"


def test_directional_derivative() -> None:
    """Raw directions are normalized to unit vectors. NOTE: the round-9 doc's second
    exhibit printed `17*y/13`; the correct value is `77*y/13` (−5(2x−y)+12(−x+6y) =
    −22x+77y) — reported to Chisel. The engine's finite-difference check confirms it."""
    f = "x**2 - x*y + 3*y**2"
    template = _op("directional_derivative", "directional_derivative", [
        {"setup": {"expression": f, "variables": ["x", "y"], "direction": [3, 4],
                   "point": {"x": -1, "y": 2}},
         "expect": {"directional_derivative": [8]}},
        {"setup": {"expression": f, "variables": ["x", "y"], "direction": [-5, 12]},
         "expect": {"directional_derivative": "-22*x/13 + 77*y/13"}},
        {"setup": {"expression": "5*x**2 - 2*x*y + y**2 - 4*y*z + z**2 + 3*x*z",
                   "variables": ["x", "y", "z"], "direction": [-1, 2, 2],
                   "point": {"x": 1, "y": -2, "z": 3}},
         "expect": {"directional_derivative": [-8.3333333333]}},
        {"setup": {"expression": "log(x**2 + y**2)", "variables": ["x", "y"],
                   "direction": [3, 4], "point": {"x": 1, "y": 2}},
         "expect": {"directional_derivative": [0.88]}},
    ])
    assert all(r.ok for r in run_fixtures(template))
    verified = verify_execution(
        template, {}, setup={"expression": f, "variables": ["x", "y"], "direction": [-5, 12]}
    )
    assert verified.verification.ok
    assert verified.verification.checks[0].name == "directional-difference"


def test_zero_direction_refuses_by_name() -> None:
    template = _op("directional_derivative", "directional_derivative", [
        {"setup": {"expression": "x + y", "variables": ["x", "y"], "direction": [1, 0]},
         "expect": {"directional_derivative": "sqrt(2)/2"}},
    ])
    assert run_fixtures(template)  # the valid fixture builds
    with pytest.raises(ExecutionError, match="zero vector"):
        execute_template(template, {}, setup={
            "expression": "x + y", "variables": ["x", "y"], "direction": [0, 0]})


# --- Op 2: iterated integration (m53961/63/65/66/67/70) ----------------------------


def test_multiple_integrals() -> None:
    template = _op("integrate_multiple", "integral", [
        {"setup": {"expression": "3*x**2 - y", "limits": [["y", 0, 3], ["x", 0, 2]]},
         "expect": {"integral": [15]}},
        {"setup": {"expression": "exp(y)*cos(x)", "limits": [["x", 0, "pi/2"], ["y", 0, 1]]},
         "expect": {"integral": "exp(1) - 1"}},
        {"setup": {"expression": "x**2*exp(x*y)", "limits": [["y", "x/2", 1], ["x", 0, 2]]},
         "expect": {"integral": [2]}},
        {"setup": {"expression": "3*x**2 + y**2",
                   "limits": [["x", "y**2 - 3", "y + 3"], ["y", -2, 3]]},
         "expect": {"integral": "2375/7"}},
        {"setup": {"expression": "x + y*z**2",
                   "limits": [["x", -1, 5], ["y", 2, 4], ["z", 0, 1]]},
         "expect": {"integral": [36]}},
        {"setup": {"expression": "5*x - 3*y",
                   "limits": [["z", 0, "1 - x - y"], ["y", 0, "1 - x"], ["x", 0, 1]]},
         "expect": {"integral": "1/12"}},
        {"setup": {"expression": "(1 - r**2)*r", "limits": [["r", 0, 1], ["theta", 0, "2*pi"]]},
         "expect": {"integral": "pi/2"}},  # polar, Jacobian folded
        {"setup": {"expression": "rho**2*sin(phi)",
                   "limits": [["rho", 0, 1], ["phi", 0, "pi/2"], ["theta", 0, "2*pi"]]},
         "expect": {"integral": "2*pi/3"}},  # spherical, Jacobian folded
        {"setup": {"expression": "x*y*exp(-x**2 - y**2)",
                   "limits": [["y", 0, "oo"], ["x", 0, "oo"]]},
         "expect": {"integral": "1/4"}},  # improper
    ])
    assert all(r.ok for r in run_fixtures(template))
    # the cubature cross-check runs on a finite region
    verified = verify_execution(
        template, {}, setup={"expression": "3*x**2 - y", "limits": [["y", 0, 3], ["x", 0, 2]]}
    )
    assert verified.verification.ok
    assert verified.verification.checks[0].name == "cross-method:cubature"


def test_change_of_variables_jacobian_folded() -> None:
    template = _op("integrate_multiple", "integral", [
        {"setup": {"expression": "(1/2)*u*exp(u*v)",
                   "limits": [["u", "-1/v", "1/v"], ["v", 1, 3]]},
         "expect": {"integral": "2/(3*exp(1))"}},
    ])
    assert all(r.ok for r in run_fixtures(template))


# --- Op 3: divergence and curl (m53986, m54001, m54009) ----------------------------


def test_divergence() -> None:
    template = _op("divergence", "divergence", [
        {"setup": {"field": ["exp(x)", "y*z", "-y*z**2"], "variables": ["x", "y", "z"]},
         "expect": {"divergence": "exp(x) + z - 2*y*z"}},
        {"setup": {"field": ["exp(x)", "y*z", "-y*z**2"], "variables": ["x", "y", "z"],
                   "point": {"x": 0, "y": 2, "z": -1}},
         "expect": {"divergence": [4]}},
        {"setup": {"field": ["x*y", "5 - z**2*y", "x**2 + y**2"], "variables": ["x", "y", "z"]},
         "expect": {"divergence": "y - z**2"}},
        {"setup": {"field": ["x - y", "x + z", "z - y"], "variables": ["x", "y", "z"]},
         "expect": {"divergence": [2]}},
        {"setup": {"field": ["x**2*y", "y - x*y**2"], "variables": ["x", "y"]},
         "expect": {"divergence": [1]}},  # 2-D, source-free test
        {"setup": {"field": ["3*x*y*z**2", "y**2*sin(z)", "x*exp(2*z)"],
                   "variables": ["x", "y", "z"]},
         "expect": {"divergence": "3*y*z**2 + 2*y*sin(z) + 2*x*exp(2*z)"}},
    ])
    assert all(r.ok for r in run_fixtures(template))
    verified = verify_execution(
        template, {},
        setup={"field": ["exp(x)", "y*z", "-y*z**2"], "variables": ["x", "y", "z"]},
    )
    assert verified.verification.ok
    assert verified.verification.checks[0].name == "divergence-difference"


def test_curl() -> None:
    template = _op("curl", "curl", [
        {"setup": {"field": ["x**2*z", "exp(y) + x*z", "x*y*z"], "variables": ["x", "y", "z"]},
         "expect": {"curl": ["x*z - x", "x**2 - y*z", "z"]}},
        {"setup": {"field": ["x**2*z", "y**2*x", "y + 2*z"], "variables": ["x", "y", "z"]},
         "expect": {"curl": ["1", "x**2", "y**2"]}},
        {"setup": {"field": ["x - y", "y - z", "z - x"], "variables": ["x", "y", "z"]},
         "expect": {"curl": [1, 1, 1]}},
        {"setup": {"field": ["x*y", "y*z", "x*z"], "variables": ["x", "y", "z"]},
         "expect": {"curl": ["-y", "-z", "-x"]}},
        {"setup": {"field": ["y", "0"], "variables": ["x", "y"]},
         "expect": {"curl": [0, 0, -1]}},  # 2-D → 3-vector
        {"setup": {"field": ["x*y*z", "y", "x"], "variables": ["x", "y", "z"],
                   "point": {"x": 1, "y": 2, "z": 3}},
         "expect": {"curl": [0, 1, -3]}},
    ])
    assert all(r.ok for r in run_fixtures(template))
    verified = verify_execution(
        template, {},
        setup={"field": ["x**2*z", "exp(y) + x*z", "x*y*z"], "variables": ["x", "y", "z"]},
    )
    assert verified.verification.ok
    assert verified.verification.checks[0].name == "curl-difference"


# --- Op 4: symbolic second-order ODE solve (m54040/44/46) --------------------------


def test_ode_homogeneous_all_root_cases() -> None:
    template = _op("ode_solve", "solution", [
        {"setup": {"equation": "y'' + 9*y' + 14*y = 0"},
         "expect": {"solution": "c1*exp(-2*x) + c2*exp(-7*x)"}},  # distinct real
        {"setup": {"equation": "y'' + 12*y' + 36*y = 0"},
         "expect": {"solution": "c1*exp(-6*x) + c2*x*exp(-6*x)"}},  # repeated
        {"setup": {"equation": "y'' - 2*y' + 5*y = 0"},
         "expect": {"solution": "exp(x)*(c1*cos(2*x) + c2*sin(2*x))"}},  # complex
        {"setup": {"equation": "y'' + 16*y = 0"},
         "expect": {"solution": "c1*cos(4*t) + c2*sin(4*t)"}},  # pure imaginary
    ])
    assert all(r.ok for r in run_fixtures(template))
    verified = verify_execution(template, {}, setup={"equation": "y'' + 9*y' + 14*y = 0"})
    assert verified.verification.ok
    assert verified.verification.checks[0].name == "ode-substitution"


def test_ode_initial_value_problems() -> None:
    template = _op("ode_solve", "solution", [
        {"setup": {"equation": "y'' + 3*y' - 4*y = 0", "ivp": {"y": [0, 1], "y'": [0, -9]}},
         "expect": {"solution": "2*exp(-4*x) - exp(x)"}},
        {"setup": {"equation": "y'' + 6*y' + 13*y = 0", "ivp": {"y": [0, 0], "y'": [0, 2]}},
         "expect": {"solution": "exp(-3*x)*sin(2*x)"}},
        {"setup": {"equation": "y'' + 2*y' + y = 0", "ivp": {"y": [0, 1], "y'": [0, 0]}},
         "expect": {"solution": "exp(-t) + t*exp(-t)"}},
    ])
    assert all(r.ok for r in run_fixtures(template))
    assert verify_execution(
        template, {},
        setup={"equation": "y'' + 3*y' - 4*y = 0", "ivp": {"y": [0, 1], "y'": [0, -9]}},
    ).verification.ok


def test_ode_nonhomogeneous_and_applications() -> None:
    template = _op("ode_solve", "solution", [
        {"setup": {"equation": "y'' + 4*y' + 3*y = 3*x"},
         "expect": {"solution": "c1*exp(-x) + c2*exp(-3*x) + x - 4/3"}},
        {"setup": {"equation": "y'' - y' - 2*y = 2*exp(3*x)"},
         "expect": {"solution": "c1*exp(-x) + c2*exp(2*x) + (1/2)*exp(3*x)"}},
        {"setup": {"equation": "y'' + 5*y' + 6*y = 3*exp(-2*x)"},  # resonant
         "expect": {"solution": "c1*exp(-2*x) + c2*exp(-3*x) + 3*x*exp(-2*x)"}},
        {"setup": {"equation": "x'' + 64*x = 0", "ivp": {"x": [0, 0], "x'": [0, -16]}},
         "expect": {"solution": "-2*sin(8*t)"}},  # SHM, x-dependent → t
        {"setup": {"equation": "x'' + 5*x' + 6*x = 0", "ivp": {"x": [0, 0], "x'": [0, -5]}},
         "expect": {"solution": "-5*exp(-2*t) + 5*exp(-3*t)"}},  # overdamped
    ])
    assert all(r.ok for r in run_fixtures(template))


def test_ode_ivp_float_coefficients_pass() -> None:
    """Round-9 follow-up: a correct IVP answer whose coefficients aren't short decimals
    (1/3, 19/6) must pass — the two comparisons are toleranced to the float-residual
    ceiling, and string IC values (`"1/3"`) are kept exact at the source."""
    template = _op("ode_solve", "solution", [
        # motorcycle suspension (m54044): distinct roots, string IC 1/3 — Site 2
        {"setup": {"equation": "x'' + 20*x' + 96*x = 0", "ivp": {"x": [0, "1/3"], "x'": [0, 10]}},
         "expect": {"solution": "(7/2)*exp(-8*t) - (19/6)*exp(-12*t)"}},
        # the same with a decimal IC — the tolerance path (not exact rationals)
        {"setup": {"equation": "x'' + 20*x' + 96*x = 0",
                   "ivp": {"x": [0, 0.3333333333333333], "x'": [0, 10]}},
         "expect": {"solution": "(7/2)*exp(-8*t) - (19/6)*exp(-12*t)"}},
        # repeated-root IVP (m54040): Site 1 — the float coefficient leaves tiny residual
        {"setup": {"equation": "25*y'' + 10*y' + y = 0", "ivp": {"y": [0, 2], "y'": [0, 1]}},
         "expect": {"solution": "2*exp(-x/5) + (7/5)*x*exp(-x/5)"}},
    ])
    assert all(r.ok for r in run_fixtures(template))
    assert verify_execution(
        template, {},
        setup={"equation": "25*y'' + 10*y' + y = 0", "ivp": {"y": [0, 2], "y'": [0, 1]}},
    ).verification.ok


def test_ode_wrong_printed_answer_is_caught() -> None:
    """The substitution check gives the expect field teeth: a reference that does not
    satisfy the ODE fails, even though the string is well-formed."""
    template_ok = _op("ode_solve", "solution", [
        {"setup": {"equation": "y'' - y = 0"}, "expect": {"solution": "c1*exp(x) + c2*exp(-x)"}},
    ])
    assert all(r.ok for r in run_fixtures(template_ok))
    template_bad = _op("ode_solve", "solution", [
        {"setup": {"equation": "y'' - y = 0"},
         "expect": {"solution": "c1*exp(2*x) + c2*exp(-x)"}},  # exp(2x) is not a solution
    ])
    assert not all(r.ok for r in run_fixtures(template_bad))


# --- the shared gate ---------------------------------------------------------------


def test_multivariable_parse_rejects_stray_symbols() -> None:
    template = _op("gradient", "gradient", [
        {"setup": {"expression": "x*y + z", "variables": ["x", "y"]},  # z not declared
         "expect": {"gradient": ["y", "x"]}},
    ])
    results = run_fixtures(template)
    assert not results[0].ok and "outside the declared variables" in results[0].detail


def test_setup_shapes_are_gated() -> None:
    with pytest.raises(TemplateValidationError, match="'wrt'"):
        _op("partial_derivative", "partial_derivative", [
            {"setup": {"expression": "x*y"}, "expect": {"partial_derivative": "y"}},
        ])
    with pytest.raises(TemplateValidationError, match="'variables'"):
        _op("gradient", "gradient", [
            {"setup": {"expression": "x*y"}, "expect": {"gradient": ["y", "x"]}},
        ])
    with pytest.raises(TemplateValidationError, match="'field'"):
        _op("divergence", "divergence", [
            {"setup": {"variables": ["x", "y"]}, "expect": {"divergence": [0]}},
        ])
    with pytest.raises(TemplateValidationError, match="y', y'' notation"):
        _op("ode_solve", "solution", [
            {"setup": {"equation": "y + 1 = 0"}, "expect": {"solution": "c1"}},
        ])


def test_curl_is_vector_valued_in_the_contract() -> None:
    with pytest.raises(TemplateValidationError, match="vector-valued"):
        _op("curl", "curl", [
            {"setup": {"field": ["y", "0"], "variables": ["x", "y"]},
             "expect": {"curl": "not a list"}},
        ])


def test_the_whitelist_is_unchanged() -> None:
    """The keystone relaxed the variable COUNT, not the function whitelist — a
    non-whitelisted function (sec) is still rejected."""
    with pytest.raises((TemplateValidationError, InputError, ExecutionError)):
        template = _op("partial_derivative", "partial_derivative", [
            {"setup": {"expression": "sec(x*y)", "wrt": ["x"]},
             "expect": {"partial_derivative": "y*tan(x*y)/cos(x*y)"}},
        ])
        run_fixtures(template)
