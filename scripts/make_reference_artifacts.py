#!/usr/bin/env python3
"""Regenerate the committed reference artifacts (E3.4, NFR-2).

These artifacts are the cross-platform reproducibility contract:
``tests/test_reproducibility.py`` reruns every one of them on the full CI matrix and
refuses ``failed``. Regenerate them ONLY deliberately — after a schema change, a
template change, or a deliberate dependency re-pin — never to silence a red rerun
(a red rerun is the finding, not the noise):

    uv run python scripts/make_reference_artifacts.py

The set spans every method kind and the resolver: symbolic (exact everywhere),
formula (float + resolved facts), a multi-step DAG, and all four solver bindings.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from assay.artifact import create_artifact, save_artifact
from assay.ir import IR
from assay.resolver import Resolver
from assay.templates import golden_templates

DESTINATION = Path(__file__).resolve().parent.parent / "tests" / "reference_artifacts"

REFERENCE_IRS: list[dict[str, Any]] = [
    {
        "domain": "algebra",
        "task": "solve_equation.univariate",
        "setup": {"expression": "x**2 - 5*x + 6 = 0"},
    },
    {
        "domain": "calculus",
        "task": "integrate.univariate",
        "setup": {"expression": "sin(x)**2", "variable": "x"},
    },
    {
        "domain": "calculus",
        "task": "differentiate.univariate",
        "setup": {"expression": "x**3 - x"},
    },
    {
        "domain": "structural_mechanics",
        "task": "beam_deflection.simply_supported.center_point",
        "setup": {"material": "steel.structural"},
        "inputs": {
            "P": {"value": 5000, "unit": "N"},
            "L": {"value": 2, "unit": "m"},
            "I": {"value": 8.33e-6, "unit": "m**4"},
        },
        "missing_inputs": ["E"],
    },
    {
        "domain": "oscillation",
        "task": "pendulum.period.simple",
        "inputs": {"L": {"value": 1, "unit": "m"}},
        "missing_inputs": ["g"],
    },
    {
        "domain": "materials",
        "task": "principal_stress.plane.max",
        "inputs": {
            "sx": {"value": 80e6, "unit": "Pa"},
            "sy": {"value": 20e6, "unit": "Pa"},
            "txy": {"value": 25e6, "unit": "Pa"},
        },
    },
    {
        "domain": "numerical_methods",
        "task": "root_find.univariate.numeric",
        "setup": {"expression": "x**5 - x + 1 = 0", "bracket": [-2, 0]},
    },
    {
        "domain": "calculus",
        "task": "integrate.definite.numeric",
        "setup": {"expression": "exp(-x**2)", "limits": [0, 1]},
    },
    {
        "domain": "optimization",
        "task": "minimize.univariate.numeric",
        "setup": {"expression": "sin(x) + x**2 / 10", "bounds": [-3, 3]},
    },
    {
        "domain": "differential_equations",
        "task": "ode.initial_value.numeric",
        "setup": {"expression": "-2 * t * y", "y0": 1, "t_span": [0, 1]},
    },
]


def main() -> int:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    for stale in DESTINATION.glob("*.result.json"):
        stale.unlink()
    catalog = {template.id: template for template in golden_templates()}
    resolver = Resolver()
    for record in REFERENCE_IRS:
        ir = IR.model_validate(record)
        template = catalog[ir.task]
        if ir.missing_inputs:
            ir = resolver.resolve_missing(ir, template).ir
            assert not ir.missing_inputs, (ir.task, ir.missing_inputs)
        artifact = create_artifact(ir, template)
        assert artifact.answer.verified.ok, (ir.task, artifact.answer.verified)
        name = ir.task.replace(".", "_") + ".result.json"
        path = save_artifact(artifact, DESTINATION / name)
        print(f"  wrote {path.name}")
    print(f"{len(REFERENCE_IRS)} reference artifacts in {DESTINATION}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
