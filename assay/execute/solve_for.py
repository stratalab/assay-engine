"""The solve-for execution mode (E2.10, chisel round 5 §1): recover an input.

Physics pedagogy solves equations in arbitrary directions; the book-printed
rearrangements are templates (content), and this mode covers the rest: a **formula
template + a target input + values for the other inputs and the output** → the
composed expression is inverted **symbolically through the same gated machinery as
everything else** (the expression was safe-parsed at validation; ``sympy.solve`` works
the tree; the roots are evaluated by the existing unit-carrying walk — no eval, no new
grammar), and each root is filtered by the **target input's declared dimension**.

Honesty rules, from the design fixtures:

- **Both-roots**: every surviving root is returned (the freeway-ramp quadratic yields
  t = 10 s *and* t = −20 s); physical selection is the caller's stated act, never a
  silent choice here.
- **Refusal**: an inversion that leaves the safe grammar (Lambert-W shapes), yields no
  roots, or yields none of the right dimension fails clear with the reasons (A-12).
- **Verification** is forward substitution — the recovered input runs back through the
  *forward* template and must reproduce the stated output (``assay.verify``); the
  print-precision comparison against a book's expected value belongs to the fixture's
  ``tol`` (set from the recovered input's precision — Chisel's tolerance-provenance
  rule).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import sympy

from assay.execute import (
    ExecutedValue,
    ExecutionError,
    ExecutionResult,
    InputError,
    _evaluate,
    _registry,
    declared_dimensionality,
    parse_formula,
)
from assay.templates import FormulaMethod, Template

__all__ = ["composed_expression", "solve_for_input"]

# The stated forward output binds to this reserved symbol during root evaluation.
_GIVEN = "_assay_given_output"


def composed_expression(template: Template) -> tuple[Any, set[str]]:
    """The template's method as ONE expression over its declared inputs — steps (E2.9
    DAGs) are substituted through, so inversion sees the whole chain."""
    if not isinstance(template.method, FormulaMethod):
        raise ExecutionError(
            f"template {template.id!r} is not a formula template — only formula"
            " methods invert (solver/symbolic templates own their own operations)"
        )
    input_names = {declared.name for declared in template.inputs}
    if not template.method.steps:
        assert template.method.expr is not None  # the schema enforces exactly one
        return parse_formula(template.method.expr, input_names), input_names
    substitutions: dict[Any, Any] = {}
    allowed = set(input_names)
    parsed = None
    for step in template.method.steps:
        parsed = parse_formula(step.expr, allowed).subs(substitutions)
        substitutions[sympy.Symbol(step.name)] = parsed
        allowed.add(step.name)
    return parsed, input_names


def solve_for_input(
    template: Template,
    target: str,
    inputs: Mapping[str, tuple[float, str]],
    output_value: tuple[float, str],
) -> ExecutionResult:
    """Recover ``target`` from the other inputs and the stated output.

    Returns every root that evaluates real and carries the target's declared
    dimension, in base SI units, sorted — all of them (both-roots honesty). Refuses
    with reasons when nothing survives.
    """
    declared = {inp.name: inp for inp in template.inputs}
    if any(inp.many for inp in template.inputs):
        raise ExecutionError(
            f"template {template.id!r} has list inputs — solve-for over lists is not"
            " defined (schema v2 boundary)"
        )
    if target not in declared:
        raise InputError(f"solve_for target {target!r} is not an input of {template.id!r}")
    if target in inputs:
        raise InputError(f"solve_for target {target!r} must not also be supplied")
    others = {name for name in declared if name != target}
    if missing := sorted(others - set(inputs)):
        raise InputError(
            f"solve-for needs every other input: missing {', '.join(missing)}"
            " — nothing will be fabricated (A-2)"
        )
    ureg = _registry()
    expression, _ = composed_expression(template)

    # Bind the knowns: the other inputs (dimension-checked) and the stated output.
    bindings: dict[str, Any] = {}
    for name in others:
        value, unit = inputs[name]
        try:
            quantity = ureg.Quantity(value, unit or "")
        except Exception as exc:
            raise InputError(f"input {name!r}: unknown unit {unit!r}") from exc
        expected_dim = declared_dimensionality(declared[name].dimension, ureg)
        if quantity.dimensionality != expected_dim:
            raise InputError(
                f"input {name!r} must have dimension {declared[name].dimension!r};"
                f" got {unit!r}"
            )
        bindings[name] = quantity
    out_value, out_unit = output_value
    try:
        given = ureg.Quantity(out_value, out_unit or "")
    except Exception as exc:
        raise InputError(f"stated output: unknown unit {out_unit!r}") from exc
    output_dim = declared_dimensionality(template.output.dimension, ureg)
    if given.dimensionality != output_dim:
        raise InputError(
            f"the stated output must have dimension {template.output.dimension!r};"
            f" got {out_unit!r}"
        )
    bindings[_GIVEN] = given

    try:
        solutions = sympy.solve(expression - sympy.Symbol(_GIVEN), sympy.Symbol(target))
    except Exception as exc:
        raise ExecutionError(f"could not invert {template.id!r} for {target!r}: {exc}") from exc
    if not solutions:
        raise ExecutionError(
            f"no symbolic inverse of {template.id!r} for {target!r} — refusing to"
            " guess (A-12)"
        )

    target_dim = declared_dimensionality(declared[target].dimension, ureg)
    roots: list[Any] = []
    dropped: list[str] = []
    for solution in solutions:
        try:
            quantity = _evaluate(solution, bindings, ureg).to_base_units()
            magnitude = float(quantity.magnitude)
        except Exception as exc:
            dropped.append(f"{sympy.sstr(solution)}: not evaluable ({exc})")
            continue
        if quantity.dimensionality != target_dim:
            dropped.append(
                f"{sympy.sstr(solution)}: dimension {quantity.dimensionality} is not"
                f" {declared[target].dimension!r}"
            )
            continue
        duplicate = any(
            abs(magnitude - float(kept.magnitude)) <= 1e-12 * max(abs(magnitude), 1.0)
            for kept in roots
        )
        if not duplicate:
            roots.append(quantity)
    if not roots:
        raise ExecutionError(
            f"no admissible root for {target!r}: " + "; ".join(dropped)
            if dropped
            else f"no admissible root for {target!r}"
        )
    roots.sort(key=lambda quantity: float(quantity.magnitude))
    from assay.answer import TraceStep  # the execution trace (E2.15)

    trace = [
        TraceStep(
            label=target,
            expr=f"{template.output.name} = {sympy.sstr(expression)}",
            value=float(quantity.magnitude),
            unit=f"{quantity.units}",
            note=(
                "recovered by symbolic inversion; every root is verified by"
                " independent forward substitution"
            ),
        )
        for quantity in roots
    ]
    return ExecutionResult(
        output=target,
        trace=trace,
        values=[
            ExecutedValue(
                label=target,
                value=float(quantity.magnitude),
                unit=f"{quantity.units}",
            )
            for quantity in roots
        ],
    )
