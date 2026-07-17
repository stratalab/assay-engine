"""Rendering (E1.8, PRD §10.1): figures as faithful views of VERIFIED data — never answers.

The compute/render split, enforced by types: the *builders* (``function_plot_data``,
``geometry_data``, ``directive_data``) are **compute** — they sample the safely-parsed
expression, derive extrema, and intersect exact SymPy geometry — and produce a
``RenderData``: everything on the figure, *as data*. ``render_svg`` is the **view**: it
takes only ``RenderData`` (numbers and labels — no expressions, nothing to evaluate),
so every mark on an Assay figure traces to a computed quantity. Fixtures assert the
figure's features from the data ("roots marked at x=2, x=3"), never pixels.

Figures are deterministic (engineering NFR-2): the Agg backend is fixed at import, SVG
ids are salted (``svg.hashsalt``), and the date metadata is stripped — the same
``RenderData`` renders byte-identically on the same platform.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import matplotlib

matplotlib.use("Agg")  # the fixed, non-interactive backend — must precede pyplot

import sympy  # noqa: E402
from matplotlib import pyplot  # noqa: E402
from pydantic import BaseModel, ConfigDict, model_validator  # noqa: E402

from assay.execute import ExecutionError, parse_problem  # noqa: E402
from assay.ir import RenderDirective  # noqa: E402

__all__ = [
    "Mark",
    "RenderData",
    "RenderError",
    "Series",
    "directive_data",
    "function_plot_data",
    "geometry_data",
    "render_svg",
]

_RC = {"svg.hashsalt": "assay", "svg.fonttype": "path", "figure.figsize": (6.4, 4.8)}


class RenderError(ExecutionError):
    """The directive/spec cannot be rendered — the message says why (A-12)."""


class Series(BaseModel):
    """A sampled curve or point set — computed data, one (x, y) per point."""

    model_config = ConfigDict(extra="forbid")
    label: str = ""
    x: list[float]
    y: list[float]

    @model_validator(mode="after")
    def _check_lengths(self) -> Series:
        if len(self.x) != len(self.y):
            raise ValueError("a series needs matching x/y lengths")
        return self


class Mark(BaseModel):
    """A marked point — each one traces to a computed, verified quantity."""

    model_config = ConfigDict(extra="forbid")
    label: str
    x: float
    y: float


class RenderData(BaseModel):
    """Everything on the figure, as data — the verifiable object behind the pixels."""

    model_config = ConfigDict(extra="forbid")
    kind: Literal["function_plot", "scatter", "geometry_diagram"]
    title: str = ""
    x_label: str = ""
    y_label: str = ""
    equal_aspect: bool = False
    series: list[Series] = []
    marks: list[Mark] = []


def function_plot_data(
    expression: str,
    variable: str | None = None,
    *,
    x_range: tuple[float, float] | None = None,
    marks: Sequence[tuple[str, float, float]] = (),
    mark_extrema: bool = False,
    samples: int = 201,
) -> RenderData:
    """Sample a gated expression into plot data (compute, not drawing).

    An equation ``lhs = rhs`` plots ``y = lhs - rhs`` (its roots sit on y = 0). With no
    ``x_range``, the range derives from the marks (roots ± 1) or defaults to [-5, 5].
    ``mark_extrema`` marks real critical points (d/dx = 0) — the "vertex" of UX §5.9.
    Non-real sample points are skipped deterministically.
    """
    parsed, symbol = parse_problem(expression, variable)
    mark_objects = [Mark(label=label, x=x, y=y) for label, x, y in marks]
    if mark_extrema:
        for critical in sympy.solve(sympy.diff(parsed, symbol), symbol):
            if isinstance(critical, sympy.Rational | sympy.Float):
                cx = float(critical)
                cy = float(parsed.subs(symbol, critical))
                mark_objects.append(Mark(label=f"extremum ({cx:g}, {cy:g})", x=cx, y=cy))
    if x_range is None:
        anchor_xs = [mark.x for mark in mark_objects]
        x_range = (min(anchor_xs) - 1, max(anchor_xs) + 1) if anchor_xs else (-5.0, 5.0)
    low, high = float(x_range[0]), float(x_range[1])
    if not high > low:
        raise RenderError(f"empty x range [{low:g}, {high:g}]")
    xs: list[float] = []
    ys: list[float] = []
    for index in range(samples):
        x = low + (high - low) * index / (samples - 1)
        try:
            y = float(parsed.subs(symbol, sympy.Float(x)))
        except (TypeError, ValueError):
            continue  # non-real at this x: skipped, deterministically
        xs.append(x)
        ys.append(y)
    if not xs:
        raise RenderError(f"no real points to plot on [{low:g}, {high:g}]")
    title = f"y = {sympy.sstr(parsed)}"
    return RenderData(
        kind="function_plot",
        title=title,
        x_label=str(symbol),
        y_label="y",
        series=[Series(label=title, x=xs, y=ys)],
        marks=mark_objects,
    )


def geometry_data(entities: Sequence[Mapping[str, Any]]) -> RenderData:
    """Analytic geometry is compute (SymPy ``geometry``, exact); the diagram renders
    those computed objects. Pairwise intersections are computed exactly and marked."""
    from sympy import geometry

    shapes: list[Any] = []
    series: list[Series] = []
    for entity in entities:
        kind = entity.get("kind")
        if kind == "circle":
            center_x, center_y = (float(v) for v in entity["center"])
            radius = float(entity["radius"])
            shapes.append(geometry.Circle(geometry.Point(center_x, center_y), radius))
            steps = 128
            xs = [
                center_x + radius * math.cos(2 * math.pi * i / steps) for i in range(steps + 1)
            ]
            ys = [
                center_y + radius * math.sin(2 * math.pi * i / steps) for i in range(steps + 1)
            ]
            series.append(Series(label=f"circle r={radius:g}", x=xs, y=ys))
        elif kind in ("segment", "polygon"):
            points = [(float(p[0]), float(p[1])) for p in entity["points"]]
            if kind == "segment":
                shapes.append(geometry.Segment(*[geometry.Point(*p) for p in points]))
            else:
                shapes.append(geometry.Polygon(*[geometry.Point(*p) for p in points]))
                points = [*points, points[0]]  # close the outline
            series.append(
                Series(label=kind, x=[p[0] for p in points], y=[p[1] for p in points])
            )
        else:
            raise RenderError(f"unknown geometry entity {kind!r}")
    marks: list[Mark] = []
    for i, first in enumerate(shapes):
        for second in shapes[i + 1 :]:
            for hit in geometry.intersection(first, second):
                if isinstance(hit, geometry.Point2D):
                    x, y = float(hit.x), float(hit.y)
                    marks.append(Mark(label=f"intersection ({x:g}, {y:g})", x=x, y=y))
    return RenderData(
        kind="geometry_diagram",
        x_label="x",
        y_label="y",
        equal_aspect=True,
        series=series,
        marks=marks,
    )


def directive_data(
    directive: RenderDirective,
    *,
    expression: str | None = None,
    variable: str | None = None,
    marks: Sequence[tuple[str, float, float]] = (),
) -> RenderData:
    """Execute a validated IR render directive (PRD §10.1) — *after* compute + verify;
    the caller passes the verified quantities to mark (they are never re-derived here)."""
    spec = directive.spec
    if directive.kind == "function_plot":
        target = spec.get("expression", expression)
        if not isinstance(target, str) or not target.strip():
            raise RenderError("function_plot needs an expression")
        x_range: tuple[float, float] | None = None
        if "x_range" in spec:
            low, high = (float(v) for v in spec["x_range"])
            x_range = (low, high)
        return function_plot_data(
            target,
            spec.get("variable", variable),
            x_range=x_range,
            marks=marks,
            mark_extrema=bool(spec.get("mark_extrema")),
        )
    if directive.kind == "scatter":
        series = [Series.model_validate(entry) for entry in spec.get("series", [])]
        if not series:
            raise RenderError("scatter needs at least one series of points")
        return RenderData(
            kind="scatter",
            title=str(spec.get("title", "")),
            x_label=str(spec.get("x_label", "")),
            y_label=str(spec.get("y_label", "")),
            series=series,
            marks=[Mark(label=label, x=x, y=y) for label, x, y in marks],
        )
    if directive.kind == "geometry_diagram":
        return geometry_data(spec.get("entities", []))
    raise RenderError(f"unknown render kind {directive.kind!r}")


def render_svg(data: RenderData, path: str | Path) -> Path:
    """The view: draw ``RenderData`` — numbers in, deterministic SVG out (no dates,
    salted ids, fixed backend). Nothing here evaluates an expression."""
    destination = Path(path)
    with matplotlib.rc_context(_RC):
        figure, axes = pyplot.subplots()
        try:
            for series in data.series:
                style = "o" if data.kind == "scatter" else "-"
                axes.plot(series.x, series.y, style, markersize=3, label=series.label or None)
            for mark in data.marks:
                axes.plot([mark.x], [mark.y], "o", color="black", markersize=4)
                axes.annotate(
                    mark.label, (mark.x, mark.y), textcoords="offset points", xytext=(6, 6)
                )
            if data.title:
                axes.set_title(data.title)
            axes.set_xlabel(data.x_label)
            axes.set_ylabel(data.y_label)
            if data.equal_aspect:
                axes.set_aspect("equal")
            axes.grid(True, alpha=0.3)
            figure.savefig(destination, format="svg", metadata={"Date": None})
        finally:
            pyplot.close(figure)
    return destination
