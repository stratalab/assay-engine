"""E1.8: rendering — figures as faithful views of verified data.

The done-criteria: the plot-and-solve flow renders a figure whose marked roots come
from the verified solve; the figure reproduces byte-stably; figure fixtures assert
features from the data, never pixels.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from assay.cli import main
from assay.execute import UnsafeExpressionError
from assay.ir import RenderDirective
from assay.render import (
    RenderData,
    RenderError,
    Series,
    directive_data,
    function_plot_data,
    geometry_data,
    render_svg,
)
from assay.templates import golden_templates
from assay.verify import verify_execution

_SOLVE = {t.id: t for t in golden_templates()}["solve_equation.univariate"]


@pytest.fixture(autouse=True)
def _workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_marked_roots_come_from_the_verified_solve() -> None:
    """The done-criterion, as a figure fixture: features asserted from DATA, not pixels."""
    verified = verify_execution(_SOLVE, {}, setup={"expression": "x**2 - 5*x + 6 = 0"})
    assert verified.verification.ok and verified.result is not None
    roots = [float(v.value) for v in verified.result.values]
    data = function_plot_data(
        "x**2 - 5*x + 6 = 0",
        marks=[(f"x = {root:g}", root, 0.0) for root in roots],
        mark_extrema=True,
    )
    marked = {(mark.x, mark.y) for mark in data.marks}
    assert (2.0, 0.0) in marked and (3.0, 0.0) in marked  # the verified roots
    assert (2.5, -0.25) in marked  # the vertex, computed (d/dx = 0)
    curve = data.series[0]
    assert (curve.x[0], curve.x[-1]) == (1.0, 4.0)  # root-derived range (UX §5.9)


def test_samples_lie_on_the_curve() -> None:
    data = function_plot_data("x**2 - 5*x + 6", x_range=(0.0, 5.0), samples=11)
    curve = data.series[0]
    for x, y in zip(curve.x, curve.y, strict=True):
        assert y == pytest.approx(x * x - 5 * x + 6, rel=1e-12, abs=1e-12)


def test_figure_reproduces_byte_stably() -> None:
    data = function_plot_data("x**2 - 5*x + 6", marks=[("x = 2", 2.0, 0.0)])
    first = render_svg(data, "a.svg").read_bytes()
    second = render_svg(data, "b.svg").read_bytes()
    assert first == second
    assert b"dc:date" not in first  # no timestamps (NFR-2)


def test_geometry_intersections_are_exact_computed_marks() -> None:
    data = geometry_data(
        [
            {"kind": "circle", "center": [0, 0], "radius": 5},
            {"kind": "circle", "center": [8, 0], "radius": 5},
            {"kind": "segment", "points": [[0, 0], [8, 0]]},
        ]
    )
    assert data.equal_aspect
    marked = {(mark.x, mark.y) for mark in data.marks}
    assert (4.0, 3.0) in marked and (4.0, -3.0) in marked  # SymPy-exact intersections
    assert (5.0, 0.0) in marked  # circle ∩ segment
    render_svg(data, "geometry.svg")
    assert Path("geometry.svg").exists()


def test_scatter_directive_renders_supplied_data() -> None:
    directive = RenderDirective(
        kind="scatter",
        spec={"series": [{"label": "obs", "x": [1, 2, 3], "y": [2.0, 4.1, 5.9]}]},
    )
    data = directive_data(directive)
    assert data.kind == "scatter" and data.series[0].y[1] == 4.1
    render_svg(data, "scatter.svg")
    assert Path("scatter.svg").exists()


def test_unknown_directive_kind_fails_clear() -> None:
    with pytest.raises(RenderError, match="unknown render kind"):
        directive_data(RenderDirective(kind="hologram", spec={}))
    with pytest.raises(RenderError, match="unknown geometry entity"):
        geometry_data([{"kind": "tesseract"}])


def test_hostile_expression_is_gated_before_sampling() -> None:
    with pytest.raises(UnsafeExpressionError):
        function_plot_data("__import__('os').system('true')", "x")


def test_non_real_regions_are_skipped_deterministically() -> None:
    data = function_plot_data("sqrt(x)", x_range=(-1.0, 1.0), samples=21)
    assert all(x >= 0 for x in data.series[0].x)  # negative-x samples skipped


def test_render_data_is_numbers_only() -> None:
    """The compute/render split: RenderData carries no expressions to evaluate."""
    fields = set(RenderData.model_fields) | set(Series.model_fields)
    assert "expression" not in fields and "expr" not in fields


def test_cli_plot_and_solve_flow(capsys: pytest.CaptureFixture[str]) -> None:
    """UX §5.9 end to end (deterministic form)."""
    assert main(["plot", "x^2 - 5x + 6 = 0", "--solve"]) == 0
    out = capsys.readouterr().out
    assert "x = 2" in out and "x = 3" in out
    assert "✓ substitution" in out
    assert "(rendering of verified data — not a result)" in out
    assert "rerun: assay run" in out
    assert Path("x_2_5x_6.svg").exists()
    assert Path("x_2_5x_6.result.json").exists()


def test_cli_plot_without_solve(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["plot", "x^2", "--fig", "curve.svg"]) == 0
    out = capsys.readouterr().out
    assert "rendered: curve.svg" in out
    assert "no solve requested" in out
    assert "✓ figure:data" in out
    assert Path("curve.svg").exists()
    assert not Path("x_2.result.json").exists()  # no computation → no artifact
