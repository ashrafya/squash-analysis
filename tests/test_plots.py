"""
Tests for plot_utils.py — court diagram and histogram rendering.

The Agg backend is activated in conftest.py so no display is needed.
"""

import tempfile
import matplotlib.pyplot as plt
import pytest
from plot_utils import draw_court, plot_histograms, _build_heatmap, plot_heatmap_comparison
from config import COURT_WIDTH_M, COURT_LENGTH_M, HEATMAP_GRID_X, HEATMAP_GRID_Y


# ── 19 ────────────────────────────────────────────────────────────────────────
def test_draw_court_axis_limits():
    """draw_court sets x/y limits to contain the full WSF court plus margin."""
    fig, ax = plt.subplots()
    draw_court(ax)

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()   # note: y-axis is inverted, so ymin > ymax

    assert xmin < 0                  # left margin
    assert xmax > COURT_WIDTH_M      # right margin
    # invert_yaxis() means the stored limits are (high, low)
    assert max(ymin, ymax) > COURT_LENGTH_M
    assert min(ymin, ymax) < 0

    plt.close(fig)


# ── 20 ────────────────────────────────────────────────────────────────────────
def test_plot_histograms_no_crash():
    """plot_histograms runs without raising on valid court-space data."""
    xs1 = [1.0, 2.0, 3.0, 4.0, 5.0]
    ys1 = [1.0, 3.0, 5.0, 7.0, 9.0]
    xs2 = [0.5, 1.5, 2.5, 3.5, 4.5]
    ys2 = [2.0, 4.0, 6.0, 8.0, 9.5]
    plot_histograms(xs1, ys1, xs2, ys2)   # must not raise
    plt.close("all")


# ── _build_heatmap tests ───────────────────────────────────────────────────────

# ── 36 ────────────────────────────────────────────────────────────────────────
def test_build_heatmap_shape_and_normalized():
    """Output shape matches grid config; values are in [0, 1]."""
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    ys = [1.0, 3.0, 5.0, 7.0, 9.0]
    h = _build_heatmap(xs, ys)
    assert h.shape == (HEATMAP_GRID_Y, HEATMAP_GRID_X)
    assert h.max() <= 1.0 + 1e-9
    assert h.min() >= 0.0


# ── 37 ────────────────────────────────────────────────────────────────────────
def test_build_heatmap_empty_input_is_zeros():
    """Empty position lists → heatmap is all zeros (no division-by-zero crash)."""
    h = _build_heatmap([], [])
    assert h.shape == (HEATMAP_GRID_Y, HEATMAP_GRID_X)
    assert h.max() == 0.0


# ── 38 ────────────────────────────────────────────────────────────────────────
def test_plot_heatmap_comparison_no_crash():
    """plot_heatmap_comparison renders both methods for both players without error."""
    xs1 = [1.0, 2.0, 3.0]
    ys1 = [2.0, 4.0, 6.0]
    xs2 = [2.0, 3.0, 4.0]
    ys2 = [3.0, 5.0, 8.0]
    with tempfile.TemporaryDirectory() as tmpdir:
        plot_heatmap_comparison(xs1, ys1, xs2, ys2, xs1, ys1, xs2, ys2,
                                output_dir=tmpdir)
    plt.close("all")
