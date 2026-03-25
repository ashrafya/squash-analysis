"""
Tests for compute_zone_stats (stats.py) — court zone breakdown.

Zone grid (3×3):
  Columns (x): Left [0, 2.133), Centre [2.133, 4.267), Right [4.267, 6.4]
  Rows    (y): Front [0, 3.25),  Mid    [3.25, 6.5),   Back  [6.5, 9.75]
  Centre-mid zone is labelled 'T'.
"""

import pytest
from analysis.stats import compute_zone_stats
from config import (
    COURT_WIDTH_M, COURT_LENGTH_M,
    ZONE_COL_EDGES, ZONE_ROW_EDGES, ZONE_NAMES, T_X, T_Y,
)

ALL_ZONES = [name for row in ZONE_NAMES for name in row]


# ── 1 ─────────────────────────────────────────────────────────────────────────
def test_all_zones_present():
    """Result dict always contains all 9 zone keys."""
    result = compute_zone_stats([T_X], [T_Y])
    assert set(result.keys()) == set(ALL_ZONES)


# ── 2 ─────────────────────────────────────────────────────────────────────────
def test_percentages_sum_to_100():
    """Zone percentages sum to 100 for any mix of positions."""
    import random
    random.seed(7)
    xs = [random.uniform(0, COURT_WIDTH_M)  for _ in range(200)]
    ys = [random.uniform(0, COURT_LENGTH_M) for _ in range(200)]
    result = compute_zone_stats(xs, ys)
    assert sum(result.values()) == pytest.approx(100.0, abs=0.2)


# ── 3 ─────────────────────────────────────────────────────────────────────────
def test_empty_input_returns_zeros():
    """Empty position lists → all zones at 0 %."""
    result = compute_zone_stats([], [])
    assert all(v == 0.0 for v in result.values())


# ── 4 ─────────────────────────────────────────────────────────────────────────
def test_t_position_lands_in_t_zone():
    """The T junction coordinates fall in the 'T' zone."""
    result = compute_zone_stats([T_X], [T_Y])
    assert result["T"] == pytest.approx(100.0)


# ── 5 ─────────────────────────────────────────────────────────────────────────
def test_front_left_corner_lands_in_front_l():
    """Position near the front-left corner → Front-L zone."""
    result = compute_zone_stats([0.1], [0.1])
    assert result["Front-L"] == pytest.approx(100.0)


# ── 6 ─────────────────────────────────────────────────────────────────────────
def test_back_right_corner_lands_in_back_r():
    """Position near the back-right corner → Back-R zone."""
    result = compute_zone_stats([COURT_WIDTH_M - 0.1], [COURT_LENGTH_M - 0.1])
    assert result["Back-R"] == pytest.approx(100.0)


# ── 7 ─────────────────────────────────────────────────────────────────────────
def test_equal_split_across_all_zones():
    """One position in each zone → all zones at ~11.1 %."""
    xs, ys = [], []
    for ri in range(3):
        y_mid = (ZONE_ROW_EDGES[ri] + ZONE_ROW_EDGES[ri + 1]) / 2
        for ci in range(3):
            x_mid = (ZONE_COL_EDGES[ci] + ZONE_COL_EDGES[ci + 1]) / 2
            xs.append(x_mid)
            ys.append(y_mid)
    result = compute_zone_stats(xs, ys)
    for pct in result.values():
        assert pct == pytest.approx(100 / 9, abs=0.2)


# ── 8 ─────────────────────────────────────────────────────────────────────────
def test_all_positions_in_front_court():
    """All positions in front row → Front-L + Front-C + Front-R sum to 100 %."""
    front_y = ZONE_ROW_EDGES[1] / 2   # midpoint of first row
    xs = [1.0, 3.2, 5.5]
    ys = [front_y] * 3
    result = compute_zone_stats(xs, ys)
    front_total = result["Front-L"] + result["Front-C"] + result["Front-R"]
    assert front_total == pytest.approx(100.0, abs=0.5)


# ── 9 ─────────────────────────────────────────────────────────────────────────
def test_single_position_100_percent_in_one_zone():
    """A single position gives 100 % to exactly one zone, 0 % to all others."""
    result = compute_zone_stats([1.0], [1.0])   # Front-L
    assert result["Front-L"] == pytest.approx(100.0)
    assert sum(v for k, v in result.items() if k != "Front-L") == pytest.approx(0.0)
