"""
Tests for compute_timeseries (stats.py).
All tests use fps=25, frame_skip=1 → time_per_step = 0.04 s.
"""

import numpy as np
import pytest
from analysis.stats import compute_timeseries
from config import T_X, T_Y

FPS  = 25
SKIP = 1


def _ts(xs, ys):
    return compute_timeseries(xs, ys, FPS, SKIP)


# ── 1 ─────────────────────────────────────────────────────────────────────────
def test_returns_none_for_single_position():
    assert _ts([1.0], [1.0]) is None


# ── 2 ─────────────────────────────────────────────────────────────────────────
def test_all_arrays_same_length():
    xs = [0.0, 1.0, 2.0, 3.0]
    ys = [0.0, 0.0, 0.0, 0.0]
    ts = _ts(xs, ys)
    n = len(xs)
    assert len(ts["time_s"])      == n
    assert len(ts["speed_ms"])    == n
    assert len(ts["dist_to_t_m"]) == n
    assert len(ts["y_m"])         == n


# ── 3 ─────────────────────────────────────────────────────────────────────────
def test_time_axis_starts_at_zero_and_increments():
    ts = _ts([0.0, 1.0, 2.0], [0.0, 0.0, 0.0])
    dt = SKIP / FPS   # 0.04 s
    assert ts["time_s"][0] == pytest.approx(0.0)
    assert ts["time_s"][1] == pytest.approx(dt)
    assert ts["time_s"][2] == pytest.approx(2 * dt)


# ── 4 ─────────────────────────────────────────────────────────────────────────
def test_first_speed_is_zero():
    """First frame has no prior step, so speed is padded to 0."""
    ts = _ts([0.0, 3.0], [0.0, 4.0])
    assert ts["speed_ms"][0] == pytest.approx(0.0)


# ── 5 ─────────────────────────────────────────────────────────────────────────
def test_speed_correct_known_step():
    """3-4-5 triangle step (5 m) at time_per_step=0.04 s → 125 m/s."""
    ts = _ts([0.0, 3.0], [0.0, 4.0])
    expected = 5.0 / (SKIP / FPS)
    assert ts["speed_ms"][1] == pytest.approx(expected, rel=1e-3)


# ── 6 ─────────────────────────────────────────────────────────────────────────
def test_stationary_player_speed_all_zero():
    xs = [3.2] * 20
    ys = [5.5] * 20
    ts = _ts(xs, ys)
    assert np.all(ts["speed_ms"] == pytest.approx(0.0))


# ── 7 ─────────────────────────────────────────────────────────────────────────
def test_dist_to_t_at_t_junction():
    """Player exactly at the T → distance = 0."""
    ts = _ts([T_X, T_X], [T_Y, T_Y])
    assert np.all(ts["dist_to_t_m"] == pytest.approx(0.0))


# ── 8 ─────────────────────────────────────────────────────────────────────────
def test_dist_to_t_known_value():
    """Player 3 m to the right of T (x) → dist = 3."""
    ts = _ts([T_X + 3.0], [T_Y, T_Y])
    # Only 1 position → None; use 2 positions at same x offset
    ts = _ts([T_X + 3.0, T_X + 3.0], [T_Y, T_Y])
    assert ts["dist_to_t_m"][0] == pytest.approx(3.0, abs=1e-6)


# ── 9 ─────────────────────────────────────────────────────────────────────────
def test_y_m_matches_input():
    """y_m should be identical to the input court_ys."""
    ys = [0.5, 2.3, 7.1, 9.0]
    ts = _ts([0.0] * 4, ys)
    np.testing.assert_array_almost_equal(ts["y_m"], ys)


# ── 10 ────────────────────────────────────────────────────────────────────────
def test_frame_skip_scales_time_axis():
    """frame_skip=5 should produce a time axis 5× larger than frame_skip=1."""
    xs = [float(i) for i in range(10)]
    ys = [0.0] * 10
    ts1 = compute_timeseries(xs, ys, FPS, 1)
    ts5 = compute_timeseries(xs, ys, FPS, 5)
    np.testing.assert_array_almost_equal(ts5["time_s"], ts1["time_s"] * 5)
