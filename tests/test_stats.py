"""
Tests for stats.py — movement stat calculations.

All tests use fps=25, frame_skip=1 so time_per_step = 0.04 s.
"""

import json
import pytest
from analysis.stats import compute_movement_stats, save_run_history
from config import T_X, T_Y, T_RADIUS_M

FPS  = 25
SKIP = 1


def _s(xs, ys):
    return compute_movement_stats(xs, ys, FPS, SKIP)


# ── 1 ─────────────────────────────────────────────────────────────────────────
def test_distance_between_two_points():
    """Euclidean distance between two court positions is computed correctly."""
    stats = _s([0.0, 3.0], [0.0, 4.0])   # 3-4-5 triangle → 5.0 m
    assert stats["total_distance_m"] == pytest.approx(5.0, abs=0.05)


# ── 2 ─────────────────────────────────────────────────────────────────────────
def test_single_position_returns_none():
    """Fewer than 2 positions → None (can't compute steps)."""
    assert _s([3.2], [5.5]) is None


# ── 3 ─────────────────────────────────────────────────────────────────────────
def test_avg_speed_known_value():
    """avg_speed = total_distance / ((n_positions - 1) * time_per_step).

    2 positions, 2.5 m apart: duration = 1 * (1/25) = 0.04 s → 62.5 m/s.
    """
    stats = _s([0.0, 2.5], [0.0, 0.0])
    expected = 2.5 / (1 / FPS)   # 62.5 m/s
    assert stats["avg_speed_ms"] == pytest.approx(expected, rel=1e-3)


# ── 4 ─────────────────────────────────────────────────────────────────────────
def test_peak_speed_p95_ignores_outlier():
    """A single extreme spike should not dominate the p95 peak speed.

    99 normal steps + 1 absurd spike: p95 falls within the normal steps.
    """
    normal_step = 0.5   # m per step
    xs = [i * normal_step for i in range(100)]
    ys = [0.0] * 100
    xs[-1] += 50_000.0  # giant spike on the last step

    stats = _s(xs, ys)
    normal_speed = normal_step * FPS   # 12.5 m/s

    # p95 of 99 identical small values + 1 outlier → still near normal_speed
    assert stats["peak_speed_ms"] < normal_speed * 2


# ── 5 ─────────────────────────────────────────────────────────────────────────
def test_t_time_all_at_t():
    """All positions exactly at the T → 100 % T-time."""
    xs = [T_X] * 20
    ys = [T_Y] * 20
    stats = _s(xs, ys)
    assert stats["t_time_pct"] == pytest.approx(100.0)


# ── 6 ─────────────────────────────────────────────────────────────────────────
def test_front_back_split_half_each():
    """5 positions in front court, 5 in back → 50 % each."""
    front_y = T_Y / 2       # clearly in front half
    back_y  = T_Y + 1.0     # clearly in back half
    xs = [3.2] * 10
    ys = [front_y] * 5 + [back_y] * 5
    stats = _s(xs, ys)
    assert stats["front_pct"] == pytest.approx(50.0)
    assert stats["back_pct"]  == pytest.approx(50.0)


# ── save_run_history tests ─────────────────────────────────────────────────────

_DUMMY_STATS = {
    "duration_s": 10.0, "total_distance_m": 50.0, "avg_speed_ms": 5.0,
    "peak_speed_ms": 8.0, "t_time_pct": 30.0, "front_pct": 45.0, "back_pct": 55.0,
}


# ── 21 ────────────────────────────────────────────────────────────────────────
def test_save_run_history_creates_file(tmp_path, monkeypatch):
    """save_run_history creates run_history.json in OUTPUT_DIR."""
    monkeypatch.setattr("analysis.stats.OUTPUT_DIR", str(tmp_path))
    save_run_history(_DUMMY_STATS, _DUMMY_STATS, 100, 100, "test.mp4", 1000, 5)
    assert (tmp_path / "run_history.json").exists()


# ── 22 ────────────────────────────────────────────────────────────────────────
def test_save_run_history_has_expected_keys(tmp_path, monkeypatch):
    """Each run entry contains the required schema keys."""
    monkeypatch.setattr("analysis.stats.OUTPUT_DIR", str(tmp_path))
    save_run_history(_DUMMY_STATS, _DUMMY_STATS, 100, 100, "test.mp4", 1000, 5)
    data = json.loads((tmp_path / "run_history.json").read_text())
    entry = data[0]
    for key in ("timestamp", "video", "frame_cap", "frame_skip", "n_positions", "player1", "player2"):
        assert key in entry


# ── 23 ────────────────────────────────────────────────────────────────────────
def test_save_run_history_appends_multiple(tmp_path, monkeypatch):
    """Calling twice appends a second entry without overwriting the first."""
    monkeypatch.setattr("analysis.stats.OUTPUT_DIR", str(tmp_path))
    save_run_history(_DUMMY_STATS, _DUMMY_STATS, 100, 100, "a.mp4", 1000, 5)
    save_run_history(_DUMMY_STATS, _DUMMY_STATS, 200, 200, "b.mp4", 1000, 5)
    data = json.loads((tmp_path / "run_history.json").read_text())
    assert len(data) == 2
    assert data[0]["video"] == "a.mp4"
    assert data[1]["video"] == "b.mp4"


# ── extra stats edge cases ─────────────────────────────────────────────────────

# ── 24 ────────────────────────────────────────────────────────────────────────
def test_zero_movement_all_same_position():
    """All positions identical → distance 0, speeds 0."""
    stats = _s([3.2] * 20, [5.5] * 20)
    assert stats["total_distance_m"] == 0.0
    assert stats["avg_speed_ms"]     == 0.0
    assert stats["peak_speed_ms"]    == 0.0


# ── 25 ────────────────────────────────────────────────────────────────────────
def test_t_time_zero_when_far_from_t():
    """Positions far from T → 0 % T-time."""
    stats = _s([0.0] * 10, [0.0] * 10)   # front-left corner, well outside T_RADIUS_M
    assert stats["t_time_pct"] == pytest.approx(0.0)


# ── 26 ────────────────────────────────────────────────────────────────────────
def test_frame_skip_scales_timing():
    """frame_skip=5 produces 5× the duration of frame_skip=1 for the same positions.

    Uses 25 positions so both durations (1.0 s and 5.0 s) round cleanly to 1 dp
    and the 5× ratio is preserved after rounding.
    """
    xs = list(range(26))
    ys = [0.0] * 26
    s1 = compute_movement_stats(xs, ys, FPS, 1)   # 25 * (1/25) = 1.0 s
    s5 = compute_movement_stats(xs, ys, FPS, 5)   # 25 * (5/25) = 5.0 s
    assert s5["duration_s"] == pytest.approx(s1["duration_s"] * 5, rel=1e-3)


# ── 27 ────────────────────────────────────────────────────────────────────────
def test_front_back_always_sum_to_100():
    """front_pct + back_pct == 100 for any mixture of positions."""
    import random
    random.seed(42)
    xs = [random.uniform(0, 6.4)  for _ in range(50)]
    ys = [random.uniform(0, 9.75) for _ in range(50)]
    stats = _s(xs, ys)
    assert stats["front_pct"] + stats["back_pct"] == pytest.approx(100.0)
