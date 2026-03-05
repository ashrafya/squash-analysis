"""
Tests for validate_ground_fix.py — validation-only helper functions.

Only pure functions are tested here; the full run() pipeline requires a
real video and is not suitable for unit testing.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock

from validate_ground_fix import _midpoint, count_jump_artifacts


# ── Helpers ───────────────────────────────────────────────────────────────────

def _lm(x=0.0, y=0.0, vis=0.0):
    lm = MagicMock()
    lm.x, lm.y, lm.visibility = x, y, vis
    return lm


# ── count_jump_artifacts tests (2) ───────────────────────────────────────────

# ── 39 ────────────────────────────────────────────────────────────────────────
def test_count_jump_artifacts_detects_spike():
    """Hip Y spikes while ground Y stays flat → artifacts are counted.

    hip_y differences:  [0.0, 0.5, -0.5]
    grnd_y differences: [0.0, 0.0,  0.0]
    Frames where |dh| > 0.3 AND |dg| < 0.1 → indices 1 and 2 → count = 2.
    """
    hip_y  = np.array([5.0, 5.0, 5.5, 5.0])
    grnd_y = np.array([5.0, 5.0, 5.0, 5.0])
    assert count_jump_artifacts(hip_y, grnd_y) == 2


# ── 40 ────────────────────────────────────────────────────────────────────────
def test_count_jump_artifacts_correlated_movement():
    """Hip and ground move together (genuine court movement) → 0 artifacts."""
    hip_y  = np.array([5.0, 6.0, 7.0, 8.0])
    grnd_y = np.array([5.0, 6.0, 7.0, 8.0])
    assert count_jump_artifacts(hip_y, grnd_y) == 0


# ── 41 ────────────────────────────────────────────────────────────────────────
def test_count_jump_artifacts_no_movement():
    """Both signals stationary → 0 artifacts."""
    hip_y  = np.array([5.0, 5.0, 5.0, 5.0])
    grnd_y = np.array([5.0, 5.0, 5.0, 5.0])
    assert count_jump_artifacts(hip_y, grnd_y) == 0


# ── _midpoint tests (3) ───────────────────────────────────────────────────────

# ── 42 ────────────────────────────────────────────────────────────────────────
def test_midpoint_all_visible():
    """Average of all landmarks when all are above threshold."""
    lms = [_lm(0.2, 0.5, 1.0), _lm(0.4, 0.5, 1.0)]
    result = _midpoint(lms, 100, 100, threshold=0.5)
    # x: (0.2+0.4)/2 * 100 = 30.0   y: 0.5 * 100 = 50.0
    assert result == pytest.approx((30.0, 50.0))


# ── 43 ────────────────────────────────────────────────────────────────────────
def test_midpoint_filters_below_threshold():
    """Landmark below threshold is excluded from the average."""
    lms = [_lm(0.2, 0.5, 0.8), _lm(0.8, 0.5, 0.1)]   # second below threshold=0.5
    result = _midpoint(lms, 100, 100, threshold=0.5)
    assert result == pytest.approx((20.0, 50.0))   # only first landmark used


# ── 44 ────────────────────────────────────────────────────────────────────────
def test_midpoint_no_visible_returns_none():
    """All landmarks below threshold → returns None."""
    lms = [_lm(0.2, 0.5, 0.1), _lm(0.8, 0.5, 0.2)]
    assert _midpoint(lms, 100, 100, threshold=0.5) is None
