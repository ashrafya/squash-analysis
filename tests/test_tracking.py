"""
Tests for extract_pose.py — landmark position extraction, smoothing, and
detection guards.

Importing extract_pose initialises two MediaPipe Pose models at module level.
This is a one-time cost per pytest session (~2 s); all tests share the models.
"""

import numpy as np
import pytest
from tracking import extract_pose
from unittest.mock import MagicMock, patch

import mediapipe as mp

# mediapipe >= 0.10.30 removed the solutions API; guard gracefully.
_mp_solutions_available = hasattr(mp, 'solutions')
if _mp_solutions_available:
    mp_pose = mp.solutions.pose
    _LHEEL = mp_pose.PoseLandmark.LEFT_HEEL
    _RHEEL = mp_pose.PoseLandmark.RIGHT_HEEL
    _LANK  = mp_pose.PoseLandmark.LEFT_ANKLE
    _RANK  = mp_pose.PoseLandmark.RIGHT_ANKLE
    _LHIP  = mp_pose.PoseLandmark.LEFT_HIP
    _RHIP  = mp_pose.PoseLandmark.RIGHT_HIP
else:
    mp_pose = None
    # BlazePose landmark indices (hard-coded fallback so collection succeeds)
    _LHEEL, _RHEEL = 29, 30
    _LANK,  _RANK  = 27, 28
    _LHIP,  _RHIP  = 23, 24

pytestmark = pytest.mark.skipif(
    not _mp_solutions_available,
    reason="mediapipe.solutions removed in >= 0.10.30; legacy MediaPipe tracker not supported on this install",
)

from tracking.extract_pose import get_ground_position, smooth_positions, _detect_in_crop, detect_in_region
from config import FOOT_VISIBILITY_MIN, CROP_MARGIN

_FALLBACK_VIS = 0.35   # must match extract_pose._FALLBACK_VIS


# ── Helpers ───────────────────────────────────────────────────────────────────

def _lm(x=0.0, y=0.0, vis=0.0):
    lm = MagicMock()
    lm.x, lm.y, lm.visibility = x, y, vis
    return lm


def _make_landmarks(overrides=None):
    """33 fully-invisible landmarks; overrides = {PoseLandmark: (x, y, vis)}."""
    lms = [_lm() for _ in range(33)]
    for lm_id, (x, y, vis) in (overrides or {}).items():
        lms[lm_id] = _lm(x, y, vis)
    return lms


# ── ground_position tests (5) ─────────────────────────────────────────────────

# ── 7 ─────────────────────────────────────────────────────────────────────────
def test_ground_tier0_both_heels():
    """Tier 0: both heels ≥ FOOT_VISIBILITY_MIN → returns heel midpoint."""
    lms = _make_landmarks({
        _LHEEL: (0.2, 0.8, FOOT_VISIBILITY_MIN),
        _RHEEL: (0.4, 0.8, FOOT_VISIBILITY_MIN),
    })
    pos = get_ground_position(lms, 100, 100)
    assert pos == pytest.approx((30.0, 80.0))  # midpoint of (20,80) and (40,80)


# ── 8 ─────────────────────────────────────────────────────────────────────────
def test_ground_tier1_ankle_fallback():
    """Tier 1: heels invisible but one ankle visible → returns ankle position."""
    lms = _make_landmarks({
        _LANK: (0.5, 0.7, _FALLBACK_VIS + 0.01),
    })
    pos = get_ground_position(lms, 100, 100)
    assert pos == pytest.approx((50.0, 70.0))


# ── 9 ─────────────────────────────────────────────────────────────────────────
def test_ground_tier2_hip_fallback():
    """Tier 2: all feet invisible → returns hip midpoint."""
    lms = _make_landmarks({
        _LHIP: (0.4, 0.5, _FALLBACK_VIS + 0.01),
        _RHIP: (0.6, 0.5, _FALLBACK_VIS + 0.01),
    })
    pos = get_ground_position(lms, 100, 100)
    assert pos == pytest.approx((50.0, 50.0))


# ── 10 ────────────────────────────────────────────────────────────────────────
def test_ground_all_invisible_returns_none():
    """All landmarks below threshold → returns None (no usable signal)."""
    lms = _make_landmarks()   # all visibility = 0.0
    assert get_ground_position(lms, 100, 100) is None


# ── 11 ────────────────────────────────────────────────────────────────────────
def test_ground_tier0_beats_tier1():
    """Tier 0 (heels) takes priority over higher-visibility ankles."""
    lms = _make_landmarks({
        _LHEEL: (0.3, 0.9, FOOT_VISIBILITY_MIN),      # heel y=90
        _RHEEL: (0.5, 0.9, FOOT_VISIBILITY_MIN),
        _LANK:  (0.3, 0.6, 0.99),                     # ankle y=60, more visible
        _RANK:  (0.5, 0.6, 0.99),
    })
    pos = get_ground_position(lms, 100, 100)
    # Heel midpoint y = 90, ankle midpoint y = 60; tier 0 must win
    assert pos[1] == pytest.approx(90.0)


# ── smooth_positions tests (2) ────────────────────────────────────────────────

# ── 12 ────────────────────────────────────────────────────────────────────────
def test_smooth_preserves_length():
    """Output arrays have the same length as the input."""
    xs = list(range(30))
    ys = list(range(30))
    xs_s, ys_s = smooth_positions(xs, ys, window=5)
    assert len(xs_s) == 30
    assert len(ys_s) == 30


# ── 13 ────────────────────────────────────────────────────────────────────────
def test_smooth_reduces_spike():
    """Median filter should significantly reduce a single large spike."""
    ys = [5.0] * 21
    ys[10] = 200.0       # isolated spike
    _, ys_s = smooth_positions([0.0] * 21, ys, window=5)
    assert ys_s[10] < 50.0


# ── _detect_in_crop guards (2) ────────────────────────────────────────────────

# ── 14 ────────────────────────────────────────────────────────────────────────
def test_detect_empty_crop_returns_none():
    """margin=0 produces an empty crop; function returns None without calling pose."""
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    pose  = MagicMock()
    result = _detect_in_crop(frame, (50, 50), 0, pose)
    assert result is None
    pose.process.assert_not_called()


# ── 15 ────────────────────────────────────────────────────────────────────────
def test_detect_no_pose_landmarks_returns_none():
    """When pose model finds no person, function returns None."""
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    pose  = MagicMock()
    pose.process.return_value.pose_landmarks = None
    result = _detect_in_crop(frame, (50, 50), 20, pose)
    assert result is None


# ── detect_in_region fallback tests (2) ──────────────────────────────────────

# ── 28 ────────────────────────────────────────────────────────────────────────
def test_detect_in_region_uses_wider_fallback(monkeypatch):
    """When the first crop returns None, detect_in_region tries CROP_MARGIN×2."""
    frame = np.zeros((400, 400, 3), dtype=np.uint8)
    pose  = MagicMock()
    margins_tried = []

    def mock_crop(frm, pos, margin, mdl):
        margins_tried.append(margin)
        return (200.0, 200.0) if margin == CROP_MARGIN * 2 else None

    monkeypatch.setattr(extract_pose, "_detect_in_crop", mock_crop)
    result = detect_in_region(frame, (200, 200), pose)

    assert result == (200.0, 200.0)
    assert CROP_MARGIN     in margins_tried
    assert CROP_MARGIN * 2 in margins_tried


# ── 29 ────────────────────────────────────────────────────────────────────────
def test_detect_in_region_first_crop_succeeds(monkeypatch):
    """When the first crop succeeds, the wider crop is never attempted."""
    frame = np.zeros((400, 400, 3), dtype=np.uint8)
    pose  = MagicMock()
    call_count = [0]

    def mock_crop(frm, pos, margin, mdl):
        call_count[0] += 1
        return (50.0, 50.0)   # always succeeds

    monkeypatch.setattr(extract_pose, "_detect_in_crop", mock_crop)
    result = detect_in_region(frame, (50, 50), pose)

    assert result == (50.0, 50.0)
    assert call_count[0] == 1   # second crop not tried


# ── extra ground_position / smooth tests ──────────────────────────────────────

# ── 30 ────────────────────────────────────────────────────────────────────────
def test_ground_tier1_averages_two_ankles():
    """Tier 1: two visible ankles → position is their midpoint."""
    lms = _make_landmarks({
        _LANK: (0.2, 0.6, _FALLBACK_VIS + 0.01),
        _RANK: (0.6, 0.6, _FALLBACK_VIS + 0.01),
    })
    pos = get_ground_position(lms, 100, 100)
    assert pos == pytest.approx((40.0, 60.0))   # (20+60)/2=40, (60+60)/2=60


# ── 31 ────────────────────────────────────────────────────────────────────────
def test_ground_only_one_heel_falls_to_tier1():
    """Only one heel visible (not both) → tier 0 fails, tier 1 averages what it can."""
    lms = _make_landmarks({
        _LHEEL: (0.3, 0.8, FOOT_VISIBILITY_MIN),   # left heel above FOOT_VISIBILITY_MIN
        # right heel invisible → tier 0 condition (both heels) fails
        _LANK:  (0.5, 0.7, _FALLBACK_VIS + 0.01),
    })
    pos = get_ground_position(lms, 100, 100)
    # Tier 1: lheel (vis 0.6 ≥ 0.35) + lank (vis 0.36 ≥ 0.35) averaged
    # x = (0.3+0.5)/2*100 = 40, y = (0.8+0.7)/2*100 = 75
    assert pos == pytest.approx((40.0, 75.0))


# ── 32 ────────────────────────────────────────────────────────────────────────
def test_smooth_window_1_passthrough():
    """median_filter with size=1 is the identity — output equals input."""
    xs = [1.5, 3.2, 0.8, 4.1, 2.0]
    ys = [5.0, 2.3, 7.1, 1.0, 6.5]
    xs_s, ys_s = smooth_positions(xs, ys, window=1)
    assert xs_s == pytest.approx(xs, abs=1e-9)
    assert ys_s == pytest.approx(ys, abs=1e-9)
