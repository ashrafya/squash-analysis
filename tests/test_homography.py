"""
Tests for calibrate.apply_homography — coordinate transform math.

No video or calibration file required; we construct H matrices directly.
"""

import numpy as np
import pytest
from calibration.calibrate import apply_homography


# ── 16 ────────────────────────────────────────────────────────────────────────
def test_apply_identity_homography():
    """Identity matrix → input coordinates pass through unchanged."""
    H  = np.eye(3, dtype=np.float32)
    xs = [1.0, 2.5, 6.4]
    ys = [0.0, 5.49, 9.75]
    cx, cy = apply_homography(xs, ys, H)
    for got, exp in zip(cx, xs):
        assert abs(got - exp) < 1e-4
    for got, exp in zip(cy, ys):
        assert abs(got - exp) < 1e-4


# ── 17 ────────────────────────────────────────────────────────────────────────
def test_apply_scale_homography():
    """2× scale matrix → all coordinates doubled."""
    H = np.diag([2.0, 2.0, 1.0]).astype(np.float32)
    cx, cy = apply_homography([1.0, 3.0], [2.0, 4.0], H)
    assert cx == pytest.approx([2.0, 6.0], abs=1e-4)
    assert cy == pytest.approx([4.0, 8.0], abs=1e-4)


# ── 18 ────────────────────────────────────────────────────────────────────────
def test_apply_accepts_numpy_arrays():
    """Function handles both Python lists and numpy arrays as input."""
    H = np.eye(3, dtype=np.float32)
    xs = np.array([1.0, 2.0])
    ys = np.array([3.0, 4.0])
    cx_arr, cy_arr = apply_homography(xs, ys, H)
    cx_lst, cy_lst = apply_homography(xs.tolist(), ys.tolist(), H)
    np.testing.assert_allclose(cx_arr, cx_lst, atol=1e-4)
    np.testing.assert_allclose(cy_arr, cy_lst, atol=1e-4)


# ── 33 ────────────────────────────────────────────────────────────────────────
def test_apply_homography_translation():
    """Pure translation matrix shifts coordinates by the expected offset."""
    # Translate by (tx=3, ty=4)
    H = np.array([[1, 0, 3],
                  [0, 1, 4],
                  [0, 0, 1]], dtype=np.float32)
    cx, cy = apply_homography([1.0, 2.0], [0.0, 1.0], H)
    assert cx == pytest.approx([4.0, 5.0], abs=1e-4)
    assert cy == pytest.approx([4.0, 5.0], abs=1e-4)


# ── 34 ────────────────────────────────────────────────────────────────────────
def test_apply_homography_single_point():
    """Single-element input is handled without errors."""
    H = np.eye(3, dtype=np.float32)
    cx, cy = apply_homography([3.2], [5.49], H)
    assert len(cx) == 1
    assert cx[0] == pytest.approx(3.2,  abs=1e-4)
    assert cy[0] == pytest.approx(5.49, abs=1e-4)


# ── 35 ────────────────────────────────────────────────────────────────────────
def test_apply_homography_returns_lists():
    """Return values are Python lists, not numpy arrays."""
    H = np.eye(3, dtype=np.float32)
    cx, cy = apply_homography([1.0, 2.0], [3.0, 4.0], H)
    assert isinstance(cx, list)
    assert isinstance(cy, list)
