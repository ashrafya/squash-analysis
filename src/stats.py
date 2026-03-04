import numpy as np

from config import T_X, T_Y, T_RADIUS_M


def compute_movement_stats(court_xs, court_ys, fps, frame_skip):
    """
    Compute per-player movement stats from court-space positions (metres).

    Parameters
    ----------
    court_xs, court_ys : list of float
        Player positions in court space (metres), one per processed frame.
    fps : float
        Video frame rate.
    frame_skip : int
        Every Nth frame was processed, so time between positions = frame_skip / fps.

    Returns
    -------
    dict with keys:
        duration_s       — total time on court (seconds)
        total_distance_m — total distance covered (metres)
        avg_speed_ms     — average speed (m/s)
        peak_speed_ms    — peak speed over any single inter-frame step (m/s)
        t_time_pct       — % of frames spent within T_RADIUS_M of the T
    """
    if len(court_xs) < 2:
        return None

    xs = np.array(court_xs)
    ys = np.array(court_ys)

    time_per_step = frame_skip / fps
    duration_s = len(xs) * time_per_step

    dx = np.diff(xs)
    dy = np.diff(ys)
    step_distances = np.hypot(dx, dy)
    total_distance_m = float(np.sum(step_distances))

    step_speeds = step_distances / time_per_step
    avg_speed_ms = total_distance_m / duration_s if duration_s > 0 else 0.0
    peak_speed_ms = float(np.max(step_speeds)) if len(step_speeds) > 0 else 0.0

    dist_to_t = np.hypot(xs - T_X, ys - T_Y)
    t_time_pct = float(np.mean(dist_to_t <= T_RADIUS_M) * 100)

    return {
        "duration_s":       round(duration_s, 1),
        "total_distance_m": round(total_distance_m, 1),
        "avg_speed_ms":     round(avg_speed_ms, 2),
        "peak_speed_ms":    round(peak_speed_ms, 2),
        "t_time_pct":       round(t_time_pct, 1),
    }


def print_stats_table(stats_p1, stats_p2):
    """Print a side-by-side stats table for both players."""
    labels = {
        "duration_s":       ("Duration",        "s"),
        "total_distance_m": ("Distance covered", "m"),
        "avg_speed_ms":     ("Avg speed",        "m/s"),
        "peak_speed_ms":    ("Peak speed",       "m/s"),
        "t_time_pct":       ("T-position time",  "%"),
    }

    col_w = 18
    print()
    print("=" * (col_w * 3 + 4))
    print(f"  {'Metric':<{col_w}} {'Player 1':>{col_w}} {'Player 2':>{col_w}}")
    print("=" * (col_w * 3 + 4))

    for key, (label, unit) in labels.items():
        v1 = f"{stats_p1[key]} {unit}" if stats_p1 else "—"
        v2 = f"{stats_p2[key]} {unit}" if stats_p2 else "—"
        print(f"  {label:<{col_w}} {v1:>{col_w}} {v2:>{col_w}}")

    print("=" * (col_w * 3 + 4))
    print()
