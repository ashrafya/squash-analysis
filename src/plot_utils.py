import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter

from config import (
    COURT_WIDTH_M,
    COURT_LENGTH_M,
    SHORT_LINE_M,
    HALF_COURT_M,
    SERVICE_BOX_M,
    OUTPUT_DIR,
    HEATMAP_GRID_X,
    HEATMAP_GRID_Y,
    HEATMAP_GAMMA,
    PLAYER_COLORS,
    PLAYER_LABELS,
    ZONE_COL_EDGES,
    ZONE_ROW_EDGES,
    ZONE_NAMES,
    T_Y,
    T_RADIUS_M,
)


def _save(fig, filename):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")


def draw_court(ax):
    """Draw a top-down squash court diagram on a matplotlib Axes.

    Coordinate system (matches homography output):
      x : 0 = left wall,  6.4 = right wall
      y : 0 = front wall, 9.75 = back wall
    Display is y-inverted so front wall appears at the top (standard orientation).
    """
    line_kw = dict(color="black", linewidth=1.5)

    # Court floor (light background)
    ax.add_patch(mpatches.Rectangle(
        (0, 0), COURT_WIDTH_M, COURT_LENGTH_M,
        facecolor="#f5e6c8", edgecolor="black", linewidth=2, zorder=1,
    ))

    # Short line — full width, 4.26 m from back wall = 5.49 m from front wall
    short_y = COURT_LENGTH_M - SHORT_LINE_M  # 5.49
    ax.plot([0, COURT_WIDTH_M], [short_y, short_y], **line_kw, zorder=2)

    # Half-court line — short line to back wall only
    ax.plot([HALF_COURT_M, HALF_COURT_M], [short_y, COURT_LENGTH_M], **line_kw, zorder=2)

    # Service boxes — 1.6 m × 1.6 m, adjacent to side walls, in the back half
    # bounded at front by the short line, extending toward the back wall
    box_back = short_y + SERVICE_BOX_M  # 7.09
    right_box_x = COURT_WIDTH_M - SERVICE_BOX_M  # 4.8

    # Left service box (inner vertical edge + back horizontal edge)
    ax.plot([SERVICE_BOX_M, SERVICE_BOX_M], [short_y, box_back], **line_kw, zorder=2)
    ax.plot([0, SERVICE_BOX_M],             [box_back, box_back], **line_kw, zorder=2)

    # Right service box
    ax.plot([right_box_x, right_box_x],    [short_y, box_back], **line_kw, zorder=2)
    ax.plot([right_box_x, COURT_WIDTH_M],  [box_back, box_back], **line_kw, zorder=2)

    ax.set_xlim(-0.4, COURT_WIDTH_M + 0.4)
    ax.set_ylim(-0.4, COURT_LENGTH_M + 0.4)
    ax.invert_yaxis()  # front wall (y=0) at top, back wall (y=9.75) at bottom
    ax.set_aspect("equal")
    ax.set_xlabel("Court width (m)  —  left wall (0) to right wall (6.4)")
    ax.set_ylabel("Court depth (m)  —  front wall (top) to back wall (bottom)")

    ax.text(COURT_WIDTH_M / 2, -0.25, "Front wall", ha="center", fontsize=8, color="gray")
    ax.text(COURT_WIDTH_M / 2, COURT_LENGTH_M + 0.15, "Back wall", ha="center", fontsize=8, color="gray")


def draw_zone_overlay(ax, zone_stats):
    """Draw 9-zone grid lines and percentage labels on an existing court axes.

    The T zone gets a highlighted label; all others get a plain text box.
    """
    line_kw = dict(color="gray", linewidth=0.8, linestyle="--", alpha=0.6, zorder=5)
    for x in ZONE_COL_EDGES[1:-1]:
        ax.plot([x, x], [0, COURT_LENGTH_M], **line_kw)
    for y in ZONE_ROW_EDGES[1:-1]:
        ax.plot([0, COURT_WIDTH_M], [y, y], **line_kw)

    for ri, row in enumerate(ZONE_NAMES):
        y_mid = (ZONE_ROW_EDGES[ri] + ZONE_ROW_EDGES[ri + 1]) / 2
        for ci, name in enumerate(row):
            x_mid = (ZONE_COL_EDGES[ci] + ZONE_COL_EDGES[ci + 1]) / 2
            pct = zone_stats.get(name, 0.0)
            if name == "T":
                ax.text(x_mid, y_mid, f"T\n{pct:.1f}%",
                        ha="center", va="center", fontsize=8, fontweight="bold",
                        color="white", zorder=7,
                        bbox=dict(boxstyle="round,pad=0.25", facecolor="black", alpha=0.55))
            else:
                ax.text(x_mid, y_mid, f"{name}\n{pct:.1f}%",
                        ha="center", va="center", fontsize=7, color="black", zorder=7,
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.6))


# ── Private draw helpers (render onto a provided axes, no figure management) ──

def _draw_court_positions(ax, xs1, ys1, xs2, ys2, title="Player Movement (Court Space)"):
    draw_court(ax)
    if xs1:
        ax.scatter(xs1, ys1, s=4, alpha=0.5, color=PLAYER_COLORS[0], zorder=5, label=PLAYER_LABELS[0])
    if xs2:
        ax.scatter(xs2, ys2, s=4, alpha=0.5, color=PLAYER_COLORS[1], zorder=5, label=PLAYER_LABELS[1])
    ax.legend(loc="upper right", fontsize=7)
    ax.set_title(title)


# Per-player colormaps: fully transparent at 0 density, opaque at peak
# RGBA tuples: (R, G, B, alpha)
_PLAYER_CMAPS = [
    LinearSegmentedColormap.from_list("p1", [
        (1.0, 1.0, 0.85, 0.0),   # 0  — transparent pale yellow
        (1.0, 0.5, 0.05, 0.65),  # 0.5 — semi-transparent orange
        (0.6, 0.0, 0.0,  1.0),   # 1  — opaque deep red
    ]),
    LinearSegmentedColormap.from_list("p2", [
        (0.88, 0.95, 1.0, 0.0),  # 0  — transparent pale blue
        (0.05, 0.45, 1.0, 0.65), # 0.5 — semi-transparent medium blue
        (0.0,  0.05, 0.65, 1.0), # 1  — opaque deep blue
    ]),
]

_X_CENTERS = (np.linspace(0, COURT_WIDTH_M,  HEATMAP_GRID_X + 1)[:-1] +
              np.linspace(0, COURT_WIDTH_M,  HEATMAP_GRID_X + 1)[1:]) / 2
_Y_CENTERS = (np.linspace(0, COURT_LENGTH_M, HEATMAP_GRID_Y + 1)[:-1] +
              np.linspace(0, COURT_LENGTH_M, HEATMAP_GRID_Y + 1)[1:]) / 2


def _build_heatmap(xs, ys):
    x_bins = np.linspace(0, COURT_WIDTH_M,  HEATMAP_GRID_X + 1)
    y_bins = np.linspace(0, COURT_LENGTH_M, HEATMAP_GRID_Y + 1)
    # histogram2d(y, x) → H[y_row, x_col], matching imshow row=y convention
    h, _, _ = np.histogram2d(ys, xs, bins=[y_bins, x_bins])
    h = gaussian_filter(h, sigma=3.0)
    if h.max() > 0:
        h /= h.max()
        h = h ** HEATMAP_GAMMA  # compress dynamic range so sparse areas are visible
    return h


def _draw_heatmap_on_ax(ax, xs, ys, cmap, zone_stats=None, title=""):
    draw_court(ax)
    h = _build_heatmap(xs, ys)
    ax.imshow(
        h,
        extent=[0, COURT_WIDTH_M, COURT_LENGTH_M, 0],
        origin="upper",
        aspect="auto",
        cmap=cmap,
        vmin=0, vmax=1,
        zorder=1.5,
    )
    if h.max() > 0:
        ax.contour(_X_CENTERS, _Y_CENTERS, h,
                   levels=[0.5, 0.8],
                   colors=["white", "white"],
                   linewidths=[1.0, 1.8],
                   linestyles=["dashed", "solid"],
                   zorder=4)
    if zone_stats:
        draw_zone_overlay(ax, zone_stats)
    if title:
        ax.set_title(title)


def _draw_histograms(ax_y, ax_x, xs1, ys1, xs2, ys2):
    """Draw overlaid Y and X court-position distributions for both players."""
    datasets = [
        (PLAYER_LABELS[0], PLAYER_COLORS[0], np.array(xs1), np.array(ys1)),
        (PLAYER_LABELS[1], PLAYER_COLORS[1], np.array(xs2), np.array(ys2)),
    ]
    for label, color, xs, ys in datasets:
        if len(ys):
            ax_y.hist(ys, bins=np.linspace(0, COURT_LENGTH_M, 50),
                      color=color, alpha=0.55, label=f"{label}  μ={np.mean(ys):.2f} m")
            ax_y.axvline(np.mean(ys), color=color, lw=1.5, ls="--")
        if len(xs):
            ax_x.hist(xs, bins=np.linspace(0, COURT_WIDTH_M, 40),
                      color=color, alpha=0.55, label=f"{label}  μ={np.mean(xs):.2f} m")
            ax_x.axvline(np.mean(xs), color=color, lw=1.5, ls="--")

    ax_y.set_xlabel("Court Y (m)  ←front  |  back→")
    ax_y.set_ylabel("Frame count")
    ax_y.set_title("Y Distribution (both players)")
    ax_y.legend(fontsize=8)

    ax_x.set_xlabel("Court X (m)  ←left  |  right→")
    ax_x.set_ylabel("Frame count")
    ax_x.set_title("X Distribution (both players)")
    ax_x.legend(fontsize=8)


# ── Public plot functions — save to file, no plt.show() ──────────────────────

def plot_court_positions(xs1, ys1, xs2, ys2, title="Player Movement (Court Space)"):
    fig, ax = plt.subplots(figsize=(6, 9))
    _draw_court_positions(ax, xs1, ys1, xs2, ys2, title)
    plt.tight_layout()
    _save(fig, "court_positions.png")
    plt.close(fig)


def plot_heatmap(xs1, ys1, xs2, ys2, title="Player Heatmap",
                 zone_stats1=None, zone_stats2=None):
    """Save a per-player Gaussian heatmap PNG for each player."""
    datasets = [
        (xs1, ys1, PLAYER_LABELS[0], _PLAYER_CMAPS[0], "heatmap_player1.png", zone_stats1),
        (xs2, ys2, PLAYER_LABELS[1], _PLAYER_CMAPS[1], "heatmap_player2.png", zone_stats2),
    ]
    for xs, ys, label, cmap, fname, zstats in datasets:
        if not xs:
            continue
        fig, ax = plt.subplots(figsize=(6, 9))
        _draw_heatmap_on_ax(ax, xs, ys, cmap, zstats, title=f"{title} — {label}")
        plt.tight_layout()
        _save(fig, fname)
        plt.close(fig)


def plot_zone_breakdown(zone_stats1, zone_stats2):
    """Save a standalone zone breakdown chart for both players.

    Each cell is filled with a colour proportional to the time spent in that
    zone, with the zone name and percentage printed inside.  The T zone gets
    a gold border so coaches can immediately spot T-position dominance.
    Saved to output/zone_breakdown.png.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 9))
    fig.suptitle("Court Zone Breakdown", fontsize=14, fontweight="bold")

    # Base RGB values per player (player 1 = red family, player 2 = blue family)
    base_colors = [(0.8, 0.1, 0.1), (0.05, 0.35, 0.85)]

    for ax, zone_stats, label, base in zip(
        axes,
        [zone_stats1, zone_stats2],
        PLAYER_LABELS,
        base_colors,
    ):
        draw_court(ax)

        all_pcts = [zone_stats.get(name, 0.0) for row in ZONE_NAMES for name in row]
        max_pct = max(all_pcts) if max(all_pcts) > 0 else 1.0

        for ri, row in enumerate(ZONE_NAMES):
            y0 = ZONE_ROW_EDGES[ri]
            y1 = ZONE_ROW_EDGES[ri + 1]
            for ci, name in enumerate(row):
                x0 = ZONE_COL_EDGES[ci]
                x1 = ZONE_COL_EDGES[ci + 1]
                pct = zone_stats.get(name, 0.0)

                # Alpha scales linearly from 0.08 (empty) to 0.82 (max)
                alpha = 0.08 + 0.74 * (pct / max_pct)
                is_t = (name == "T")

                ax.add_patch(mpatches.Rectangle(
                    (x0, y0), x1 - x0, y1 - y0,
                    facecolor=(*base, alpha),
                    edgecolor="gold" if is_t else "none",
                    linewidth=3 if is_t else 0,
                    zorder=3,
                ))

                x_mid = (x0 + x1) / 2
                y_mid = (y0 + y1) / 2
                text_color = "white" if alpha > 0.45 else "black"
                ax.text(
                    x_mid, y_mid,
                    f"{name}\n{pct:.1f}%",
                    ha="center", va="center",
                    fontsize=10,
                    fontweight="bold" if is_t else "normal",
                    color=text_color,
                    zorder=6,
                )

        # Draw zone boundary lines on top of fills
        line_kw = dict(color="gray", linewidth=1.2, linestyle="--", alpha=0.7, zorder=5)
        for x in ZONE_COL_EDGES[1:-1]:
            ax.plot([x, x], [0, COURT_LENGTH_M], **line_kw)
        for y in ZONE_ROW_EDGES[1:-1]:
            ax.plot([0, COURT_WIDTH_M], [y, y], **line_kw)

        ax.set_title(f"{label} — Zone Breakdown", fontsize=12)

    plt.tight_layout()
    _save(fig, "zone_breakdown.png")
    plt.close(fig)


def plot_histograms(xs1, ys1, xs2, ys2):
    """Save overlaid Y and X position distribution histograms for both players."""
    fig, (ax_y, ax_x) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Position Distributions")
    _draw_histograms(ax_y, ax_x, xs1, ys1, xs2, ys2)
    plt.tight_layout()
    _save(fig, "histograms.png")
    plt.close(fig)


def plot_positions(xs1, ys1, xs2, ys2, background=None, title="Player Movement (Pixel Space)"):
    fig = plt.figure(figsize=(10, 7))
    if background is not None:
        rgb = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
        plt.imshow(rgb)
    else:
        plt.gca().invert_yaxis()
    if xs1:
        plt.scatter(xs1, ys1, s=2, alpha=0.5, color=PLAYER_COLORS[0], label=PLAYER_LABELS[0])
    if xs2:
        plt.scatter(xs2, ys2, s=2, alpha=0.5, color=PLAYER_COLORS[1], label=PLAYER_LABELS[1])
    plt.legend()
    plt.title(title)
    plt.xlabel("X (pixels)")
    plt.ylabel("Y (pixels)")
    plt.tight_layout()
    _save(fig, "pixel_positions.png")
    plt.close(fig)


_SMOOTH_WIN = 15   # rolling-average window for speed series display


def _draw_timeseries(ax_speed, ax_dist, ax_y, ts1, ts2):
    """Draw speed, distance-from-T, and court-depth time series for both players."""
    for ts, label, color in [
        (ts1, PLAYER_LABELS[0], PLAYER_COLORS[0]),
        (ts2, PLAYER_LABELS[1], PLAYER_COLORS[1]),
    ]:
        if ts is None:
            continue
        t     = ts["time_s"]
        speed = ts["speed_ms"]

        # Light rolling average so the speed trace is readable
        if len(speed) >= _SMOOTH_WIN:
            kernel = np.ones(_SMOOTH_WIN) / _SMOOTH_WIN
            speed  = np.convolve(speed, kernel, mode="same")

        ax_speed.plot(t, speed,            color=color, lw=1.2, alpha=0.85, label=label)
        ax_dist.plot( t, ts["dist_to_t_m"], color=color, lw=1.0, alpha=0.8,  label=label)
        ax_y.plot(    t, ts["y_m"],          color=color, lw=1.0, alpha=0.8,  label=label)

    ax_speed.set_xlabel("Time (s)")
    ax_speed.set_ylabel("Speed (m/s)")
    ax_speed.set_title("Speed over time")
    ax_speed.legend(fontsize=8)

    ax_dist.axhline(T_RADIUS_M, color="gray", lw=0.8, ls="--",
                    label=f"T radius ({T_RADIUS_M} m)")
    ax_dist.set_xlabel("Time (s)")
    ax_dist.set_ylabel("Distance from T (m)")
    ax_dist.set_title("Distance from T over time")
    ax_dist.legend(fontsize=8)

    short_y = T_Y
    ax_y.axhline(short_y, color="gray", lw=0.8, ls="--",
                 label=f"Short line ({short_y:.2f} m)")
    ax_y.invert_yaxis()   # front wall (y=0) at top, matching court diagram
    ax_y.set_xlabel("Time (s)")
    ax_y.set_ylabel("Court depth Y (m)")
    ax_y.set_title("Court depth over time")
    ax_y.legend(fontsize=8)


def plot_timeseries(ts1, ts2):
    """Save time-series plots (speed, dist-from-T, Y position) to file."""
    fig, (ax_speed, ax_dist, ax_y) = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Time Series Analysis")
    _draw_timeseries(ax_speed, ax_dist, ax_y, ts1, ts2)
    plt.tight_layout()
    _save(fig, "timeseries.png")
    plt.close(fig)


def show_summary(court_xs1, court_ys1, court_xs2, court_ys2,
                 zone_stats1=None, zone_stats2=None,
                 ts1=None, ts2=None):
    """Display all key analysis plots in a single window.

    Layout (3 rows × 6 virtual columns):
      Row 0 (portrait): Court scatter | Heatmap P1 | Heatmap P2
      Row 1 (landscape): Y histogram (both) | X histogram (both)
      Row 2 (landscape): Speed | Dist from T | Court depth  [if ts provided]
    """
    has_ts = ts1 is not None or ts2 is not None
    n_rows = 3 if has_ts else 2
    height_ratios = [1.6, 1, 1] if has_ts else [1.6, 1]

    fig = plt.figure(figsize=(18, 26 if has_ts else 20))
    gs = fig.add_gridspec(
        n_rows, 6,
        height_ratios=height_ratios,
        hspace=0.45,
        wspace=0.35,
    )

    # Row 0 — three portrait court panels
    ax_court = fig.add_subplot(gs[0, 0:2])
    ax_h1    = fig.add_subplot(gs[0, 2:4])
    ax_h2    = fig.add_subplot(gs[0, 4:6])

    # Row 1 — two landscape histogram panels
    ax_y = fig.add_subplot(gs[1, 0:3])
    ax_x = fig.add_subplot(gs[1, 3:6])

    _draw_court_positions(ax_court, court_xs1, court_ys1, court_xs2, court_ys2,
                          title="Player Positions")
    if court_xs1:
        _draw_heatmap_on_ax(ax_h1, court_xs1, court_ys1, _PLAYER_CMAPS[0],
                            zone_stats1, title=f"Heatmap — {PLAYER_LABELS[0]}")
    if court_xs2:
        _draw_heatmap_on_ax(ax_h2, court_xs2, court_ys2, _PLAYER_CMAPS[1],
                            zone_stats2, title=f"Heatmap — {PLAYER_LABELS[1]}")
    _draw_histograms(ax_y, ax_x, court_xs1, court_ys1, court_xs2, court_ys2)

    # Row 2 — three time series panels (optional)
    if has_ts:
        ax_speed = fig.add_subplot(gs[2, 0:2])
        ax_dist  = fig.add_subplot(gs[2, 2:4])
        ax_depth = fig.add_subplot(gs[2, 4:6])
        _draw_timeseries(ax_speed, ax_dist, ax_depth, ts1, ts2)

    fig.suptitle("Match Analysis Summary", fontsize=15, fontweight="bold")
    plt.show()


# ── Validation-only function (used by validate_ground_fix.py) ─────────────────

def plot_heatmap_comparison(hxs1, hys1, hxs2, hys2, gxs1, gys1, gxs2, gys2,
                             output_dir=None):
    """2×2 heatmap grid: rows=players, cols=hip(old)/heel-ankle(new). Saves and closes."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 18))
    fig.suptitle("Heatmap Comparison: Hip (old) vs Heel/Ankle (new)", fontsize=13)

    for row, (label, hip_data, grnd_data, cmap) in enumerate([
        ("Player 1", (hxs1, hys1), (gxs1, gys1), _PLAYER_CMAPS[0]),
        ("Player 2", (hxs2, hys2), (gxs2, gys2), _PLAYER_CMAPS[1]),
    ]):
        for col, ((xs, ys), method) in enumerate([
            (hip_data,  "Hip (old)"),
            (grnd_data, "Heel/Ankle (new)"),
        ]):
            ax = axes[row, col]
            _draw_heatmap_on_ax(ax, xs, ys, cmap, title=f"{label} — {method}")

    plt.tight_layout()
    fname = "heatmap_comparison.png"
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, fname)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    else:
        _save(fig, fname)
    plt.close(fig)
