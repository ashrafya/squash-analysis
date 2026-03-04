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


def plot_court_positions(xs1, ys1, xs2, ys2, title="Player Movement (Court Space)"):
    fig, ax = plt.subplots(figsize=(6, 9))
    draw_court(ax)
    if xs1:
        ax.scatter(xs1, ys1, s=4, alpha=0.5, color=PLAYER_COLORS[0], zorder=5, label=PLAYER_LABELS[0])
    if xs2:
        ax.scatter(xs2, ys2, s=4, alpha=0.5, color=PLAYER_COLORS[1], zorder=5, label=PLAYER_LABELS[1])
    ax.legend(loc="upper right")
    ax.set_title(title)
    plt.tight_layout()
    _save(fig, "court_positions.png")
    plt.show()


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


def plot_heatmap(xs1, ys1, xs2, ys2, title="Player Heatmap"):
    """Overlay a per-player Gaussian heatmap on the court diagram."""
    x_centers = (np.linspace(0, COURT_WIDTH_M,  HEATMAP_GRID_X + 1)[:-1] +
                 np.linspace(0, COURT_WIDTH_M,  HEATMAP_GRID_X + 1)[1:]) / 2
    y_centers = (np.linspace(0, COURT_LENGTH_M, HEATMAP_GRID_Y + 1)[:-1] +
                 np.linspace(0, COURT_LENGTH_M, HEATMAP_GRID_Y + 1)[1:]) / 2

    datasets = [
        (xs1, ys1, PLAYER_LABELS[0], _PLAYER_CMAPS[0], "heatmap_player1.png"),
        (xs2, ys2, PLAYER_LABELS[1], _PLAYER_CMAPS[1], "heatmap_player2.png"),
    ]
    for xs, ys, label, cmap, fname in datasets:
        if not xs:
            continue
        fig, ax = plt.subplots(figsize=(6, 9))
        draw_court(ax)
        h = _build_heatmap(xs, ys)

        # Heatmap sits between court floor (zorder=1) and court lines (zorder=2)
        ax.imshow(
            h,
            extent=[0, COURT_WIDTH_M, COURT_LENGTH_M, 0],
            origin="upper",
            aspect="auto",
            cmap=cmap,
            vmin=0, vmax=1,
            zorder=1.5,
        )

        # Contour lines at 50 % and 80 % density, drawn above court lines
        if h.max() > 0:
            ax.contour(x_centers, y_centers, h,
                       levels=[0.5, 0.8],
                       colors=["white", "white"],
                       linewidths=[1.0, 1.8],
                       linestyles=["dashed", "solid"],
                       zorder=4)

        ax.set_title(f"{title} — {label}")
        plt.tight_layout()
        _save(fig, fname)
        plt.show()


def plot_positions(xs1, ys1, xs2, ys2, background=None, title="Player Movement (Pixel Space)"):
    fig = plt.figure(figsize=(10, 7))

    if background is not None:
        rgb = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
        plt.imshow(rgb)
    else:
        plt.gca().invert_yaxis()  # image coordinate system

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
    plt.show()