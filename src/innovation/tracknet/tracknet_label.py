"""
Interactive ball labeling tool for TrackNetV4 training data.

Displays video frames one at a time.  Click on the ball to label it.
Keyboard shortcuts:
  Left-click      — mark ball position on current frame
  Right arrow / D — next frame  (by step)
  Left arrow  / A — previous frame  (by step)
  F               — fast-forward step frames
  B               — fast-backward step frames
  G               — go to a specific frame number (type in terminal)
  U               — mark ball as not visible (Visibility=0) on current frame
  Delete          — clear label on current frame
  S               — save labels to CSV
  Q / Escape      — save and quit

Labels are saved to:  assets/ball_labels.csv
CSV columns: Video, Frame, Visibility, X, Y
  Video      — filename of the source video (e.g. men360.mp4)
  Visibility — 1 = ball labeled, 0 = explicitly not visible
  X, Y       — pixel coordinates in original video resolution
  Unlabeled frames are simply absent from the CSV.
  Old CSVs without a Video column are treated as belonging to the current video.

Usage:
  python src/tracknet_label.py [--start FRAME] [--end FRAME] [--step N]

  --start : first frame to show (default 0)
  --end   : last  frame to show (default: all frames)
  --step  : frame step for navigation (default 1)
"""

import os
import sys
import argparse
import cv2
import numpy as np
import pandas as pd

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, _SRC)  # reach src/ for core imports
from config import VIDEO_PATH
if not os.path.isabs(VIDEO_PATH):
    VIDEO_PATH = os.path.normpath(os.path.join(_SRC, VIDEO_PATH))

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
LABEL_PATH = os.path.join(_ROOT, "assets", "ball_labels.csv")
WIN = "TrackNetV4 — Ball Labeler  [click=label  U=not-visible  Del=clear  A/D=prev/next  S=save  Q=quit]"

# Overlay colours
COL_LABELED    = (0, 255, 0)     # green  — ball position
COL_NOT_VIS    = (0, 100, 255)   # orange — not visible
COL_CROSSHAIR  = (255, 255, 255)


def _load_existing(path: str, current_video: str) -> tuple[dict, list]:
    """Load label CSV and split into current-video labels and other-video rows.

    Returns
    -------
    labels       : {frame_idx: (vis, x, y)} for the current video only
    other_rows   : list of raw row dicts for all other videos (preserved as-is on save)
    """
    labels: dict = {}
    other_rows: list = []
    if not os.path.exists(path):
        return labels, other_rows

    df = pd.read_csv(path)

    if "Video" not in df.columns:
        # Legacy CSV with no Video column — treat everything as current video
        for _, row in df.iterrows():
            labels[int(row["Frame"])] = (int(row["Visibility"]),
                                         float(row["X"]), float(row["Y"]))
    else:
        for _, row in df.iterrows():
            if row["Video"] == current_video:
                labels[int(row["Frame"])] = (int(row["Visibility"]),
                                             float(row["X"]), float(row["Y"]))
            else:
                other_rows.append(row.to_dict())

    return labels, other_rows


def _save(labels: dict, other_rows: list, current_video: str, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    # Current video rows
    current_rows = [
        {"Video": current_video, "Frame": fi,
         "Visibility": vis, "X": x, "Y": y}
        for fi, (vis, x, y) in sorted(labels.items())
    ]
    df = pd.DataFrame(current_rows + other_rows,
                      columns=["Video", "Frame", "Visibility", "X", "Y"])
    df = df.sort_values(["Video", "Frame"]).reset_index(drop=True)
    df.to_csv(path, index=False)

    n_other = len(other_rows)
    print(f"  Saved {len(current_rows)} labels for '{current_video}' "
          f"+ {n_other} labels for other videos → {path}")


def _draw_overlay(frame: np.ndarray, fi: int, labels: dict,
                  mouse_pos: tuple, current_video: str) -> np.ndarray:
    vis_frame = frame.copy()
    h, w = vis_frame.shape[:2]

    # Draw cursor: small ring + short tick marks (doesn't cover the ball)
    mx, my = mouse_pos
    if 0 <= mx < w and 0 <= my < h:
        R    = 10   # ring radius — sits just outside the ball
        TICK = 8    # length of tick marks beyond the ring
        cv2.circle(vis_frame, (mx, my), R, COL_CROSSHAIR, 1, cv2.LINE_AA)
        cv2.line(vis_frame, (mx, my - R - TICK), (mx, my - R - 1), COL_CROSSHAIR, 1)
        cv2.line(vis_frame, (mx, my + R + 1),    (mx, my + R + TICK), COL_CROSSHAIR, 1)
        cv2.line(vis_frame, (mx - R - TICK, my), (mx - R - 1, my),    COL_CROSSHAIR, 1)
        cv2.line(vis_frame, (mx + R + 1,    my), (mx + R + TICK, my), COL_CROSSHAIR, 1)

    # Draw existing label for this frame
    if fi in labels:
        vis, x, y = labels[fi]
        ix, iy = int(x), int(y)
        if vis == 1:
            cv2.circle(vis_frame, (ix, iy), 8, COL_LABELED, 2)
            cv2.drawMarker(vis_frame, (ix, iy), COL_LABELED,
                           cv2.MARKER_CROSS, 16, 1)
            cv2.putText(vis_frame, f"({ix},{iy})", (ix + 12, iy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, COL_LABELED, 1)
        else:
            cv2.putText(vis_frame, "NOT VISIBLE", (20, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, COL_NOT_VIS, 2)

    # Status bar
    n_labeled = sum(1 for v, _, _ in labels.values() if v == 1)
    n_not_vis = sum(1 for v, _, _ in labels.values() if v == 0)
    status = f"{current_video}   Frame {fi}   labeled={n_labeled}  not-visible={n_not_vis}"
    cv2.rectangle(vis_frame, (0, 0), (w, 24), (30, 30, 30), -1)
    cv2.putText(vis_frame, status, (6, 17),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    return vis_frame


def main(start: int = 0, end: int | None = None, step: int = 1):
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_end = min(end, total_frames - 1) if end is not None else total_frames - 1

    current_video = os.path.basename(VIDEO_PATH)
    labels, other_rows = _load_existing(LABEL_PATH, current_video)
    dirty = False
    print(f"Video: {current_video}")

    # Shared state updated by the mouse callback
    mouse_xy    = [0, 0]
    click_state = {"fired": False, "x": 0, "y": 0}

    # Frame cache: avoid re-seeking for recently seen frames
    frame_cache: dict[int, np.ndarray] = {}

    def _get_frame(fi: int) -> np.ndarray | None:
        if fi in frame_cache:
            return frame_cache[fi]
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frm = cap.read()
        if not ret:
            return None
        if len(frame_cache) > 50:
            del frame_cache[next(iter(frame_cache))]
        frame_cache[fi] = frm
        return frm

    def _on_mouse(event, x, y, flags, param):
        mouse_xy[0], mouse_xy[1] = x, y
        if event == cv2.EVENT_LBUTTONDOWN:
            click_state["fired"] = True
            click_state["x"]     = x
            click_state["y"]     = y

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1280, 720)
    cv2.setMouseCallback(WIN, _on_mouse)

    fi = start
    print(f"Labeling frames {start}–{frame_end} (step={step})")
    print(f"Loaded {len(labels)} existing labels from {LABEL_PATH}")

    while True:
        # Reset click flag at top of each iteration
        click_state["fired"] = False

        frame = _get_frame(fi)
        if frame is None:
            print(f"Could not read frame {fi}")
            fi = max(start, fi - step)
            continue

        overlay = _draw_overlay(frame, fi, labels, tuple(mouse_xy), current_video)
        cv2.imshow(WIN, overlay)

        key = cv2.waitKey(20) & 0xFF

        if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
            break

        # Left-click: label ball position and auto-advance
        if click_state["fired"]:
            cx, cy = click_state["x"], click_state["y"]
            labels[fi] = (1, float(cx), float(cy))
            dirty = True
            print(f"  Frame {fi}: ball at ({cx}, {cy})")
            fi = min(frame_end, fi + step)

        if key == ord("q") or key == 27:       # Q / Escape — save and quit
            break
        elif key == ord("s"):                  # S — save
            _save(labels, other_rows, current_video, LABEL_PATH)
            dirty = False
        elif key == ord("u"):                  # U — mark not visible
            labels[fi] = (0, 0.0, 0.0)
            dirty = True
            print(f"  Frame {fi}: not visible")
            fi = min(frame_end, fi + step)
        elif key in (255, 0, 8):               # Delete / Backspace — clear label
            if fi in labels:
                del labels[fi]
                dirty = True
                print(f"  Frame {fi}: label cleared")
        elif key in (83, ord("d")):            # Right / D — next frame
            fi = min(frame_end, fi + step)
        elif key in (81, ord("a")):            # Left / A — previous frame
            fi = max(start, fi - step)
        elif key == ord("f"):                  # F — fast-forward 10×step
            fi = min(frame_end, fi + step)
        elif key == ord("b"):                  # B — fast-backward 10×step
            fi = max(start, fi - step)
        elif key == ord("g"):                  # G — go to specific frame
            try:
                target = int(input(f"  Go to frame (current={fi}): "))
                fi = max(start, min(frame_end, target))
            except ValueError:
                pass

    cap.release()
    cv2.destroyAllWindows()

    if dirty:
        _save(labels, other_rows, current_video, LABEL_PATH)

    n_vis    = sum(1 for v, _, _ in labels.values() if v == 1)
    n_no_vis = sum(1 for v, _, _ in labels.values() if v == 0)
    print(f"\nDone.  '{current_video}': {n_vis} visible + {n_no_vis} not-visible = {len(labels)} frames")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ball labeling tool for TrackNetV4")
    parser.add_argument("--start", type=int, default=0,  help="First frame index")
    parser.add_argument("--end",   type=int, default=None, help="Last frame index")
    parser.add_argument("--step",  type=int, default=1,  help="Frame step for navigation")
    args = parser.parse_args()
    main(start=args.start, end=args.end, step=args.step)
