"""
Ground truth annotator for squash player position benchmarking.

Usage
-----
    python src/annotate_ground_truth.py              # 500 frames, default video
    python src/annotate_ground_truth.py --frames 200
    python src/annotate_ground_truth.py --resume     # continue a previous session

Workflow per frame
------------------
    1. Click Player 1's feet  (red crosshair)
    2. Click Player 2's feet  (blue crosshair)
       → auto-advances to next frame

Keyboard
--------
    U   undo last click (within current frame, or go back one frame)
    S   skip frame (too blurry / player off-court)
    Q   save and quit

Output
------
    assets/ground_truth.json
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from calibration.calibrate import load_best_calibration, project_to_court, apply_homography, CalibData
from config import VIDEO_PATH, FRAME_CAP, COURT_WIDTH_M, COURT_LENGTH_M, SHORT_LINE_M, HALF_COURT_M, SERVICE_BOX_M
from utils.video_utils import load_video

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GT_PATH = os.path.join(_PROJECT_ROOT, "assets", "ground_truth.json")

# ── colours (BGR) ─────────────────────────────────────────────────────────────
C_P1     = (60,  60,  220)   # red    — Player 1
C_P2     = (220, 140,  60)   # blue   — Player 2
C_CURSOR = (180, 180, 180)   # grey   — mouse projection
C_WHITE  = (230, 230, 230)
C_GREEN  = (60,  200,  60)
C_CYAN   = (220, 220,   0)


# ── frame sampling ─────────────────────────────────────────────────────────────

def _sample_frames(total: int, n: int) -> list[int]:
    """Uniform stride sample of n frame indices from [0, total)."""
    if n >= total:
        return list(range(total))
    step = total / n
    indices = sorted({int(i * step) for i in range(n)})
    # pad to exactly n if rounding created duplicates
    extra = n - len(indices)
    if extra > 0:
        all_idx = set(indices)
        for i in range(total):
            if i not in all_idx:
                indices.append(i)
                extra -= 1
                if extra == 0:
                    break
        indices.sort()
    return indices[:n]


# ── court diagram (OpenCV, no matplotlib) ─────────────────────────────────────

_SHORT_Y = COURT_LENGTH_M - SHORT_LINE_M   # 5.49 m
_BOX_Y   = _SHORT_Y + SERVICE_BOX_M        # 7.09 m
_RIGHT_X = COURT_WIDTH_M - SERVICE_BOX_M   # 4.80 m


def _draw_court_panel(w: int, h: int,
                      p1_court=None, p2_court=None,
                      cursor_court=None,
                      next_player: int = 0) -> np.ndarray:
    """Return an OpenCV court diagram panel (top-down, front wall at top)."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (30, 30, 30)

    M = 20   # margin px
    cw = w - 2 * M
    ch = h - 2 * M - 40   # leave room for legend at bottom

    def to_px(x_m, y_m):
        # y=0 front wall → top of panel (invert y)
        px = int(M + x_m / COURT_WIDTH_M * cw)
        py = int(M + y_m / COURT_LENGTH_M * ch)
        return px, py

    line_kw = dict(color=(150, 150, 150), thickness=1)
    # court outline
    cv2.rectangle(img, to_px(0, 0), to_px(COURT_WIDTH_M, COURT_LENGTH_M), (180, 180, 180), 1)
    # short line
    cv2.line(img, to_px(0, _SHORT_Y), to_px(COURT_WIDTH_M, _SHORT_Y), **line_kw)
    # half-court (T to back only)
    cv2.line(img, to_px(HALF_COURT_M, _SHORT_Y), to_px(HALF_COURT_M, COURT_LENGTH_M), **line_kw)
    # service boxes
    cv2.line(img, to_px(SERVICE_BOX_M, _SHORT_Y), to_px(SERVICE_BOX_M, _BOX_Y), **line_kw)
    cv2.line(img, to_px(_RIGHT_X, _SHORT_Y), to_px(_RIGHT_X, _BOX_Y), **line_kw)
    cv2.line(img, to_px(0, _BOX_Y), to_px(COURT_WIDTH_M, _BOX_Y), **line_kw)

    # labels
    cv2.putText(img, "Front wall", (M, M - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (100, 100, 100), 1)
    cv2.putText(img, "Back wall",  (M, M + ch + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (100, 100, 100), 1)

    # cursor (mouse hover projection)
    if cursor_court is not None:
        cx, cy = cursor_court
        if not (np.isnan(cx) or np.isnan(cy)):
            cv2.circle(img, to_px(cx, cy), 5, C_CURSOR, -1)

    # player dots
    for pos, col, label in [(p1_court, C_P1, "P1"), (p2_court, C_P2, "P2")]:
        if pos is not None:
            px, py = to_px(pos[0], pos[1])
            cv2.circle(img, (px, py), 8, col, -1)
            cv2.putText(img, label, (px + 10, py + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)

    # bottom legend: which player to click next
    legend_y = h - 18
    cv2.rectangle(img, (0, h - 32), (w, h), (20, 20, 20), -1)
    label = f"Click {'P1' if next_player == 0 else 'P2'} feet"
    col   = C_P1 if next_player == 0 else C_P2
    cv2.putText(img, label, (M, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)

    return img


# ── video renderer ─────────────────────────────────────────────────────────────

def _draw_video_overlay(frame: np.ndarray,
                        clicks: list,
                        mouse_xy: tuple,
                        frame_no: int,
                        done: int,
                        total: int) -> np.ndarray:
    fh, fw = frame.shape[:2]
    disp = frame.copy()

    # existing clicks
    colors = [C_P1, C_P2]
    labels = ["P1", "P2"]
    for i, (cx, cy) in enumerate(clicks):
        cv2.circle(disp, (cx, cy), 10, colors[i], -1)
        cv2.circle(disp, (cx, cy), 10, (0, 0, 0), 1)
        cv2.putText(disp, labels[i], (cx + 12, cy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, colors[i], 2)

    # crosshair for next click
    if len(clicks) < 2:
        col = colors[len(clicks)]
        mx, my = mouse_xy
        L = 20
        cv2.line(disp, (mx - L, my), (mx + L, my), col, 1)
        cv2.line(disp, (mx, my - L), (mx, my + L), col, 1)
        cv2.circle(disp, (mx, my), 5, col, 1)

    # status bar at bottom
    bh = 30
    bar = np.zeros((bh, fw, 3), dtype=np.uint8)
    bar[:] = (25, 25, 25)
    progress = f"  Frame {frame_no}   [{done}/{total} annotated]"
    cv2.putText(bar, progress, (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.50, C_WHITE, 1)
    cv2.putText(bar, "U=undo  S=skip  Q=quit", (fw - 210, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (130, 130, 130), 1)

    return np.vstack([disp, bar])


# ── projection helper ──────────────────────────────────────────────────────────

def _project(px, py, calib_or_H):
    if isinstance(calib_or_H, CalibData):
        xs, ys = project_to_court([px], [py], calib_or_H)
    else:
        xs, ys = apply_homography([px], [py], calib_or_H)
    if not xs:
        return float("nan"), float("nan")
    return xs[0], ys[0]


# ── main annotator ─────────────────────────────────────────────────────────────

def annotate(n_frames: int, video_path: str, gt_path: str):
    cap, _, frame_count = load_video(video_path)
    total = min(frame_count, FRAME_CAP) if FRAME_CAP else frame_count

    ret, first_frame = cap.read()
    if not ret:
        raise RuntimeError("Could not read video")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    calib, H = load_best_calibration(first_frame)
    calib_or_H = calib if calib is not None else H

    # ── load or create ground truth file ──────────────────────────────────────
    if os.path.exists(gt_path):
        with open(gt_path) as f:
            gt = json.load(f)
        annotations: dict = gt.get("annotations", {})
        skipped: list     = gt.get("skipped", [])
        frame_indices: list = gt.get("frame_indices", _sample_frames(total, n_frames))
        print(f"Resuming: {len(annotations)} frames already annotated, "
              f"{len(skipped)} skipped.")
    else:
        frame_indices = _sample_frames(total, n_frames)
        annotations   = {}
        skipped       = []
        print(f"New session: {len(frame_indices)} frames sampled.")

    def _save():
        os.makedirs(os.path.dirname(gt_path), exist_ok=True)
        with open(gt_path, "w") as f:
            json.dump({
                "video_path":    video_path,
                "frame_indices": frame_indices,
                "annotations":   annotations,
                "skipped":       skipped,
            }, f, indent=2)

    # ── find first unannotated frame ───────────────────────────────────────────
    annotated_set = set(int(k) for k in annotations) | set(skipped)
    cursor = next(
        (i for i, fi in enumerate(frame_indices) if fi not in annotated_set),
        len(frame_indices)
    )

    if cursor >= len(frame_indices):
        print("All frames already annotated!")
        cap.release()
        return

    # ── OpenCV windows ─────────────────────────────────────────────────────────
    WIN_VID   = "Annotator — click P1 then P2 feet"
    WIN_COURT = "Court projection (live)"

    fh, fw = first_frame.shape[:2]
    # scale video to fit screen (max 900px wide)
    scale = min(1.0, 900 / fw)
    dw, dh = int(fw * scale), int(fh * scale)

    court_w, court_h = 260, 440

    cv2.namedWindow(WIN_VID,   cv2.WINDOW_NORMAL)
    cv2.namedWindow(WIN_COURT, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_VID,   dw, dh + 30)
    cv2.resizeWindow(WIN_COURT, court_w, court_h)
    cv2.moveWindow(WIN_VID,   40,       50)
    cv2.moveWindow(WIN_COURT, dw + 70,  50)

    # ── state ──────────────────────────────────────────────────────────────────
    clicks     = []          # pixel (x, y) for current frame; len 0, 1, or 2
    mouse_xy   = (0, 0)
    frame_cache: dict = {}   # frame_idx → BGR image (small LRU)
    current_frame_img = None

    def _load_frame(fi: int) -> np.ndarray:
        if fi in frame_cache:
            return frame_cache[fi]
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, f = cap.read()
        if not ret:
            return np.zeros((fh, fw, 3), dtype=np.uint8)
        if scale < 1.0:
            f = cv2.resize(f, (dw, dh))
        if len(frame_cache) > 10:
            frame_cache.pop(next(iter(frame_cache)))
        frame_cache[fi] = f
        return f

    def _render():
        fi  = frame_indices[cursor]
        img = _load_frame(fi)
        done = len(annotations)

        # video window
        vid_disp = _draw_video_overlay(img, clicks, mouse_xy, fi, done, len(frame_indices))
        cv2.imshow(WIN_VID, vid_disp)

        # court window
        p1_court = p2_court = None
        cursor_court = None
        if len(clicks) >= 1:
            x, y = _project(clicks[0][0] / scale, clicks[0][1] / scale, calib_or_H)
            p1_court = (x, y)
        if len(clicks) >= 2:
            x, y = _project(clicks[1][0] / scale, clicks[1][1] / scale, calib_or_H)
            p2_court = (x, y)
        mx, my = mouse_xy
        cx, cy = _project(mx / scale, my / scale, calib_or_H)
        cursor_court = (cx, cy)

        court_img = _draw_court_panel(court_w, court_h,
                                       p1_court, p2_court, cursor_court,
                                       next_player=len(clicks))
        cv2.imshow(WIN_COURT, court_img)

    def on_mouse(event, x, y, flags, _param):
        nonlocal mouse_xy, clicks, cursor
        mouse_xy = (x, min(y, dh - 1))

        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 2:
            clicks.append((x, min(y, dh - 1)))

            if len(clicks) == 2:
                # both players clicked — save and advance
                fi = frame_indices[cursor]
                p1_px = (clicks[0][0] / scale, clicks[0][1] / scale)
                p2_px = (clicks[1][0] / scale, clicks[1][1] / scale)
                p1_cx, p1_cy = _project(*p1_px, calib_or_H)
                p2_cx, p2_cy = _project(*p2_px, calib_or_H)
                annotations[str(fi)] = {
                    "p1_pixel": list(p1_px),
                    "p2_pixel": list(p2_px),
                    "p1_court": [p1_cx, p1_cy],
                    "p2_court": [p2_cx, p2_cy],
                }
                _save()

                # advance cursor past already-done frames
                cursor += 1
                annotated_set.add(fi)
                while (cursor < len(frame_indices) and
                       frame_indices[cursor] in annotated_set):
                    cursor += 1
                clicks = []

        _render()

    cv2.setMouseCallback(WIN_VID, on_mouse)
    _render()

    print("\nControls: click P1 feet → P2 feet (auto-advances)  |  U=undo  S=skip  Q=quit\n")

    while cursor < len(frame_indices):
        key = cv2.waitKey(30) & 0xFF

        if key in (ord('q'), ord('Q'), 27):   # Q or ESC
            print("\nSaving and quitting…")
            break

        elif key in (ord('u'), ord('U')):
            if clicks:
                clicks.pop()
            elif cursor > 0:
                # undo entire previous frame
                cursor -= 1
                fi_prev = frame_indices[cursor]
                annotations.pop(str(fi_prev), None)
                annotated_set.discard(fi_prev)
                _save()
                clicks = []
            _render()

        elif key in (ord('s'), ord('S')):
            fi = frame_indices[cursor]
            if fi not in skipped:
                skipped.append(fi)
            annotated_set.add(fi)
            _save()
            cursor += 1
            while (cursor < len(frame_indices) and
                   frame_indices[cursor] in annotated_set):
                cursor += 1
            clicks = []
            _render()

    cv2.destroyAllWindows()
    cap.release()

    done = len(annotations)
    print(f"\nDone. {done} frames annotated, {len(skipped)} skipped.")
    print(f"Saved → {gt_path}")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Annotate ground truth player positions.")
    ap.add_argument("--frames",  type=int, default=500,    help="Number of frames to sample (default 500)")
    ap.add_argument("--video",   type=str, default=VIDEO_PATH, help="Path to video file")
    ap.add_argument("--out",     type=str, default=GT_PATH,    help="Output JSON path")
    ap.add_argument("--resume",  action="store_true",      help="Resume existing annotation session")
    args = ap.parse_args()

    if not args.resume and os.path.exists(args.out):
        ans = input(f"Annotation file already exists at {args.out}. Resume? [Y/n]: ").strip().lower()
        if ans in ("", "y"):
            pass   # resume anyway
        else:
            import shutil
            bak = args.out + ".bak"
            shutil.copy2(args.out, bak)
            print(f"Backed up existing file to {bak}")
            os.remove(args.out)

    annotate(args.frames, args.video, args.out)
