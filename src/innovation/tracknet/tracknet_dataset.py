"""
PyTorch dataset for TrackNetV4 training.

Each sample is built from a triplet of consecutive labeled frames (t-1, t, t+1).
Only frames with explicit labels (visible or not-visible) in the CSV are used.
The center frame of each triplet is the one whose heatmap is the primary target.

The dataset is multi-video aware: it reads all unique video names from the CSV,
opens each video file from `video_dir`, and pools samples across all of them.
This means training automatically uses every labeled video — regardless of which
video config.py currently points to.

Input tensor  : (11, 288, 512) — 9 RGB channels + 2 grayscale difference maps
Target tensor : (3,  288, 512) — Gaussian heatmap for each of the 3 frames
                                  (sigma=5 px at model resolution; zeros if not visible)

Label CSV format (produced by tracknet_label.py):
  Video, Frame, Visibility, X, Y
  Visibility=1 → ball is at (X, Y) in original video pixel coordinates
  Visibility=0 → ball is not visible (heatmap = all zeros)
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split

_PKG = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_PKG, "..", "..")
sys.path.insert(0, _PKG)   # for tracknet_model import
sys.path.insert(0, _SRC)   # for config / core imports
from tracknet_model import HEIGHT, WIDTH

_ROOT = os.path.join(_PKG, "..", "..", "..")
LABEL_PATH = os.path.join(_ROOT, "assets", "ball_labels.csv")
VIDEO_DIR  = os.path.join(_ROOT, "assets", "video")

# Gaussian heatmap parameters
_SIGMA = 5.0   # pixels at model resolution (288×512)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_heatmap(cx: float, cy: float, h: int = HEIGHT, w: int = WIDTH,
                  sigma: float = _SIGMA) -> np.ndarray:
    """Return a (H, W) float32 Gaussian heatmap centred at (cx, cy).

    Returns all-zeros if cx <= 0 or cy <= 0 (ball not visible).
    """
    if cx <= 0 and cy <= 0:
        return np.zeros((h, w), dtype=np.float32)
    xs = np.arange(w, dtype=np.float32)
    ys = np.arange(h, dtype=np.float32)
    xg, yg = np.meshgrid(xs, ys)
    hm = np.exp(-((xg - cx) ** 2 + (yg - cy) ** 2) / (2 * sigma ** 2))
    return hm.astype(np.float32)


def _load_frame(cap: cv2.VideoCapture, fi: int, cache: dict) -> np.ndarray | None:
    if fi in cache:
        return cache[fi]
    cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
    ret, frame = cap.read()
    if not ret:
        return None
    if len(cache) > 200:
        del cache[next(iter(cache))]
    cache[fi] = frame
    return frame


def _frames_to_tensor(frames_bgr: list) -> tuple[np.ndarray, np.ndarray]:
    """Convert 3 BGR frames to (visual[9,H,W], diff[2,H,W]), both float32 /255."""
    resized = [cv2.resize(f, (WIDTH, HEIGHT)) for f in frames_bgr]
    visual = np.concatenate(
        [np.moveaxis(f, -1, 0) for f in resized], axis=0
    ).astype(np.float32) / 255.0
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
             for f in resized]
    diff = np.stack([
        np.abs(grays[1] - grays[0]),
        np.abs(grays[2] - grays[1]),
    ], axis=0)
    return visual, diff


# ── Dataset ───────────────────────────────────────────────────────────────────

class SquashBallDataset(Dataset):
    """Multi-video dataset for TrackNetV4 training.

    Parameters
    ----------
    label_path : str  — path to ball_labels.csv (must have a Video column)
    video_dir  : str  — directory containing the video files
    augment    : bool — apply horizontal flip augmentation
    """

    def __init__(self, label_path: str = LABEL_PATH,
                 video_dir: str = VIDEO_DIR,
                 augment: bool = False):
        self.augment = augment

        df = pd.read_csv(label_path)
        if "Video" not in df.columns:
            raise ValueError(
                "Label CSV has no 'Video' column.\n"
                "Open tracknet_label.py, re-save the CSV — it will add the column automatically."
            )

        # Per-video metadata keyed by video filename
        self._meta: dict[str, dict] = {}
        self._samples: list[tuple[str, int]] = []   # (video_name, center_frame_idx)

        videos_found = []
        for video_name, group in df.groupby("Video"):
            video_path = os.path.join(video_dir, video_name)
            if not os.path.exists(video_path):
                print(f"[Warning] Video file not found, skipping labels for: {video_name}")
                print(f"          Expected at: {video_path}")
                continue

            cap = cv2.VideoCapture(video_path)
            orig_w       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            orig_h       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            label_map = {
                int(row["Frame"]): (int(row["Visibility"]),
                                    float(row["X"]), float(row["Y"]))
                for _, row in group.iterrows()
            }

            self._meta[video_name] = {
                "path":         video_path,
                "orig_w":       orig_w,
                "orig_h":       orig_h,
                "total_frames": total_frames,
                "label_map":    label_map,
            }

            valid = [(video_name, fi) for fi in sorted(label_map)
                     if 0 < fi < total_frames - 1]
            self._samples.extend(valid)
            videos_found.append(f"{video_name} ({len(valid)} samples)")

        print(f"Dataset: {len(self._samples)} total samples from {len(self._meta)} video(s)")
        for v in videos_found:
            print(f"  {v}")

        if len(self._samples) == 0:
            raise RuntimeError(
                "No samples found. Check that:\n"
                f"  1. Labels exist in: {label_path}\n"
                f"  2. Video files are in: {video_dir}\n"
                f"  3. Video filenames in the CSV match the files on disk."
            )

        # Lazy VideoCapture + frame cache, one per video (populated in __getitem__)
        self._caps:   dict[str, cv2.VideoCapture] = {}
        self._caches: dict[str, dict]             = {}

    def _get_cap(self, video_name: str) -> cv2.VideoCapture:
        if video_name not in self._caps or not self._caps[video_name].isOpened():
            self._caps[video_name] = cv2.VideoCapture(self._meta[video_name]["path"])
            self._caches[video_name] = {}
        return self._caps[video_name]

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        video_name, fi = self._samples[idx]
        meta  = self._meta[video_name]
        cap   = self._get_cap(video_name)
        cache = self._caches[video_name]

        # Load frame triplet; fall back to centre frame if neighbour is unreadable
        frames = []
        for offset in (-1, 0, 1):
            frame = _load_frame(cap, fi + offset, cache)
            if frame is None:
                frame = _load_frame(cap, fi, cache)
            frames.append(frame)

        visual, diff = _frames_to_tensor(frames)
        inp = np.concatenate([visual, diff], axis=0)   # (11, H, W)

        # Build target heatmaps for each frame in the triplet
        label_map = meta["label_map"]
        orig_w, orig_h = meta["orig_w"], meta["orig_h"]
        heatmaps = []
        for offset in (-1, 0, 1):
            entry = label_map.get(fi + offset)
            if entry is not None and entry[0] == 1:
                xm = entry[1] * WIDTH  / orig_w
                ym = entry[2] * HEIGHT / orig_h
                hm = _make_heatmap(xm, ym)
            else:
                hm = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
            heatmaps.append(hm)

        target = np.stack(heatmaps, axis=0)   # (3, H, W)

        if self.augment and np.random.rand() > 0.5:
            inp    = inp[:, :, ::-1].copy()
            target = target[:, :, ::-1].copy()

        return torch.from_numpy(inp), torch.from_numpy(target)

    def __del__(self):
        for cap in self._caps.values():
            cap.release()


# ── Convenience factory ───────────────────────────────────────────────────────

def make_dataloaders(label_path: str = LABEL_PATH,
                     video_dir: str = VIDEO_DIR,
                     val_fraction: float = 0.15,
                     batch_size: int = 4,
                     num_workers: int = 0) -> tuple[DataLoader, DataLoader]:
    """Build train/val DataLoaders from all labeled videos in the CSV.

    Parameters
    ----------
    label_path   : path to ball_labels.csv
    video_dir    : directory containing video files
    val_fraction : fraction of samples used for validation
    batch_size   : samples per batch
    num_workers  : DataLoader workers (keep 0 on Windows)
    """
    full = SquashBallDataset(label_path, video_dir, augment=True)

    n_val   = max(1, int(len(full) * val_fraction))
    n_train = len(full) - n_val
    if n_train < 1:
        raise RuntimeError(
            f"Only {len(full)} sample(s) found — not enough to split into train/val. "
            "Label more frames."
        )

    train_ds, val_ds = random_split(full, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))
    val_ds.dataset.augment = False

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=num_workers > 0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=num_workers > 0)
    return train_loader, val_loader


if __name__ == "__main__":
    ds = SquashBallDataset()
    print(f"\nFirst sample shapes:")
    if len(ds) > 0:
        inp, tgt = ds[0]
        print(f"  Input  {tuple(inp.shape)}  range [{inp.min():.3f}, {inp.max():.3f}]")
        print(f"  Target {tuple(tgt.shape)}  max {tgt.max():.3f}")
