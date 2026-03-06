"""
PyTorch dataset for TrackNetV4 training — Colab-ready version.

Default paths point to /content/ (Google Colab working directory).
Override LABEL_PATH and VIDEO_DIR at the top of this file or pass them
directly to SquashBallDataset / make_dataloaders.

Label CSV format (produced by tracknet_label.py):
  Video, Frame, Visibility, X, Y
  Visibility=1 → ball is at (X, Y) in original video pixel coordinates
  Visibility=0 → ball is not visible (heatmap = all zeros)
"""

import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split

from tracknet_model import HEIGHT, WIDTH

# ── Default paths (Colab) ──────────────────────────────────────────────────────
LABEL_PATH = '/content/ball_labels.csv'
VIDEO_DIR  = '/content'

_SIGMA = 5.0   # Gaussian heatmap sigma in pixels at model resolution


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_heatmap(cx: float, cy: float, h: int = HEIGHT, w: int = WIDTH,
                  sigma: float = _SIGMA) -> np.ndarray:
    if cx <= 0 and cy <= 0:
        return np.zeros((h, w), dtype=np.float32)
    xs = np.arange(w, dtype=np.float32)
    ys = np.arange(h, dtype=np.float32)
    xg, yg = np.meshgrid(xs, ys)
    hm = np.exp(-((xg - cx) ** 2 + (yg - cy) ** 2) / (2 * sigma ** 2))
    return hm.astype(np.float32)


def _load_frame(cap: cv2.VideoCapture, fi: int, cache: dict):
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


def _frames_to_tensor(frames_bgr: list) -> tuple:
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


# ── Dataset ────────────────────────────────────────────────────────────────────

class SquashBallDataset(Dataset):
    """Multi-video TrackNetV4 dataset.

    Parameters
    ----------
    label_path : path to ball_labels.csv
    video_dir  : directory containing the video files
    augment    : apply horizontal flip augmentation
    """

    def __init__(self, label_path: str = LABEL_PATH,
                 video_dir: str = VIDEO_DIR,
                 augment: bool = False):
        self.augment = augment

        if not os.path.exists(label_path):
            raise FileNotFoundError(
                f"Label file not found: {label_path}\n"
                "Make sure ball_labels.csv is uploaded to /content/"
            )

        df = pd.read_csv(label_path)
        if "Video" not in df.columns:
            raise ValueError(
                "Label CSV has no 'Video' column.\n"
                "Re-save using tracknet_label.py — it adds the column automatically."
            )

        self._meta: dict = {}
        self._samples: list = []

        for video_name, group in df.groupby("Video"):
            video_path = os.path.join(video_dir, video_name)
            if not os.path.exists(video_path):
                print(f"[Warning] Video not found, skipping: {video_path}")
                continue

            cap = cv2.VideoCapture(video_path)
            orig_w       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            orig_h       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            label_map = {
                int(row["Frame"]): (int(row["Visibility"]), float(row["X"]), float(row["Y"]))
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
            print(f"  {video_name}: {len(valid)} samples")

        print(f"Total: {len(self._samples)} samples from {len(self._meta)} video(s)")

        if len(self._samples) == 0:
            raise RuntimeError(
                "No samples found. Check:\n"
                f"  1. Labels exist in: {label_path}\n"
                f"  2. Video files are in: {video_dir}\n"
                f"  3. Video filenames in the CSV match files on disk."
            )

        self._caps:   dict = {}
        self._caches: dict = {}

    def _get_cap(self, video_name: str) -> cv2.VideoCapture:
        if video_name not in self._caps or not self._caps[video_name].isOpened():
            self._caps[video_name] = cv2.VideoCapture(self._meta[video_name]["path"])
            self._caches[video_name] = {}
        return self._caps[video_name]

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int):
        video_name, fi = self._samples[idx]
        meta  = self._meta[video_name]
        cap   = self._get_cap(video_name)
        cache = self._caches[video_name]

        frames = []
        for offset in (-1, 0, 1):
            frame = _load_frame(cap, fi + offset, cache)
            if frame is None:
                frame = _load_frame(cap, fi, cache)
            frames.append(frame)

        visual, diff = _frames_to_tensor(frames)
        inp = np.concatenate([visual, diff], axis=0)   # (11, H, W)

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


# ── Factory ────────────────────────────────────────────────────────────────────

def make_dataloaders(label_path: str = LABEL_PATH,
                     video_dir: str = VIDEO_DIR,
                     val_fraction: float = 0.15,
                     batch_size: int = 4,
                     num_workers: int = 0):
    """Build train/val DataLoaders."""
    full = SquashBallDataset(label_path, video_dir, augment=True)

    n_val   = max(1, int(len(full) * val_fraction))
    n_train = len(full) - n_val
    if n_train < 1:
        raise RuntimeError(
            f"Only {len(full)} sample(s) — not enough for train/val split. Label more frames."
        )

    train_ds, val_ds = random_split(
        full, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    val_ds.dataset.augment = False

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader
