"""
TrackNetV4 training script.

Trains on labeled squash ball data produced by tracknet_label.py.
Uses the Weighted Binary Cross-Entropy loss from the original TrackNet papers,
which heavily penalises false negatives (missing the ball) over false positives.

Checkpoints are saved to:  assets/tracknet_ckpt/
  best.pt   — model with lowest validation loss
  last.pt   — model after the most recent epoch

Usage:
  python src/tracknet_train.py [--epochs N] [--lr LR] [--batch B] [--resume]

  --epochs N  : number of training epochs (default 50)
  --lr LR     : learning rate (default 1e-3)
  --batch  B  : batch size    (default 4)
  --resume    : resume from last.pt if it exists

Typical workflow:
  1. Label ~300+ frames with tracknet_label.py
  2. Run this script on a GPU machine (or CPU, just slower)
  3. Use best.pt with tracknet_infer.py
"""

import os
import sys
import argparse
import json
import time

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

_PKG = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_PKG, "..", "..")
sys.path.insert(0, _PKG)   # for tracknet_* imports
sys.path.insert(0, _SRC)   # for config / core imports
from tracknet_model import TrackNetV4, HEIGHT, WIDTH
from tracknet_dataset import make_dataloaders, LABEL_PATH, VIDEO_DIR

_ROOT = os.path.join(_PKG, "..", "..", "..")
CKPT_DIR  = os.path.join(_ROOT, "assets", "tracknet_ckpt")
BEST_PT   = os.path.join(CKPT_DIR, "best.pt")
LAST_PT   = os.path.join(CKPT_DIR, "last.pt")
LOG_JSON  = os.path.join(CKPT_DIR, "train_log.json")


# ── Loss ──────────────────────────────────────────────────────────────────────

def weighted_bce(y_pred: torch.Tensor, y: torch.Tensor,
                eps: float = 1e-7) -> torch.Tensor:
    """Weighted Binary Cross-Entropy from TrackNet papers.

    Weights false negatives by (1 - y_pred)² and false positives by y_pred²,
    which focuses training on uncertain predictions and penalises missed balls
    more than phantom detections.
    """
    loss = -(
        torch.square(1.0 - y_pred) * y * torch.log(y_pred.clamp(eps, 1.0))
        + torch.square(y_pred) * (1.0 - y) * torch.log((1.0 - y_pred).clamp(eps, 1.0))
    )
    return loss.mean()


# ── Metrics ───────────────────────────────────────────────────────────────────

def _tp_fp_fn(y_pred: torch.Tensor, y: torch.Tensor,
                threshold: float = 0.5) -> tuple[int, int, int]:
    """Pixel-level TP / FP / FN counts for a batch."""
    pred_bin = (y_pred >= threshold).float()
    gt_bin   = (y >= threshold).float()
    tp = int((pred_bin * gt_bin).sum().item())
    fp = int((pred_bin * (1.0 - gt_bin)).sum().item())
    fn = int(((1.0 - pred_bin) * gt_bin).sum().item())
    return tp, fp, fn


# ── One epoch ─────────────────────────────────────────────────────────────────

def _run_epoch(model: nn.Module, loader, optimizer,
               device: torch.device, train: bool) -> dict:
    model.train(train)
    total_loss = 0.0
    tp = fp = fn = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for inp, target in tqdm(loader, leave=False,
                                desc="train" if train else "val  "):
            inp    = inp.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            y_pred = model(inp)
            loss   = weighted_bce(y_pred, target)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * inp.size(0)
            b_tp, b_fp, b_fn = _tp_fp_fn(y_pred.detach(), target)
            tp += b_tp; fp += b_fp; fn += b_fn

    n = len(loader.dataset)
    avg_loss = total_loss / max(1, n)
    precision = tp / max(1, tp + fp)
    recall    = tp / max(1, tp + fn)
    f1        = 2 * precision * recall / max(1e-8, precision + recall)

    return {"loss": avg_loss, "precision": precision, "recall": recall, "f1": f1}


# ── Main ──────────────────────────────────────────────────────────────────────

def main(epochs: int = 50, lr: float = 1e-3, batch_size: int = 4,
         resume: bool = False):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Data ────────────────────────────────────────────────────────────────
    if not os.path.exists(LABEL_PATH):
        print(f"[Error] Label file not found: {LABEL_PATH}")
        print("Run tracknet_label.py first to create training data.")
        return

    train_loader, val_loader = make_dataloaders(
        label_path=LABEL_PATH,
        video_dir=VIDEO_DIR,
        batch_size=batch_size,
        num_workers=0,    # 0 is safest on Windows
    )
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val   samples: {len(val_loader.dataset)}")

    if len(train_loader.dataset) < 10:
        print("[Warning] Very few training samples — label more frames for good results.")

    # ── Model ───────────────────────────────────────────────────────────────
    model = TrackNetV4().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"TrackNetV4 parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    start_epoch = 0
    best_val_loss = float("inf")
    log: list[dict] = []

    # ── Resume ───────────────────────────────────────────────────────────────
    os.makedirs(CKPT_DIR, exist_ok=True)
    if resume and os.path.exists(LAST_PT):
        ckpt = torch.load(LAST_PT, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch  = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        log          = ckpt.get("log", [])
        print(f"Resumed from epoch {start_epoch} (best val loss={best_val_loss:.4f})")

    # ── Training loop ────────────────────────────────────────────────────────
    print(f"\nTraining for {epochs} epochs (lr={lr}, batch={batch_size})\n")
    for epoch in range(start_epoch, start_epoch + epochs):
        t0 = time.time()
        tr = _run_epoch(model, train_loader, optimizer, device, train=True)
        va = _run_epoch(model, val_loader,   optimizer, device, train=False)
        dt = time.time() - t0

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch+1:3d}/{start_epoch+epochs}  "
            f"train loss={tr['loss']:.4f}  "
            f"val loss={va['loss']:.4f}  "
            f"val F1={va['f1']:.3f}  "
            f"val recall={va['recall']:.3f}  "
            f"lr={current_lr:.2e}  "
            f"({dt:.0f}s)"
        )

        scheduler.step(va["loss"])

        entry = {"epoch": epoch + 1, "train": tr, "val": va, "lr": current_lr}
        log.append(entry)

        # Save last checkpoint
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "log": log,
            "model_params": {"in_dim": 9, "out_dim": 3, "n_diff": 2},
        }
        torch.save(ckpt, LAST_PT)

        # Save best checkpoint
        if va["loss"] < best_val_loss:
            best_val_loss = va["loss"]
            ckpt["best_val_loss"] = best_val_loss
            torch.save(ckpt, BEST_PT)
            print(f"  ✓ New best val loss: {best_val_loss:.4f} → saved {BEST_PT}")

    # Save log JSON
    with open(LOG_JSON, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\nTraining complete. Log: {LOG_JSON}")
    print(f"Best model: {BEST_PT}  (val loss={best_val_loss:.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TrackNetV4 training")
    parser.add_argument("--epochs", type=int,   default=50,   help="Training epochs")
    parser.add_argument("--lr",     type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch",  type=int,   default=4,    help="Batch size")
    parser.add_argument("--resume", action="store_true",      help="Resume from last.pt")
    args = parser.parse_args()
    main(epochs=args.epochs, lr=args.lr, batch_size=args.batch, resume=args.resume)
