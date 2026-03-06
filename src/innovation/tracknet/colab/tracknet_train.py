"""
TrackNetV4 training script — Colab-ready version.

All paths default to /content/ (Google Colab working directory).

Usage (in a Colab cell):
    from tracknet_train import train
    train(epochs=30, batch_size=4, lr=5e-4)

Or from the command line:
    python tracknet_train.py --epochs 30 --batch 4 --lr 5e-4

Checkpoints are saved to /content/weights/:
    best.pt  — lowest validation loss
    last.pt  — most recent epoch
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

sys.path.insert(0, '/content')

from tracknet_model   import TrackNetV4, HEIGHT, WIDTH
from tracknet_dataset import make_dataloaders

# ── Default paths (all under /content/) ───────────────────────────────────────
LABEL_PATH = '/content/ball_labels.csv'
VIDEO_DIR  = '/content'
CKPT_DIR   = '/content/weights'
BEST_PT    = '/content/weights/best.pt'
LAST_PT    = '/content/weights/last.pt'
LOG_JSON   = '/content/weights/train_log.json'


# ── Loss ───────────────────────────────────────────────────────────────────────

def weighted_bce(y_pred: torch.Tensor, y: torch.Tensor,
                 eps: float = 1e-7) -> torch.Tensor:
    """Weighted Binary Cross-Entropy from the TrackNet papers.

    Penalises missed balls (false negatives) more than phantom detections.
    """
    loss = -(
        torch.square(1.0 - y_pred) * y * torch.log(y_pred.clamp(eps, 1.0))
        + torch.square(y_pred) * (1.0 - y) * torch.log((1.0 - y_pred).clamp(eps, 1.0))
    )
    return loss.mean()


# ── Metrics ────────────────────────────────────────────────────────────────────

def _tp_fp_fn(y_pred: torch.Tensor, y: torch.Tensor,
              threshold: float = 0.5) -> tuple:
    pred_bin = (y_pred >= threshold).float()
    gt_bin   = (y >= threshold).float()
    tp = int((pred_bin * gt_bin).sum().item())
    fp = int((pred_bin * (1.0 - gt_bin)).sum().item())
    fn = int(((1.0 - pred_bin) * gt_bin).sum().item())
    return tp, fp, fn


# ── One epoch ──────────────────────────────────────────────────────────────────

def _run_epoch(model, loader, optimizer, device, is_train: bool) -> dict:
    model.train(is_train)
    total_loss = 0.0
    tp = fp = fn = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for inp, target in tqdm(loader, leave=False, desc="train" if is_train else "val  "):
            inp    = inp.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            y_pred = model(inp)
            loss   = weighted_bce(y_pred, target)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * inp.size(0)
            b_tp, b_fp, b_fn = _tp_fp_fn(y_pred.detach(), target)
            tp += b_tp; fp += b_fp; fn += b_fn

    n         = len(loader.dataset)
    avg_loss  = total_loss / max(1, n)
    precision = tp / max(1, tp + fp)
    recall    = tp / max(1, tp + fn)
    f1        = 2 * precision * recall / max(1e-8, precision + recall)
    return {"loss": avg_loss, "precision": precision, "recall": recall, "f1": f1}


# ── Main training function ─────────────────────────────────────────────────────

def train(
    label_path: str  = LABEL_PATH,
    video_dir:  str  = VIDEO_DIR,
    out_dir:    str  = CKPT_DIR,
    epochs:     int  = 30,
    lr:         float = 5e-4,
    batch_size: int  = 4,
    resume:     bool = False,
):
    """Train TrackNetV4 and save best.pt / last.pt to out_dir.

    Parameters
    ----------
    label_path : path to ball_labels.csv
    video_dir  : directory containing video file(s)
    out_dir    : where to save checkpoints
    epochs     : number of training epochs
    lr         : learning rate
    batch_size : samples per batch
    resume     : if True, resume from out_dir/last.pt
    """
    best_pt  = os.path.join(out_dir, 'best.pt')
    last_pt  = os.path.join(out_dir, 'last.pt')
    log_json = os.path.join(out_dir, 'train_log.json')

    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if not os.path.exists(label_path):
        print(f"[Error] Label file not found: {label_path}")
        return

    train_loader, val_loader = make_dataloaders(
        label_path=label_path,
        video_dir=video_dir,
        batch_size=batch_size,
        num_workers=0,
    )
    print(f"Train: {len(train_loader.dataset)} samples  |  Val: {len(val_loader.dataset)} samples")

    if len(train_loader.dataset) < 10:
        print("[Warning] Very few training samples — label more frames for good results.")

    model = TrackNetV4().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"TrackNetV4 parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    start_epoch   = 0
    best_val_loss = float("inf")
    log: list     = []

    if resume and os.path.exists(last_pt):
        ckpt = torch.load(last_pt, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch   = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        log           = ckpt.get("log", [])
        print(f"Resumed from epoch {start_epoch}  (best val loss={best_val_loss:.4f})")

    print(f"\nTraining for {epochs} epochs  (lr={lr}, batch={batch_size})\n")

    for epoch in range(start_epoch, start_epoch + epochs):
        t0 = time.time()
        tr = _run_epoch(model, train_loader, optimizer, device, is_train=True)
        va = _run_epoch(model, val_loader,   optimizer, device, is_train=False)
        dt = time.time() - t0

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch+1:3d}/{start_epoch+epochs}  "
            f"train={tr['loss']:.4f}  val={va['loss']:.4f}  "
            f"F1={va['f1']:.3f}  recall={va['recall']:.3f}  "
            f"lr={current_lr:.2e}  ({dt:.0f}s)"
        )

        scheduler.step(va["loss"])

        ckpt = {
            "epoch":        epoch,
            "model":        model.state_dict(),
            "optimizer":    optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "log":          log,
            "model_params": {"in_dim": 9, "out_dim": 3, "n_diff": 2},
        }
        torch.save(ckpt, last_pt)

        if va["loss"] < best_val_loss:
            best_val_loss = va["loss"]
            ckpt["best_val_loss"] = best_val_loss
            torch.save(ckpt, best_pt)
            print(f"  -> New best val loss: {best_val_loss:.4f}  saved to {best_pt}")

        log.append({"epoch": epoch + 1, "train": tr, "val": va, "lr": current_lr})

    with open(log_json, "w") as f:
        json.dump(log, f, indent=2)

    print(f"\nDone. Best model: {best_pt}  (val loss={best_val_loss:.4f})")
    print(f"Log:  {log_json}")


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TrackNetV4 training")
    parser.add_argument("--label_path", default=LABEL_PATH)
    parser.add_argument("--video_dir",  default=VIDEO_DIR)
    parser.add_argument("--out_dir",    default=CKPT_DIR)
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--lr",         type=float, default=5e-4)
    parser.add_argument("--batch",      type=int,   default=4)
    parser.add_argument("--resume",     action="store_true")
    args = parser.parse_args()
    train(
        label_path=args.label_path,
        video_dir=args.video_dir,
        out_dir=args.out_dir,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch,
        resume=args.resume,
    )
