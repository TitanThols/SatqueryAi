"""
training/train_change.py
Trains Siamese U-Net on LEVIR-CD dataset.
Standalone, self-contained, resumable. Seeded. Config-driven via CLI or config file.
Expected runtime: ~2-3 hours on Kaggle/Colab T4.
Output: checkpoints/change_model_best.pt  (load into SiameseUNet)
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Config ────────────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "data_root":   "./data/LEVIR-CD",
    "out_dir":     "./checkpoints",
    "img_size":    256,
    "batch_size":  8,
    "lr":          1e-4,
    "epochs":      50,
    "seed":        42,
    "resume":      None,         # path to checkpoint to resume from
    "val_every":   1,
    "save_every":  5,
    "grad_clip":   1.0,
    "device":      "auto",       # auto | cuda | mps | cpu
    "num_workers": 2,
}


# ── Dataset ────────────────────────────────────────────────────────────────────

class LEVIRDataset(Dataset):
    """
    LEVIR-CD dataset loader.
    Expected structure:
      data_root/
        train/
          A/   (T1 images, PNG)
          B/   (T2 images, PNG)
          label/ (change masks, PNG, 0=no-change, 255=change)
        val/
          A/ B/ label/
        test/
          A/ B/ label/
    """

    def __init__(self, root: Path, split: str, img_size: int = 256, augment: bool = True):
        self.root     = root / split
        self.img_size = img_size
        self.augment  = augment

        self.a_dir    = self.root / "A"
        self.b_dir    = self.root / "B"
        self.lbl_dir  = self.root / "label"

        if not self.a_dir.exists():
            raise FileNotFoundError(
                f"LEVIR-CD split '{split}' not found at {self.root}. "
                f"Download from: https://huggingface.co/datasets/blanchon/LEVIR_CDPlus\n"
                f"or: kaggle datasets download -d levir-cd"
            )

        self.samples = sorted(self.a_dir.glob("*.png"))
        logger.info("LEVIR-CD %s: %d samples", split, len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        name = self.samples[idx].name
        img_a = Image.open(self.a_dir / name).convert("RGB")
        img_b = Image.open(self.b_dir / name).convert("RGB")
        mask  = Image.open(self.lbl_dir / name).convert("L")

        # Resize
        img_a = img_a.resize((self.img_size, self.img_size), Image.BILINEAR)
        img_b = img_b.resize((self.img_size, self.img_size), Image.BILINEAR)
        mask  = mask.resize((self.img_size, self.img_size), Image.NEAREST)

        # Augmentation
        if self.augment:
            if random.random() > 0.5:
                img_a = img_a.transpose(Image.FLIP_LEFT_RIGHT)
                img_b = img_b.transpose(Image.FLIP_LEFT_RIGHT)
                mask  = mask.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() > 0.5:
                img_a = img_a.transpose(Image.FLIP_TOP_BOTTOM)
                img_b = img_b.transpose(Image.FLIP_TOP_BOTTOM)
                mask  = mask.transpose(Image.FLIP_TOP_BOTTOM)
            # Random 90° rotation
            k = random.randint(0, 3)
            img_a = img_a.rotate(k * 90)
            img_b = img_b.rotate(k * 90)
            mask  = mask.rotate(k * 90)

        # To tensor
        t_a  = torch.from_numpy(np.array(img_a, np.float32).transpose(2, 0, 1) / 255.0)
        t_b  = torch.from_numpy(np.array(img_b, np.float32).transpose(2, 0, 1) / 255.0)
        lbl  = torch.from_numpy((np.array(mask) > 127).astype(np.float32)).unsqueeze(0)

        return t_a, t_b, lbl


# ── Metrics ───────────────────────────────────────────────────────────────────

class BinaryMetrics:
    def __init__(self):
        self.tp = self.fp = self.fn = self.tn = 0.0

    def update(self, pred_logits: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5):
        pred = (torch.sigmoid(pred_logits) >= threshold).float()
        self.tp += (pred * labels).sum().item()
        self.fp += (pred * (1 - labels)).sum().item()
        self.fn += ((1 - pred) * labels).sum().item()
        self.tn += ((1 - pred) * (1 - labels)).sum().item()

    def precision(self):
        return self.tp / (self.tp + self.fp + 1e-8)

    def recall(self):
        return self.tp / (self.tp + self.fn + 1e-8)

    def f1(self):
        p, r = self.precision(), self.recall()
        return 2 * p * r / (p + r + 1e-8)

    def iou(self):
        return self.tp / (self.tp + self.fp + self.fn + 1e-8)


# ── Loss ──────────────────────────────────────────────────────────────────────

class DiceBCELoss(nn.Module):
    def __init__(self, dice_weight: float = 0.5):
        super().__init__()
        self.dice_w = dice_weight
        self.bce    = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        bce_loss = self.bce(logits, labels)
        probs    = torch.sigmoid(logits)
        inter    = (probs * labels).sum(dim=(2, 3))
        union    = probs.sum(dim=(2, 3)) + labels.sum(dim=(2, 3))
        dice     = 1 - (2 * inter + 1) / (union + 1)
        dice_loss = dice.mean()
        return (1 - self.dice_w) * bce_loss + self.dice_w * dice_loss


# ── Training loop ─────────────────────────────────────────────────────────────

def train(cfg: dict):
    # Seed
    seed = cfg["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Device
    if cfg["device"] == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else
                              "mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(cfg["device"])
    logger.info("Device: %s", device)

    # Data
    data_root = Path(cfg["data_root"])
    train_ds = LEVIRDataset(data_root, "train", cfg["img_size"], augment=True)
    val_ds   = LEVIRDataset(data_root, "val",   cfg["img_size"], augment=False)

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                              num_workers=cfg["num_workers"], pin_memory=device.type == "cuda")
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"], shuffle=False,
                              num_workers=cfg["num_workers"])

    # Model
    from agent.tools.change_tool import SiameseUNet
    model = SiameseUNet(in_channels=3, pretrained=True).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
    criterion = DiceBCELoss(dice_weight=0.5)

    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 0
    best_f1     = 0.0
    log_entries = []

    # Resume
    if cfg.get("resume") and Path(cfg["resume"]).exists():
        ckpt = torch.load(cfg["resume"], map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_f1     = ckpt.get("best_f1", 0.0)
        log_entries = ckpt.get("log", [])
        logger.info("Resumed from epoch %d (best_f1=%.4f)", start_epoch, best_f1)

    for epoch in range(start_epoch, cfg["epochs"]):
        t0 = time.time()

        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for batch_idx, (t1, t2, lbl) in enumerate(train_loader):
            t1, t2, lbl = t1.to(device), t2.to(device), lbl.to(device)
            optimizer.zero_grad()
            logits = model(t1, t2)
            loss   = criterion(logits, lbl)
            loss.backward()
            if cfg["grad_clip"] > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            optimizer.step()
            train_loss += loss.item()

            if batch_idx % 20 == 0:
                logger.info("Epoch %d [%d/%d] loss=%.4f", epoch, batch_idx, len(train_loader), loss.item())

        scheduler.step()
        train_loss /= len(train_loader)

        # ── Validate ──────────────────────────────────────────────────────────
        if (epoch + 1) % cfg["val_every"] == 0:
            model.eval()
            metrics = BinaryMetrics()
            val_loss = 0.0
            with torch.no_grad():
                for t1, t2, lbl in val_loader:
                    t1, t2, lbl = t1.to(device), t2.to(device), lbl.to(device)
                    logits = model(t1, t2)
                    val_loss += criterion(logits, lbl).item()
                    metrics.update(logits, lbl)

            val_loss /= len(val_loader)
            f1  = metrics.f1()
            iou = metrics.iou()
            elapsed = time.time() - t0

            logger.info(
                "Epoch %d | train_loss=%.4f | val_loss=%.4f | F1=%.4f | IoU=%.4f | %.1fs",
                epoch, train_loss, val_loss, f1, iou, elapsed,
            )

            entry = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "f1": f1,
                "iou": iou,
                "elapsed_s": elapsed,
            }
            log_entries.append(entry)

            # Save best
            if f1 > best_f1:
                best_f1 = f1
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_f1": best_f1,
                    "log": log_entries,
                    "config": cfg,
                }, out_dir / "change_model_best.pt")
                logger.info("✓ Best model saved (F1=%.4f)", best_f1)

        # Periodic checkpoint
        if (epoch + 1) % cfg["save_every"] == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_f1": best_f1,
                "log": log_entries,
                "config": cfg,
            }, out_dir / f"change_model_epoch{epoch:03d}.pt")

    # Save log
    log_path = out_dir / "change_training_log.json"
    log_path.write_text(json.dumps(log_entries, indent=2))
    logger.info("Training complete. Log: %s", log_path)
    logger.info("Best F1 on val: %.4f", best_f1)
    logger.info("Best model: %s", out_dir / "change_model_best.pt")
    logger.info(
        "\n\n══ REMOTE GPU STEP COMPLETE ══\n"
        "Copy to Mac:  scp user@gpu-box:./checkpoints/change_model_best.pt ./checkpoints/\n"
        "Then set:     export SATQUERY_CHANGE_WEIGHTS=./checkpoints/change_model_best.pt\n"
    )


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train Siamese U-Net on LEVIR-CD")
    for k, v in DEFAULT_CONFIG.items():
        t = type(v) if v is not None else str
        p.add_argument(f"--{k.replace('_', '-')}", type=t, default=v)
    p.add_argument("--config", type=str, default=None, help="JSON config file path")
    return p.parse_args()


if __name__ == "__main__":
    args   = parse_args()
    cfg    = DEFAULT_CONFIG.copy()
    if args.config and Path(args.config).exists():
        cfg.update(json.loads(Path(args.config).read_text()))
    # Override with CLI args
    for k in DEFAULT_CONFIG:
        v = getattr(args, k, None)
        if v is not None:
            cfg[k] = v
    train(cfg)
