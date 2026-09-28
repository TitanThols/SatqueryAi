"""
training/train_fusion.py
Trains Dual-Stream ResNet-18 on BigEarthNet v2.0 (10% stratified subset).
Standalone, seeded, resumable.
Expected runtime: ~3-5 hours on Colab/Kaggle T4.
Output: checkpoints/fusion_model_best.pt
"""

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))

DEFAULT_CONFIG = {
    "data_root":    "./data/BigEarthNet",
    "out_dir":      "./checkpoints",
    "img_size":     120,
    "batch_size":   64,
    "lr":           3e-4,
    "epochs":       30,
    "seed":         42,
    "resume":       None,
    "val_every":    1,
    "save_every":   5,
    "subset_frac":  0.10,     # use 10% of BigEarthNet
    "n_classes":    19,
    "device":       "auto",
    "num_workers":  4,
}


# ── BigEarthNet Dataset ───────────────────────────────────────────────────────

class BigEarthNetDataset(torch.utils.data.Dataset):
    """
    BigEarthNet v2.0 loader.
    Uses torchgeo or local GeoTIFF patches.

    Structure option 1 (torchgeo):
      torchgeo will download and cache automatically.

    Structure option 2 (local):
      data_root/
        BigEarthNet-S2/  (optical patches, each dir = one patch)
        BigEarthNet-S1/  (SAR patches)
        metadata.parquet (v2.0 metadata with splits + labels)
    """

    def __init__(self, root: Path, split: str, img_size: int, subset_frac: float = 0.1):
        self.img_size    = img_size
        self.subset_frac = subset_frac

        # Try torchgeo first
        try:
            from torchgeo.datasets import BigEarthNet
            logger.info("Using torchgeo BigEarthNet (split=%s, subset=%.0f%%)", split, subset_frac * 100)
            self._use_torchgeo = True
            self._tg_dataset = BigEarthNet(
                root=str(root),
                split=split,
                bands="all",
                num_classes=19,
                download=True,
            )
            # Subsample
            n = len(self._tg_dataset)
            k = max(1, int(n * subset_frac))
            indices = random.sample(range(n), k)
            self._indices = indices
            logger.info("BigEarthNet torchgeo %s: %d/%d samples", split, k, n)
        except Exception as e:
            logger.warning("torchgeo not available or download failed: %s", e)
            logger.warning("Falling back to local GeoTIFF loading.")
            self._use_torchgeo = False
            self._load_local(root, split, subset_frac)

    def _load_local(self, root: Path, split: str, subset_frac: float):
        s2_root = root / "BigEarthNet-S2"
        s1_root = root / "BigEarthNet-S1"
        meta_path = root / "metadata.parquet"

        if not meta_path.exists():
            raise FileNotFoundError(
                f"metadata.parquet not found at {meta_path}.\n"
                f"Download BigEarthNet v2.0 from https://bigearth.net/\n"
                f"Or use torchgeo: pip install torchgeo"
            )

        import pandas as pd
        meta = pd.read_parquet(meta_path)
        meta = meta[meta["split"] == split]

        # Subsample
        n = len(meta)
        k = max(1, int(n * subset_frac))
        meta = meta.sample(k, random_state=42)

        self._samples = []
        for _, row in meta.iterrows():
            patch = row["patch_id"]
            s2_dir = s2_root / patch
            s1_dir = s1_root / patch.replace("S2", "S1") if "S2" in patch else s1_root / patch
            labels = [int(x) for x in str(row.get("labels", "[]")).strip("[]").split(",") if x.strip().isdigit()]
            self._samples.append({"s2_dir": s2_dir, "s1_dir": s1_dir, "labels": labels})

        logger.info("BigEarthNet local %s: %d/%d samples", split, k, n)

    def _multilabel_tensor(self, labels: list[int], n: int = 19) -> torch.Tensor:
        t = torch.zeros(n)
        for l in labels:
            if 0 <= l < n:
                t[l] = 1.0
        return t

    def __len__(self):
        if self._use_torchgeo:
            return len(self._indices)
        return len(self._samples)

    def __getitem__(self, idx):
        if self._use_torchgeo:
            item = self._tg_dataset[self._indices[idx]]
            # torchgeo returns {"image": (C, H, W), "label": tensor}
            img = item["image"].float()
            # Split S2 (13 bands) and S1 (2 bands) — BigEarthNet "all" = S1(2) + S2(13)
            if img.shape[0] >= 15:
                s1 = img[:2]         # VV, VH
                s2 = img[2:6]        # B2,B3,B4,B8 (indices 2-5 in concatenated)
            elif img.shape[0] == 13:
                s2 = img[[1, 2, 3, 7]]   # B2,B3,B4,B8 → indices 1,2,3,7
                s1 = torch.zeros(2, img.shape[1], img.shape[2])
            else:
                s2 = img[:4] if img.shape[0] >= 4 else torch.cat([img, img[:1]], dim=0)[:4]
                s1 = torch.zeros(2, img.shape[1], img.shape[2])
            labels = item["label"].float()
            return s2, s1, labels

        # Local GeoTIFF loading
        sample = self._samples[idx]
        s2 = self._read_s2_geotiff(Path(sample["s2_dir"]))
        s1 = self._read_s1_geotiff(Path(sample["s1_dir"]))
        labels = self._multilabel_tensor(sample["labels"])
        return s2, s1, labels

    def _read_s2_geotiff(self, patch_dir: Path) -> torch.Tensor:
        """Read 4 S2 bands (B2,B3,B4,B8) from patch directory."""
        try:
            import rasterio
            band_files = sorted(patch_dir.glob("*_B*.tif"))
            bands_of_interest = ["B02", "B03", "B04", "B08"]
            arrays = []
            for bf in band_files:
                for b in bands_of_interest:
                    if b in bf.name:
                        with rasterio.open(bf) as ds:
                            arr = ds.read(1).astype(np.float32) / 10000.0
                            arrays.append(arr)
                        break
            if not arrays:
                return torch.zeros(4, self.img_size, self.img_size)
            # Stack and resize
            stacked = np.stack(arrays[:4], axis=0)   # (4, H, W)
            if stacked.shape[1] != self.img_size:
                from PIL import Image
                out = []
                for c in range(stacked.shape[0]):
                    ch = (stacked[c] * 255).clip(0, 255).astype(np.uint8)
                    pil = Image.fromarray(ch).resize((self.img_size, self.img_size))
                    out.append(np.array(pil) / 255.0)
                stacked = np.stack(out, axis=0).astype(np.float32)
            return torch.from_numpy(stacked)
        except Exception:
            return torch.zeros(4, self.img_size, self.img_size)

    def _read_s1_geotiff(self, patch_dir: Path) -> torch.Tensor:
        """Read S1 VV+VH from patch directory."""
        try:
            import rasterio
            vv_file = next(patch_dir.glob("*VV*.tif"), None)
            vh_file = next(patch_dir.glob("*VH*.tif"), None)
            arrs = []
            for f in [vv_file, vh_file]:
                if f and f.exists():
                    with rasterio.open(f) as ds:
                        arr = ds.read(1).astype(np.float32)
                        # dB conversion
                        arr = np.clip(10 * np.log10(np.maximum(arr, 1e-10)), -25, 0) / 25.0 + 1.0
                        arrs.append(arr)
            if not arrs:
                return torch.zeros(2, self.img_size, self.img_size)
            while len(arrs) < 2:
                arrs.append(arrs[-1] * 0.7)
            stacked = np.stack(arrs[:2], axis=0)
            return torch.from_numpy(stacked)
        except Exception:
            return torch.zeros(2, self.img_size, self.img_size)


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_multilabel_f1(preds: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5) -> dict:
    preds  = (torch.sigmoid(preds) >= threshold).float()
    tp = (preds * labels).sum(0)
    fp = (preds * (1 - labels)).sum(0)
    fn = ((1 - preds) * labels).sum(0)

    precision = (tp / (tp + fp + 1e-8)).mean().item()
    recall    = (tp / (tp + fn + 1e-8)).mean().item()
    f1_macro  = (2 * tp / (2 * tp + fp + fn + 1e-8)).mean().item()
    return {"macro_f1": f1_macro, "precision": precision, "recall": recall}


# ── Training ──────────────────────────────────────────────────────────────────

def train(cfg: dict):
    seed = cfg["seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    if cfg["device"] == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else
                              "mps"  if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(cfg["device"])
    logger.info("Device: %s", device)

    data_root = Path(cfg["data_root"])
    train_ds  = BigEarthNetDataset(data_root, "train", cfg["img_size"], cfg["subset_frac"])
    val_ds    = BigEarthNetDataset(data_root, "val",   cfg["img_size"], cfg["subset_frac"])

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=cfg["num_workers"], pin_memory=device.type == "cuda",
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=cfg["num_workers"],
    )

    from agent.tools.sar_optical_tool import DualStreamFusion
    model = DualStreamFusion(optical_in=4, sar_in=2, n_classes=cfg["n_classes"], pretrained=True)
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])

    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    best_f1 = 0.0
    log_entries = []
    start_epoch = 0

    if cfg.get("resume") and Path(cfg["resume"]).exists():
        ckpt = torch.load(cfg["resume"], map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_f1     = ckpt.get("best_f1", 0.0)
        log_entries = ckpt.get("log", [])
        logger.info("Resumed from epoch %d", start_epoch)

    for epoch in range(start_epoch, cfg["epochs"]):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for s2, s1, labels in train_loader:
            s2, s1, labels = s2.to(device), s1.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(s2, s1)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        epoch_loss /= len(train_loader)

        if (epoch + 1) % cfg["val_every"] == 0:
            model.eval()
            all_logits, all_labels = [], []
            val_loss = 0.0
            with torch.no_grad():
                for s2, s1, labels in val_loader:
                    s2, s1, labels = s2.to(device), s1.to(device), labels.to(device)
                    logits = model(s2, s1)
                    val_loss += criterion(logits, labels).item()
                    all_logits.append(logits.cpu())
                    all_labels.append(labels.cpu())

            val_loss /= len(val_loader)
            all_logits = torch.cat(all_logits, 0)
            all_labels = torch.cat(all_labels, 0)
            m = compute_multilabel_f1(all_logits, all_labels)

            logger.info(
                "Epoch %d | train_loss=%.4f | val_loss=%.4f | macro_F1=%.4f | %.1fs",
                epoch, epoch_loss, val_loss, m["macro_f1"], time.time() - t0,
            )

            entry = {"epoch": epoch, "train_loss": epoch_loss,
                     "val_loss": val_loss, **m}
            log_entries.append(entry)

            if m["macro_f1"] > best_f1:
                best_f1 = m["macro_f1"]
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_f1": best_f1,
                    "log": log_entries,
                    "config": cfg,
                }, out_dir / "fusion_model_best.pt")
                logger.info("✓ Best fusion model saved (macro_F1=%.4f)", best_f1)

    (out_dir / "fusion_training_log.json").write_text(json.dumps(log_entries, indent=2))
    logger.info("Best macro-F1 on val: %.4f", best_f1)
    logger.info(
        "\n\n══ FUSION TRAINING COMPLETE ══\n"
        "Copy to Mac:  scp user@gpu-box:./checkpoints/fusion_model_best.pt ./checkpoints/\n"
        "Then set:     export SATQUERY_FUSION_WEIGHTS=./checkpoints/fusion_model_best.pt\n"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for k, v in DEFAULT_CONFIG.items():
        t = type(v) if v is not None else str
        p.add_argument(f"--{k.replace('_','-')}", type=t, default=v)
    p.add_argument("--config", type=str, default=None)
    args = p.parse_args()

    cfg = DEFAULT_CONFIG.copy()
    if args.config and Path(args.config).exists():
        cfg.update(json.loads(Path(args.config).read_text()))
    for k in DEFAULT_CONFIG:
        v = getattr(args, k.replace("-","_"), None)
        if v is not None:
            cfg[k] = v

    train(cfg)
