"""
agent/tools/change_tool.py
Change detection pipeline:
1. Loads the Siamese U-Net (change_map tool)
2. Runs change VQA by feeding both images + change map to the VLM (change_vqa tool)
3. Exports binary change mask as GeoTIFF (georeferenced if CRS available)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)

# ── Device ────────────────────────────────────────────────────────────────────
from agent.tools.vlm_tool import get_device
DEVICE = get_device()


# ── Siamese U-Net with EfficientNet-B0 encoder ───────────────────────────────

class ConvBnRelu(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, padding=kernel // 2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up  = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = ConvBnRelu(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # Handle size mismatch
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class SiameseUNet(nn.Module):
    """
    Siamese U-Net for binary change detection.
    Shared EfficientNet-B0 encoder for both images; difference of features fed to decoder.
    Falls back to a simple CNN if timm not available.
    """

    def __init__(self, in_channels: int = 3, pretrained: bool = True):
        super().__init__()
        try:
            import timm
            self.encoder = timm.create_model(
                "efficientnet_b0",
                pretrained=pretrained,
                features_only=True,
                out_indices=(0, 1, 2, 3, 4),
                in_chans=in_channels,
            )
            enc_channels = self.encoder.feature_info.channels()   # e.g. [16,24,40,112,320]
            self._use_timm = True
        except Exception:
            logger.warning("timm not available; using fallback simple CNN encoder.")
            self.encoder = _SimpleCNNEncoder(in_channels)
            enc_channels = [16, 32, 64, 128, 256]
            self._use_timm = False

        # Decoder: takes absolute diff of siamese features
        self.dec3 = DecoderBlock(enc_channels[4] * 2, enc_channels[3] * 2, 256)
        self.dec2 = DecoderBlock(256, enc_channels[2] * 2, 128)
        self.dec1 = DecoderBlock(128, enc_channels[1] * 2, 64)
        self.dec0 = DecoderBlock(64, enc_channels[0] * 2, 32)

        # Final head → binary change probability
        self.head = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
        )

    def _encode(self, x):
        if self._use_timm:
            return self.encoder(x)
        return self.encoder(x)

    def forward(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        """
        t1, t2: (B, C, H, W) normalized images.
        Returns (B, 1, H, W) change probability map (logits).
        """
        f1 = self._encode(t1)
        f2 = self._encode(t2)

        # Difference features at each scale
        diff = [torch.abs(a - b) for a, b in zip(f1, f2)]
        # Concatenate both at each scale for richer features
        cat  = [torch.cat([a, b], dim=1) for a, b in zip(f1, f2)]

        # Decoder uses concat features
        x = torch.cat([f1[-1], f2[-1]], dim=1)   # bottleneck
        x = self.dec3(x, cat[3])
        x = self.dec2(x, cat[2])
        x = self.dec1(x, cat[1])
        x = self.dec0(x, cat[0])

        # Upsample to input size
        x = F.interpolate(x, size=t1.shape[2:], mode="bilinear", align_corners=False)
        return self.head(x)   # logits


class _SimpleCNNEncoder(nn.Module):
    """Fallback encoder when timm is unavailable."""
    def __init__(self, in_ch: int):
        super().__init__()
        self.stages = nn.ModuleList([
            nn.Sequential(ConvBnRelu(in_ch, 16), nn.MaxPool2d(2)),
            nn.Sequential(ConvBnRelu(16, 32), nn.MaxPool2d(2)),
            nn.Sequential(ConvBnRelu(32, 64), nn.MaxPool2d(2)),
            nn.Sequential(ConvBnRelu(64, 128), nn.MaxPool2d(2)),
            nn.Sequential(ConvBnRelu(128, 256), nn.MaxPool2d(2)),
        ])

    def forward(self, x):
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features


# ── Model loading ─────────────────────────────────────────────────────────────

_change_model_cache: dict = {}


def _load_change_model(weights_path: Optional[str] = None) -> SiameseUNet:
    key = str(weights_path)
    if key not in _change_model_cache:
        model = SiameseUNet(in_channels=3, pretrained=(weights_path is None))
        if weights_path and Path(weights_path).exists():
            ckpt = torch.load(weights_path, map_location=DEVICE)
            state = ckpt.get("model_state_dict", ckpt)
            model.load_state_dict(state, strict=False)
            logger.info("Loaded change model weights: %s", weights_path)
        else:
            logger.warning(
                "No change model weights found at %s — using random weights. "
                "Run training/train_change.py first.", weights_path
            )
        model = model.to(DEVICE).eval()
        _change_model_cache[key] = model
    return _change_model_cache[key]


# ── Inference ─────────────────────────────────────────────────────────────────

def _image_to_tensor(img: Image.Image, size: int = 512) -> torch.Tensor:
    """PIL Image → (1, 3, H, W) tensor, normalized to [0,1]."""
    img = img.resize((size, size), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    arr = arr[:, :, :3].transpose(2, 0, 1)   # (3, H, W)
    return torch.from_numpy(arr).unsqueeze(0)  # (1, 3, H, W)


def run_change_map(
    img_t1: Image.Image,
    img_t2: Image.Image,
    threshold: float = 0.5,
    output_path: Optional[Path] = None,
    meta_t1=None,   # ImageMeta for georeferencing
) -> dict:
    """
    Run the Siamese U-Net change detector.
    Returns binary change mask + stats.
    Saves GeoTIFF if output_path provided and CRS available.
    """
    import os
    weights_path = os.environ.get("SATQUERY_CHANGE_WEIGHTS")
    adapter_status = (
        "LEVIR-CD trained" if weights_path and Path(weights_path).exists()
        else "baseline (not adapted — random weights)"
    )

    model = _load_change_model(weights_path)

    t1 = _image_to_tensor(img_t1).to(DEVICE)
    t2 = _image_to_tensor(img_t2).to(DEVICE)

    with torch.no_grad():
        logits = model(t1, t2)           # (1, 1, H, W)
        probs  = torch.sigmoid(logits)   # (1, 1, H, W)

    prob_map = probs[0, 0].cpu().numpy()   # (H, W) float [0,1]
    binary   = (prob_map >= threshold).astype(np.uint8)

    changed_pct = float(binary.mean() * 100)
    mean_prob   = float(prob_map.mean())

    # Save mask
    mask_path = None
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if meta_t1 and meta_t1.crs:
            _save_as_geotiff(binary, meta_t1, output_path)
        else:
            Image.fromarray(binary * 255).save(output_path.with_suffix(".png"))
            output_path = output_path.with_suffix(".png")
        mask_path = str(output_path)

    return {
        "change_mask_path": mask_path,
        "change_mask_array": binary,  # kept in memory for VQA
        "prob_map": prob_map,
        "changed_area_pct": changed_pct,
        "mean_prob": mean_prob,
        "confidence": mean_prob,        # confidence = mean change probability
        "model_used": "SiameseUNet-EfficientB0",
        "adapter_status": adapter_status,
    }


def run_change_vqa(
    img_t1: Image.Image,
    img_t2: Image.Image,
    change_result: dict,
    query: str,
    model_size: str = "small",
) -> dict:
    """
    VQA over change: creates composite image (T1 | T2 | mask overlay) and asks VLM.
    """
    from agent.tools.vlm_tool import run_vqa

    mask = change_result.get("change_mask_array")
    changed_pct = change_result.get("changed_area_pct", 0.0)

    # Build composite: side-by-side T1, T2, mask overlay
    size = (256, 256)
    t1_r = img_t1.resize(size)
    t2_r = img_t2.resize(size)

    if mask is not None:
        # Red overlay on T2
        mask_r = Image.fromarray(
            (mask * 255).astype(np.uint8), "L"
        ).resize(size)
        overlay = t2_r.copy().convert("RGB")
        red = Image.new("RGB", size, (255, 0, 0))
        overlay = Image.composite(red, overlay, mask_r)
    else:
        overlay = t2_r

    composite = Image.new("RGB", (size[0] * 3, size[1]))
    composite.paste(t1_r, (0, 0))
    composite.paste(t2_r, (size[0], 0))
    composite.paste(overlay, (size[0] * 2, 0))

    augmented_query = (
        f"{query}\n\n"
        f"Context: Left image = T1 (older), Middle = T2 (newer), "
        f"Right = change overlay (red = changed regions). "
        f"Changed area: {changed_pct:.1f}% of the scene."
    )

    vqa_result = run_vqa(composite, augmented_query, model_size=model_size)

    # Generate change description
    desc_query = (
        f"Describe the changes between the left (older) and middle (newer) satellite images. "
        f"Red regions in the right panel indicate detected changes ({changed_pct:.1f}% of scene)."
    )
    desc_result = run_vqa(composite, desc_query, model_size=model_size)

    return {
        "answer": vqa_result["answer"],
        "change_description": desc_result["answer"],
        "changed_area_pct": changed_pct,
        "confidence": vqa_result["confidence"],
        "model_used": vqa_result["model_used"],
        "adapter_status": vqa_result["adapter_status"],
    }


# ── GeoTIFF export helper ─────────────────────────────────────────────────────

def _save_as_geotiff(binary_mask: np.ndarray, meta, out_path: Path) -> None:
    """Save binary change mask as single-band uint8 GeoTIFF with proper georef."""
    try:
        import rasterio
        from rasterio.transform import from_bounds

        with rasterio.open(meta.path) as src:
            profile = src.profile.copy()
            profile.update(count=1, dtype="uint8", nodata=255)

        # Resample mask to source dimensions if needed
        from PIL import Image as PILImage
        if binary_mask.shape != (profile["height"], profile["width"]):
            pil = PILImage.fromarray(binary_mask * 255, "L")
            pil = pil.resize((profile["width"], profile["height"]), PILImage.NEAREST)
            binary_mask = (np.array(pil) > 127).astype(np.uint8)

        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(binary_mask, 1)
        logger.info("Saved change GeoTIFF: %s", out_path)
    except Exception as e:
        logger.warning("GeoTIFF export failed: %s — saving PNG instead.", e)
        out_png = out_path.with_suffix(".png")
        Image.fromarray(binary_mask * 255, "L").save(out_png)
