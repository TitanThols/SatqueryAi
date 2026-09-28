"""
agent/tools/sar_optical_tool.py
Dual-stream optical+SAR fusion model for land cover classification.
Architecture: ResNet-18 dual-branch, BigEarthNet 19-class multilabel.
Confidence: mean max-class probability per pixel (semantic mask) or per image (classification).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)
from agent.tools.vlm_tool import get_device
DEVICE = get_device()

# BigEarthNet 19-class labels (compatible with v1.0 and v2.0 19-class split)
BIGEARTH_CLASSES_19 = [
    "Urban fabric",
    "Industrial or commercial units",
    "Arable land",
    "Permanent crops",
    "Pastures",
    "Complex cultivation patterns",
    "Agriculture + natural vegetation",
    "Agro-forestry areas",
    "Broad-leaved forest",
    "Coniferous forest",
    "Mixed forest",
    "Natural grasslands",
    "Moors and heathland",
    "Sclerophyllous vegetation",
    "Transitional woodland-shrub",
    "Beaches, dunes, sands",
    "Inland wetlands",
    "Water bodies",
    "Bare rocks and sparse vegetation",
]

# Simplified 4-class mapping for easier reporting
CLASS_MAP_4 = {
    "built-up":    ["Urban fabric", "Industrial or commercial units"],
    "vegetation":  ["Broad-leaved forest", "Coniferous forest", "Mixed forest",
                    "Natural grasslands", "Moors and heathland", "Sclerophyllous vegetation",
                    "Transitional woodland-shrub", "Agro-forestry areas", "Pastures",
                    "Complex cultivation patterns", "Agriculture + natural vegetation",
                    "Permanent crops", "Arable land"],
    "water":       ["Water bodies", "Inland wetlands"],
    "bare/other":  ["Beaches, dunes, sands", "Bare rocks and sparse vegetation"],
}


# ── Dual-stream ResNet-18 ─────────────────────────────────────────────────────

class _ResNetStream(nn.Module):
    """Single-stream ResNet-18 feature extractor."""

    def __init__(self, in_channels: int, pretrained: bool = True):
        super().__init__()
        try:
            import timm
            self.backbone = timm.create_model(
                "resnet18",
                pretrained=pretrained,
                num_classes=0,            # remove classifier
                global_pool="avg",
                in_chans=in_channels,
            )
            self.out_dim = 512
        except Exception:
            # Fallback minimal CNN
            self.backbone = nn.Sequential(
                nn.Conv2d(in_channels, 64, 7, stride=2, padding=3), nn.ReLU(),
                nn.MaxPool2d(3, stride=2, padding=1),
                nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
            )
            self.out_dim = 128

    def forward(self, x):
        return self.backbone(x)


class DualStreamFusion(nn.Module):
    """
    Dual-stream fusion model.
    optical_stream: processes S2 / optical bands.
    sar_stream:     processes S1 / SAR bands.
    Fusion: concatenation → FC head → 19-class multilabel sigmoid.
    """

    def __init__(
        self,
        optical_in: int = 4,   # S2: B2,B3,B4,B8 (or 3 for RGB)
        sar_in:     int = 2,   # S1: VV, VH
        n_classes:  int = 19,
        pretrained: bool = True,
    ):
        super().__init__()
        self.optical_stream = _ResNetStream(optical_in, pretrained)
        self.sar_stream     = _ResNetStream(sar_in, pretrained=False)  # no pretrained for SAR

        fuse_dim = self.optical_stream.out_dim + self.sar_stream.out_dim
        self.fusion_head = nn.Sequential(
            nn.Linear(fuse_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes),
        )

    def forward(self, optical: torch.Tensor, sar: torch.Tensor):
        """
        optical: (B, optical_in, H, W)
        sar:     (B, sar_in, H, W)
        Returns: (B, n_classes) logits (before sigmoid)
        """
        f_opt = self.optical_stream(optical)   # (B, 512)
        f_sar = self.sar_stream(sar)           # (B, 512)
        fused = torch.cat([f_opt, f_sar], dim=1)
        return self.fusion_head(fused)

    def forward_optical_only(self, optical: torch.Tensor):
        """Optical-only baseline path."""
        f_opt = self.optical_stream(optical)
        dummy_sar = torch.zeros(f_opt.shape[0], self.sar_stream.out_dim, device=f_opt.device)
        fused = torch.cat([f_opt, dummy_sar], dim=1)
        return self.fusion_head(fused)

    def forward_sar_only(self, sar: torch.Tensor):
        """SAR-only baseline path."""
        f_sar = self.sar_stream(sar)
        dummy_opt = torch.zeros(f_sar.shape[0], self.optical_stream.out_dim, device=f_sar.device)
        fused = torch.cat([dummy_opt, f_sar], dim=1)
        return self.fusion_head(fused)


# ── Model loading ─────────────────────────────────────────────────────────────

_fusion_model_cache: dict = {}


def _load_fusion_model(weights_path: Optional[str] = None) -> DualStreamFusion:
    key = str(weights_path)
    if key not in _fusion_model_cache:
        model = DualStreamFusion(optical_in=4, sar_in=2, n_classes=19, pretrained=True)
        if weights_path and Path(weights_path).exists():
            ckpt = torch.load(weights_path, map_location=DEVICE)
            state = ckpt.get("model_state_dict", ckpt)
            model.load_state_dict(state, strict=False)
            logger.info("Loaded fusion model weights: %s", weights_path)
        else:
            logger.warning(
                "No fusion model weights at %s — using ImageNet-pretrained optical stream only. "
                "Run training/train_fusion.py first.", weights_path
            )
        model = model.to(DEVICE).eval()
        _fusion_model_cache[key] = model
    return _fusion_model_cache[key]


# ── Inference ──────────────────────────────────────────────────────────────────

def _prep_optical(img: Image.Image, target_size: int = 120) -> torch.Tensor:
    """PIL RGB → (1, 4, H, W) with zero-filled NIR channel."""
    img = img.resize((target_size, target_size), Image.LANCZOS).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0  # (H, W, 3)
    # Append a synthetic NIR ≈ R channel (coarse approximation when NIR missing)
    nir = arr[:, :, 0:1]
    arr = np.concatenate([arr, nir], axis=-1)       # (H, W, 4)
    arr = arr.transpose(2, 0, 1)                    # (4, H, W)
    return torch.from_numpy(arr).unsqueeze(0)       # (1, 4, H, W)


def _prep_sar(img: Image.Image, target_size: int = 120) -> torch.Tensor:
    """PIL greyscale or RGB → (1, 2, H, W) dual-pol approximation."""
    img = img.resize((target_size, target_size), Image.LANCZOS)
    arr = np.array(img.convert("L"), dtype=np.float32) / 255.0  # (H, W)
    # Approximate VV and VH as first two channels
    vv = arr
    vh = arr * 0.7  # coarse approximation when dual-pol not available
    arr = np.stack([vv, vh], axis=0)                # (2, H, W)
    return torch.from_numpy(arr).unsqueeze(0)       # (1, 2, H, W)


def _logits_to_class_probs(logits: torch.Tensor) -> dict[str, float]:
    """Convert 19-class logits → probability per simplified class."""
    probs = torch.sigmoid(logits[0]).cpu().numpy()  # (19,)
    result = {}
    for cls4, cls19_list in CLASS_MAP_4.items():
        indices = [BIGEARTH_CLASSES_19.index(c) for c in cls19_list if c in BIGEARTH_CLASSES_19]
        if indices:
            result[cls4] = float(np.max(probs[indices]))
        else:
            result[cls4] = 0.0
    return result


def run_sar_optical_fusion(
    optical_img: Image.Image,
    sar_img: Image.Image,
    output_path: Optional[Path] = None,
    meta_optical=None,
) -> dict:
    """
    Run dual-stream fusion and return land-cover class probabilities.
    Also runs optical-only and SAR-only baselines for comparison.
    """
    weights_path = os.environ.get("SATQUERY_FUSION_WEIGHTS")
    adapter_status = (
        "BigEarthNet-trained (10% subset)"
        if weights_path and Path(weights_path).exists()
        else "baseline (not adapted — ImageNet pretrained optical only)"
    )

    model = _load_fusion_model(weights_path)

    opt_t = _prep_optical(optical_img).to(DEVICE)
    sar_t = _prep_sar(sar_img).to(DEVICE)

    with torch.no_grad():
        logits_fused   = model(opt_t, sar_t)
        logits_opt     = model.forward_optical_only(opt_t)
        logits_sar     = model.forward_sar_only(sar_t)

    class_probs_fused = _logits_to_class_probs(logits_fused)
    class_probs_opt   = _logits_to_class_probs(logits_opt)
    class_probs_sar   = _logits_to_class_probs(logits_sar)

    mean_prob = float(torch.sigmoid(logits_fused).max(dim=1).values.mean().item())

    # Generate simple colour mask for visualization
    mask_path = None
    if output_path:
        mask_img = _generate_class_overlay(optical_img, class_probs_fused)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mask_img.save(output_path.with_suffix(".png"))
        mask_path = str(output_path.with_suffix(".png"))

    return {
        "class_areas_pct": class_probs_fused,
        "class_areas_pct_optical_only": class_probs_opt,
        "class_areas_pct_sar_only": class_probs_sar,
        "class_mask_path": mask_path,
        "mean_prob": mean_prob,
        "confidence": mean_prob,
        "model_used": "DualStreamResNet18",
        "adapter_status": adapter_status,
        "answer": _format_land_cover_summary(class_probs_fused),
    }


def _format_land_cover_summary(class_probs: dict[str, float]) -> str:
    """Convert class probabilities to a natural-language summary."""
    dominant = sorted(class_probs.items(), key=lambda x: -x[1])
    lines = ["Land cover analysis (optical+SAR fusion):"]
    for cls, prob in dominant:
        if prob > 0.1:
            lines.append(f"  • {cls}: {prob * 100:.1f}% confidence")
    if not any(v > 0.1 for v in class_probs.values()):
        lines.append("  No dominant land class detected (confidence low).")
    return "\n".join(lines)


def _generate_class_overlay(base_img: Image.Image, class_probs: dict) -> Image.Image:
    """Create a colour-coded overlay on the base image."""
    COLOR_MAP = {
        "built-up":    (231, 76,  60),   # red
        "vegetation":  (46,  204, 113),  # green
        "water":       (52,  152, 219),  # blue
        "bare/other":  (243, 156, 18),   # orange
    }
    base = base_img.resize((512, 512)).convert("RGBA")
    overlay = Image.new("RGBA", (512, 512), (0, 0, 0, 0))

    # Simple block representation — one dominant class
    dominant = max(class_probs.items(), key=lambda x: x[1])
    dom_cls, dom_prob = dominant
    color = COLOR_MAP.get(dom_cls, (127, 127, 127))
    alpha = int(dom_prob * 120)   # transparency proportional to confidence

    color_layer = Image.new("RGBA", (512, 512), (*color, alpha))
    overlay = Image.alpha_composite(overlay, color_layer)
    result = Image.alpha_composite(base, overlay)
    return result.convert("RGB")
