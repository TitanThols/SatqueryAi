`"""
agent/validators/input_validator.py
Validates uploaded files and query against task requirements.
Returns a structured ValidationResult – never raises directly, always returns.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ── Enums ─────────────────────────────────────────────────────────────────────

class Modality(str, Enum):
    OPTICAL = "optical"
    SAR     = "sar"
    UNKNOWN = "unknown


class ImageRole(str, Enum):
    SINGLE  = "single"
    T1      = "t1"        # temporal 1 (older)
    T2      = "t2"        # temporal 2 (newer)
    OPTICAL = "optical"   # for optical/SAR pair
    SAR     = "sar"


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class ImageMeta:
    path:     Path
    role:     ImageRole
    modality: Modality
    width:    int
    height:   int
    bands:    int
    crs:      Optional[str]
    res_x:    Optional[float]   # metres / pixel (approx)
    res_y:    Optional[float]
    dtype:    str
    nodata:   Optional[float]


@dataclass
class ValidationResult:
    ok:      bool
    errors:  list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metas:   list[ImageMeta] = field(default_factory=list)

    def fail(self, msg: str) -> "ValidationResult":
        self.ok = False
        self.errors.append(msg)
        return self

    def warn(self, msg: str) -> "ValidationResult":
        self.warnings.append(msg)
        return self


# ── Helpers ────────────────────────────────────────────────────────────────────

ALLOWED_SUFFIXES = {".tif", ".tiff", ".geotiff", ".png", ".jpg", ".jpeg"}
MAX_FILE_BYTES   = 500 * 1024 * 1024   # 500 MB per file


def _try_rasterio(path: Path, role: ImageRole, modality: Modality) -> tuple[Optional[ImageMeta], Optional[str]]:
    """Open file with rasterio and extract metadata. Returns (meta, error_msg)."""
    try:
        import rasterio
        with rasterio.open(path) as ds:
            res = ds.res  # (pixel_x_metres, pixel_y_metres) if CRS known
            crs_str = str(ds.crs) if ds.crs else None
            # Convert res to metres: if CRS is geographic (degrees), approximate
            rx, ry = res
            if ds.crs and ds.crs.is_geographic:
                # rough conversion: 1 degree ≈ 111_320 m
                rx = abs(rx) * 111_320
                ry = abs(ry) * 111_320
            return ImageMeta(
                path=path,
                role=role,
                modality=modality,
                width=ds.width,
                height=ds.height,
                bands=ds.count,
                crs=crs_str,
                res_x=rx if rx else None,
                res_y=ry if ry else None,
                dtype=str(ds.dtypes[0]),
                nodata=ds.nodata,
            ), None
    except Exception as exc:
        return None, str(exc)


def _try_pillow(path: Path, role: ImageRole, modality: Modality) -> tuple[Optional[ImageMeta], Optional[str]]:
    """Fallback: open with Pillow (PNG/JPEG, no georef)."""
    try:
        from PIL import Image
        img = Image.open(path)
        bands = len(img.getbands())
        return ImageMeta(
            path=path, role=role, modality=modality,
            width=img.width, height=img.height, bands=bands,
            crs=None, res_x=None, res_y=None,
            dtype="uint8", nodata=None,
        ), None
    except Exception as exc:
        return None, str(exc)


def _infer_modality(path: Path, role: ImageRole, meta: Optional[ImageMeta]) -> Modality:
    """Try to infer optical vs SAR from filename and band count."""
    stem = path.stem.lower()
    if any(k in stem for k in ("sar", "s1", "vv", "vh", "risat", "asar")):
        return Modality.SAR
    if any(k in stem for k in ("s2", "optical", "rgb", "nir", "cartosat", "pan")):
        return Modality.OPTICAL
    if role in (ImageRole.SAR,):
        return Modality.SAR
    if meta and meta.bands in (1, 2):
        # Single or dual-pol → likely SAR
        return Modality.SAR
    if meta and meta.bands >= 3:
        return Modality.OPTICAL
    return Modality.UNKNOWN


# ── Extent overlap check ───────────────────────────────────────────────────────

def _extents_overlap(m1: ImageMeta, m2: ImageMeta, threshold: float = 0.0) -> bool:
    """Check if two georeferenced images overlap spatially. Returns True if CRS is unavailable."""
    try:
        import rasterio
        from rasterio.crs import CRS
        if not m1.crs or not m2.crs:
            return True  # can't verify, assume ok and warn
        with rasterio.open(m1.path) as d1, rasterio.open(m2.path) as d2:
            b1 = d1.bounds
            b2 = d2.bounds
            # Reproject b2 to m1 CRS if different
            if d1.crs != d2.crs:
                from rasterio.warp import transform_bounds
                b2 = transform_bounds(d2.crs, d1.crs, *b2)
            # Check overlap
            overlap = (
                max(b1.left, b2.left) < min(b1.right, b2.right) and
                max(b1.bottom, b2.bottom) < min(b1.top, b2.top)
            )
            return overlap
    except Exception:
        return True   # permissive on error, emit warning


# ── Public API ────────────────────────────────────────────────────────────────

def validate_inputs(
    file_paths:  list[Path],
    roles:       list[str],
    task:        str,
) -> ValidationResult:
    """
    Validate a set of uploaded images for a given task.

    Parameters
    ----------
    file_paths : list of Path objects (already saved to disk)
    roles      : matching list of role strings ("single","t1","t2","optical","sar")
    task       : one of "single_vqa","caption","ground","change","sar_optical"
    """
    vr = ValidationResult(ok=True)

    # ── 1. Basic count / extension checks ────────────────────────────────────
    if len(file_paths) == 0:
        return vr.fail("No images uploaded.")
    if len(file_paths) != len(roles):
        return vr.fail("file_paths and roles must have the same length.")

    for p in file_paths:
        if p.suffix.lower() not in ALLOWED_SUFFIXES:
            vr.fail(f"Unsupported file type: {p.suffix}. Allowed: {ALLOWED_SUFFIXES}")
        if p.stat().st_size > MAX_FILE_BYTES:
            vr.fail(f"{p.name} exceeds 500 MB limit.")

    if not vr.ok:
        return vr

    # ── 2. Read metadata ──────────────────────────────────────────────────────
    metas: list[ImageMeta] = []
    for p, role_str in zip(file_paths, roles):
        role = ImageRole(role_str)
        meta, err = _try_rasterio(p, role, Modality.UNKNOWN)
        if meta is None:
            meta, err2 = _try_pillow(p, role, Modality.UNKNOWN)
            if meta is None:
                vr.fail(f"Cannot open {p.name}: {err}; fallback PIL: {err2}")
                continue
            vr.warn(f"{p.name} opened via PIL (no georeferencing). Rasterio error: {err}")
        # Infer modality
        meta.modality = _infer_modality(p, role, meta)
        metas.append(meta)

    if not vr.ok:
        return vr
    vr.metas = metas

    # ── 3. Task-specific structural checks ────────────────────────────────────
    roles_set = {r for r in roles}

    if task in ("single_vqa", "caption", "ground"):
        if len(metas) != 1:
            vr.fail(f"Task '{task}' requires exactly 1 image; got {len(metas)}.")

    elif task == "change":
        if len(metas) != 2:
            vr.fail(f"Change task requires exactly 2 images (t1, t2); got {len(metas)}.")
        elif {"t1", "t2"} - roles_set:
            vr.fail("Change task requires images tagged 't1' (older) and 't2' (newer).")
        else:
            # Spatial extent overlap
            m_t1 = next(m for m in metas if m.role == ImageRole.T1)
            m_t2 = next(m for m in metas if m.role == ImageRole.T2)
            if not _extents_overlap(m_t1, m_t2):
                vr.fail("T1 and T2 images do not overlap spatially. Cannot perform change detection.")
            else:
                # Resolution mismatch
                if m_t1.res_x and m_t2.res_x:
                    ratio = max(m_t1.res_x, m_t2.res_x) / min(m_t1.res_x, m_t2.res_x)
                    if ratio > 4.0:
                        vr.warn(f"Large resolution mismatch between T1 ({m_t1.res_x:.1f}m) and T2 ({m_t2.res_x:.1f}m). Resampling applied.")

    elif task == "sar_optical":
        if len(metas) != 2:
            vr.fail(f"SAR+Optical task requires exactly 2 images; got {len(metas)}.")
        elif {"optical", "sar"} - roles_set:
            vr.fail("SAR+Optical task requires one image tagged 'optical' and one tagged 'sar'.")
        else:
            modalities = {m.modality for m in metas}
            if Modality.SAR not in modalities:
                vr.warn("Could not confirm SAR modality from file metadata. Proceeding with user-specified roles.")
            if not _extents_overlap(metas[0], metas[1]):
                vr.fail("Optical and SAR images do not overlap spatially.")

    else:
        vr.fail(f"Unknown task: '{task}'. Valid: single_vqa, caption, ground, change, sar_optical.")

    # ── 4. Georeferencing warnings ────────────────────────────────────────────
    for m in metas:
        if not m.crs:
            vr.warn(f"{m.path.name} has no CRS. Georeferenced outputs (GeoTIFF masks) will be disabled.")
        if m.res_x and m.res_x < 0.1:
            vr.warn(f"{m.path.name}: very high resolution ({m.res_x:.3f}m). Consider tiling for large files.")

    return vr
