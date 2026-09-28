"""
agent/tools/image_preprocessor.py
GeoTIFF / raster preprocessing utilities:
 - Band selection + normalization
 - dB conversion for SAR
 - Resampling to common grid
 - Tiling for large images
 - Generates RGB preview for VLM input
All functions return numpy arrays; no side effects unless explicitly asked.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# ── Constants ──────────────────────────────────────────────────────────────────

# Sentinel-2 band indices (0-based) for common use:
# B2=Blue(0), B3=Green(1), B4=Red(2), B8=NIR(3)  (in a 10m 4-band stack)
S2_RGB_BANDS   = [2, 1, 0]   # B4, B3, B2 → RGB (index in 13-band order)
S2_NIRGB_BANDS = [3, 2, 1, 0]
S2_ALL_BANDS   = list(range(13))

TARGET_SIZE = (512, 512)   # default VLM input size


# ── Core readers ──────────────────────────────────────────────────────────────

def read_geotiff(path: Path, bands: Optional[list[int]] = None) -> np.ndarray:
    """
    Read a GeoTIFF and return array of shape (C, H, W) as float32.
    bands: 0-based band indices to select; None = all bands.
    """
    try:
        import rasterio
        with rasterio.open(path) as ds:
            if bands is None:
                bands = list(range(ds.count))
            # rasterio uses 1-based band indexing
            data = ds.read([b + 1 for b in bands]).astype(np.float32)
        return data  # (C, H, W)
    except Exception as e:
        raise RuntimeError(f"Failed to read {path}: {e}") from e


def read_image_auto(path: Path, bands: Optional[list[int]] = None) -> np.ndarray:
    """
    Auto-detect format: try rasterio first, fall back to PIL.
    Returns (C, H, W) float32.
    """
    try:
        return read_geotiff(path, bands)
    except Exception:
        img = Image.open(path).convert("RGB")
        arr = np.array(img, dtype=np.float32).transpose(2, 0, 1)  # (3, H, W)
        return arr


# ── Normalization ──────────────────────────────────────────────────────────────

def normalize_optical(arr: np.ndarray, quantile: float = 0.98) -> np.ndarray:
    """
    Per-image, per-channel percentile normalization → [0, 1].
    quantile: clip at this quantile to remove outliers.
    """
    out = np.empty_like(arr)
    for c in range(arr.shape[0]):
        ch = arr[c]
        lo = np.nanpercentile(ch, 2)
        hi = np.nanpercentile(ch, quantile * 100)
        if hi > lo:
            out[c] = np.clip((ch - lo) / (hi - lo), 0, 1)
        else:
            out[c] = np.zeros_like(ch)
    return out


def sar_to_db(arr: np.ndarray, min_db: float = -25.0, max_db: float = 0.0) -> np.ndarray:
    """
    Convert SAR linear power to dB and clip/normalize to [0, 1].
    arr: (C, H, W) linear backscatter (e.g. VV, VH).
    """
    eps = 1e-10
    db = 10.0 * np.log10(np.maximum(arr, eps))
    normalized = np.clip((db - min_db) / (max_db - min_db), 0, 1)
    return normalized.astype(np.float32)


def normalize_for_model(arr: np.ndarray, modality: str = "optical") -> np.ndarray:
    """Dispatch to the right normalization based on modality."""
    if modality == "sar":
        return sar_to_db(arr)
    return normalize_optical(arr)


# ── Band selection ─────────────────────────────────────────────────────────────

def select_rgb_bands(arr: np.ndarray, n_bands_in_file: int) -> np.ndarray:
    """
    Select 3 bands for visualization/VLM.
    Handles various common cases automatically.
    Returns (3, H, W).
    """
    C = arr.shape[0]
    if C == 1:
        # Greyscale — repeat channel
        return np.concatenate([arr, arr, arr], axis=0)
    if C == 2:
        # Dual-pol SAR — pad with zeros
        pad = np.zeros_like(arr[:1])
        return np.concatenate([arr, pad], axis=0)
    if C == 3:
        return arr
    if C == 4:
        # RGBA or NIRGB — use RGB channels (indices 0,1,2)
        return arr[:3]
    if C >= 13:
        # Sentinel-2 13-band: select B4(idx3), B3(idx2), B2(idx1) in band order
        # Standard Sentinel-2 band order: B1,B2,B3,B4,...
        return arr[[3, 2, 1], :, :]   # R, G, B
    if C >= 4:
        # Try indices 2, 1, 0
        return arr[[2, 1, 0], :, :]
    return arr[:3]


# ── Resampling ─────────────────────────────────────────────────────────────────

def resample_to_target(
    arr: np.ndarray,
    target_hw: Tuple[int, int] = TARGET_SIZE,
    antialias: bool = True,
) -> np.ndarray:
    """
    Bilinear resample a (C, H, W) array to target (H, W).
    Uses PIL for speed; rasterio for large files.
    """
    C, H, W = arr.shape
    tH, tW = target_hw
    if H == tH and W == tW:
        return arr

    out_channels = []
    for c in range(C):
        ch = arr[c]
        ch_uint8 = (np.clip(ch, 0, 1) * 255).astype(np.uint8)
        pil_img = Image.fromarray(ch_uint8, mode="L")
        resamp = Image.LANCZOS if antialias else Image.NEAREST
        pil_img = pil_img.resize((tW, tH), resamp)
        out_channels.append(np.array(pil_img, dtype=np.float32) / 255.0)

    return np.stack(out_channels, axis=0)


# ── Tiling ────────────────────────────────────────────────────────────────────

def tile_image(
    arr: np.ndarray,
    tile_size: int = 512,
    overlap: int = 64,
) -> list[Tuple[np.ndarray, Tuple[int, int]]]:
    """
    Split (C, H, W) into overlapping tiles.
    Returns list of (tile_array, (row_offset, col_offset)).
    Useful for large GeoTIFFs.
    """
    C, H, W = arr.shape
    tiles = []
    stride = tile_size - overlap
    for r in range(0, H, stride):
        for c in range(0, W, stride):
            r_end = min(r + tile_size, H)
            c_end = min(c + tile_size, W)
            tile = arr[:, r:r_end, c:c_end]
            # Pad if needed
            if tile.shape[1] < tile_size or tile.shape[2] < tile_size:
                padded = np.zeros((C, tile_size, tile_size), dtype=tile.dtype)
                padded[:, :tile.shape[1], :tile.shape[2]] = tile
                tile = padded
            tiles.append((tile, (r, c)))
    return tiles


# ── VLM-ready PIL image ───────────────────────────────────────────────────────

def prepare_vlm_image(
    path: Path,
    modality: str = "optical",
    target_size: Tuple[int, int] = TARGET_SIZE,
) -> Image.Image:
    """
    Full pipeline: read → select RGB → normalize → resample → PIL Image (RGB, uint8).
    This is what gets fed to Florence-2 / Qwen2-VL.
    """
    arr = read_image_auto(path)
    arr = normalize_for_model(arr, modality)
    arr = select_rgb_bands(arr, arr.shape[0])
    arr = resample_to_target(arr, target_size)
    # Convert to uint8 PIL
    rgb = (arr.transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(rgb, "RGB")


# ── Co-registration check ─────────────────────────────────────────────────────

def check_and_align(
    path1: Path,
    path2: Path,
    out_dir: Path,
) -> Tuple[Path, Path]:
    """
    Check if two GeoTIFFs have matching CRS/grid.
    If not, reproject path2 to match path1 and save to out_dir.
    Returns (path1, aligned_path2).
    """
    try:
        import rasterio
        from rasterio.warp import calculate_default_transform, reproject, Resampling

        with rasterio.open(path1) as src1, rasterio.open(path2) as src2:
            if src1.crs and src2.crs and src1.crs == src2.crs:
                if src1.res == src2.res:
                    return path1, path2  # already aligned

            # Reproject path2 → path1's CRS + grid
            transform, width, height = calculate_default_transform(
                src2.crs, src1.crs, src2.width, src2.height, *src2.bounds
            )
            profile = src1.profile.copy()
            profile.update(
                crs=src1.crs, transform=src1.transform,
                width=src1.width, height=src1.height,
                count=src2.count,
            )
            out_path = out_dir / (path2.stem + "_aligned.tif")
            with rasterio.open(out_path, "w", **profile) as dst:
                for i in range(1, src2.count + 1):
                    reproject(
                        source=rasterio.band(src2, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src2.transform,
                        src_crs=src2.crs,
                        dst_transform=src1.transform,
                        dst_crs=src1.crs,
                        resampling=Resampling.bilinear,
                    )
            logger.info("Co-registered %s → %s", path2.name, out_path.name)
            return path1, out_path

    except Exception as e:
        logger.warning("Co-registration failed: %s — using originals.", e)
        return path1, path2
