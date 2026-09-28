"""
data/scripts/download_datasets.py
Download scripts for all required datasets.
Usage:
  python data/scripts/download_datasets.py --dataset levir-cd --out ./data
  python data/scripts/download_datasets.py --dataset vrsbench  --out ./data
  python data/scripts/download_datasets.py --dataset rsvqa-lr  --out ./data
  python data/scripts/download_datasets.py --dataset oscd      --out ./data
  python data/scripts/download_datasets.py --dataset demo      --out ./data  (generates demo GeoTIFFs)
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def download_vrsbench(out_dir: Path):
    """Download VRSBench from HuggingFace hub."""
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading VRSBench from HuggingFace (xiang709/VRSBench)...")
    try:
        from datasets import load_dataset
        ds = load_dataset("xiang709/VRSBench", streaming=False, split="train")
        qa_pairs = []
        img_dir = out_dir / "VRSBench" / "images"
        img_dir.mkdir(parents=True, exist_ok=True)

        for i, item in enumerate(ds):
            if i >= 10000:   # cap at 10k for feasibility
                break
            img = item.get("image")
            if img:
                img.save(img_dir / f"{i:06d}.png")
            qa_pairs.append({
                "image": f"{i:06d}.png",
                "question": item.get("question", ""),
                "answer": item.get("answer", ""),
            })

        import json
        (out_dir / "VRSBench" / "QA_train.json").write_text(json.dumps(qa_pairs[:8000], indent=2))
        (out_dir / "VRSBench" / "QA_val.json").write_text(json.dumps(qa_pairs[8000:], indent=2))
        logger.info("VRSBench saved to %s/VRSBench", out_dir)
    except ImportError:
        logger.error("pip install datasets  first.")
        sys.exit(1)


def download_levir_cd(out_dir: Path):
    """Download LEVIR-CD+ from HuggingFace."""
    logger.info("Downloading LEVIR-CD+ from HuggingFace (blanchon/LEVIR_CDPlus)...")
    try:
        from datasets import load_dataset
        ds = load_dataset("blanchon/LEVIR_CDPlus", split="train")

        split_dir = out_dir / "LEVIR-CD"
        for sp, split_data in [("train", ds), ("val", load_dataset("blanchon/LEVIR_CDPlus", split="validation")),
                                ("test", load_dataset("blanchon/LEVIR_CDPlus", split="test"))]:
            a_dir = split_dir / sp / "A"; a_dir.mkdir(parents=True, exist_ok=True)
            b_dir = split_dir / sp / "B"; b_dir.mkdir(parents=True, exist_ok=True)
            l_dir = split_dir / sp / "label"; l_dir.mkdir(parents=True, exist_ok=True)
            for i, item in enumerate(split_data):
                name = f"{i:05d}.png"
                item["image_A"].save(a_dir / name)
                item["image_B"].save(b_dir / name)
                item["mask"].save(l_dir / name)
            logger.info("LEVIR-CD %s: %d pairs", sp, i + 1)
        logger.info("LEVIR-CD saved to %s/LEVIR-CD", out_dir)
    except Exception as e:
        logger.error("LEVIR-CD download failed: %s", e)
        logger.info("Manual download: https://levir.buaa.edu.cn/ or Kaggle: levir-cd")


def download_rsvqa_lr(out_dir: Path):
    """Instructions for RSVQA-LR (requires manual download)."""
    logger.info(
        "RSVQA-LR requires registration at https://rsvqa.sylvainlobry.com/\n"
        "Steps:\n"
        "  1. Go to https://rsvqa.sylvainlobry.com/\n"
        "  2. Download 'Low Resolution' dataset\n"
        "  3. Extract to %s/RSVQA-LR/\n"
        "  Expected structure:\n"
        "    RSVQA-LR/\n"
        "      Images_LR/\n"
        "      QA_LR_train.json\n"
        "      QA_LR_val.json\n"
        "      QA_LR_test.json\n",
        out_dir,
    )


def download_oscd(out_dir: Path):
    """Instructions for OSCD."""
    logger.info(
        "OSCD requires free registration at IEEE Dataport:\n"
        "  https://ieee-dataport.org/open-access/oscd-onera-satellite-change-detection\n"
        "  Or use Kaggle: kaggle datasets download -d oscd-satellite-change-detection\n"
        "  Extract to %s/OSCD/\n",
        out_dir,
    )


def generate_demo_geotiffs(out_dir: Path):
    """
    Generate synthetic demo GeoTIFFs for local testing.
    Creates: single_optical.tif, sar_image.tif, t1.tif, t2.tif
    All are 512x512, georeferenced to a dummy WGS84 bounding box.
    """
    logger.info("Generating demo GeoTIFFs...")
    try:
        import numpy as np
        import rasterio
        from rasterio.transform import from_bounds
        from rasterio.crs import CRS

        demo_dir = out_dir / "demo"
        demo_dir.mkdir(parents=True, exist_ok=True)

        crs       = CRS.from_epsg(4326)
        transform = from_bounds(77.0, 28.0, 77.1, 28.1, 512, 512)
        rng       = np.random.default_rng(42)

        def write_geotiff(path, bands_data, dtype="float32"):
            n_bands, H, W = bands_data.shape
            profile = {
                "driver":    "GTiff",
                "dtype":     dtype,
                "width":     W,
                "height":    H,
                "count":     n_bands,
                "crs":       crs,
                "transform": transform,
            }
            with rasterio.open(path, "w", **profile) as dst:
                for b in range(n_bands):
                    dst.write(bands_data[b].astype(dtype), b + 1)
            logger.info("Created: %s", path)

        # Single optical (4-band: B2,B3,B4,B8 Sentinel-2 style)
        optical = rng.uniform(0, 10000, (4, 512, 512)).astype("float32")
        # Fake some structure: bright rectangle = "urban"
        optical[:, 100:200, 100:200] *= 3.0
        write_geotiff(demo_dir / "single_optical.tif", optical)

        # SAR image (2-band: VV, VH)
        sar = rng.exponential(0.2, (2, 512, 512)).astype("float32")
        sar[:, 100:200, 100:200] *= 2.5   # bright urban backscatter
        write_geotiff(demo_dir / "sar_image.tif", sar)

        # Bi-temporal pair: T1 and T2 (optical)
        t1 = rng.uniform(0, 10000, (4, 512, 512)).astype("float32")
        t2 = t1.copy()
        # Simulate new construction: add a bright block in T2
        t2[:, 300:380, 300:380] += 5000
        t2 = t2.clip(0, 15000)
        write_geotiff(demo_dir / "t1.tif", t1)
        write_geotiff(demo_dir / "t2.tif", t2)

        logger.info(
            "\nDemo data ready in %s/\n"
            "  single_optical.tif — 4-band Sentinel-2 style (512x512)\n"
            "  sar_image.tif       — 2-band SAR VV+VH (512x512)\n"
            "  t1.tif, t2.tif      — bi-temporal pair (change at [300:380, 300:380])\n",
            demo_dir,
        )
    except ImportError:
        logger.warning("rasterio not installed — creating PNG demos instead.")
        from PIL import Image
        demo_dir = out_dir / "demo"; demo_dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(42)
        for name in ["single_optical", "sar_image", "t1", "t2"]:
            arr = (rng.uniform(0, 255, (256, 256, 3))).astype(np.uint8)
            Image.fromarray(arr).save(demo_dir / f"{name}.png")
        logger.info("PNG demos created in %s/", demo_dir)


DOWNLOAD_FNS = {
    "vrsbench":  download_vrsbench,
    "levir-cd":  download_levir_cd,
    "rsvqa-lr":  download_rsvqa_lr,
    "oscd":      download_oscd,
    "demo":      generate_demo_geotiffs,
}

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=list(DOWNLOAD_FNS.keys()), required=True)
    p.add_argument("--out", type=str, default="./data")
    args = p.parse_args()

    fn = DOWNLOAD_FNS[args.dataset]
    fn(Path(args.out))
