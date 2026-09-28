"""
agent/tools/vlm_tool.py
VLM inference: Florence-2 (small/local) and Qwen2-VL-2B (full/remote LoRA).
Supports: VQA, captioning, text-guided grounding (bounding boxes).
Auto-detects device: cuda > mps > cpu.
NEVER uses flash-attn or bitsandbytes in the inference path.
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Any, Optional

import torch
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# ── Device selection ──────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = get_device()
logger.info("VLM device: %s", DEVICE)


# ── Model cache (singleton per process) ───────────────────────────────────────

_model_cache: dict[str, Any] = {}


def _load_florence2(model_id: str = "microsoft/Florence-2-base") -> tuple:
    """Load Florence-2 model + processor. Cached."""
    key = f"florence2_{model_id}"
    if key not in _model_cache:
        from transformers import AutoProcessor, AutoModelForCausalLM
        logger.info("Loading Florence-2: %s on %s", model_id, DEVICE)
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.float16 if DEVICE.type != "cpu" else torch.float32,
            attn_implementation="eager",   # no flash_attn
        ).to(DEVICE).eval()
        _model_cache[key] = (model, processor)
        logger.info("Florence-2 loaded.")
    return _model_cache[key]


def _load_qwen2vl(
    model_id: str = "Qwen/Qwen2-VL-2B-Instruct",
    adapter_path: Optional[str] = None,
) -> tuple:
    """Load Qwen2-VL-2B. If adapter_path provided, load LoRA adapter via PEFT."""
    key = f"qwen2vl_{model_id}_{adapter_path}"
    if key not in _model_cache:
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
        logger.info("Loading Qwen2-VL: %s on %s (adapter=%s)", model_id, DEVICE, adapter_path)
        dtype = torch.float16 if DEVICE.type != "cpu" else torch.float32
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            attn_implementation="eager",   # no flash_attn; MPS/CPU safe
            device_map=None,
        ).to(DEVICE).eval()
        if adapter_path and Path(adapter_path).exists():
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter_path)
            model = model.merge_and_unload()
            logger.info("LoRA adapter merged from: %s", adapter_path)
        processor = AutoProcessor.from_pretrained(model_id)
        _model_cache[key] = (model, processor)
        logger.info("Qwen2-VL loaded.")
    return _model_cache[key]


# ── Confidence: token log-prob ─────────────────────────────────────────────────

def _compute_confidence_logprob(logits: torch.Tensor, token_ids: torch.Tensor) -> float:
    """
    Compute mean log-probability over generated tokens.
    Maps to [0, 1] via exp(mean_log_prob).
    logits: (seq_len, vocab_size)
    token_ids: (seq_len,)
    """
    try:
        with torch.no_grad():
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(token_ids)), token_ids]
            mean_lp = selected.mean().item()
            return float(math.exp(mean_lp))  # in [0,1]
    except Exception:
        return 0.0


# ── Florence-2 inference ───────────────────────────────────────────────────────

def _florence2_vqa(
    image: Image.Image,
    query: str,
    max_new_tokens: int = 128,
) -> dict:
    model, processor = _load_florence2()
    task_prompt = "<VQA>"
    inputs = processor(
        text=task_prompt + query,
        images=image,
        return_tensors="pt",
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            output_scores=True,
            return_dict_in_generate=True,
        )

    # Decode
    gen_ids = outputs.sequences[0, inputs["input_ids"].shape[-1]:]
    answer = processor.batch_decode([gen_ids], skip_special_tokens=True)[0].strip()

    # Confidence from scores
    if outputs.scores:
        stacked = torch.stack(outputs.scores, dim=0)  # (T, 1, vocab)
        logits = stacked[:, 0, :]                      # (T, vocab)
        conf = _compute_confidence_logprob(logits, gen_ids[:len(outputs.scores)])
    else:
        conf = 0.0

    return {"answer": answer, "confidence": conf}


def _florence2_caption(image: Image.Image) -> dict:
    model, processor = _load_florence2()
    task_prompt = "<DETAILED_CAPTION>"
    inputs = processor(text=task_prompt, images=image, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            output_scores=True,
            return_dict_in_generate=True,
        )
    gen_ids = outputs.sequences[0, inputs["input_ids"].shape[-1]:]
    caption = processor.batch_decode([gen_ids], skip_special_tokens=True)[0].strip()

    # Post-process Florence result dict format
    parsed = processor.post_process_generation(
        {"<DETAILED_CAPTION>": caption},
        task="<DETAILED_CAPTION>",
        image_size=(image.width, image.height),
    )
    caption_text = parsed.get("<DETAILED_CAPTION>", caption)
    if isinstance(caption_text, dict):
        caption_text = caption_text.get("caption", caption)

    if outputs.scores:
        stacked = torch.stack(outputs.scores, dim=0)
        logits = stacked[:, 0, :]
        conf = _compute_confidence_logprob(logits, gen_ids[:len(outputs.scores)])
    else:
        conf = 0.0

    return {"caption": str(caption_text), "confidence": conf}


def _florence2_grounding(
    image: Image.Image,
    query: str,
    score_threshold: float = 0.3,
) -> dict:
    """
    Text-guided grounding via Florence-2 REFERRING_EXPRESSION_SEGMENTATION
    or OPEN_VOCABULARY_DETECTION.
    Returns normalized (0-1000 scale) box coords as Florence native format.
    """
    model, processor = _load_florence2()

    # Try grounded generation
    task = "<OPEN_VOCABULARY_DETECTION>"
    text_input = task + query
    inputs = processor(text=text_input, images=image, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            output_scores=True,
            return_dict_in_generate=True,
        )

    gen_ids = outputs.sequences[0, inputs["input_ids"].shape[-1]:]
    raw_out = processor.batch_decode([gen_ids], skip_special_tokens=False)[0]

    parsed = processor.post_process_generation(
        {task: raw_out},
        task=task,
        image_size=(image.width, image.height),
    )
    result = parsed.get(task, {})
    boxes  = result.get("bboxes", [])
    labels = result.get("bboxes_labels", [])

    if outputs.scores:
        stacked = torch.stack(outputs.scores, dim=0)
        logits = stacked[:, 0, :]
        conf = _compute_confidence_logprob(logits, gen_ids[:len(outputs.scores)])
    else:
        conf = 0.5

    # Normalize boxes to 0-1000 scale (Florence uses pixel coords)
    W, H = image.width, image.height
    norm_boxes = []
    for box in boxes:
        x1, y1, x2, y2 = box
        norm_boxes.append([
            int(x1 / W * 1000), int(y1 / H * 1000),
            int(x2 / W * 1000), int(y2 / H * 1000),
        ])

    # Scores not available from Florence grounding — use uniform derived from conf
    scores = [conf] * len(norm_boxes)

    return {
        "boxes": norm_boxes,
        "labels": labels,
        "scores": scores,
        "geo_boxes": None,  # filled by caller if CRS available
    }


# ── Qwen2-VL inference ────────────────────────────────────────────────────────

def _qwen2vl_vqa(
    image: Image.Image,
    query: str,
    max_new_tokens: int = 256,
    adapter_path: Optional[str] = None,
) -> dict:
    from transformers import AutoProcessor

    model, processor = _load_qwen2vl(adapter_path=adapter_path)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": query},
            ],
        }
    ]
    # Qwen2VL chat template
    text_input = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text_input],
        images=[image],
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            output_scores=True,
            return_dict_in_generate=True,
        )

    input_len = inputs["input_ids"].shape[-1]
    gen_ids   = outputs.sequences[0, input_len:]
    answer    = processor.batch_decode([gen_ids], skip_special_tokens=True)[0].strip()

    if outputs.scores:
        stacked = torch.stack(outputs.scores, dim=0)
        logits  = stacked[:, 0, :]
        conf    = _compute_confidence_logprob(logits, gen_ids[:len(outputs.scores)])
    else:
        conf = 0.0

    return {"answer": answer, "confidence": conf}


# ── Public interface ──────────────────────────────────────────────────────────

def run_vqa(
    image: Image.Image,
    query: str,
    model_size: str = "small",
    max_new_tokens: int = 128,
) -> dict:
    """
    Run VQA and return {answer, confidence, model_used, adapter_status}.
    model_size: "small" (Florence-2) or "full" (Qwen2-VL-2B).
    """
    adapter_path = os.environ.get("SATQUERY_ADAPTER_PATH")
    adapter_status = (
        f"LoRA-adapted ({os.path.basename(adapter_path)})"
        if adapter_path and Path(adapter_path).exists()
        else "baseline (not adapted)"
    )

    if model_size == "small":
        result = _florence2_vqa(image, query, max_new_tokens)
        model_used = "Florence-2-base"
    else:
        result = _qwen2vl_vqa(image, query, max_new_tokens, adapter_path)
        model_used = "Qwen2-VL-2B-Instruct"

    return {
        **result,
        "model_used": model_used,
        "adapter_status": adapter_status,
    }


def run_caption(image: Image.Image, model_size: str = "small") -> dict:
    if model_size == "small":
        result = _florence2_caption(image)
        model_used = "Florence-2-base"
    else:
        result = _qwen2vl_vqa(image, "Describe this satellite image in detail.", max_new_tokens=256)
        result["caption"] = result.pop("answer", "")
        model_used = "Qwen2-VL-2B-Instruct"
    return {**result, "model_used": model_used}


def run_grounding(
    image: Image.Image,
    query: str,
    model_size: str = "small",
    score_threshold: float = 0.3,
    meta=None,
) -> dict:
    """
    Text-guided grounding. Returns boxes in pixel coords (0-1000 scale) + optional geo-boxes.
    """
    result = _florence2_grounding(image, query, score_threshold)
    result["model_used"] = "Florence-2-base"

    # If CRS available, compute geo-boxes
    if meta and meta.crs:
        try:
            import rasterio
            geo_boxes = []
            W, H = image.width, image.height
            with rasterio.open(meta.path) as ds:
                for box in result["boxes"]:
                    # box in 0-1000 scale → pixel coords in original image
                    px1 = int(box[0] / 1000 * ds.width)
                    py1 = int(box[1] / 1000 * ds.height)
                    px2 = int(box[2] / 1000 * ds.width)
                    py2 = int(box[3] / 1000 * ds.height)
                    lon1, lat1 = ds.xy(py1, px1)
                    lon2, lat2 = ds.xy(py2, px2)
                    geo_boxes.append([lon1, lat1, lon2, lat2])
            result["geo_boxes"] = geo_boxes
        except Exception as e:
            logger.warning("Failed to compute geo-boxes: %s", e)

    return result
