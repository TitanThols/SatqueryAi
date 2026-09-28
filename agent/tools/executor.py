"""
agent/tools/executor.py
Central dispatcher: maps tool names to implementations.
Called by the AgentController. Never raises — returns error dict instead.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


def execute_tool(
    tool_name: str,
    params:    dict,
    metas:     list,     # list[ImageMeta]
    upload_dir: Path,
) -> dict:
    """
    Dispatch a tool call. Returns output dict or raises on fatal error.
    """
    try:
        return _dispatch(tool_name, params, metas, upload_dir)
    except Exception as exc:
        logger.exception("Tool '%s' fatal error: %s", tool_name, exc)
        raise


def _dispatch(tool_name: str, params: dict, metas: list, upload_dir: Path) -> dict:
    from agent.tools.image_preprocessor import prepare_vlm_image, check_and_align
    from agent.validators.input_validator import ImageRole, Modality

    if tool_name == "vqa":
        from agent.tools.vlm_tool import run_vqa
        meta = metas[0]
        image = prepare_vlm_image(meta.path, modality=meta.modality.value)
        return run_vqa(
            image=image,
            query=params["query"],
            model_size=params.get("model_size", "small"),
            max_new_tokens=params.get("max_new_tokens", 128),
        )

    elif tool_name == "caption":
        from agent.tools.vlm_tool import run_caption
        meta = metas[0]
        image = prepare_vlm_image(meta.path, modality=meta.modality.value)
        return run_caption(image=image, model_size=params.get("model_size", "small"))

    elif tool_name == "ground":
        from agent.tools.vlm_tool import run_grounding
        meta = metas[0]
        image = prepare_vlm_image(meta.path, modality=meta.modality.value)
        result = run_grounding(
            image=image,
            query=params["query"],
            model_size=params.get("model_size", "small"),
            score_threshold=params.get("score_threshold", 0.3),
            meta=meta,
        )
        # Generate overlay image with bounding boxes
        overlay_path = upload_dir / "results" / "grounding_overlay.png"
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        _draw_boxes(image, result["boxes"], result["labels"], overlay_path)
        result["overlay_path"] = str(overlay_path)
        result["answer"] = _boxes_to_text(result["boxes"], result["labels"])
        return result

    elif tool_name == "change_map":
        from agent.tools.change_tool import run_change_map
        # Expect exactly 2 images: t1 and t2
        meta_t1 = next((m for m in metas if m.role.value == "t1"), metas[0])
        meta_t2 = next((m for m in metas if m.role.value == "t2"), metas[1])

        # Co-register if needed
        path_t1, path_t2 = check_and_align(meta_t1.path, meta_t2.path, upload_dir / "aligned")

        img_t1 = prepare_vlm_image(path_t1, modality=meta_t1.modality.value)
        img_t2 = prepare_vlm_image(path_t2, modality=meta_t2.modality.value)

        out_path = upload_dir / "results" / "change_mask.tif"
        return run_change_map(
            img_t1=img_t1,
            img_t2=img_t2,
            threshold=params.get("threshold", 0.5),
            output_path=out_path,
            meta_t1=meta_t1,
        )

    elif tool_name == "change_vqa":
        from agent.tools.change_tool import run_change_vqa
        from agent.tools.image_preprocessor import prepare_vlm_image as pvi
        meta_t1 = next((m for m in metas if m.role.value == "t1"), metas[0])
        meta_t2 = next((m for m in metas if m.role.value == "t2"), metas[1])

        img_t1 = pvi(meta_t1.path, modality=meta_t1.modality.value)
        img_t2 = pvi(meta_t2.path, modality=meta_t2.modality.value)

        # Retrieve change_map result from accumulated params
        change_result = {
            "change_mask_array": params.get("change_mask_array"),
            "changed_area_pct": params.get("changed_area_pct", 0.0),
        }
        return run_change_vqa(
            img_t1=img_t1,
            img_t2=img_t2,
            change_result=change_result,
            query=params["query"],
            model_size=params.get("model_size", "small"),
        )

    elif tool_name == "sar_optical_fusion":
        from agent.tools.sar_optical_tool import run_sar_optical_fusion
        meta_opt = next((m for m in metas if m.role.value in ("optical",) or
                         m.modality == Modality.OPTICAL), metas[0])
        meta_sar = next((m for m in metas if m.role.value in ("sar",) or
                         m.modality == Modality.SAR), metas[1])

        img_opt = prepare_vlm_image(meta_opt.path, modality="optical")
        img_sar = prepare_vlm_image(meta_sar.path, modality="sar")

        out_path = upload_dir / "results" / "fusion_mask.png"
        return run_sar_optical_fusion(
            optical_img=img_opt,
            sar_img=img_sar,
            output_path=out_path,
            meta_optical=meta_opt,
        )

    else:
        raise ValueError(f"Unknown tool: '{tool_name}'")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _draw_boxes(
    image,
    boxes: list,
    labels: list,
    out_path: Path,
) -> None:
    """Draw bounding boxes on image and save."""
    try:
        from PIL import ImageDraw, ImageFont
        import PIL

        draw_img = image.copy()
        draw = ImageDraw.Draw(draw_img)
        W, H = image.size

        for i, box in enumerate(boxes):
            x1 = box[0] / 1000 * W
            y1 = box[1] / 1000 * H
            x2 = box[2] / 1000 * W
            y2 = box[3] / 1000 * H
            draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
            label = labels[i] if i < len(labels) else ""
            if label:
                draw.text((x1 + 2, y1 + 2), label, fill=(255, 255, 0))

        draw_img.save(out_path)
    except Exception as e:
        logger.warning("Box drawing failed: %s", e)


def _boxes_to_text(boxes: list, labels: list) -> str:
    if not boxes:
        return "No regions detected for the given description."
    parts = []
    for i, (box, label) in enumerate(zip(boxes, labels)):
        parts.append(f"Region {i+1}: '{label}' at ({box[0]}, {box[1]})–({box[2]}, {box[3]})")
    return "Detected: " + "; ".join(parts)
