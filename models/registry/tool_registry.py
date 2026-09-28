"""
models/registry/tool_registry.py
JSON-schema-gated tool registry.
Each tool has: name, description, accepted_input_types, modalities, params schema, output schema.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# ── Tool definitions ──────────────────────────────────────────────────────────

TOOLS: list[dict[str, Any]] = [
    {
        "name": "vqa",
        "description": "Answer a natural-language question about a single remote sensing image.",
        "accepted_tasks": ["single_vqa"],
        "accepted_modalities": ["optical", "sar", "unknown"],
        "min_images": 1,
        "max_images": 1,
        "params": {
            "query": {
                "type": "string",
                "description": "The natural language question.",
                "required": True,
            },
            "model_size": {
                "type": "string",
                "enum": ["small", "full"],
                "default": "small",
                "description": "small=Florence-2-base; full=Qwen2-VL-2B with LoRA adapter.",
            },
            "max_new_tokens": {
                "type": "integer",
                "minimum": 10,
                "maximum": 512,
                "default": 128,
            },
        },
        "output_schema": {
            "answer": "string",
            "confidence": "float [0-1] — token log-prob derived",
            "model_used": "string",
            "adapter_status": "string — 'baseline (not adapted)' or 'LoRA-adapted (VRSBench+RSVQA-LR)'",
        },
    },
    {
        "name": "caption",
        "description": "Generate a detailed caption for a single remote sensing image.",
        "accepted_tasks": ["caption"],
        "accepted_modalities": ["optical", "sar", "unknown"],
        "min_images": 1,
        "max_images": 1,
        "params": {
            "model_size": {
                "type": "string",
                "enum": ["small", "full"],
                "default": "small",
            },
        },
        "output_schema": {
            "caption": "string",
            "confidence": "float — mean log-prob",
            "model_used": "string",
        },
    },
    {
        "name": "ground",
        "description": "Text-guided region grounding: find bounding boxes for described objects.",
        "accepted_tasks": ["ground"],
        "accepted_modalities": ["optical", "sar", "unknown"],
        "min_images": 1,
        "max_images": 1,
        "params": {
            "query": {
                "type": "string",
                "description": "Description of the region/object to locate.",
                "required": True,
            },
            "model_size": {
                "type": "string",
                "enum": ["small", "full"],
                "default": "small",
            },
            "score_threshold": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "default": 0.3,
            },
        },
        "output_schema": {
            "boxes": "list of [x1,y1,x2,y2] in pixel coords (normalized 0-1000)",
            "labels": "list of strings",
            "scores": "list of float",
            "geo_boxes": "list of [lon1,lat1,lon2,lat2] if CRS available else null",
            "model_used": "string",
        },
    },
    {
        "name": "change_map",
        "description": "Produce a pixel-wise binary change map from a bi-temporal image pair.",
        "accepted_tasks": ["change"],
        "accepted_modalities": ["optical"],
        "min_images": 2,
        "max_images": 2,
        "params": {
            "threshold": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "default": 0.5,
                "description": "Probability threshold for change vs no-change.",
            },
            "output_geotiff": {
                "type": "boolean",
                "default": True,
            },
        },
        "output_schema": {
            "change_mask_path": "string — path to binary GeoTIFF mask",
            "changed_area_pct": "float — % pixels classified as changed",
            "mean_prob": "float — mean change probability (confidence)",
            "model_used": "string",
            "adapter_status": "string",
        },
    },
    {
        "name": "change_vqa",
        "description": "Answer a question about change between two temporal images, using the change map.",
        "accepted_tasks": ["change"],
        "accepted_modalities": ["optical"],
        "min_images": 2,
        "max_images": 2,
        "params": {
            "query": {
                "type": "string",
                "required": True,
            },
            "model_size": {
                "type": "string",
                "enum": ["small", "full"],
                "default": "small",
            },
        },
        "output_schema": {
            "answer": "string",
            "change_description": "string — VLM-generated change description",
            "changed_area_pct": "float",
            "confidence": "float",
            "model_used": "string",
        },
    },
    {
        "name": "sar_optical_fusion",
        "description": "Fuse optical + SAR to produce land-cover / built-up / water masks.",
        "accepted_tasks": ["sar_optical"],
        "accepted_modalities": ["optical", "sar"],
        "min_images": 2,
        "max_images": 2,
        "params": {
            "output_classes": {
                "type": "array",
                "items": {"type": "string"},
                "default": ["built-up", "water", "vegetation", "bare soil"],
                "description": "Land cover classes to report.",
            },
            "output_geotiff": {
                "type": "boolean",
                "default": True,
            },
        },
        "output_schema": {
            "class_mask_path": "string — multiclass GeoTIFF",
            "class_areas_pct": "dict[class → float]",
            "mean_prob": "float — mean max-class probability",
            "model_used": "string",
            "adapter_status": "string",
        },
    },
]

# Index by name for fast lookup
_TOOL_INDEX: dict[str, dict] = {t["name"]: t for t in TOOLS}


def get_tool(name: str) -> dict:
    """Return tool schema or raise KeyError."""
    if name not in _TOOL_INDEX:
        raise KeyError(f"Tool '{name}' not in registry. Available: {list(_TOOL_INDEX)}")
    return _TOOL_INDEX[name]


def list_tools() -> list[str]:
    return list(_TOOL_INDEX.keys())


def validate_tool_call(name: str, params: dict) -> list[str]:
    """
    Validate a proposed tool call against its schema.
    Returns list of error strings (empty = OK).
    """
    try:
        tool = get_tool(name)
    except KeyError as e:
        return [str(e)]

    errors = []
    schema = tool.get("params", {})

    for pname, pschema in schema.items():
        if pschema.get("required") and pname not in params:
            errors.append(f"Tool '{name}': required param '{pname}' missing.")
            continue
        if pname not in params:
            continue
        val = params[pname]
        ptype = pschema.get("type")
        if ptype == "string" and not isinstance(val, str):
            errors.append(f"Tool '{name}': param '{pname}' must be string.")
        if ptype in ("number", "integer") and not isinstance(val, (int, float)):
            errors.append(f"Tool '{name}': param '{pname}' must be numeric.")
        if ptype == "integer" and isinstance(val, float) and not val.is_integer():
            errors.append(f"Tool '{name}': param '{pname}' must be integer.")
        if "minimum" in pschema and val < pschema["minimum"]:
            errors.append(f"Tool '{name}': param '{pname}'={val} < minimum {pschema['minimum']}.")
        if "maximum" in pschema and val > pschema["maximum"]:
            errors.append(f"Tool '{name}': param '{pname}'={val} > maximum {pschema['maximum']}.")
        if "enum" in pschema and val not in pschema["enum"]:
            errors.append(f"Tool '{name}': param '{pname}'={val!r} not in {pschema['enum']}.")

    return errors


def export_registry_json(out_path: Path) -> None:
    """Write the full registry to a JSON file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(TOOLS, indent=2))
