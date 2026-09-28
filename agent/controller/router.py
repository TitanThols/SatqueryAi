"""
agent/controller/router.py
Rule-based controller that:
1. Classifies query → task
2. Validates inputs (delegates to input_validator)
3. Selects tools from registry
4. Executes tools in order, collecting outputs + timings
5. Merges outputs
6. Returns ExecutionTrace (JSON-serializable)

Optional: LLM augmentation via Ollama or OpenAI (configurable via env vars).
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ── Execution trace data structure ────────────────────────────────────────────

@dataclass
class ToolCall:
    tool_name:   str
    params:      dict
    output:      Optional[dict]
    error:       Optional[str]
    started_at:  float   # unix timestamp
    duration_s:  float


@dataclass
class ExecutionTrace:
    query:          str
    task:           str
    confidence:     float
    answer:         Optional[str]
    image_outputs:  list[str]           # paths to generated overlays / masks
    tool_calls:     list[ToolCall] = field(default_factory=list)
    validation_errors: list[str]  = field(default_factory=list)
    validation_warnings: list[str] = field(default_factory=list)
    total_duration_s: float = 0.0
    model_label:    str = "baseline (not adapted)"
    error:          Optional[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


# ── Keyword-based task classifier ─────────────────────────────────────────────

_CHANGE_KEYWORDS = [
    "change", "changed", "differ", "before", "after", "temporal",
    "growth", "shrink", "deforestation", "urban expansion", "flood",
    "between", "t1", "t2", "bi-temporal", "bitemporal"
]
_SAR_KEYWORDS = [
    "sar", "radar", "s1", "sentinel-1", "backscatter", "vv", "vh",
    "risat", "asar", "synthetic aperture"
]
_OPTICAL_SAR_KEYWORDS = [
    "fuse", "fusion", "optical", "complement", "built-up", "land cover",
    "water body", "vegetation"
]
_GROUNDING_KEYWORDS = [
    "where is", "where are", "locate", "find", "detect", "show me",
    "highlight", "bounding box", "region", "ground", "grounding",
    "segment", "segmentation", "industrial zone", "industrial areas",
    "provide visual grounding", "visual grounding",
]
_CAPTION_KEYWORDS = [
    "describe", "caption", "summarize",
    "explain this", "explain the",
    "tell me everything", "generate caption",
    "explain this satellite",
]


def classify_task(query: str, n_images: int, roles: list[str]) -> str:
    """
    Pure rule-based task classification.
    Returns one of: single_vqa, caption, ground, change, sar_optical
    """
    q = query.lower()
    roles_set = set(roles)

    # Structural signals take priority
    if "t1" in roles_set and "t2" in roles_set:
        return "change"
    if "optical" in roles_set and "sar" in roles_set:
        return "sar_optical"

    if n_images == 2:
        # Two images with no role tags → heuristic
        for kw in _CHANGE_KEYWORDS:
            if kw in q:
                return "change"
        return "change"    # default for 2 images

    # Single image → keyword-based
    for kw in _GROUNDING_KEYWORDS:
        if kw in q:
            return "ground"
    for kw in _CAPTION_KEYWORDS:
        if kw in q:
            return "caption"

    return "single_vqa"   # default single-image task


# ── Tool selector ─────────────────────────────────────────────────────────────

def select_tools(task: str) -> list[str]:
    """Return ordered list of tool names for a given task."""
    return {
        "single_vqa":  ["vqa"],
        "caption":     ["caption"],
        "ground":      ["ground"],
        "change":      ["change_map", "change_vqa"],
        "sar_optical": ["sar_optical_fusion"],
    }.get(task, [])


# ── LLM-augmented classifier (optional, non-blocking) ─────────────────────────

def _llm_classify(query: str, n_images: int, roles: list[str]) -> Optional[str]:
    """
    Optional: use Ollama (local) or OpenAI to refine classification.
    Returns task string or None if unavailable.
    """
    llm_backend = os.environ.get("SATQUERY_LLM_BACKEND", "none").lower()
    if llm_backend == "none":
        return None

    system_prompt = (
        "You are a remote sensing task classifier. "
        "Given a user query and image context, output ONLY one of: "
        "single_vqa, caption, ground, change, sar_optical\n"
        "No explanation."
    )
    user_msg = (
        f"Query: {query}\n"
        f"Number of images: {n_images}\n"
        f"Image roles: {roles}\n"
        "Task:"
    )

    try:
        if llm_backend == "ollama":
            import requests
            model = os.environ.get("SATQUERY_LLM_MODEL", "qwen2.5:1.5b")
            resp = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": model, "prompt": f"{system_prompt}\n{user_msg}", "stream": False},
                timeout=30,
            )
            resp.raise_for_status()
            raw = resp.json().get("response", "").strip().lower()
            valid = {"single_vqa", "caption", "ground", "change", "sar_optical"}
            for word in raw.split():
                if word in valid:
                    return word

        elif llm_backend == "openai":
            import openai
            client = openai.OpenAI()
            resp = client.chat.completions.create(
                model=os.environ.get("SATQUERY_LLM_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=10,
                temperature=0,
            )
            raw = resp.choices[0].message.content.strip().lower()
            valid = {"single_vqa", "caption", "ground", "change", "sar_optical"}
            if raw in valid:
                return raw

    except Exception as exc:
        logger.warning("LLM classifier failed: %s — falling back to rule-based.", exc)

    return None


# ── Main controller ───────────────────────────────────────────────────────────

class AgentController:
    """
    Orchestrates the full agent pipeline.
    Usage:
        controller = AgentController()
        trace = controller.run(query, file_paths, roles, upload_dir)
    """

    def __init__(self, tool_executor=None):
        """
        tool_executor: callable(tool_name, params, metas) → dict
        If None, tools are imported from agent/tools/executor.py at runtime.
        """
        self._executor = tool_executor

    def _get_executor(self):
        if self._executor:
            return self._executor
        from agent.tools.executor import execute_tool
        return execute_tool

    def run(
        self,
        query:       str,
        file_paths:  list[Path],
        roles:       list[str],
        upload_dir:  Path,
    ) -> ExecutionTrace:
        from agent.validators.input_validator import validate_inputs
        from models.registry.tool_registry import validate_tool_call

        t_start = time.monotonic()

        # 1. Classify task (LLM-augmented if configured, else rule-based)
        llm_task = _llm_classify(query, len(file_paths), roles)
        rule_task = classify_task(query, len(file_paths), roles)
        task = llm_task if llm_task else rule_task
        logger.info("Task classified as '%s' (llm=%s, rule=%s)", task, llm_task, rule_task)

        trace = ExecutionTrace(
            query=query,
            task=task,
            confidence=0.0,
            answer=None,
            image_outputs=[],
        )

        # 2. Validate inputs
        vr = validate_inputs(file_paths, roles, task)
        trace.validation_errors   = vr.errors
        trace.validation_warnings = vr.warnings

        if not vr.ok:
            trace.error = "Input validation failed: " + "; ".join(vr.errors)
            trace.total_duration_s = time.monotonic() - t_start
            return trace

        # 3. Select + execute tools
        tools = select_tools(task)
        if not tools:
            trace.error = f"No tools available for task '{task}'."
            trace.total_duration_s = time.monotonic() - t_start
            return trace

        # Build common params
        common_params = {
            "query": query,
            "upload_dir": str(upload_dir),
        }

        accumulated_outputs: dict[str, Any] = {}
        executor = self._get_executor()

        for tool_name in tools:
            # Merge accumulated_outputs into params (e.g. change_map feeds change_vqa)
            params = {**common_params, **accumulated_outputs}

            # Schema validation
            schema_errors = validate_tool_call(tool_name, params)
            if schema_errors:
                logger.warning("Schema errors for %s: %s", tool_name, schema_errors)
                # Non-fatal: proceed with warning
                trace.validation_warnings.extend(schema_errors)

            t_tool = time.monotonic()
            try:
                output = executor(tool_name, params, vr.metas, upload_dir)
                error = None
            except Exception as exc:
                logger.exception("Tool '%s' raised: %s", tool_name, exc)
                output = None
                error = str(exc)

            tc = ToolCall(
                tool_name=tool_name,
                params={k: v for k, v in params.items() if k != "upload_dir"},
                output=output,
                error=error,
                started_at=t_tool,
                duration_s=time.monotonic() - t_tool,
            )
            trace.tool_calls.append(tc)

            if output:
                accumulated_outputs.update(output)
            elif error:
                trace.error = f"Tool '{tool_name}' failed: {error}"
                break  # stop pipeline on tool failure

        # 4. Merge outputs into trace
        trace.answer       = accumulated_outputs.get("answer") or accumulated_outputs.get("caption")
        trace.confidence   = float(accumulated_outputs.get("confidence", 0.0))
        trace.model_label  = accumulated_outputs.get("adapter_status", "baseline (not adapted)")

        for key in ("change_mask_path", "class_mask_path", "overlay_path"):
            if val := accumulated_outputs.get(key):
                trace.image_outputs.append(str(val))

        trace.total_duration_s = time.monotonic() - t_start
        return trace
