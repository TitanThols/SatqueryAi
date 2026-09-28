"""
tests/test_validators.py
Unit tests for input validator and routing logic.
Run: pytest tests/ -v
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
from PIL import Image


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_images(tmp_path):
    """Create temporary PNG test images."""
    paths = []
    for i in range(3):
        img = Image.fromarray(
            (np.random.rand(256, 256, 3) * 255).astype(np.uint8), "RGB"
        )
        p = tmp_path / f"test_{i}.png"
        img.save(p)
        paths.append(p)
    return paths, tmp_path


# ── Validator tests ────────────────────────────────────────────────────────────

class TestInputValidator:

    def test_single_image_vqa_valid(self, tmp_images):
        paths, _ = tmp_images
        from agent.validators.input_validator import validate_inputs
        vr = validate_inputs([paths[0]], ["single"], "single_vqa")
        assert vr.ok, f"Expected OK but got errors: {vr.errors}"
        assert len(vr.metas) == 1

    def test_single_image_caption_valid(self, tmp_images):
        paths, _ = tmp_images
        from agent.validators.input_validator import validate_inputs
        vr = validate_inputs([paths[0]], ["single"], "caption")
        assert vr.ok

    def test_single_image_grounding_valid(self, tmp_images):
        paths, _ = tmp_images
        from agent.validators.input_validator import validate_inputs
        vr = validate_inputs([paths[0]], ["single"], "ground")
        assert vr.ok

    def test_change_task_requires_two_images(self, tmp_images):
        paths, _ = tmp_images
        from agent.validators.input_validator import validate_inputs
        # Only 1 image for change task → should fail
        vr = validate_inputs([paths[0]], ["t1"], "change")
        assert not vr.ok
        assert any("2" in e for e in vr.errors)

    def test_change_task_requires_t1_t2_roles(self, tmp_images):
        paths, _ = tmp_images
        from agent.validators.input_validator import validate_inputs
        # Wrong roles
        vr = validate_inputs([paths[0], paths[1]], ["single", "single"], "change")
        assert not vr.ok

    def test_change_task_valid_with_t1_t2(self, tmp_images):
        paths, _ = tmp_images
        from agent.validators.input_validator import validate_inputs
        vr = validate_inputs([paths[0], paths[1]], ["t1", "t2"], "change")
        assert vr.ok, f"Expected OK but got: {vr.errors}"

    def test_sar_optical_requires_two_images(self, tmp_images):
        paths, _ = tmp_images
        from agent.validators.input_validator import validate_inputs
        vr = validate_inputs([paths[0]], ["optical"], "sar_optical")
        assert not vr.ok

    def test_sar_optical_valid(self, tmp_images):
        paths, _ = tmp_images
        from agent.validators.input_validator import validate_inputs
        vr = validate_inputs([paths[0], paths[1]], ["optical", "sar"], "sar_optical")
        assert vr.ok, f"Expected OK but got: {vr.errors}"

    def test_no_images_fails(self):
        from agent.validators.input_validator import validate_inputs
        vr = validate_inputs([], [], "single_vqa")
        assert not vr.ok
        assert "No images" in vr.errors[0]

    def test_unsupported_extension(self, tmp_path):
        from agent.validators.input_validator import validate_inputs
        bad_file = tmp_path / "test.xyz"
        bad_file.write_bytes(b"fake data")
        vr = validate_inputs([bad_file], ["single"], "single_vqa")
        assert not vr.ok

    def test_mismatched_roles_length(self, tmp_images):
        paths, _ = tmp_images
        from agent.validators.input_validator import validate_inputs
        vr = validate_inputs([paths[0], paths[1]], ["single"], "single_vqa")
        assert not vr.ok

    def test_unknown_task_fails(self, tmp_images):
        paths, _ = tmp_images
        from agent.validators.input_validator import validate_inputs
        vr = validate_inputs([paths[0]], ["single"], "totally_invalid_task")
        assert not vr.ok

    def test_meta_extraction_png(self, tmp_images):
        paths, _ = tmp_images
        from agent.validators.input_validator import validate_inputs
        vr = validate_inputs([paths[0]], ["single"], "single_vqa")
        assert vr.ok
        assert vr.metas[0].width == 256
        assert vr.metas[0].height == 256


# ── Router tests ───────────────────────────────────────────────────────────────

class TestRouter:

    def test_single_vqa_default(self):
        from agent.controller.router import classify_task
        task = classify_task("How many buildings are there?", 1, ["single"])
        assert task == "single_vqa"

    def test_caption_keyword(self):
        from agent.controller.router import classify_task
        task = classify_task("Describe this satellite image.", 1, ["single"])
        assert task == "caption"

    def test_grounding_where_keyword(self):
        from agent.controller.router import classify_task
        task = classify_task("Where is the water body?", 1, ["single"])
        assert task == "ground"

    def test_grounding_locate_keyword(self):
        from agent.controller.router import classify_task
        task = classify_task("Locate the buildings.", 1, ["single"])
        assert task == "ground"

    def test_change_from_roles(self):
        from agent.controller.router import classify_task
        task = classify_task("What do you see?", 2, ["t1", "t2"])
        assert task == "change"

    def test_change_from_keyword(self):
        from agent.controller.router import classify_task
        task = classify_task("What has changed between these images?", 2, ["t1", "t2"])
        assert task == "change"

    def test_sar_optical_from_roles(self):
        from agent.controller.router import classify_task
        task = classify_task("Analyze the scene.", 2, ["optical", "sar"])
        assert task == "sar_optical"

    def test_two_images_no_roles_defaults_to_change(self):
        from agent.controller.router import classify_task
        task = classify_task("What changed here?", 2, ["single", "single"])
        assert task == "change"


# ── Tool registry tests ────────────────────────────────────────────────────────

class TestToolRegistry:

    def test_list_tools(self):
        from models.registry.tool_registry import list_tools
        tools = list_tools()
        assert "vqa" in tools
        assert "change_map" in tools
        assert "sar_optical_fusion" in tools

    def test_get_valid_tool(self):
        from models.registry.tool_registry import get_tool
        tool = get_tool("vqa")
        assert tool["name"] == "vqa"

    def test_get_invalid_tool(self):
        from models.registry.tool_registry import get_tool
        with pytest.raises(KeyError):
            get_tool("nonexistent_tool")

    def test_validate_vqa_params_valid(self):
        from models.registry.tool_registry import validate_tool_call
        errors = validate_tool_call("vqa", {"query": "What is this?", "model_size": "small"})
        assert errors == []

    def test_validate_vqa_params_missing_required(self):
        from models.registry.tool_registry import validate_tool_call
        errors = validate_tool_call("vqa", {"model_size": "small"})
        assert any("query" in e for e in errors)

    def test_validate_vqa_params_invalid_enum(self):
        from models.registry.tool_registry import validate_tool_call
        errors = validate_tool_call("vqa", {"query": "Q?", "model_size": "xlarge"})
        assert any("model_size" in e for e in errors)

    def test_validate_change_map_threshold_range(self):
        from models.registry.tool_registry import validate_tool_call
        errors = validate_tool_call("change_map", {"threshold": 1.5})
        assert any("threshold" in e for e in errors)

    def test_validate_change_map_threshold_valid(self):
        from models.registry.tool_registry import validate_tool_call
        errors = validate_tool_call("change_map", {"threshold": 0.5})
        assert errors == []
