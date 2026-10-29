# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import sys
import types
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file
from torch import nn

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(32, 16)
        self.head = nn.Linear(16, 4)

    def forward(self, x):
        return self.head(torch.tanh(self.proj(x)))


class FakeDAOSResult:
    def __init__(self, weight):
        self.weight = weight.to(torch.float8_e4m3fn)
        self.weight_scale = torch.full((weight.shape[0], weight.shape[1] // 32), 127, dtype=torch.uint8)
        self.mul_scale = torch.full((weight.shape[1],), 0.5)
        self.oasq_scale = torch.full((weight.shape[1],), 2.0)
        self.outlier_ratio = 0.0625
        self.initial_normalized_rmse = 0.3
        self.normalized_rmse = 0.2
        self.completed_iters = 5


def test_daos_calibration_exports_and_merges_mxfp8_tensors(tmp_path: Path):
    from vllm_omni.quantization.tools.mxfp8_daos import (
        calibrate_mxfp8_daos,
        load_mxfp8_daos_artifact,
        merge_mxfp8_daos_state_dict,
    )

    model = TinyModel().eval()
    seen = {}

    def fake_search(weight, activations, **kwargs):
        seen["weight_device"] = weight.device
        seen["activations"] = activations
        seen["kwargs"] = kwargs
        return FakeDAOSResult(weight)

    output_path = tmp_path / "daos-mxfp8.safetensors"
    artifact = calibrate_mxfp8_daos(
        model,
        [torch.randn(2, 8, 32)],
        output_path=output_path,
        include=("proj",),
        max_calibration_tokens=10,
        quantizer=fake_search,
        train_iters=5,
        learning_rate=0.002,
    )

    assert seen["weight_device"].type == "cpu"
    assert seen["activations"].shape == (10, 32)
    assert seen["kwargs"]["train_iters"] == 5
    assert seen["kwargs"]["learning_rate"] == 0.002
    assert seen["kwargs"]["bias"].shape == (16,)
    assert artifact.metadata["format"] == "mxfp8_daos_v1"
    assert artifact.metadata["quantized_layers"] == ["proj"]
    assert artifact.metadata["ignored_layers"] == ["head"]
    assert artifact.metadata["layers"]["proj"]["outlier_ratio"] == 0.0625
    assert artifact.metadata["layers"]["proj"]["initial_normalized_rmse"] == 0.3
    assert artifact.metadata["layers"]["proj"]["completed_iters"] == 5
    assert artifact.state_dict["proj.weight"].dtype == torch.float8_e4m3fn
    assert artifact.state_dict["proj.weight_scale"].dtype == torch.uint8
    assert artifact.state_dict["proj.mul_scale"].shape == (32,)
    assert output_path.exists()
    assert "proj.weight_scale" in load_file(output_path)

    loaded = load_mxfp8_daos_artifact(output_path)
    merged = merge_mxfp8_daos_state_dict(model.state_dict(), loaded)
    assert merged["proj.weight"].dtype == torch.float8_e4m3fn
    assert torch.equal(merged["head.weight"], model.state_dict()["head.weight"])


def test_daos_default_searcher_calls_mindiesd_public_api(monkeypatch):
    from vllm_omni.quantization.tools.mxfp8_daos import _default_daos_searcher

    seen = {}
    mindiesd_module = types.ModuleType("mindiesd")
    mindiesd_module.__path__ = []
    quantization_module = types.ModuleType("mindiesd.quantization")

    def search_mxfp8_daos(weight, activations, **kwargs):
        seen["weight"] = weight
        seen["activations"] = activations
        seen["kwargs"] = kwargs
        return "result"

    quantization_module.search_mxfp8_daos = search_mxfp8_daos
    monkeypatch.setitem(sys.modules, "mindiesd", mindiesd_module)
    monkeypatch.setitem(sys.modules, "mindiesd.quantization", quantization_module)

    weight = torch.randn(4, 32)
    activations = torch.randn(3, 32)
    result = _default_daos_searcher(
        weight,
        activations,
        bias=None,
        max_calibration_tokens=3,
        train_iters=7,
        learning_rate=0.002,
        base_z_threshold=2.0,
        target_outlier_min=0.02,
        target_outlier_max=0.2,
        oasq_max_iters=6,
        z_step=0.1,
    )

    assert result == "result"
    assert seen["weight"] is weight
    assert seen["activations"] is activations
    assert seen["kwargs"] == {
        "bias": None,
        "train_iters": 7,
        "learning_rate": 0.002,
        "max_calibration_tokens": 3,
        "base_z_threshold": 2.0,
        "target_outlier_min": 0.02,
        "target_outlier_max": 0.2,
        "oasq_max_iters": 6,
        "z_step": 0.1,
    }
