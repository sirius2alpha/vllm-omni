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


class FakeSearchResult:
    def __init__(self, weight):
        self.weight = weight.to(torch.float8_e4m3fn)
        self.weight_scale = torch.full((weight.shape[0], weight.shape[1] // 32), 127, dtype=torch.uint8)
        self.mul_scale = torch.ones(weight.shape[1])
        self.alpha = 0.5
        self.beta = 0.25
        self.normalized_rmse = 0.125


def fake_search(weight, activations, **kwargs):
    assert activations.ndim == 2
    assert activations.shape[1] == weight.shape[1]
    return FakeSearchResult(weight)


def test_calibration_collects_generic_linear_inputs_and_exports(tmp_path: Path):
    from vllm_omni.quantization.tools.mxfp8_flex_smooth import calibrate_mxfp8_flex_smooth

    model = TinyModel().eval()
    batches = [torch.randn(2, 8, 32), torch.randn(1, 4, 32)]
    output_path = tmp_path / "flex-mxfp8.safetensors"

    artifact = calibrate_mxfp8_flex_smooth(
        model,
        batches,
        output_path=output_path,
        quantizer=fake_search,
        max_calibration_tokens=10,
    )

    assert artifact.metadata["quantized_layers"] == ["proj"]
    assert artifact.metadata["ignored_layers"] == ["head"]
    assert artifact.metadata["quantization_config"]["ignored_layers"] == ["head"]
    assert artifact.metadata["layers"]["proj"]["calibration_tokens"] == 10
    assert artifact.state_dict["proj.weight"].dtype == torch.float8_e4m3fn
    assert artifact.state_dict["proj.weight_scale"].dtype == torch.uint8
    assert artifact.state_dict["proj.mul_scale"].shape == (32,)
    assert "head.weight" not in artifact.state_dict
    assert output_path.exists()
    loaded = load_file(output_path)
    assert "proj.weight_scale" in loaded
    assert "proj.mul_scale" in loaded
    from vllm_omni.quantization.tools.mxfp8_flex_smooth import (
        load_mxfp8_flex_smooth_artifact,
        merge_mxfp8_flex_smooth_state_dict,
    )

    loaded_artifact = load_mxfp8_flex_smooth_artifact(output_path)
    merged = merge_mxfp8_flex_smooth_state_dict(model.state_dict(), loaded_artifact)
    assert merged["proj.weight"].dtype == torch.float8_e4m3fn
    assert torch.equal(merged["head.weight"], model.state_dict()["head.weight"])


def test_calibration_restores_model_mode_and_removes_hooks():
    from vllm_omni.quantization.tools.mxfp8_flex_smooth import calibrate_mxfp8_flex_smooth

    model = TinyModel().train()
    calibrate_mxfp8_flex_smooth(model, [torch.randn(2, 32)], quantizer=fake_search)

    assert model.training
    assert all(len(module._forward_pre_hooks) == 0 for module in model.modules())


def test_calibration_filters_modules_and_rejects_empty_capture():
    from vllm_omni.quantization.tools.mxfp8_flex_smooth import calibrate_mxfp8_flex_smooth

    model = TinyModel().eval()
    artifact = calibrate_mxfp8_flex_smooth(
        model,
        [torch.randn(2, 32)],
        include=("proj",),
        quantizer=fake_search,
    )
    assert artifact.metadata["quantized_layers"] == ["proj"]
    with pytest.raises(ValueError, match="No MXFP8-compatible Linear modules"):
        calibrate_mxfp8_flex_smooth(model, [torch.randn(2, 32)], include=("missing",), quantizer=fake_search)


def test_calibration_passes_bias_and_maps_runtime_names():
    from vllm_omni.quantization.tools.mxfp8_flex_smooth import calibrate_mxfp8_flex_smooth

    model = nn.Sequential(nn.Linear(32, 8))
    seen = {}

    def search(weight, activations, **kwargs):
        seen["bias"] = kwargs["bias"]
        return fake_search(weight, activations, **kwargs)

    artifact = calibrate_mxfp8_flex_smooth(
        model,
        [torch.randn(2, 32)],
        quantizer=search,
        module_name_mapper=lambda name: f"mapped.{name}",
    )

    assert seen["bias"].shape == (8,)
    assert artifact.metadata["quantized_layers"] == ["mapped.0"]
    assert "mapped.0.weight" in artifact.state_dict
    assert artifact.metadata["layers"]["0"]["checkpoint_name"] == "mapped.0"


def test_merge_artifact_overlays_only_quantized_tensors():
    from vllm_omni.quantization.tools.mxfp8_flex_smooth import (
        calibrate_mxfp8_flex_smooth,
        merge_mxfp8_flex_smooth_state_dict,
    )

    model = TinyModel()
    artifact = calibrate_mxfp8_flex_smooth(
        model,
        [torch.randn(2, 32)],
        quantizer=fake_search,
    )
    base_state_dict = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    merged = merge_mxfp8_flex_smooth_state_dict(base_state_dict, artifact)

    assert merged["proj.weight"].dtype == torch.float8_e4m3fn
    assert merged["proj.weight_scale"].dtype == torch.uint8
    assert merged["proj.mul_scale"].shape == (32,)
    assert torch.equal(merged["head.weight"], base_state_dict["head.weight"])


def test_merge_rejects_artifact_shape_mismatch():
    from vllm_omni.quantization.tools.mxfp8_flex_smooth import (
        calibrate_mxfp8_flex_smooth,
        merge_mxfp8_flex_smooth_state_dict,
    )

    model = TinyModel()
    artifact = calibrate_mxfp8_flex_smooth(model, [torch.randn(2, 32)], quantizer=fake_search)
    with pytest.raises(ValueError, match="shape does not match"):
        merge_mxfp8_flex_smooth_state_dict({"proj.weight": torch.empty(16, 64)}, artifact)


def test_calibration_cleans_hooks_after_model_forward_failure():
    from vllm_omni.quantization.tools.mxfp8_flex_smooth import calibrate_mxfp8_flex_smooth

    model = TinyModel().eval()
    with pytest.raises(RuntimeError, match="calibration failed"):
        calibrate_mxfp8_flex_smooth(
            model,
            [torch.randn(2, 32)],
            quantizer=fake_search,
            forward_batch=lambda _model, _batch: (_ for _ in ()).throw(RuntimeError("calibration failed")),
        )
    assert all(len(module._forward_pre_hooks) == 0 for module in model.modules())
