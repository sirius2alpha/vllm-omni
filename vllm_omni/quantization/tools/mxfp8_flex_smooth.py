# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Model-independent calibration and export for MindIE-SD MXFP8 Flex Smooth."""

from __future__ import annotations

import fnmatch
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import nn


@dataclass(frozen=True)
class MXFP8FlexSmoothArtifact:
    state_dict: dict[str, torch.Tensor]
    metadata: dict[str, object]
    output_path: Path | None


class _SearchResult(Protocol):
    weight: torch.Tensor
    weight_scale: torch.Tensor
    mul_scale: torch.Tensor
    alpha: float
    beta: float
    normalized_rmse: float


class _Searcher(Protocol):
    def __call__(
        self,
        weight: torch.Tensor,
        activations: torch.Tensor,
        *,
        bias: torch.Tensor | None,
        alpha_candidates: Sequence[float] | None,
        beta_candidates: Sequence[float] | None,
        search_step: float,
        max_calibration_tokens: int,
        max_output_features: int,
    ) -> _SearchResult: ...


def _matches(name: str, patterns: Sequence[str] | None) -> bool:
    return patterns is not None and any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns)


def _default_searcher(
    weight: torch.Tensor,
    activations: torch.Tensor,
    *,
    bias: torch.Tensor | None,
    alpha_candidates: Sequence[float] | None,
    beta_candidates: Sequence[float] | None,
    search_step: float,
    max_calibration_tokens: int,
    max_output_features: int,
) -> _SearchResult:
    try:
        from mindiesd.quantization import search_mxfp8_flex_smooth
    except ImportError as exc:
        raise ImportError("MXFP8 Flex calibration requires MindIE-SD with search_mxfp8_flex_smooth installed") from exc
    return search_mxfp8_flex_smooth(
        weight,
        activations,
        bias=bias,
        alpha_candidates=alpha_candidates,
        beta_candidates=beta_candidates,
        search_step=search_step,
        max_calibration_tokens=max_calibration_tokens,
        max_output_features=max_output_features,
    )


def _default_forward(model: nn.Module, batch: object) -> object:
    if isinstance(batch, torch.Tensor):
        return model(batch)
    if isinstance(batch, tuple):
        return model(*batch)
    if isinstance(batch, Mapping):
        return model(**batch)
    raise TypeError("Calibration batches must be tensors, tuples, mappings, or use forward_batch")


def _tensor_key(module_name: str, parameter_name: str) -> str:
    return f"{module_name}.{parameter_name}" if module_name else parameter_name


def _linear_input_width(module: nn.Module) -> int | None:
    """Return the K dimension for nn.Linear and compatible module wrappers."""
    weight = getattr(module, "weight", None)
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
        return None
    if isinstance(module, nn.Linear):
        width = module.in_features
    elif hasattr(module, "input_size_per_partition"):
        width = module.input_size_per_partition
    elif hasattr(module, "in_features"):
        width = module.in_features
    else:
        return None
    if not isinstance(width, int) or width != weight.shape[1]:
        return None
    return width


def load_mxfp8_flex_smooth_artifact(path: str | Path) -> MXFP8FlexSmoothArtifact:
    """Load a calibration artifact previously written by this module."""
    artifact_path = Path(path)
    with safe_open(str(artifact_path), framework="pt", device="cpu") as safetensor_file:
        file_metadata = safetensor_file.metadata() or {}
    try:
        metadata = json.loads(file_metadata["calibration"])
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{artifact_path} is not an MXFP8 Flex Smooth artifact") from exc
    if not isinstance(metadata, dict):
        raise ValueError(f"{artifact_path} has invalid MXFP8 Flex Smooth metadata")
    if metadata.get("format") != "mxfp8_flex_smooth_v1":
        raise ValueError(f"Unsupported MXFP8 Flex Smooth artifact format: {metadata.get('format')!r}")
    return MXFP8FlexSmoothArtifact(
        state_dict=load_file(str(artifact_path), device="cpu"), metadata=metadata, output_path=artifact_path
    )


def merge_mxfp8_flex_smooth_state_dict(
    base_state_dict: Mapping[str, torch.Tensor],
    artifact: MXFP8FlexSmoothArtifact | str | Path,
) -> dict[str, torch.Tensor]:
    """Overlay calibrated MXFP8 tensors on a base checkpoint state dict.

    The artifact's ``quantization_config`` must also be copied to the model
    component config. ``ignored_layers`` in that config leaves all unselected
    Linear modules in the base checkpoint dtype.
    """
    loaded_artifact = load_mxfp8_flex_smooth_artifact(artifact) if isinstance(artifact, (str, Path)) else artifact
    quantized_layers = loaded_artifact.metadata.get("quantized_layers")
    if not isinstance(quantized_layers, list) or not quantized_layers:
        raise ValueError("artifact metadata must contain non-empty quantized_layers")
    quantization_config = loaded_artifact.metadata.get("quantization_config")
    if (
        not isinstance(quantization_config, dict)
        or quantization_config.get("quant_method") != "mxfp8"
        or quantization_config.get("is_checkpoint_mxfp8_serialized") is not True
        or quantization_config.get("require_smooth_scale") is not True
        or quantization_config.get("ignored_layers") != loaded_artifact.metadata.get("ignored_layers")
    ):
        raise ValueError("artifact metadata contains an invalid MXFP8 quantization_config")

    for name in quantized_layers:
        if not isinstance(name, str):
            raise ValueError("artifact quantized layer names must be strings")
        weight_key = _tensor_key(name, "weight")
        scale_key = _tensor_key(name, "weight_scale")
        mul_scale_key = _tensor_key(name, "mul_scale")
        if weight_key not in base_state_dict:
            raise KeyError(f"Base checkpoint is missing {weight_key!r}")
        if any(key not in loaded_artifact.state_dict for key in (weight_key, scale_key, mul_scale_key)):
            raise ValueError(f"Artifact is missing MXFP8 tensors for layer {name!r}")
        base_weight = base_state_dict[weight_key]
        quantized_weight = loaded_artifact.state_dict[weight_key]
        if (
            base_weight.ndim != 2
            or quantized_weight.shape != base_weight.shape
            or quantized_weight.dtype != torch.float8_e4m3fn
        ):
            raise ValueError(f"MXFP8 weight shape does not match the base checkpoint for layer {name!r}")
        expected_scales = (base_weight.shape[0], (base_weight.shape[1] + 31) // 32)
        if (
            loaded_artifact.state_dict[scale_key].shape != expected_scales
            or loaded_artifact.state_dict[scale_key].dtype != torch.uint8
        ):
            raise ValueError(f"MXFP8 weight_scale shape is invalid for layer {name!r}")
        multiplier = loaded_artifact.state_dict[mul_scale_key]
        if (
            multiplier.shape != (base_weight.shape[1],)
            or not multiplier.is_floating_point()
            or not bool(torch.isfinite(multiplier).all())
            or not bool((multiplier > 0).all())
        ):
            raise ValueError(f"MXFP8 mul_scale shape is invalid for layer {name!r}")

    return {**base_state_dict, **loaded_artifact.state_dict}


def calibrate_mxfp8_flex_smooth(
    model: nn.Module,
    calibration_batches: Iterable[object],
    *,
    output_path: str | Path | None = None,
    include: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
    max_calibration_tokens: int = 128,
    max_output_features: int = 128,
    search_step: float = 0.1,
    alpha_candidates: Sequence[float] | None = None,
    beta_candidates: Sequence[float] | None = None,
    quantizer: _Searcher | None = None,
    forward_batch: Callable[[nn.Module, object], object] | None = None,
    module_name_mapper: Callable[[str], str] | None = None,
) -> MXFP8FlexSmoothArtifact:
    """Calibrate selected Linear-like modules and export Omni MXFP8 tensors.

    Batches may be tensors, positional tuples, or keyword mappings. Frameworks
    with custom input protocols can provide ``forward_batch``. The model is only
    observed; its weights are not changed. Exported tensor names use standard
    module names followed by ``weight``, ``weight_scale`` and ``mul_scale``.
    The returned quantization config lists all unselected Linear modules as
    ``ignored_layers`` so the Omni loader keeps those base-checkpoint weights
    in their original dtype. ``module_name_mapper`` can align calibration names
    with the target checkpoint/runtime names.
    """
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if max_calibration_tokens <= 0 or max_output_features <= 0:
        raise ValueError("max_calibration_tokens and max_output_features must be positive")

    linears: dict[str, tuple[nn.Module, int]] = {}
    for name, module in model.named_modules():
        if (width := _linear_input_width(module)) is not None:
            linears[name] = (module, width)
    selected = {
        name: module_and_width
        for name, module_and_width in linears.items()
        if (include is None or _matches(name, include))
        and not _matches(name, exclude)
        and module_and_width[1] % 32 == 0
    }
    if not selected:
        raise ValueError("No MXFP8-compatible Linear modules matched include/exclude patterns")

    captures: dict[str, list[torch.Tensor]] = {name: [] for name in selected}
    captured_tokens = {name: 0 for name in selected}
    handles = []
    search: _Searcher = quantizer or _default_searcher
    forward = forward_batch or _default_forward
    module_modes = {module: module.training for module in model.modules()}

    def make_hook(name: str):
        def capture(_module: nn.Module, args: tuple[object, ...], kwargs: dict[str, object]) -> None:
            if captured_tokens[name] >= max_calibration_tokens:
                return
            value = (
                args[0]
                if args and isinstance(args[0], torch.Tensor)
                else next((item for item in kwargs.values() if isinstance(item, torch.Tensor)), None)
            )
            if not isinstance(value, torch.Tensor) or value.ndim < 1:
                raise TypeError(f"Calibration input for {name} must be a tensor")
            if value.shape[-1] != selected[name][1]:
                raise ValueError(
                    f"Calibration input for {name} has width {value.shape[-1]}, expected {selected[name][1]}"
                )
            rows = value.detach().reshape(-1, value.shape[-1])
            remaining = max_calibration_tokens - captured_tokens[name]
            rows = rows[:remaining].to(device="cpu", dtype=torch.float32)
            if rows.shape[0]:
                captures[name].append(rows)
                captured_tokens[name] += rows.shape[0]

        return capture

    try:
        for name, (module, _) in selected.items():
            handles.append(module.register_forward_pre_hook(make_hook(name), with_kwargs=True))
        model.eval()
        with torch.no_grad():
            for batch in calibration_batches:
                forward(model, batch)
        missing = [name for name, rows in captures.items() if not rows]
        if missing:
            raise ValueError(f"No calibration inputs were captured for Linear modules: {missing}")
    finally:
        for handle in handles:
            handle.remove()
        for module, training in module_modes.items():
            module.training = training

    state_dict: dict[str, torch.Tensor] = {}
    layer_metadata: dict[str, dict[str, object]] = {}
    mapped_names = {name: name if module_name_mapper is None else module_name_mapper(name) for name in linears}
    if any(not isinstance(mapped, str) or (name and not mapped) for name, mapped in mapped_names.items()):
        raise TypeError("module_name_mapper must return strings")
    if len(set(mapped_names.values())) != len(mapped_names):
        raise ValueError("module_name_mapper must return unique names for all Linear modules")
    for name, (module, _) in selected.items():
        activations = torch.cat(captures[name], dim=0)
        bias = getattr(module, "bias", None)
        result = search(
            module.weight.detach().to(device="cpu", dtype=torch.float32),
            activations,
            bias=None if bias is None else bias.detach().to(device="cpu", dtype=torch.float32),
            search_step=search_step,
            alpha_candidates=alpha_candidates,
            beta_candidates=beta_candidates,
            max_calibration_tokens=max_calibration_tokens,
            max_output_features=max_output_features,
        )
        target_name = mapped_names[name]
        state_dict[_tensor_key(target_name, "weight")] = result.weight.contiguous().cpu()
        state_dict[_tensor_key(target_name, "weight_scale")] = result.weight_scale.contiguous().cpu()
        state_dict[_tensor_key(target_name, "mul_scale")] = result.mul_scale.contiguous().to(torch.float32).cpu()
        layer_metadata[name] = {
            "checkpoint_name": target_name,
            "alpha": float(result.alpha),
            "beta": float(result.beta),
            "normalized_rmse": float(result.normalized_rmse),
            "calibration_tokens": int(captured_tokens[name]),
        }

    mapped_quantized_layers = [mapped_names[name] for name in selected]
    ignored_layers = [mapped_names[name] for name in linears if name not in selected]
    metadata: dict[str, object] = {
        "format": "mxfp8_flex_smooth_v1",
        "quantization_config": {
            "quant_method": "mxfp8",
            "is_checkpoint_mxfp8_serialized": True,
            "require_smooth_scale": True,
            "ignored_layers": ignored_layers,
        },
        "quantized_layers": mapped_quantized_layers,
        "ignored_layers": ignored_layers,
        "layers": layer_metadata,
    }
    artifact_path = None if output_path is None else Path(output_path)
    if artifact_path is not None:
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        save_file(
            state_dict,
            str(artifact_path),
            metadata={
                "format": "mxfp8_flex_smooth_v1",
                "quantization_config": json.dumps(metadata["quantization_config"], sort_keys=True),
                "calibration": json.dumps(metadata, sort_keys=True),
            },
        )
    return MXFP8FlexSmoothArtifact(state_dict=state_dict, metadata=metadata, output_path=artifact_path)
