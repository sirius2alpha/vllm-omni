# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Model calibration and export for MindIE-SD MXFP8 DAOS."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

import torch
from torch import nn

from .mxfp8_flex_smooth import (
    MXFP8CalibrationArtifact,
    _calibrate_mxfp8,
    _load_mxfp8_artifact,
    _merge_mxfp8_state_dict,
    _Searcher,
    _SearchResult,
)

MXFP8DAOSArtifact = MXFP8CalibrationArtifact


def _default_daos_searcher(
    weight: torch.Tensor,
    activations: torch.Tensor,
    *,
    bias: torch.Tensor | None,
    max_calibration_tokens: int,
    train_iters: int,
    learning_rate: float,
    base_z_threshold: float,
    target_outlier_min: float,
    target_outlier_max: float,
    oasq_max_iters: int,
    z_step: float,
) -> _SearchResult:
    try:
        from mindiesd.quantization import search_mxfp8_daos
    except ImportError as exc:
        raise ImportError("MXFP8 DAOS calibration requires MindIE-SD with search_mxfp8_daos installed") from exc
    return search_mxfp8_daos(
        weight,
        activations,
        bias=bias,
        train_iters=train_iters,
        learning_rate=learning_rate,
        max_calibration_tokens=max_calibration_tokens,
        base_z_threshold=base_z_threshold,
        target_outlier_min=target_outlier_min,
        target_outlier_max=target_outlier_max,
        oasq_max_iters=oasq_max_iters,
        z_step=z_step,
    )


def load_mxfp8_daos_artifact(path: str | Path) -> MXFP8DAOSArtifact:
    """Load a DAOS calibration artifact written by this module."""
    return _load_mxfp8_artifact(path, "mxfp8_daos_v1", "DAOS")


def merge_mxfp8_daos_state_dict(
    base_state_dict: dict[str, torch.Tensor],
    artifact: MXFP8DAOSArtifact | str | Path,
) -> dict[str, torch.Tensor]:
    """Overlay DAOS MXFP8 tensors on a base checkpoint state dict."""
    return _merge_mxfp8_state_dict(base_state_dict, artifact, algorithm_name="DAOS")


def calibrate_mxfp8_daos(
    model: nn.Module,
    calibration_batches: Iterable[object],
    *,
    output_path: str | Path | None = None,
    include: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
    max_calibration_tokens: int = 128,
    train_iters: int = 100,
    learning_rate: float = 1e-3,
    base_z_threshold: float = 2.5,
    target_outlier_min: float = 0.01,
    target_outlier_max: float = 0.15,
    oasq_max_iters: int = 8,
    z_step: float = 0.2,
    quantizer: _Searcher | None = None,
    search_device: str | torch.device | None = None,
    forward_batch: Callable[[nn.Module, object], object] | None = None,
    module_name_mapper: Callable[[str], str] | None = None,
) -> MXFP8DAOSArtifact:
    """Capture selected Linear inputs, call MindIE-SD DAOS and export tensors.

    When ``search_device`` is omitted, the layer search runs on the module's
    current device. Model execution is observational and the model weights are
    not changed.
    """
    return _calibrate_mxfp8(
        model,
        calibration_batches,
        result_format="mxfp8_daos_v1",
        searcher=quantizer or _default_daos_searcher,
        search_options={
            "train_iters": train_iters,
            "learning_rate": learning_rate,
            "base_z_threshold": base_z_threshold,
            "target_outlier_min": target_outlier_min,
            "target_outlier_max": target_outlier_max,
            "oasq_max_iters": oasq_max_iters,
            "z_step": z_step,
        },
        max_output_features=None,
        search_device=search_device,
        output_path=output_path,
        include=include,
        exclude=exclude,
        max_calibration_tokens=max_calibration_tokens,
        forward_batch=forward_batch,
        module_name_mapper=module_name_mapper,
    )


__all__ = [
    "MXFP8DAOSArtifact",
    "calibrate_mxfp8_daos",
    "load_mxfp8_daos_artifact",
    "merge_mxfp8_daos_state_dict",
]
