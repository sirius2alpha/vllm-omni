# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Lazy bridge to the MindIE-SD functional MXFP8 linear API.

MindIE-SD owns the reusable, framework-neutral deployment math (smooth-scale
validation/application, MXFP8 activation quantization and the NPU quantized
GEMM). vLLM-Omni keeps the vLLM-specific concerns (configuration, checkpoint
naming/remapping, parameter registration, TP sharding, loading/offload) and
calls into MindIE-SD only on the Ascend NPU path.

The import is deferred so that importing ``vllm_omni.quantization`` never pulls
in MindIE-SD (and its torch_npu dependency) on non-NPU platforms.
"""

from __future__ import annotations

_MINDIE_MXFP8_MODULE = "mindiesd.layers.quant.mxfp8_functional"

_COMPATIBILITY_ERROR = (
    "NPU MXFP8 SmoothQuant execution requires a compatible MindIE-SD. "
    f"Install or upgrade the `mindiesd` package (module `{_MINDIE_MXFP8_MODULE}`), "
    "or add it to PYTHONPATH before serving an MXFP8 checkpoint on Ascend."
)

_REQUIRED_API = (
    "MXFP8PreparedWeight",
    "forward_2d",
    "normalize_weight_scale",
    "prepare_smooth_scale",
    "quant_matmul",
    "quantize_activation",
    "validate_smooth_scale",
)


def get_mxfp8_ops():
    """Return the MindIE-SD MXFP8 functional module, with compatibility errors."""
    import importlib

    try:
        ops = importlib.import_module(_MINDIE_MXFP8_MODULE)
    except ImportError as exc:
        raise ImportError(_COMPATIBILITY_ERROR) from exc
    missing = [name for name in _REQUIRED_API if not hasattr(ops, name)]
    if missing:
        raise ImportError(f"{_COMPATIBILITY_ERROR} Missing API(s): {', '.join(missing)}.")
    return ops
