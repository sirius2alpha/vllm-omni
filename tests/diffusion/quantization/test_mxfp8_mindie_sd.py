# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU contract tests for the lazy MindIE-SD MXFP8 compatibility bridge."""

import importlib
from types import ModuleType

import pytest

from vllm_omni.quantization import _mindie_sd

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def test_bridge_imports_mindie_sd_only_when_requested(monkeypatch):
    calls = []
    ops = ModuleType("mindiesd.layers.quant.mxfp8_functional")
    for name in _mindie_sd._REQUIRED_API:
        setattr(ops, name, object())

    def import_module(name):
        calls.append(name)
        return ops

    monkeypatch.setattr(importlib, "import_module", import_module)
    assert calls == []
    assert _mindie_sd.get_mxfp8_ops() is ops
    assert calls == [_mindie_sd._MINDIE_MXFP8_MODULE]


def test_bridge_reports_missing_mindie_sd(monkeypatch):
    def missing_module(name):
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(importlib, "import_module", missing_module)
    with pytest.raises(ImportError, match="requires a compatible MindIE-SD"):
        _mindie_sd.get_mxfp8_ops()


def test_bridge_reports_incompatible_mindie_sd(monkeypatch):
    monkeypatch.setattr(importlib, "import_module", lambda name: ModuleType(name))
    with pytest.raises(ImportError, match="Missing API.*forward_2d"):
        _mindie_sd.get_mxfp8_ops()
