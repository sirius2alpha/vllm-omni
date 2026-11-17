# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
#!/usr/bin/env python3
"""Rebuild the two-level MXFP4 scales for an H3 delivery that shipped only placeholders.

Situation
---------
`MiniMax-H3-mxfp4mxfp8mixed-ref2va/transformer_ref` declares
`W4A4_MXFP4_DUALSCALE` but ships

* `weight_scale` tensors that are constant 127 (E8M0 == 1.0), and
* no `weight_dual_scale` tensors at all,

so the FP4 codes cannot be reconstructed (raw codes are ~30x too large) and the
checkpoint is rejected by the offline reader.

What this tool does
-------------------
The E2M1 codes themselves are intact, so the missing magnitude information is
recovered from the BF16 reference partition: for every group of 32 inputs the
least-squares scale s = <codes, w> / <codes, codes> is the scale the quantizer
used. It is split into the two levels the format expects

    weight_scale      = E8M0(floor(log2(s)))        uint8  [N, K/32]
    weight_dual_scale = s / weight_scale            float32[N, K/32, 1]

so the reader's normal MXFP4 path reconstructs the matrix with roughly the
intrinsic MXFP4 error. Codes, shard layout and index are left untouched; only the
degenerate scale tensors are replaced and the description is extended.

Nothing here is used at inference time: the file is an offline conversion helper
and is not imported by the serving path.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

_MX4 = "W4A4_MXFP4_DUALSCALE"
_VALUES = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float32)


def _decode_codes(weight: torch.Tensor) -> torch.Tensor:
    codes = torch.stack((weight & 15, weight >> 4), dim=-1).flatten(-2).long()
    return _VALUES[codes & 7] * torch.where(codes & 8 != 0, -1.0, 1.0)


def _split_scales(codes: torch.Tensor, ref: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-group (E8M0 code, float32 residual) for [N, groups, 32] blocks."""
    denom = (codes * codes).sum(dim=-1).clamp_min(1e-12)
    scale = (codes * ref).sum(dim=-1) / denom
    exponent = torch.floor(torch.log2(scale.clamp_min(1e-30))).to(torch.int32) + 127
    exponent = exponent.clamp(0, 254)
    fine = torch.ldexp(torch.ones((), dtype=torch.float32), exponent - 127)
    dual = (scale / fine).clamp_min(1e-30)
    zero = ref.abs().amax(dim=-1) <= 0
    exponent = torch.where(zero, torch.full_like(exponent, 127), exponent)
    dual = torch.where(zero, torch.ones_like(dual), dual)
    return exponent.to(torch.uint8), dual.unsqueeze(-1).contiguous()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--delivery", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    src, ref_dir, out = Path(args.delivery), Path(args.reference), Path(args.output)
    if out.exists():
        raise SystemExit(f"output already exists: {out}")

    desc = json.loads((src / "quant_model_description.json").read_text(encoding="utf-8"))
    index_doc = json.loads((src / "model.safetensors.index.json").read_text(encoding="utf-8"))
    index = index_doc["weight_map"]
    ref_index = json.loads((ref_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]
    by_file: dict[str, list[str]] = defaultdict(list)
    for name, file in index.items():
        by_file[file].append(name)

    derived: dict[str, torch.Tensor] = {}
    with ExitStack() as stack:
        readers: dict[str, object] = {}

        def read_ref(name: str) -> torch.Tensor:
            file = ref_index[name]
            if file not in readers:
                readers[file] = stack.enter_context(safe_open(str(ref_dir / file), framework="pt", device="cpu"))
            return readers[file].get_tensor(name)  # type: ignore[union-attr]

        for file, names in by_file.items():
            with safe_open(str(src / file), framework="pt", device="cpu") as fh:
                for name in names:
                    if desc.get(name) != _MX4 or not name.endswith(".weight"):
                        continue
                    weight = fh.get_tensor(name)
                    ref = read_ref(name).float()
                    if ref.shape[1] != weight.shape[1] * 2:
                        raise SystemExit(f"{name}: bad shapes {tuple(weight.shape)} vs {tuple(ref.shape)}")
                    n, k = ref.shape
                    groups = (k + 31) // 32
                    pad = groups * 32 - k
                    codes = torch.nn.functional.pad(_decode_codes(weight), (0, pad)).reshape(n, groups, 32)
                    padded_ref = torch.nn.functional.pad(ref, (0, pad)).reshape(n, groups, 32)
                    fine, dual = _split_scales(codes, padded_ref)
                    stem = name.removesuffix("weight")
                    derived[stem + "weight_scale"] = fine.reshape(n, groups).contiguous()
                    derived[stem + "weight_dual_scale"] = dual
                    del weight, ref, codes, padded_ref, fine, dual
            print(f"scales: {len(derived) // 2} MXFP4 layers fitted ({file})", flush=True)

    # the index gains the new tensors; scales keep living in their weight's shard
    new_index = dict(index)
    for name in derived:
        if name.endswith("weight_dual_scale"):
            new_index[name] = index[name.removesuffix("weight_dual_scale") + "weight"]

    out.mkdir(parents=True)
    new_desc = dict(desc)
    dual_by_file: dict[str, list[str]] = defaultdict(list)
    for name in new_index:
        if name.endswith("weight_dual_scale"):
            dual_by_file[new_index[name]].append(name)
            new_desc[name] = desc[name.removesuffix("weight_dual_scale") + "weight"]
    for file, names in by_file.items():
        with safe_open(str(src / file), framework="pt", device="cpu") as fh:
            tensors = {name: (derived[name] if name in derived else fh.get_tensor(name)) for name in names}
        for name in dual_by_file[file]:
            tensors[name] = derived[name]
        save_file(tensors, str(out / file), metadata={"format": "pt"})
        print(f"wrote {file}: {len(tensors)} tensors", flush=True)
        del tensors

    index_doc = {"metadata": {"format": "pt"}, "weight_map": new_index}
    (out / "model.safetensors.index.json").write_text(json.dumps(index_doc, indent=2), encoding="utf-8")
    shutil.copy2(out / "model.safetensors.index.json", out / "quant_model_weights.safetensors.index.json")
    (out / "quant_model_description.json").write_text(json.dumps(new_desc, indent=2), encoding="utf-8")
    for extra in ("config.json", "minimax_h3_best_practice.yaml"):
        if (src / extra).is_file():
            shutil.copy2(src / extra, out / extra)
    print(f"done -> {out} ({len(derived) // 2} layers, two-level scales rebuilt)")


if __name__ == "__main__":
    main()
