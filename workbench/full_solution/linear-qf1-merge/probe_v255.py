"""v255 verification: what exactly changed when block-smooth sizes gained 32/64.

Calibrates every attention layer with v254 (workspace root) and v255 and reports
block_smooth_size / block_smooth_seed / block_smooth_signs hash, the A-GR1 arm,
and whether the deployed dynamic q/k outputs differ on one real sample.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def tensor_hash(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        return hashlib.sha256(
            value.detach().to("cpu").contiguous().numpy().tobytes()
        ).hexdigest()[:12]
    return repr(value)


def main() -> int:
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    v254 = load("p_v254", ROOT / "solution.py")
    v255 = load("p_v255", HERE / "candidate_v255" / "solution.py")
    sys.path.insert(0, str(ROOT / "evaluator"))
    import official_eval as v2  # noqa: PLC0415

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pack = torch.load(
        ROOT / "artifacts/official_eval/cache/qwen3.5-4b-proxy-v2.pt",
        map_location="cpu", mmap=True, weights_only=False,
    )
    qh = int(pack["q_heads"]); kvh = int(pack["kv_heads"]); hd = int(pack["head_dim"])
    splits = len(pack["calibration_qkv"])
    layers = [
        layer for layer in range(len(pack["calibration_qkv"][0]))
        if all(pack["calibration_qkv"][s][layer] is not None for s in range(splits))
    ]
    for layer in layers:
        calibration = [
            {
                role: tuple(t.to(dev) for t in v2._pair(pack["calibration_qkv"][s][layer][i]))
                for i, role in enumerate(("q", "k", "v"))
            }
            for s in range(splits)
        ]
        states = {
            "v254": v254.hif4_calibration_attention(calibration, qh, kvh, hd),
            "v255": v255.hif4_calibration_attention(calibration, qh, kvh, hd),
        }
        row = []
        for name in ("v254", "v255"):
            q = states[name]["q_state"]
            row.append(
                f"{name}: size={q.get('block_smooth_size')} seed={q.get('block_smooth_seed')} "
                f"signs={tensor_hash(q.get('block_smooth_signs'))} agr1={q.get('agr1_arm')}"
            )
        # deployed dynamic outputs on one calibration sample
        diffs = {}
        sample = pack["calibration_qkv"][0][layer]
        for role, heads in (("q", qh), ("k", kvh)):
            pair = tuple(t.to(dev) for t in v2._pair(sample[("q", "k").index(role)]))
            outs = []
            for name in ("v254", "v255"):
                out = getattr(v255 if name == "v255" else v254, f"hif4_dynamic_quantize_{role}")(
                    pair[0], pair[1], heads, hd, states[name][f"{role}_state"]
                )
                outs.append(out)
            same = all(
                torch.equal(outs[0][k].detach().cpu(), outs[1][k].detach().cpu())
                for k in outs[0]
            )
            diffs[role] = "SAME" if same else "DIFFER"
        print(f"layer {layer:>3}: {' | '.join(row)} | dynamic {diffs}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
