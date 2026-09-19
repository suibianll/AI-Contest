"""v252 attention states must equal v246's (A-GR1 block composed unchanged).

v251's two changes are Linear-side, so the attention calibration path of v252
is v245's wrapper plus the same appended A-GR1 block v246 carries.  Byte-equal
q/k/v states on all six real attention layers is the falsifiable check.
"""

from __future__ import annotations

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


def signature(state):
    if torch.is_tensor(state):
        return (
            "tensor",
            tuple(state.shape),
            str(state.dtype),
            state.detach().to("cpu").contiguous().numpy().tobytes(),
        )
    if isinstance(state, dict):
        return {str(k): signature(v) for k, v in state.items()}
    if isinstance(state, (list, tuple)):
        return ["seq", [signature(v) for v in state]]
    return state


def main() -> int:
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    v246 = load("v246", ROOT / "solutions/20260911_v246_attention-agr1-on-v245_scoreNA_timeNA/solution.py")
    v252 = load("v252", HERE / "candidate_v252" / "solution.py")
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
    all_ok = True
    for layer in layers:
        calibration = [
            {
                role: tuple(t.to(dev) for t in v2._pair(pack["calibration_qkv"][s][layer][i]))
                for i, role in enumerate(("q", "k", "v"))
            }
            for s in range(splits)
        ]
        states_a = v246.hif4_calibration_attention(calibration, qh, kvh, hd)
        states_b = v252.hif4_calibration_attention(calibration, qh, kvh, hd)
        same = all(
            signature(states_a[side]) == signature(states_b[side])
            for side in ("q_state", "k_state", "v_state")
        )
        arm_a = states_a["q_state"].get("agr1_arm")
        arm_b = states_b["q_state"].get("agr1_arm")
        print(f"layer {layer}: states {'IDENTICAL' if same else 'DIFFER'}  agr1_arm v246={arm_a} v252={arm_b}", flush=True)
        all_ok = all_ok and same
    print("OVERALL:", "PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
