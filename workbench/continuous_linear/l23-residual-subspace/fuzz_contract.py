"""Linear-side official-contract fuzz for the L23 candidate (WA discipline).

The official surface includes unseen shapes, inference_mode/no_grad harnesses,
CPU-only execution, extreme magnitudes, and repeat calibration determinism.
This hunts for any exception on that surface for the weight/activation APIs
(the changed side). Exit code 1 lists repro cases.
"""

from __future__ import annotations

import importlib.util
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))

sys.path.insert(0, os.path.join(ROOT, "evaluator"))
import reference_hif4 as ref  # noqa: E402

SOLUTION = os.path.join(HERE, "candidate", "solution.py")
spec = importlib.util.spec_from_file_location("fuzz_l23", SOLUTION)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

FP4_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
DEVICE = "cpu"
torch.set_default_device(DEVICE)


def make_pair(tokens: int, channels: int, seed: int, extreme: bool = False):
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    dense = torch.randn(tokens, channels, generator=generator)
    if extreme:
        dense = dense * 330000.0
    blocks = channels // 16
    grouped = dense.unflatten(-1, (blocks, 16))
    scale = (grouped.abs().amax(-1) / 6.0).clamp(min=1e-6, max=1e6)
    normalized = grouped / scale[..., None]
    sign = normalized.sign()
    index = torch.argmin((normalized.abs()[..., None] - FP4_GRID).abs(), dim=-1)
    quant = (sign * FP4_GRID[index]).flatten(-2, -1)
    return quant, scale


SHAPES = [
    ("narrow", 16, 256),
    ("wide", 32, 8192),
    ("nonpow2in", 24, 1280),
    ("big", 64, 4096),
]


def run() -> list[str]:
    failures: list[str] = []

    for tag, out_f, in_f in SHAPES:
        label = f"l23/{tag}"
        try:
            wq, ws = make_pair(out_f, in_f, 100 + in_f)
            act_pairs = [make_pair(48, in_f, 200 + in_f + k) for k in range(2)]
            state = module.hif4_calibration_and_quantize_weight(wq, ws, act_pairs)
            ref.validate_hif4_params(state["weight_params"], (out_f, in_f))
            for pair in act_pairs:
                act = module.hif4_dynamic_quantize_activation(
                    pair[0], pair[1], state["activation_state"]
                )
                ref.validate_hif4_params(act, pair[0].shape)
        except Exception as error:  # noqa: BLE001
            failures.append(f"{label} calib: {type(error).__name__}: {error}")

    # extreme magnitudes
    try:
        wq, ws = make_pair(16, 256, 777, extreme=True)
        act_pairs = [make_pair(24, 256, 778 + k) for k in range(2)]
        state = module.hif4_calibration_and_quantize_weight(wq, ws, act_pairs)
        ref.validate_hif4_params(state["weight_params"], (16, 256))
    except Exception as error:  # noqa: BLE001
        failures.append(f"l23/extreme: {type(error).__name__}: {error}")

    # inference_mode + no_grad harness
    for mode_name, ctx in (("inference_mode", torch.inference_mode()), ("no_grad", torch.no_grad())):
        try:
            with ctx:
                wq, ws = make_pair(16, 512, 900 + len(mode_name))
                act_pairs = [make_pair(32, 512, 950 + len(mode_name) + k) for k in range(2)]
                state = module.hif4_calibration_and_quantize_weight(wq, ws, act_pairs)
                ref.validate_hif4_params(state["weight_params"], (16, 512))
                act = module.hif4_dynamic_quantize_activation(
                    act_pairs[0][0], act_pairs[0][1], state["activation_state"]
                )
                ref.validate_hif4_params(act, act_pairs[0][0].shape)
        except Exception as error:  # noqa: BLE001
            failures.append(f"l23/{mode_name}: {type(error).__name__}: {error}")

    # repeat calibration determinism
    try:
        wq, ws = make_pair(16, 1024, 21)
        act_pairs = [make_pair(40, 1024, 22 + k) for k in range(2)]
        s1 = module.hif4_calibration_and_quantize_weight(wq, ws, act_pairs)
        s2 = module.hif4_calibration_and_quantize_weight(wq, ws, act_pairs)
        p1, p2 = s1["weight_params"], s2["weight_params"]
        for key in p1:
            if not torch.equal(p1[key], p2[key]):
                failures.append(f"l23/repeat: {key} differs between identical runs")
                break
    except Exception as error:  # noqa: BLE001
        failures.append(f"l23/repeat: {type(error).__name__}: {error}")

    return failures


def main() -> int:
    failures = run()
    if failures:
        print(f"--- l23: {len(failures)} failure(s)")
        for item in failures:
            print("   ", item)
        return 1
    print("l23: all fuzz checks PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())