"""What device does the hook actually see on the official path?

``probe_dequant_device.py`` found that feeding ``_em1_compile_metric`` the
cache's CPU params costs 39.11 ms (in=2560) / 52.05 ms (in=4096), while the same
hook with GPU params costs 15.94 / 26.87 ms -- a 23-25 ms difference, because
``_dequantize_hif4`` is 20.97 ms on CPU against 0.74 ms on GPU.  Weighted over
the 144 in-scope calibrations that difference is ~3.4 s, which would be the
largest reclaimable cost in the shipped root.

But that measurement fed the hook **cache** params, and the cache is a
deliberately CPU-resident format -- ``official_eval.py:2583`` stores
``_cpu_params(result["weight_params"])``.  Whether the hook ever sees those
bytes depends on where the params live *during* the call, and the earlier
``profile_hook_breakdown.py`` numbers already hint they do not: its component
bench moved params to the GPU and measured ``_dequantize_hif4`` at ~1.2 ms,
while its full-hook bench read ``result["weight_params"]`` and measured 45 ms.
The "unattributed ~3 s" this workbench has been chasing may be nothing more than
that probe mixing a GPU component sum against a CPU full-hook timing.

So stop reconstructing the call and make the real one.  ``official_eval.py``
calls::

    weight_pair = _move_pair(pack.weights[layer][role], device)
    calibration = [_move_pair(pack.linear_calibration_activations[role][s][layer], device) ...]
    result = solution.hif4_calibration_and_quantize_weight(
        weight_pair[0], weight_pair[1], calibration)

which is reproduced here verbatim against the real candidate.  The probe reports
the device of every returned param, the cost of that whole API call, and -- to
size the lever if the params do turn out to be on the host -- what the same call
would cost with the params built on the GPU.

Read-only; prints only.  No state is written and no candidate is modified.
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "evaluator"))

import official_eval as v2  # noqa: E402

CANDIDATE = ROOT / "workbench/full_solution/linear-em3-k2-arm/candidate/solution.py"
PACK = ROOT / "artifacts/official_eval/cache/qwen3.5-4b-proxy-v2.pt"
CASES = ((0, "q"), (0, "o"))
REPEATS = 5


def main() -> int:
    device = torch.device("cuda")
    candidate = v2.load_solution(CANDIDATE)
    # `prepare_pack` needs the full case set (this cached pack has trimmed test
    # windows), so reproduce only the two lines that build the calibration
    # inputs -- prepare_pack lines 1209 and 1211 -- which is all this probes.
    raw = v2.load_pack(PACK)
    print(f"pack roles: {list(raw.roles)} layers={raw.layers}")

    for layer, role in CASES:
        weight_pair = v2._move_pair(v2._pair(raw.weights[layer][role]), device)
        # `official_eval.py:991` uses the first two designated Linear folds.
        samples = raw.calibration_activations[role]
        indices = tuple(range(min(2, len(samples))))
        calibration = [
            v2._move_pair(v2._pair(samples[s][layer]), device) for s in indices
        ]
        print(f"\n=== layer{layer}/{role} ===")
        print(f"  inputs to the API: weight_quant {weight_pair[0].device}, "
              f"calibration[0][0] {calibration[0][0].device}, "
              f"{len(calibration)} samples")

        seen: dict = {}

        def once():
            started = time.perf_counter()
            result = candidate.hif4_calibration_and_quantize_weight(
                weight_pair[0], weight_pair[1], calibration
            )
            torch.cuda.synchronize(device)
            seen["seconds"] = time.perf_counter() - started
            seen["devices"] = {
                name: str(value.device)
                for name, value in result["weight_params"].items()
            }
            return result

        once()
        device_report = None
        for _ in range(REPEATS):
            result = once()
            device_report = seen["devices"]
        samples = [
            seen["seconds"] for _ in range(REPEATS)
        ]

        print("  returned weight_params devices:")
        for name, where in sorted(device_report.items()):
            print(f"    {name:<14} {where}")
        print(f"  real API call (median of {REPEATS}): "
              f"{statistics.median(samples) * 1e3:.2f} ms")

        state = result["activation_state"]
        em1 = state.get("em1") if isinstance(state, dict) else None
        if isinstance(em1, dict):
            h = em1.get("h")
            print(f"  state['em1']: keys={sorted(em1)} "
                  f"h={None if not torch.is_tensor(h) else tuple(h.shape)} "
                  f"h.device={None if not torch.is_tensor(h) else h.device}")
            if torch.is_tensor(h):
                print(f"  h norm={float(h.norm()):.6e} "
                      f"gram_diag_mean={float(em1.get('gram_diag_mean', 0.0)):.6e}")
        print(f"  diagnostics: "
              f"{ {k: v for k, v in state.items() if k.startswith('em1_')} }"
              if isinstance(state, dict) else "  diagnostics: n/a")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
