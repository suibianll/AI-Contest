"""Profile the dynamic API -- the Linear line's dominant and K-scaling cost.

The L-EM4 reconnaissance set out to reclaim the calibration hook and found
instead that the hook is roughly 2.2x cheaper than the figure carried in v231's
archive (see FINDINGS.md): its components had been timed with GPU params while
the full hook was timed with the cache's CPU params, and the CPU-side
``_dequantize_hif4`` (20.97 ms) is 28x the GPU one (0.74 ms).  Corrected, the
hook costs ~2.6 s over the 144 in-scope calibrations, not 5.8-6.2 s, and the
~3.2 s "unattributed gap" this workbench had been chasing does not exist.

That correction moves the target.  ``hif4_dynamic_quantize_activation`` is the
other API L-EM2 touched, it runs once per Linear test case rather than once per
layer/role, and -- unlike the hook, which is paid once -- its cost is
**multiplied by the pass count K**: the L-EM3 card measured the K=2 arm at
+3.2 s over K=1, which is what makes K=3 unaffordable at a 292 s root with a
300 s gate.  Any reclaim here buys accuracy back, because it is the only lever
that lowers the price of the next K.

So this probe measures where the dynamic call's time actually goes, using the
evaluator's own call shape (``official_eval.py:2619``)::

    state, weight_params = weight_states[(case.layer, case.role)]
    activation_pair = _move_pair(pack.test_activations[role][window][layer], device)
    solution.hif4_dynamic_quantize_activation(activation_pair[0], activation_pair[1], state)

It reports the wall cost of that call and a kernel-level breakdown of it, so the
next card can name a target instead of a hypothesis.

Read-only; prints only.
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
CAL_CACHE = (
    ROOT
    / "artifacts/official_eval/cache/proxy-v3-calibration"
    / "56dc805d6e5a3aef-linear-452e6fe4aa5747590535.pt"
)
CASES = ((0, "q"), (0, "o"))
WARMUP = 2
REPEATS = 5


def main() -> int:
    from torch.profiler import ProfilerActivity, profile

    device = torch.device("cuda")
    candidate = v2.load_solution(CANDIDATE)
    raw = v2.load_pack(PACK)
    calibration = torch.load(CAL_CACHE, map_location="cpu", mmap=True, weights_only=False)

    # The evaluator reads weight_states[(layer, role)] = (activation_state, params);
    # the *state* -- not the params -- is what the dynamic API receives.
    states = {
        (int(item["layer"]), str(item["role"])): item
        for item in calibration["weight_states"]
    }

    def populated(role, layer):
        """First test window that actually carries activations for this layer.

        This cached pack has most windows trimmed to None; the evaluator
        materializes them at run time.  Cost depends only on the shape, so the
        first populated window is a faithful stand-in.
        """

        for window, entries in enumerate(raw.test_activations[role]):
            if entries is not None and entries[layer] is not None:
                return window, entries[layer]
        raise SystemExit(f"no populated test activation for {role} layer {layer}")

    for layer, role in CASES:
        item = states[(layer, role)]
        channels = int(item["state"]["in_features"])
        window, activation = populated(role, layer)
        activation_pair = v2._move_pair(v2._pair(activation), device)
        print(f"  (using test window {window} for {role} layer {layer})")
        rows = int(activation_pair[0].shape[0])
        print(f"\n=== layer{layer}/{role} in={channels} activation rows={rows} ===",
              flush=True)

        def once():
            return candidate.hif4_dynamic_quantize_activation(
                activation_pair[0], activation_pair[1], dict(item["state"])
            )

        for _ in range(WARMUP):
            once()
        torch.cuda.synchronize(device)

        samples = []
        for _ in range(REPEATS):
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            once()
            torch.cuda.synchronize(device)
            samples.append(time.perf_counter() - started)
        median = statistics.median(samples)
        print(f"  wall: median {median * 1e3:.2f} ms over {REPEATS} "
              f"(min {min(samples) * 1e3:.2f}, max {max(samples) * 1e3:.2f})")

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        ) as prof:
            for _ in range(REPEATS):
                once()
            torch.cuda.synchronize(device)

        averages = list(prof.key_averages())
        cuda = sum(e.device_time_total for e in averages) / REPEATS / 1e3
        cpu = sum(e.cpu_time_total for e in averages) / REPEATS / 1e3
        print(f"  profiled device {cuda:.2f} ms/call, CPU {cpu:.2f} ms/call")
        print(prof.key_averages().table(sort_by="self_device_time_total", row_limit=12))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
