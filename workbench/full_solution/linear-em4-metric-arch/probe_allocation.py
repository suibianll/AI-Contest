"""Where the calibration hook's unattributed ~3 s goes.

``profile_hook_breakdown.py`` times every component of the deployed hook on real
states and gets a puzzle: the named parts -- two dequantizations, two n^2
matmuls, an n^2 subtraction, two n^2 diagnostic scans and the D2H copy -- sum to
about 2.4 s over the official 144 in-scope calibrations, while the hook itself
measures 5.6 s.  Ablating a whole matmul plus both diagnostic scans moves the
end-to-end time by only 4.6 ms in 45.  So ~3 s sits somewhere none of the parts
account for.

The hypothesis this probe tests is **host-side allocation**.  ``_cpu_state_tensor``
hands back a fresh n^2 CPU tensor that the hook then *retains* in the activation
state, so the previous call's buffer is still referenced when the next one is
allocated: unlike a discarded D2H result, it cannot be served from a warm reuse
buffer, and every call page-faults a cold 26-67 MB of host memory.

Three timings per state, in the same process, tell them apart:

  A  D2H result discarded        -- the warm-buffer case (what the parts measured)
  B  D2H result retained in a list  -- forces a fresh host allocation each call
  C  full hook, result retained  -- the deployed behaviour

If B >> A the hypothesis holds and the reclaim is a buffer-reuse change in the
state plumbing, which is bit-identical by construction: the bytes copied do not
change, only where they land.

Read-only CPU/GPU probe on the real shard cache; prints only.
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
CACHE = (
    ROOT
    / "artifacts/official_eval/cache/proxy-v3-calibration"
    / "56dc805d6e5a3aef-linear-452e6fe4aa5747590535.pt"
)
PACK = ROOT / "artifacts/official_eval/cache/qwen3.5-4b-proxy-v2.pt"
CASES = ((0, "q"), (0, "o"))
REPEATS = 12
OFFICIAL_BY_CHANNELS = {2560: 120, 4096: 24}


def bench(fn, device, repeats: int = REPEATS) -> float:
    fn()
    torch.cuda.synchronize(device)
    samples = []
    for _ in range(repeats):
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        fn()
        torch.cuda.synchronize(device)
        samples.append(time.perf_counter() - started)
    return statistics.median(samples)


def main() -> int:
    device = torch.device("cuda")
    payload = torch.load(CACHE, map_location="cpu", mmap=True, weights_only=False)
    pack = torch.load(PACK, map_location="cpu", mmap=True, weights_only=False)
    keep: list[torch.Tensor] = []
    candidate = v2.load_solution(CANDIDATE)

    for layer, role in CASES:
        entry = next(
            item
            for item in payload["weight_states"]
            if int(item["layer"]) == layer and str(item["role"]) == role
        )
        channels = int(entry["state"]["in_features"])
        params = {key: value.to(device) for key, value in entry["params"].items()}
        rows = int(params["mant"].shape[0])
        weight_quant, weight_scale = v2._pair(
            pack["weights"][layer][role].to(torch.float32)
        )
        weight_quant = weight_quant.to(device)
        weight_scale = weight_scale.to(device)

        dense = candidate._dequantize_nvfp4_float32(weight_quant, weight_scale).to(
            torch.float32
        )
        deployed = candidate._dequantize_hif4(params).to(
            device=device, dtype=torch.float32
        )
        h_matrix = deployed.transpose(0, 1).mm(deployed) - dense.transpose(0, 1).mm(
            deployed
        )

        def discarded():
            return candidate._cpu_state_tensor(h_matrix)

        def retained():
            keep.append(candidate._cpu_state_tensor(h_matrix))
            if len(keep) > 8:
                keep.pop(0)

        def retained_prealloc():
            buffer = buffers[0]
            buffer.copy_(h_matrix)
            return buffer

        buffers = [torch.empty_like(h_matrix, device="cpu")]

        print(f"\n=== layer{layer}/{role} in={channels} rows={rows} "
              f"h={tuple(h_matrix.shape)} ({h_matrix.numel() * 4 / 1e6:.0f} MB) ===",
              flush=True)
        a = bench(discarded, device)
        b = bench(retained, device)
        c = bench(retained_prealloc, device)
        print(f"  A D2H discarded (warm reuse)      {a * 1e3:8.2f} ms")
        print(f"  B D2H retained (fresh host alloc) {b * 1e3:8.2f} ms")
        print(f"  C D2H into a preallocated buffer  {c * 1e3:8.2f} ms")
        print(f"  -> retaining costs {(b - a) * 1e3:+.2f} ms, "
              f"preallocating saves {(b - c) * 1e3:+.2f} ms")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
