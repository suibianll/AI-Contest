"""Name the kernels behind the calibration hook's unattributed ~20 ms/call.

Three probes have now boxed this in from the outside and all three came back
negative, which is itself the finding:

  * the named parts (two dequantizations, two n^2 matmuls, an n^2 subtraction,
    two n^2 diagnostic scans, the D2H copy) sum to ~26 ms of the hook's 45 ms at
    in=4096;
  * ablating a whole matmul *and* both diagnostic scans -- a lean version with
    one matmul, a column-sum diagonal and no diagnostics -- still measures
    40.43 ms against the deployed 45.00 ms, so the gap survives the ablation;
  * neither retaining the D2H result instead of discarding it (+0.07 ms) nor its
    host allocation is the cause.

Summing parts that are each timed against an idle GPU under-counts whatever the
real call does that an isolated call does not.  So stop inferring and ask the
profiler which kernels actually run, and how much CPU time sits between them --
launch gaps and allocator churn show up there and nowhere else.

Read-only: profiles on the real shard cache, prints the top entries only.
"""

from __future__ import annotations

import sys
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
CASES = ((0, "o"),)
WARMUP = 3
REPEATS = 5


def main() -> int:
    from torch.profiler import ProfilerActivity, profile

    device = torch.device("cuda")
    candidate = v2.load_solution(CANDIDATE)
    payload = torch.load(CACHE, map_location="cpu", mmap=True, weights_only=False)
    pack = torch.load(PACK, map_location="cpu", mmap=True, weights_only=False)

    for layer, role in CASES:
        entry = next(
            item
            for item in payload["weight_states"]
            if int(item["layer"]) == layer and str(item["role"]) == role
        )
        channels = int(entry["state"]["in_features"])
        # The cache stores params on the host, but the official path hands the
        # hook GPU params (verified in probe_real_calibration_path.py), so
        # profile the configuration that actually ships.
        params = {key: value.to(device) for key, value in entry["params"].items()}
        host_params = dict(entry["params"])
        weight_quant, weight_scale = v2._pair(
            pack["weights"][layer][role].to(torch.float32)
        )
        weight_quant = weight_quant.to(device)
        weight_scale = weight_scale.to(device)
        result = {
            "activation_state": dict(entry["state"]),
            "weight_params": params,
        }
        del host_params

        def once():
            result["activation_state"].pop("em1", None)
            diagnostics: dict = {"em1_arm": "unavailable"}
            candidate._em1_compile_metric(
                weight_quant, weight_scale, result, diagnostics
            )

        for _ in range(WARMUP):
            once()
        torch.cuda.synchronize(device)

        print(f"\n=== layer{layer}/{role} in={channels} "
              f"rows={int(params['mant'].shape[0])} ===", flush=True)
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=False,
        ) as prof:
            for _ in range(REPEATS):
                once()
            torch.cuda.synchronize(device)

        per_call = lambda us: us / REPEATS / 1e3  # noqa: E731
        print(f"  device time / call: "
              f"{per_call(sum(e.device_time_total for e in prof.key_averages())):.2f} ms")
        print(f"  CPU time total / call: "
              f"{per_call(sum(e.cpu_time_total for e in prof.key_averages())):.2f} ms")
        print("\n  --- top by self CUDA time ---")
        print(prof.key_averages().table(
            sort_by="self_device_time_total", row_limit=15
        ))
        print("\n  --- top by self CPU time ---")
        print(prof.key_averages().table(
            sort_by="self_cpu_time_total", row_limit=15
        ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
