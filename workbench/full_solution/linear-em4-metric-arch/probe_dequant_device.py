"""The hook's 6 aten::mul kernels: which device, and how much is reclaimable.

``probe_torch_profiler.py`` named the cost that three component-level probes had
missed: per call the hook runs 6 ``aten::mul`` kernels (20.7 ms of 52.6 ms self
CUDA time, 39%) plus 2 ``aten::nan_to_num`` (4.6 ms, 9%).  Those are exactly the
multiply chains of the two dequantizations the hook calls::

    _dequantize_nvfp4_float32   2 muls
    _dequantize_hif4            sign * mant * lv3 * lv2 * scale_factor = 4 muls

The matmuls the earlier probes focused on are only 14.2 ms (27%), and the
diagnostics and D2H that the ablations chased are smaller still.  So the dequant
chain, not the metric algebra, is the largest single cost in the shipped hook.

Before this can be a reclaim card it has to answer one question the profile
cannot: **which device do those muls run on?**  ``_em1_compile_metric`` reads
``result["weight_params"]`` and feeds it straight to ``_dequantize_hif4`` while
only moving the *result* to ``dense.device``.  If the evaluator hands the hook
CPU params, those muls are a CPU-side tensor expression whose cost is host
compute, and no GPU rewording helps; if it hands GPU params, the chain is
GPU bandwidth.  The two call for completely different fixes, so measure it
rather than assume it.

This probe prints each input's device both as the cache stores it and as the
evaluator would provide it, then times the whole hook under each arrangement.

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
CACHE = (
    ROOT
    / "artifacts/official_eval/cache/proxy-v3-calibration"
    / "56dc805d6e5a3aef-linear-452e6fe4aa5747590535.pt"
)
PACK = ROOT / "artifacts/official_eval/cache/qwen3.5-4b-proxy-v2.pt"
CASES = ((0, "q"), (0, "o"))
REPEATS = 12


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
    candidate = v2.load_solution(CANDIDATE)
    payload = torch.load(CACHE, map_location="cpu", mmap=True, weights_only=False)
    pack = torch.load(PACK, map_location="cpu", mmap=True, weights_only=False)

    # What the evaluator actually passes.  `_em1_compile_metric` is invoked from
    # the deployed quantization API, so read the caller rather than the cache.
    sites = []
    for name in dir(candidate):
        if not name.startswith("_em1"):
            continue
        fn = getattr(candidate, name)
        if callable(fn) and getattr(fn, "__doc__", None or "") is not None:
            sites.append(name)
    print(f"em1 entry points in the candidate: {sites}")

    for layer, role in CASES:
        entry = next(
            item
            for item in payload["weight_states"]
            if int(item["layer"]) == layer and str(item["role"]) == role
        )
        channels = int(entry["state"]["in_features"])
        weight_quant, weight_scale = v2._pair(
            pack["weights"][layer][role].to(torch.float32)
        )
        weight_quant = weight_quant.to(device)
        weight_scale = weight_scale.to(device)

        print(f"\n=== layer{layer}/{role} in={channels} ===")
        print("  cache params as stored:")
        for key, value in sorted(entry["params"].items()):
            print(f"    {key:<14} {str(value.device):<8} "
                  f"{str(tuple(value.shape)):<24} {value.dtype}")

        gpu_params = {k: v.to(device) for k, v in entry["params"].items()}

        def hook(params_obj, host_params):
            state = dict(entry["state"])
            result = {"activation_state": state, "weight_params": params_obj}
            diagnostics: dict = {"em1_arm": "unavailable"}
            candidate._em1_compile_metric(
                weight_quant, weight_scale, result, diagnostics
            )
            del host_params

        cpu_holder: list = []
        gpu_holder: list = []

        def as_cached():
            hook(entry["params"], cpu_holder)

        def as_gpu():
            hook(gpu_params, gpu_holder)

        def as_cpu_copy():
            # The evaluator may hand a fresh CPU dict rather than the mmap'd one.
            hook({k: v.clone() for k, v in entry["params"].items()}, cpu_holder)

        t_cached = bench(as_cached, device)
        t_gpu = bench(as_gpu, device)
        t_cpucopy = bench(as_cpu_copy, device)
        print(f"  hook with cache-resident params   {t_cached * 1e3:8.2f} ms")
        print(f"  hook with params cloned to GPU    {t_gpu * 1e3:8.2f} ms")
        print(f"  hook with fresh CPU param copies  {t_cpucopy * 1e3:8.2f} ms")

        # The two dequant chains alone, on each device, to size the lever.
        def dequant_hif4_gpu():
            candidate._dequantize_hif4(gpu_params)

        def dequant_hif4_cpu():
            candidate._dequantize_hif4(entry["params"])

        def chain_gpu():
            candidate._dequantize_nvfp4_float32(weight_quant, weight_scale).to(
                torch.float32
            )
            candidate._dequantize_hif4(gpu_params).to(
                device=device, dtype=torch.float32
            )

        def chain_cpu():
            candidate._dequantize_nvfp4_float32(weight_quant, weight_scale).to(
                torch.float32
            )
            candidate._dequantize_hif4(entry["params"]).to(
                device=device, dtype=torch.float32
            )

        print(f"  chain gpu (2 dequants)            {bench(chain_gpu, device) * 1e3:8.2f} ms")
        print(f"  chain cpu (2 dequants)            {bench(chain_cpu, device) * 1e3:8.2f} ms")
        print(f"  _dequantize_hif4 on GPU           {bench(dequant_hif4_gpu, device) * 1e3:8.2f} ms")
        print(f"  _dequantize_hif4 on CPU           {bench(dequant_hif4_cpu, device) * 1e3:8.2f} ms")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
