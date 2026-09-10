"""Attributes a whole dynamic call, to say whether the descent is even the cost.

L-TF1 removes two matrix products that are *inside the descent*.  Whether that
is worth anything depends on the descent's share of the call, and the card must
not assume it.  ``where_the_time_goes.py`` prices the descent's parts
individually and finds them cheap; this script checks that conclusion from the
other side, by measuring the whole call and then measuring it again with one
component stubbed out.

Four measurements, all on the same state and the same activation, all with CUDA
events, all warm:

  full             the shipped call, unmodified
  descent_off      the same state with the ``em1`` payload removed, so
                   ``_em1_metric`` returns None and the descent is never
                   entered -- this is the parent dynamic API plus the hook
                   overhead, and the difference from ``full`` is the descent's
                   entire contribution
  metric_stub      the full call with the cholesky and its inverse replaced by
                   cheap stubs, which prices the metric build
  mm_stub          the full call with every ``Tensor.mm`` replaced by an
                   allocation of the right shape, which prices all the matrix
                   products together -- including the two the card removes

The stubs are only used for attribution.  They are not part of any timing claim
about the candidate: they change the values the code computes, so no number
from this script is a candidate measurement.  What this script produces is the
denominator that says how much of a call the card can possibly reach.
"""

from pathlib import Path
import importlib.util
import json
import statistics
import sys

import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CACHE_DIR = ROOT / "artifacts/official_eval/cache/proxy-v3-calibration"
PACK = ROOT / "artifacts/official_eval/cache/qwen3.5-4b-proxy-v2.pt"

PARENT_SHA_PREFIX = "0f1af6dbc207ff32"
ROUNDS = 15
WARMUP = 3


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def find_state(layer: int, role: str) -> dict:
    for path in sorted(CACHE_DIR.glob(f"{PARENT_SHA_PREFIX}-linear-*.pt")):
        payload = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
        for entry in payload["weight_states"]:
            if int(entry["layer"]) == layer and str(entry["role"]) == role:
                del payload
                return dict(entry["state"])
    raise SystemExit(f"no cached state for layer {layer} / role {role}")


def timeit(rounds, function):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(WARMUP):
        function()
    torch.cuda.synchronize()
    samples = []
    for _ in range(rounds):
        start.record()
        function()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    samples.sort()
    return {
        "median_ms": statistics.median(samples),
        "min_ms": samples[0],
        "max_ms": samples[-1],
    }


def main() -> int:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    device = torch.device("cuda")

    sys.path.insert(0, str(ROOT / "evaluator"))
    import official_eval as evaluator  # noqa: PLC0415

    parent = load("attr_parent", ROOT / "solution.py")
    pack = torch.load(PACK, map_location="cpu", mmap=True, weights_only=False)
    quant_cpu, scale_cpu = evaluator._pair(
        pack["test_activations"]["q"][1][0].to(torch.float32)
    )

    report = {
        "device": torch.cuda.get_device_name(0),
        "rounds": ROUNDS,
        "note": "attribution only; the stubs change the computed values",
        "states": [],
    }

    for layer, rows in ((0, 128), (0, 512)):
        state = find_state(layer, "q")
        # The pack carries 128 rows per window; the 512-row case repeats them,
        # which changes which values the arithmetic sees but not how much of it
        # there is, and rows are what the attribution is about.
        repeat = -(-rows // quant_cpu.shape[0])
        quant = quant_cpu.repeat(repeat, 1)[:rows].contiguous().to(device)
        scale = scale_cpu.repeat(repeat, 1)[:rows].contiguous().to(device)

        def call():
            return parent.hif4_dynamic_quantize_activation(quant, scale, dict(state))

        full = timeit(ROUNDS, call)

        # Cross-check the event clock against the wall clock: the CUDA events
        # are GPU timestamps, and if the two clocks disagreed by an order of
        # magnitude the attribution below would be attributing the wrong thing.
        import time  # noqa: PLC0415

        wall = []
        for _ in range(5):
            torch.cuda.synchronize()
            begin = time.perf_counter()
            call()
            torch.cuda.synchronize()
            wall.append((time.perf_counter() - begin) * 1000.0)
        wall.sort()
        wall_median_ms = statistics.median(wall)

        stripped = {key: value for key, value in state.items() if key != "em1"}

        def call_descent_off():
            return parent.hif4_dynamic_quantize_activation(quant, scale, dict(stripped))

        descent_off = timeit(ROUNDS, call_descent_off)

        # --- metric build stubbed -------------------------------------------
        original_cholesky = torch.linalg.cholesky
        original_inverse = torch.cholesky_inverse
        identity_cache: dict = {}

        def cheap_cholesky(matrix):
            key = (tuple(matrix.shape), matrix.dtype, str(matrix.device))
            if key not in identity_cache:
                identity_cache[key] = torch.eye(
                    matrix.shape[-1], dtype=matrix.dtype, device=matrix.device
                )
            return identity_cache[key]

        def cheap_inverse(matrix, *args, **kwargs):
            key = (tuple(matrix.shape), matrix.dtype, str(matrix.device))
            if key not in identity_cache:
                identity_cache[key] = torch.eye(
                    matrix.shape[-1], dtype=matrix.dtype, device=matrix.device
                )
            return identity_cache[key]

        torch.linalg.cholesky = cheap_cholesky
        torch.cholesky_inverse = cheap_inverse
        try:
            metric_stub = timeit(ROUNDS, call)
        finally:
            torch.linalg.cholesky = original_cholesky
            torch.cholesky_inverse = original_inverse

        # --- all matrix products stubbed ------------------------------------
        original_mm = torch.Tensor.mm

        def cheap_mm(tensor, other):
            return torch.empty(
                tensor.shape[0], other.shape[-1], dtype=tensor.dtype, device=tensor.device
            )

        torch.Tensor.mm = cheap_mm
        try:
            mm_stub = timeit(ROUNDS, call)
        finally:
            torch.Tensor.mm = original_mm

        record = {
            "layer": layer,
            "role": "q",
            "rows": rows,
            "channels": int(state["in_features"]),
            "full": full,
            "full_wall_clock_median_ms": wall_median_ms,
            "wall_over_event": wall_median_ms / full["median_ms"],
            "descent_off": descent_off,
            "metric_stub": metric_stub,
            "mm_stub": mm_stub,
            "descent_share_ms": full["median_ms"] - descent_off["median_ms"],
            "descent_share_percent": 100.0
            * (full["median_ms"] - descent_off["median_ms"])
            / full["median_ms"],
            "metric_share_percent": 100.0
            * (full["median_ms"] - metric_stub["median_ms"])
            / full["median_ms"],
            "all_mm_share_percent": 100.0
            * (full["median_ms"] - mm_stub["median_ms"])
            / full["median_ms"],
        }
        report["states"].append(record)
        print(f"\n=== layer{layer}/q rows={rows} channels={record['channels']} ===")
        for name in ("full", "descent_off", "metric_stub", "mm_stub"):
            print(
                f"  {name:>14}  {record[name]['median_ms']:9.3f} ms   "
                f"[{record[name]['min_ms']:.3f}, {record[name]['max_ms']:.3f}]"
            )
        print(
            f"  wall clock agrees: {wall_median_ms:.3f} ms "
            f"({record['wall_over_event']:.3f} x the event figure)"
        )
        print(
            f"  descent contributes {record['descent_share_ms']:+.3f} ms "
            f"({record['descent_share_percent']:+.2f}% of the call)"
        )
        print(f"  metric build is {record['metric_share_percent']:.2f}% of the call")
        print(f"  all mm products are {record['all_mm_share_percent']:.2f}% of the call")

    (HERE / "attribute.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print("\nwrote attribute.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
