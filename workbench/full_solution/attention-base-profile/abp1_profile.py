"""Where does the base attention stack spend its time on the CPU path?

`_V189_CALIBRATION_ATTENTION` is ~90% of one `hif4_calibration_attention` call on
the CPU-tensor path, and that call is what the 300 s official gate actually
pays.  A-TF1 and A-TG1 only touched the A2 block; this measures the base.

The base is a single function, so the pieces are timed by monkeypatching the
module-level helpers it calls and accumulating their wall time.  That measures
the helpers, not the glue -- the run also prints the whole call so the
unattributed remainder is visible rather than assumed away.

    ABP1_DEVICE=cuda .venv/Scripts/python.exe abp1_profile.py
"""

from __future__ import annotations

import importlib.util
import os
import statistics
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
TARGETS = (
    "_dequantize_nvfp4_float32",
    "_sample_rows",
    "_dense_to_hif4",
    "_dequantize_hif4",
    "_solve_k_center_scale_aware",
)


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    root = load("abp1", ROOT / "solution.py")
    sys.path.insert(0, str(ROOT / "evaluator"))
    import official_eval as v2  # noqa: PLC0415

    dev = torch.device(os.environ.get("ABP1_DEVICE", "cpu"))
    pack = torch.load(
        ROOT / "artifacts/official_eval/cache/qwen3.5-4b-proxy-v2.pt",
        map_location="cpu", mmap=True, weights_only=False,
    )
    qh = int(pack["q_heads"]); kvh = int(pack["kv_heads"]); hd = int(pack["head_dim"])
    splits = len(pack["calibration_qkv"])
    layers = [l for l in range(len(pack["calibration_qkv"][0]))
              if all(pack["calibration_qkv"][s][l] is not None for s in range(splits))]
    layer = int(os.environ.get("ABP1_LAYER", str(layers[0])))
    print(f"device={dev} layer={layer}", flush=True)

    cl = [
        {r: tuple(t.to(dev) for t in v2._pair(pack["calibration_qkv"][s][layer][i]))
         for i, r in enumerate(("q", "k", "v"))}
        for s in range(splits)
    ]

    totals: dict[str, float] = {n: 0.0 for n in TARGETS}
    calls: dict[str, int] = {n: 0 for n in TARGETS}
    originals = {n: getattr(root, n) for n in TARGETS}

    def wrap(name):
        orig = originals[name]

        def timed(*a, **k):
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = orig(*a, **k)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            totals[name] += time.perf_counter() - t0
            calls[name] += 1
            return out

        return timed

    for name in TARGETS:
        setattr(root, name, wrap(name))
    # _solve_k_center_scale_aware calls _dense_to_hif4 / _dequantize_hif4, so its
    # own total OVERLAPS theirs; both are reported, the overlap is stated.

    root.hif4_calibration_attention(cl, qh, kvh, hd)      # warmup

    def timeit(fn):
        ts = []
        for _ in range(3):
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = fn()
            if dev.type == "cuda":
                torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
        return statistics.median(ts), out

    t_whole, _ = timeit(lambda: root.hif4_calibration_attention(cl, qh, kvh, hd))
    base_totals = {n: totals[n] / 4.0 for n in TARGETS}     # 1 warmup + 3 timed
    base_calls = {n: calls[n] // 4 for n in TARGETS}

    print(f"\n  whole hif4_calibration_attention : {t_whole:.4f}s")
    print(f"  {'helper':<34}{'calls':>7}{'seconds':>10}{'share':>8}")
    for n in TARGETS:
        print(f"  {n:<34}{base_calls[n]:>7}{base_totals[n]:>10.4f}{base_totals[n] / t_whole:>7.1%}")
    lean = sum(base_totals[n] for n in TARGETS if n != "_solve_k_center_scale_aware")
    print(f"\n  sum of leaf helpers (excl. the solver)        : {lean:.4f}s  ({lean / t_whole:.1%})")
    print(f"  _solve_k_center_scale_aware contains _dense_to_hif4/_dequantize_hif4:")
    print(f"    solver total {base_totals['_solve_k_center_scale_aware']:.4f}s,"
          f" of which the two it calls are counted above -- do not add them twice")
    print(f"  unattributed remainder                        : "
          f"{t_whole - lean - base_totals['_solve_k_center_scale_aware']:.4f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
