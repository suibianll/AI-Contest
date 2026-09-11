"""Re-measure the calibration breakdown, because two measurements disagreed 3.6x.

`atg1_probe.py` reported trainer ~1.36 s + two gate calls ~1.17 s each = ~3.7 s
per layer, while `atg1_verify.py` measured the WHOLE `hif4_calibration_attention`
call at ~1.02 s per layer.  A part cannot exceed its whole, so one of the two is
wrong.  This file measures everything in ONE process, with warmup and medians,
so the pieces and the whole are directly comparable:

    base      _V189_CALIBRATION_ATTENTION alone
    trainer   _a2_train_rotation alone
    gate      one _a2_true_path_gate_loss call (parent form: two arms)
    whole     hif4_calibration_attention
    sum       base + trainer + 2 * gate   <- must be >= whole, never below

    ATG2_DEVICE=cuda .venv/Scripts/python.exe atg2_profile.py
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
PARENT = ROOT / "solutions" / "20260911_v243_attention-atf1-cayley-hoist_scoreNA_timeNA" / "solution.py"


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    par = load("atg2_par", PARENT)
    sys.path.insert(0, str(ROOT / "evaluator"))
    import official_eval as v2  # noqa: PLC0415

    dev = torch.device(os.environ.get("ATG2_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
    rounds = int(os.environ.get("ATG2_ROUNDS", "3"))
    pack = torch.load(
        ROOT / "artifacts/official_eval/cache/qwen3.5-4b-proxy-v2.pt",
        map_location="cpu", mmap=True, weights_only=False,
    )
    qh = int(pack["q_heads"]); kvh = int(pack["kv_heads"]); hd = int(pack["head_dim"])
    splits = len(pack["calibration_qkv"])
    layers = [l for l in range(len(pack["calibration_qkv"][0]))
              if all(pack["calibration_qkv"][s][l] is not None for s in range(splits))]
    layer = int(os.environ.get("ATG2_LAYER", str(layers[0])))
    print(f"device={dev} layers={layers} measuring layer {layer}, rounds={rounds}", flush=True)

    cl = [
        {r: v2._pair(pack["calibration_qkv"][s][layer][i].to(torch.float32))
         for i, r in enumerate(("q", "k", "v"))}
        for s in range(splits)
    ]

    def timeit(fn, n=rounds):
        outs = []
        for _ in range(n):
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = fn()
            if dev.type == "cuda":
                torch.cuda.synchronize()
            outs.append(time.perf_counter() - t0)
        return statistics.median(outs), out

    # warmup: one full pass so lazily-built kernels and caches are resident
    par.hif4_calibration_attention(cl, qh, kvh, hd)
    if dev.type == "cuda":
        torch.cuda.synchronize()

    t_base, states = timeit(lambda: par._V189_CALIBRATION_ATTENTION(cl, qh, kvh, hd), 1)
    win = [
        {r: par._dequantize_nvfp4_float32(*item[r]).to(torch.float32) for r in ("q", "k", "v")}
        for item in cl
    ]
    t_train, (rot, info, ctr) = timeit(lambda: par._a2_train_rotation(win[:-1], qh, kvh, hd, dev))
    t_gate, _ = timeit(lambda: par._a2_true_path_gate_loss(cl[-1], qh, kvh, hd, states, rot, dev, ctr))
    t_whole, _ = timeit(lambda: par.hif4_calibration_attention(cl, qh, kvh, hd))

    print(f"\n  base  _V189_CALIBRATION_ATTENTION : {t_base:.4f}s")
    print(f"  trainer _a2_train_rotation        : {t_train:.4f}s")
    print(f"  gate  _a2_true_path_gate_loss x1  : {t_gate:.4f}s   (two arms)")
    print(f"  whole hif4_calibration_attention  : {t_whole:.4f}s")
    s = t_base + t_train + 2 * t_gate
    print(f"  sum (base + trainer + 2*gate)     : {s:.4f}s")
    print(f"\n  sum / whole = {s / t_whole:.2f}x   (must be >= 1.0; a part cannot exceed its whole)")
    a2 = t_whole - t_base
    print(f"  A2 block (whole - base)           : {a2:.4f}s")
    print(f"  trainer+2*gate vs A2 block        : {t_train + 2 * t_gate:.4f}s vs {a2:.4f}s"
          f"   ratio {(t_train + 2 * t_gate) / max(a2, 1e-9):.2f}x")
    print(f"  gate share of the A2 block        : {2 * t_gate / max(a2, 1e-9):.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
