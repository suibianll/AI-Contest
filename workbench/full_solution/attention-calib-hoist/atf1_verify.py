"""A-TF1: hoist the loop-invariant Cayley pair out of the A2 training window loop.

Claim under test, in two independent halves:

  EQUIVALENCE  the candidate's hif4_calibration_attention must return states that
               are BYTE-FOR-BYTE identical to the parent's, on every real
               attention layer, and the dynamic q/k/v outputs must be
               byte-for-byte identical too.  The edit only moves two
               computations to where their inputs are already fixed, so any
               non-zero here is a defect, not a tolerance question.

  TIME         the calibration call is timed paired against a SAME-BYTE null
               (the parent module loaded a second time under another name).
               The null gives this machine's noise floor; a change smaller than
               the null is reported as "not distinguishable", not as a win.

Local seconds here are GPU seconds.  The official judge is a Kunpeng 920B CPU,
so the ratio between the two arms is the only thing this file claims -- never a
predicted official second count.

    ATF1_DEVICE=cuda .venv/Scripts/python.exe atf1_verify.py
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
PARENT = ROOT / "solutions" / "20260910_v237_linear-tf2-first-pass-gradient-reuse_scoreNA_timeNA" / "solution.py"


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def canon(obj, prefix=""):
    """Flatten a state dict into {path: bytes} so equality is byte equality."""
    out = {}
    if isinstance(obj, dict):
        for k in sorted(obj, key=str):
            out.update(canon(obj[k], f"{prefix}/{k}"))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            out.update(canon(v, f"{prefix}[{i}]"))
    elif torch.is_tensor(obj):
        out[prefix] = obj.detach().to("cpu").contiguous().numpy().tobytes()
    else:
        out[prefix] = repr(obj).encode()
    return out


def diff(a, b):
    keys = set(a) | set(b)
    return sorted(k for k in keys if a.get(k) != b.get(k))


def main() -> int:
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    root = load("atf1_cand", ROOT / "solution.py")
    par = load("atf1_par", PARENT)
    nul = load("atf1_nul", PARENT)          # same bytes, second module -> null
    sys.path.insert(0, str(ROOT / "evaluator"))
    import official_eval as v2  # noqa: PLC0415

    dev = torch.device(os.environ.get("ATF1_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
    rounds = int(os.environ.get("ATF1_ROUNDS", "3"))
    pack = torch.load(
        ROOT / "artifacts/official_eval/cache/qwen3.5-4b-proxy-v2.pt",
        map_location="cpu", mmap=True, weights_only=False,
    )
    qh = int(pack["q_heads"]); kvh = int(pack["kv_heads"]); hd = int(pack["head_dim"])
    splits = len(pack["calibration_qkv"])
    all_layers = list(range(len(pack["calibration_qkv"][0])))
    # only layers that actually carry calibration QKV are attention layers; the
    # rest of the pack's depth is Linear-only and its qkv slots are None.
    layers = [
        l for l in all_layers
        if all(pack["calibration_qkv"][s][l] is not None for s in range(splits))
    ]
    print(f"device={dev} q_heads={qh} kv_heads={kvh} head_dim={hd} splits={splits}", flush=True)
    print(f"attention layers: {layers}  (skipped {len(all_layers) - len(layers)} Linear-only)", flush=True)

    def calib_list(layer):
        return [
            {r: v2._pair(pack["calibration_qkv"][s][layer][i].to(torch.float32))
             for i, r in enumerate(("q", "k", "v"))}
            for s in range(splits)
        ]

    # --- equivalence, on every real attention layer
    bad = 0
    for layer in layers:
        cl = calib_list(layer)
        sc = root.hif4_calibration_attention(cl, qh, kvh, hd)
        sp = par.hif4_calibration_attention(cl, qh, kvh, hd)
        d = diff(canon(sc), canon(sp))
        # dynamic path too: same window, same state
        w = cl[0]
        outs = []
        for mod, st in ((root, sc), (par, sp)):
            outs.append(canon({
                "q": mod.hif4_dynamic_quantize_q(*w["q"], qh, hd, st["q_state"]),
                "k": mod.hif4_dynamic_quantize_k(*w["k"], kvh, hd, st["k_state"]),
                "v": mod.hif4_dynamic_quantize_v(*w["v"], kvh, hd, st["v_state"]),
            }))
        dd = diff(outs[0], outs[1])
        ok = not d and not dd
        bad += (not ok)
        print(f"layer {layer:>3}: state diffs={len(d)} dynamic diffs={len(dd)}  {'IDENTICAL' if ok else '*** DIFFERS ***'}"
              + (f"  e.g. {d[:3]}" if d else "") + (f"  e.g. {dd[:3]}" if dd else ""), flush=True)
    print(f"\nequivalence: {len(layers) - bad}/{len(layers)} layers byte-identical", flush=True)
    if bad:
        print("VERDICT: NOT equivalent -- the hoist changed behaviour. Do not ship.")
        return 1

    if os.environ.get("ATF1_SKIP_TIMING"):
        print("\n(ATF1_SKIP_TIMING set -- equivalence only, timing not re-measured)")
        return 0

    # --- timing, paired against the same-byte null
    times = {"cand": [], "par": [], "nul": []}
    if dev.type == "cuda":
        torch.cuda.synchronize()
    for layer in layers[:1] if os.environ.get("ATF1_LAYER0_ONLY") else layers:
        cl = calib_list(layer)
        for _ in range(rounds):
            for name, mod in (("par", par), ("cand", root), ("nul", nul)):
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                mod.hif4_calibration_attention(cl, qh, kvh, hd)
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                times[name].append(time.perf_counter() - t0)
    print()
    for name in ("par", "nul", "cand"):
        v = times[name]
        print(f"  {name:<5} n={len(v):>3}  median={statistics.median(v):.4f}s  min={min(v):.4f}s  mean={statistics.mean(v):.4f}s")
    pc = statistics.median(times["par"])
    nn = statistics.median(times["nul"])
    cd = statistics.median(times["cand"])
    print(f"\n  null spread (par vs nul, same bytes) = {abs(pc - nn):.4f}s")
    print(f"  candidate vs parent                   = {cd - pc:+.4f}s "
          f"({(cd / pc - 1.0) * 100:+.2f}%)")
    if abs(cd - pc) <= abs(pc - nn):
        print("  VERDICT: not distinguishable from this machine's noise floor.")
    else:
        print("  VERDICT: distinguishable on this machine (GPU seconds; official is CPU).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
