"""A-TG1: does sharing the standard gate arm change anything, and is it faster?

EQUIVALENCE is the only release condition and it is byte equality, not a
tolerance.  The edit rewrites the gate as

    standard_mse = arm(q_state, k_state)
    loss_identity = standard_mse / max(standard_mse, 1e-12)
    loss_rotation = arm(q_with_rotation, k_with_rotation) / max(standard_mse, 1e-12)

where the parent evaluated four full-window arms.  Two things have to hold for
that to be exact, and both were measured before the edit (`atg1_probe.py`):
the identity arm's player and standard arms are bit-equal on every layer, and
`loss_identity` is exactly 1.0.  The layer whose rotation arm LOSES the gate
(layer 8, loss_rotation 1.0286 > 1.0) is the one that would expose a changed
comparison, so a state mismatch there is the expected failure signature.

TIMING is paired against a same-byte null (the parent loaded a second time
under another name), so a change smaller than this machine's noise floor is
reported as "not distinguishable" rather than as a win.  Local seconds are GPU
seconds; the official judge is a Kunpeng 920B CPU, so only the RATIO between
arms is claimed here.

    ATG1_DEVICE=cuda .venv/Scripts/python.exe atg1_verify.py
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
    cand = load("atg1_cand", ROOT / "solution.py")
    par = load("atg1_par", PARENT)
    nul = load("atg1_nul", PARENT)
    sys.path.insert(0, str(ROOT / "evaluator"))
    import official_eval as v2  # noqa: PLC0415

    dev = torch.device(os.environ.get("ATG1_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
    rounds = int(os.environ.get("ATG1_ROUNDS", "3"))
    pack = torch.load(
        ROOT / "artifacts/official_eval/cache/qwen3.5-4b-proxy-v2.pt",
        map_location="cpu", mmap=True, weights_only=False,
    )
    qh = int(pack["q_heads"]); kvh = int(pack["kv_heads"]); hd = int(pack["head_dim"])
    splits = len(pack["calibration_qkv"])
    layers = [l for l in range(len(pack["calibration_qkv"][0]))
              if all(pack["calibration_qkv"][s][l] is not None for s in range(splits))]
    print(f"device={dev} q_heads={qh} kv_heads={kvh} head_dim={hd} splits={splits}", flush=True)
    print(f"attention layers: {layers}", flush=True)

    def calib_list(layer):
        # EXACTLY the harness construction (_move_qkv): encode with _pair, then
        # move to the algorithm device.  Leaving these on CPU -- as the earlier
        # probes did -- makes the base stack calibration run on CPU and inflates
        # the whole call ~3.6x, so the timings stop representing the panel.
        return [
            {r: tuple(t.to(dev) for t in v2._pair(pack["calibration_qkv"][s][layer][i]))
             for i, r in enumerate(("q", "k", "v"))}
            for s in range(splits)
        ]

    bad = 0
    for layer in layers:
        cl = calib_list(layer)
        sc = cand.hif4_calibration_attention(cl, qh, kvh, hd)
        sp = par.hif4_calibration_attention(cl, qh, kvh, hd)
        d = diff(canon(sc), canon(sp))
        w = cl[0]
        outs = []
        for mod, st in ((cand, sc), (par, sp)):
            outs.append(canon({
                "q": mod.hif4_dynamic_quantize_q(*w["q"], qh, hd, st["q_state"]),
                "k": mod.hif4_dynamic_quantize_k(*w["k"], kvh, hd, st["k_state"]),
                "v": mod.hif4_dynamic_quantize_v(*w["v"], kvh, hd, st["v_state"]),
            }))
        dd = diff(outs[0], outs[1])
        ok = not d and not dd
        bad += (not ok)
        arm = sc["q_state"].get("a2_arm")
        print(f"layer {layer:>3}: arm={arm:<8} state diffs={len(d)} dynamic diffs={len(dd)}  "
              f"{'IDENTICAL' if ok else '*** DIFFERS ***'}"
              + (f"  e.g. {d[:3]}" if d else "") + (f"  e.g. {dd[:3]}" if dd else ""), flush=True)
    print(f"\nequivalence: {len(layers) - bad}/{len(layers)} layers byte-identical", flush=True)
    if bad:
        print("VERDICT: NOT equivalent -- sharing the standard arm changed behaviour. Do not ship.")
        return 1

    if os.environ.get("ATG1_SKIP_TIMING"):
        print("\n(ATG1_SKIP_TIMING set -- equivalence only)")
        return 0

    times = {"cand": [], "par": [], "nul": []}
    if dev.type == "cuda":
        torch.cuda.synchronize()
    for layer in layers:
        cl = calib_list(layer)
        for _ in range(rounds):
            for name, mod in (("par", par), ("cand", cand), ("nul", nul)):
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
    pc, nn, cd = (statistics.median(times[k]) for k in ("par", "nul", "cand"))
    print(f"\n  null spread (par vs nul, same bytes) = {abs(pc - nn):.4f}s")
    print(f"  candidate vs parent                   = {cd - pc:+.4f}s ({(cd / pc - 1.0) * 100:+.2f}%)")
    if abs(cd - pc) <= abs(pc - nn):
        print("  VERDICT: not distinguishable from this machine's noise floor.")
    else:
        print(f"  VERDICT: distinguishable, {abs(cd - pc) / max(abs(pc - nn), 1e-9):.1f}x the null spread "
              f"(GPU seconds; official is CPU).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
