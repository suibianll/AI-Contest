"""v246 vs v236: how much of A-GR1's calibration cost do the three cards give back?

v236 = A-GR1 on v231 (official full package TIMEOUT, 9s of headroom on the parent).
v246 = the same A-GR1 block appended to v245, which carries A-TF1/A-TG1/A-TR1.
Both produce byte-identical attention states (v246_verify.py), so any time
difference is the margin recovered by the three cards.

Local seconds are NOT converted into official seconds -- the official judge is a
Kunpeng 920B CPU and this machine is not it.  The paired same-byte null bounds
the noise; the ratio between the two candidates is the only claim.
"""
from __future__ import annotations
import importlib.util, os, statistics, sys, time
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[3]

def load(n, p):
    s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s)
    sys.modules[n] = m; s.loader.exec_module(m); return m

def main():
    torch.set_num_threads(4); torch.set_grad_enabled(False)
    v246 = load("t246", ROOT / "solution.py")
    v236 = load("t236", ROOT / "solutions" / "20260910_v236_attention-agr1-on-v231_scoreNA_timeNA" / "solution.py")
    nul  = load("t236n", ROOT / "solutions" / "20260910_v236_attention-agr1-on-v231_scoreNA_timeNA" / "solution.py")
    sys.path.insert(0, str(ROOT / "evaluator"))
    import official_eval as v2
    dev = torch.device(os.environ.get("V246_DEVICE", "cpu"))
    rounds = int(os.environ.get("V246_ROUNDS", "1"))
    pack = torch.load(ROOT / "artifacts/official_eval/cache/qwen3.5-4b-proxy-v2.pt",
                      map_location="cpu", mmap=True, weights_only=False)
    qh, kvh, hd = int(pack["q_heads"]), int(pack["kv_heads"]), int(pack["head_dim"])
    sp = len(pack["calibration_qkv"])
    layers = [l for l in range(len(pack["calibration_qkv"][0]))
              if all(pack["calibration_qkv"][s][l] is not None for s in range(sp))]
    print(f"device={dev} layers={layers} rounds={rounds}", flush=True)

    def cl_of(L):
        return [{r: tuple(t.to(dev) for t in v2._pair(pack["calibration_qkv"][s][L][i]))
                 for i, r in enumerate(("q", "k", "v"))} for s in range(sp)]

    cls = {L: cl_of(L) for L in layers}
    t = {"v246": [], "v236": [], "null": []}
    for _ in range(rounds):
        for L in layers:
            for name, mod in (("v236", v236), ("v246", v246), ("null", nul)):
                if dev.type == "cuda": torch.cuda.synchronize()
                t0 = time.perf_counter()
                mod.hif4_calibration_attention(cls[L], qh, kvh, hd)
                if dev.type == "cuda": torch.cuda.synchronize()
                t[name].append(time.perf_counter() - t0)
    for n in ("v236", "null", "v246"):
        v = t[n]
        print(f"  {n:<5} n={len(v):>3} median={statistics.median(v):.4f}s min={min(v):.4f}s")
    b, nl, c = (statistics.median(t[k]) for k in ("v236", "null", "v246"))
    print(f"\n  null spread (v236 vs same-byte null) = {abs(b-nl):.4f}s")
    print(f"  v246 vs v236                         = {c-b:+.4f}s ({(c/b-1)*100:+.2f}%)")
    print(f"  ratio to null spread                 = {abs(c-b)/max(abs(b-nl),1e-9):.1f}x")
    print("\n  v246's parent has A-TF1+A-TG1+A-TR1; v236's does not.  Both run the")
    print("  same A-GR1 block and produce identical states, so this difference IS")
    print("  the margin the three cards recovered.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
