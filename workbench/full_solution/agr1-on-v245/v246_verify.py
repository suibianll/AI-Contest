"""v246 = A-GR1 on the v245 root.  Its Attention states must equal v236's BYTE FOR BYTE.

v236's parent is the v231 A2 wrapper; v246's parent is the v245 A2 wrapper, which
carries A-TF1/A-TG1/A-TR1 -- all three verified byte-equivalent to their own
parents.  So A-GR1 trains on an identical parent in both, and the composed
states have to match exactly.  Any difference is a composition defect.
"""
from __future__ import annotations
import importlib.util, os, sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[3]

def load(n, p):
    s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s)
    sys.modules[n] = m; s.loader.exec_module(m); return m

def canon(o, pre=""):
    out = {}
    if isinstance(o, dict):
        for k in sorted(o, key=str): out.update(canon(o[k], f"{pre}/{k}"))
    elif isinstance(o, (list, tuple)):
        for i, v in enumerate(o): out.update(canon(v, f"{pre}[{i}]"))
    elif torch.is_tensor(o):
        out[pre] = o.detach().to("cpu").contiguous().numpy().tobytes()
    else:
        out[pre] = repr(o).encode()
    return out

def diff(a, b):
    return sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))

def main():
    torch.set_num_threads(4); torch.set_grad_enabled(False)
    cand = load("v246", ROOT / "solution.py")
    ref = load("v236", ROOT / "solutions" / "20260910_v236_attention-agr1-on-v231_scoreNA_timeNA" / "solution.py")
    sys.path.insert(0, str(ROOT / "evaluator"))
    import official_eval as v2
    dev = torch.device(os.environ.get("V246_DEVICE", "cuda"))
    pack = torch.load(ROOT / "artifacts/official_eval/cache/qwen3.5-4b-proxy-v2.pt",
                      map_location="cpu", mmap=True, weights_only=False)
    qh, kvh, hd = int(pack["q_heads"]), int(pack["kv_heads"]), int(pack["head_dim"])
    sp = len(pack["calibration_qkv"])
    layers = [l for l in range(len(pack["calibration_qkv"][0]))
              if all(pack["calibration_qkv"][s][l] is not None for s in range(sp))]
    print(f"device={dev} layers={layers}", flush=True)
    bad = 0
    for L in layers:
        cl = [{r: tuple(t.to(dev) for t in v2._pair(pack["calibration_qkv"][s][L][i]))
               for i, r in enumerate(("q", "k", "v"))} for s in range(sp)]
        sc = cand.hif4_calibration_attention(cl, qh, kvh, hd)
        sr = ref.hif4_calibration_attention(cl, qh, kvh, hd)
        d = diff(canon(sc), canon(sr))
        w = cl[0]
        o = []
        for mod, st in ((cand, sc), (ref, sr)):
            o.append(canon({"q": mod.hif4_dynamic_quantize_q(*w["q"], qh, hd, st["q_state"]),
                            "k": mod.hif4_dynamic_quantize_k(*w["k"], kvh, hd, st["k_state"]),
                            "v": mod.hif4_dynamic_quantize_v(*w["v"], kvh, hd, st["v_state"])}))
        dd = diff(o[0], o[1])
        ok = not d and not dd
        bad += (not ok)
        arm = sc["q_state"].get("agr1_arm")
        print(f"layer {L:>3}: agr1_arm={arm}  state diffs={len(d)} dynamic diffs={len(dd)}  "
              f"{'IDENTICAL to v236' if ok else '*** DIFFERS ***'}"
              + (f"  e.g. {d[:4]}" if d else "") + (f"  e.g. {dd[:3]}" if dd else ""), flush=True)
    print(f"\ncomposed states identical to v236: {len(layers)-bad}/{len(layers)}")
    return 1 if bad else 0

if __name__ == "__main__":
    raise SystemExit(main())
