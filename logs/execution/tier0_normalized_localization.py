"""Tier-0 (zero-GPU): normalized error localization on the current attention
parent (v186 default 120).  Corrects aggregate-share distortions before any
new mechanism is designed.

Outputs:
  A. per-case error by layer/length with over-proportionality index
  B. layer x length cell matrix + top outlier cells
  C. discover per-case control-arm fields (Q/K/V arms) if present
  D. gain distribution stats beyond mean (median/quartile/worst/tail share)
"""
import json
from collections import defaultdict

d = json.load(open("artifacts/official_eval/v186-attn-default.json", encoding="utf-8"))
res = d["results"][0]
cases = res["case_scores"]["attention"]
print(f"cases: {len(cases)}")
print("per-case fields:", sorted(cases[0].keys()))

dec = d.get("decomposition", {}).get("attention", {})
print("\ndecomposition keys:", list(dec.keys())[:12])
if dec:
    k0 = [k for k in dec if isinstance(dec[k], dict)]
    if k0:
        print("first group key sample:", k0[0], "->", list(dec[k0[0]].keys())[:8])

total = sum(c["mse_player"] for c in cases)

# A. over-proportionality: error share vs case share
def op_index(group_fn, name):
    err = defaultdict(float)
    cnt = defaultdict(int)
    for c in cases:
        err[group_fn(c)] += c["mse_player"]
        cnt[group_fn(c)] += 1
    print(f"\n[A] {name}: (share_err / share_cases) per group")
    for g in sorted(err, key=lambda x: -err[x] / total):
        es = err[g] / total
        cs = cnt[g] / len(cases)
        print(f"  {name}={g}: err_share={es*100:5.1f}%  case_share={cs*100:5.1f}%  "
              f"OP={es/cs:4.2f}x  per-case-MSE={err[g]/cnt[g]:.3e}")

op_index(lambda c: c["layer"], "layer")
op_index(lambda c: c.get("test_length", c.get("length")), "length")

# B. layer x length cells
cell = defaultdict(float)
ccnt = defaultdict(int)
for c in cases:
    key = (c["layer"], c.get("test_length", c.get("length")))
    cell[key] += c["mse_player"]
    ccnt[key] += 1
rows = sorted(cell.items(), key=lambda kv: -(kv[1] / ccnt[kv[0]]))
print("\n[B] top-15 worst layer x length cells (per-case MSE)")
for (l, ln), e in rows[:15]:
    print(f"  L{l:2d} len={ln:5d}  per-case-MSE={e/ccnt[(l,ln)]:.3e}  "
          f"gain_mean={sum(c['gain'] for c in cases if (c['layer'], c.get('test_length', c.get('length')))==(l,ln))/ccnt[(l,ln)]:.4f}")

# D. gain distribution
gains = sorted(c["gain"] for c in cases)
n = len(gains)
print("\n[D] gain distribution (parent v186, 120 cases)")
print(f"  min={gains[0]:.4f} q25={gains[n//4]:.4f} median={gains[n//2]:.4f} "
      f"q75={gains[3*n//4]:.4f} max={gains[-1]:.4f}")
neg = [g for g in gains if g < 0.5]
print(f"  cases with gain<0.5: {len(neg)}; gain<0.6: {sum(1 for g in gains if g<0.6)}; "
      f"gain<0.7: {sum(1 for g in gains if g<0.7)}")
# worst 12 cases full identity
print("\n[D2] worst-12 cases by gain")
for c in sorted(cases, key=lambda c: c["gain"])[:12]:
    print(f"  L{c['layer']:2d} len={c.get('test_length', c.get('length')):5d} "
          f"win={c.get('test_window', c.get('window'))} split={c.get('split','?')} "
          f"gain={c['gain']:.4f} mse_p={c['mse_player']:.3e} mse_s={c['mse_standard']:.3e} "
          f"refE={c.get('reference_energy', float('nan')):.3e}")
