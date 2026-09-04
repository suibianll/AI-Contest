"""Diagnostics 2 & 3 (2026-09-04): remaining-error concentration + full64
interaction-reversal attribution.  Pure JSON analysis, zero GPU, zero quota."""
import json

ROOT = r"artifacts/official_eval"

print("=" * 72)
print("DIAG 2a: Linear remaining-error concentration (v182 default 168)")
print("=" * 72)
d = json.load(open(f"{ROOT}/v182-linear-default.json", encoding="utf-8"))
cases = d["results"][0]["case_scores"]["linear"]
total = sum(c["mse_player"] for c in cases)
std_total = sum(c["mse_standard"] for c in cases)
print(f"total player MSE={total:.6f}  standard MSE={std_total:.6f}  "
      f"global gain={1-total/std_total:.6f}")
# concentration by case
srt = sorted(cases, key=lambda c: -c["mse_player"])
for k in (10, 20, 40, 84):
    share = sum(c["mse_player"] for c in srt[:k]) / total
    print(f"top {k:3d}/{len(cases)} cases hold {share*100:5.1f}% of remaining error")
# by role
from collections import defaultdict
by = defaultdict(float)
cnt = defaultdict(int)
for c in cases:
    by[c["role"]] += c["mse_player"]
    cnt[c["role"]] += 1
for r in sorted(by, key=lambda x: -by[x]):
    print(f"role {r:8s}: {by[r]/total*100:5.1f}% of remaining err "
          f"(mean/case {by[r]/cnt[r]:.3e})")
# by layer
byL = defaultdict(float)
for c in cases:
    byL[c["layer"]] += c["mse_player"]
worst = sorted(byL.items(), key=lambda x: -x[1])[:8]
print("worst layers:", [(l, f"{v/total*100:.1f}%") for l, v in worst])
# A vs W operand concentration
wa = sum(c["activation_relative_mse"] for c in cases) / len(cases)
ww = sum(c["weight_relative_mse"] for c in cases) / len(cases)
print(f"mean activation rel-MSE={wa:.3f}  weight rel-MSE={ww:.3f} "
      "(2.0 = operand as bad as output standard)")

print()
print("=" * 72)
print("DIAG 2b: Attention remaining-error concentration (v183-attn default 120)")
print("=" * 72)
d = json.load(open(f"{ROOT}/v183-attn-default.json", encoding="utf-8"))
cases = d["results"][0]["case_scores"]["attention"]
total = sum(c["mse_player"] for c in cases)
std_total = sum(c["mse_standard"] for c in cases)
print(f"total player MSE={total:.6f}  standard MSE={std_total:.6f}  "
      f"global gain={1-total/std_total:.6f}")
srt = sorted(cases, key=lambda c: -c["mse_player"])
for k in (6, 12, 24, 60):
    share = sum(c["mse_player"] for c in srt[:k]) / total
    print(f"top {k:3d}/{len(cases)} cases hold {share*100:5.1f}% of remaining error")
byL = defaultdict(float)
for c in cases:
    byL[c["layer"]] += c["mse_player"]
worst = sorted(byL.items(), key=lambda x: -x[1])[:8]
print("worst layers:", [(l, f"{v/total*100:.1f}%") for l, v in worst])
byLen = defaultdict(float)
for c in cases:
    byLen[c.get("test_length", c.get("length", -1))] += c["mse_player"]
print("by length:", [(l, f"{v/total*100:.1f}%") for l, v in sorted(byLen.items())])

print()
print("=" * 72)
print("DIAG 3: L3 full64 W-only vs joint interaction reversal")
print("=" * 72)
d = json.load(open(f"{ROOT}/l3-full64-reachable-qwen-compact.json", encoding="utf-8"))
pe = d.get("paired_effect", {})
lin = pe.get("linear", {})
print("paired linear keys:", list(lin.keys())[:12])
ov = lin.get("overall", {})
print("overall:", json.dumps({k: ov.get(k) for k in
      ["case_count","mean_delta_gain","positive_cases","negative_cases"]},
      ensure_ascii=False))
for grp in ("by_role", "by_role_family", "by_layer"):
    g = lin.get(grp, {})
    if g:
        print(f"--- {grp} (mean_delta_gain / W-only / A-only / interaction) ---")
        for name, st in sorted(g.items(), key=lambda kv: kv[1].get("mean_delta_gain", 0)):
            cd = st.get("component_delta_mean", {})
            print(f"  {name:10s} d={st.get('mean_delta_gain',0):+.6f} "
                  f"W={cd.get('w_only_gain',0):+.4f} A={cd.get('a_only_gain',0):+.4f} "
                  f"I={cd.get('interaction_gain',0):+.4f}")
worstc = lin.get("worst_cases", [])
if worstc:
    print("worst cases:", [(w.get("layer"), w.get("role"),
          f"{w.get('delta_gain',0):+.4f}") for w in worstc[:6]])
