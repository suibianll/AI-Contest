"""Tier-0b (zero-GPU): per-case control-arm structure for the worst layers.
Uses fields already stored in v186-attn-default.json."""
import json
from collections import defaultdict

d = json.load(open("artifacts/official_eval/v186-attn-default.json", encoding="utf-8"))
cases = d["results"][0]["case_scores"]["attention"]

WORST = (21, 23, 9, 20, 16, 8)
BEST = (0, 2, 5)

def arm_table(layers, name):
    print(f"\n=== {name} layers {layers} ===")
    print("L   len   gain   q_only  k_only  v_only  qk_only  both   qk_int   "
          "logit_p/s    prob_p/s")
    for c in sorted(cases, key=lambda c: (c["layer"], c["test_length"])):
        if c["layer"] not in layers:
            continue
        lr = c["logit_mse_player"] / max(c["logit_mse_standard"], 1e-12)
        pr = c["probability_mse_player"] / max(c["probability_mse_standard"], 1e-12)
        print(f"L{c['layer']:2d} {c['test_length']:5d} {c['gain']:.3f}  "
              f"{c['gain_q_only']:+.2f}  {c['gain_k_only']:+.2f}  "
              f"{c['gain_v_only']:+.2f}  {c['gain_qk_only']:+.2f}  "
              f"{c['gain_both']:+.2f}  {c['qk_interaction_gain']:+.2f}  "
              f"{lr:8.1f}  {pr:7.3f}")

arm_table(WORST, "WORST")
arm_table(BEST, "BEST")

# aggregate: per-layer mean of arms
print("\n=== per-layer arm means (all 5 cases) ===")
print("L    gain   q_only  k_only  v_only  qk_only  qk_int")
by = defaultdict(list)
for c in cases:
    by[c["layer"]].append(c)
for l in sorted(by, key=lambda l: -sum(c["mse_player"] for c in by[l])):
    cs = by[l]
    m = lambda k: sum(c[k] for c in cs) / len(cs)
    print(f"L{l:2d} {m('gain'):.3f}  {m('gain_q_only'):+7.2f} {m('gain_k_only'):+7.2f} "
          f"{m('gain_v_only'):+7.2f} {m('gain_qk_only'):+7.2f} {m('qk_interaction_gain'):+8.2f}")
