"""Fine-grained layer x role cross analysis (2026-09-04, zero-cost JSON)."""
import json
from collections import defaultdict

print("=" * 76)
print("LINEAR: layer x role remaining-MSE cross table (v182 default 168)")
print("=" * 76)
d = json.load(open(r"artifacts\official_eval\v182-linear-default.json", encoding="utf-8"))
cases = d["results"][0]["case_scores"]["linear"]
total = sum(c["mse_player"] for c in cases)
cross = defaultdict(float)
cross_std = defaultdict(float)
wa = defaultdict(lambda: [0.0, 0.0, 0])  # act_rel, w_rel, n
for c in cases:
    key = (c["layer"], c["role"])
    cross[key] += c["mse_player"]
    cross_std[key] += c["mse_standard"]
    wa[c["role"]][0] += c["activation_relative_mse"]
    wa[c["role"]][1] += c["weight_relative_mse"]
    wa[c["role"]][2] += 1
worst = sorted(cross.items(), key=lambda kv: -kv[1])[:20]
print("worst cells (share of total remaining error):")
cum = 0.0
for (l, r), v in worst:
    cum += v
    print("  L%2d %-8s %6.2f%%  (cum %5.1f%%)  std_mse=%.5f player=%.5f"
          % (l, r, v / total * 100, cum / total * 100, cross_std[(l, r)], v))
print()
print("per-role operand structure (act_rel vs weight_rel; 2.0 = as bad as std output):")
for r in sorted(wa, key=lambda x: -wa[x][0] / wa[x][2]):
    a, w, n = wa[r]
    print("  %-8s act=%.3f weight=%.3f  %s-dominant" % (
        r, a / n, w / n, "W" if w > a else "A"))
print()
# deep-layer structure
deep = sum(v for (l, r), v in cross.items() if l >= 16)
print("layers 16-23 hold %.1f%% of remaining error (8/24 layers)" % (deep / total * 100))
fc_deep = sum(v for (l, r), v in cross.items() if l >= 16 and r in ("fc_gate", "fc_up"))
qkv_deep = sum(v for (l, r), v in cross.items() if l >= 16 and r in ("q", "k", "v"))
print("  of which fc_gate/fc_up %.1f%%, qkv %.1f%%" % (
    fc_deep / total * 100, qkv_deep / total * 100))

print()
print("=" * 76)
print("ATTENTION: layer x length remaining-MSE cross (v184 default 120)")
print("=" * 76)
d = json.load(open(r"artifacts\official_eval\v184-attn-default.json", encoding="utf-8"))
cases = d["results"][0]["case_scores"]["attention"]
total = sum(c["mse_player"] for c in cases)
cross = defaultdict(float)
for c in cases:
    cross[(c["layer"], c.get("test_length", c.get("length")))] += c["mse_player"]
worst = sorted(cross.items(), key=lambda kv: -kv[1])[:15]
cum = 0.0
print("worst cells:")
for (l, ln), v in worst:
    cum += v
    print("  L%2d len=%4d  %6.2f%%  (cum %5.1f%%)" % (l, ln, v / total * 100, cum / total * 100))
# length-normalized: mse per case
by_len = defaultdict(lambda: [0.0, 0])
for c in cases:
    ln = c.get("test_length", c.get("length"))
    by_len[ln][0] += c["mse_player"]
    by_len[ln][1] += 1
print("mean mse/case by length:",
      [(ln, "%.5f" % (s / n)) for ln, (s, n) in sorted(by_len.items())])
