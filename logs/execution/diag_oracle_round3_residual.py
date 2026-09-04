"""Oracle round 3 (2026-09-04): per-layer exhaustive-vs-deployed residual.

Pure JSON analysis, zero GPU, zero quota.  Compares the +/-24 exhaustive
oracle gain (diag-oracle-attn-default.json) against the deployed +4 gain
(diag-plus4-attn-default.json, now = v186) per layer and per (layer,length):
residual = oracle_gain - plus4_gain.  Layers with large residual are blocks
where the exhaustive search finds improvement the deployed window cannot
capture -> next mechanism hypothesis target.
"""
import json
from collections import defaultdict

ROOT = r"artifacts\official_eval"


def by_cell(path):
    d = json.load(open(path, encoding="utf-8"))
    cells = {}
    for c in d["results"][0]["case_scores"]["attention"]:
        key = (c["layer"], c.get("test_length", c.get("length")))
        cells[key] = c["gain"]
    return cells


oracle = by_cell(f"{ROOT}/diag-oracle-attn-default.json")
plus4 = by_cell(f"{ROOT}/diag-plus4-attn-default.json")

# baseline (v180 attn = v182/v186 attn without +4)
base = by_cell(f"{ROOT}/v180-attn-default.json")

common = [k for k in oracle if k in plus4 and k in base]
print("n cells:", len(common))

res_layer = defaultdict(list)
res_len = defaultdict(list)
res_cell = []
for k in common:
    o = oracle[k] - base[k]
    p = plus4[k] - base[k]
    r = o - p
    res_layer[k[0]].append(r)
    res_len[k[1]].append(r)
    res_cell.append((r, o, p, k))

print()
print("=== per-LAYER exhaustive residual (oracle minus deployed +4) ===")
rows = sorted(res_layer.items(), key=lambda kv: -sum(kv[1]))
print("layer  mean_resid  oracle_mean  plus4_mean  n")
for layer, vals in rows[:10]:
    print("L%-4d %+.6f   %+.6f   %+.6f   %d"
          % (layer, sum(vals) / len(vals),
             sum(oracle[(layer, l)] - base[(layer, l)] for l in set(k[1] for k in common if k[0] == layer)) / len(set(k[1] for k in common if k[0] == layer)),
             sum(plus4[(layer, l)] - base[(layer, l)] for l in set(k[1] for k in common if k[0] == layer)) / len(set(k[1] for k in common if k[0] == layer)),
             len(vals)))

print()
print("=== per-LENGTH residual ===")
for ln, vals in sorted(res_len.items()):
    print("len %-5d mean_resid %+.6f" % (ln, sum(vals) / len(vals)))

print()
print("=== top residual CELLS ===")
print("resid     oracle    plus4     (layer,length)")
for r, o, p, k in sorted(res_cell, reverse=True)[:15]:
    print("%+.6f %+.6f %+.6f  L%d@%d" % (r, o, p, k[0], k[1]))

print()
print("=== global ===")
tot_o = sum(oracle[k] - base[k] for k in common) / len(common)
tot_p = sum(plus4[k] - base[k] for k in common) / len(common)
print("oracle mean %+.6f  plus4 mean %+.6f  residual mean %+.6f"
      % (tot_o, tot_p, tot_o - tot_p))
