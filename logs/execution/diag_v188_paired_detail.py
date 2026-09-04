"""Extract per-case paired deltas for the v188 candidate vs v186 parent
(default attention panel) + L1 gate.  Pure JSON analysis."""
import json

def load_cases(path):
    d = json.load(open(path, encoding="utf-8"))
    out = {}
    for c in d["results"][0]["case_scores"]["attention"]:
        key = (c["layer"], c.get("test_length", c.get("length")),
               c.get("test_window", c.get("window")))
        out[key] = c["gain"]
    return out

parent = load_cases("artifacts/official_eval/v186-attn-default.json")
cand = load_cases("artifacts/official_eval/v188-attn-default.json")
common = [k for k in parent if k in cand]
print(f"paired cases: {len(common)}")

deltas = [(k, cand[k] - parent[k]) for k in common]
changed = [(k, d) for k, d in deltas if d != 0]
print(f"changed: {len(changed)}")
for k, d in sorted(changed, key=lambda x: -abs(x[1])):
    print(f"  L{k[0]:2d} len={k[1]:5d} win={k[2]}  delta={d:+.6f}  "
          f"parent={parent[k]:.6f} -> cand={cand[k]:.6f}")

mean_delta = sum(d for _, d in deltas) / len(deltas)
l1 = sum(abs(d) for _, d in deltas) / len(deltas)
pos = sum(1 for _, d in deltas if d > 0)
neg = sum(1 for _, d in deltas if d < 0)
print(f"\nmean_delta={mean_delta:+.8f}  L1={l1:.8f}  pos/neg/eq={pos}/{neg}/{len(deltas)-pos-neg}")
print(f"sign gate (mean>0 and L1<0.02): {mean_delta > 0 and l1 < 0.02}")
