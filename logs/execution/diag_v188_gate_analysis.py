"""Analyze the v188 Jacobian-port LOO gate dump (diag_v188_gate.jsonl).
Pure JSON analysis, zero GPU, zero quota."""
import json

rows = [json.loads(l) for l in open("diag_v188_gate.jsonl", encoding="utf-8")]
print(f"layers dumped: {len(rows)}")
print()
print("L   accepted  med_delta    min_delta   q_rel   k_rel   q_range          k_range          folds")
for i, r in enumerate(rows):
    deltas = r["deltas"]
    srt = sorted(deltas)
    med = srt[len(srt) // 2]
    print(
        f"L{i:2d} {str(r['accepted']):5s}  "
        f"{med:+.6f}  {min(deltas):+.6f}  "
        f"{r['q_rel_dist']:.3f}   {r['k_rel_dist']:.3f}   "
        f"[{r['q_imp_range'][0]:.3f},{r['q_imp_range'][1]:.3f}]  "
        f"[{r['k_imp_range'][0]:.3f},{r['k_imp_range'][1]:.3f}]  "
        f"{len(r['folds'])}"
    )

print()
print("per-fold candidate/parent MSE ratios (causal), layer by layer")
for i, r in enumerate(rows):
    ratios = [
        f["held_causal"] / max(p, 1e-12)
        for f, p in zip(r["folds"], r["parent_causal"])
    ]
    print(f"L{i:2d} " + " ".join(f"{x:.4f}" for x in ratios))
