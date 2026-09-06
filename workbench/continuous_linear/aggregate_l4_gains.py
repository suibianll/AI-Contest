"""Aggregate L4 eval-v3 six-shard per-case gains to locate remaining error.

Reads the paired ID artifacts produced for the L4 side package and groups the
per-case gain by shape_bucket / role / role_family / layer to identify where the
remaining quantization error concentrates (input for the next mechanism card).
"""

from __future__ import annotations

import json
from pathlib import Path

BASE = Path(
    r"d:\工作内容\AI竞赛\artifacts\proxy_v3\v162-independent\linear\l4-v189-linear-exact\id\v162-linear-l4-v189-linear-exact"
)

rows = []
for shard in range(6):
    p = BASE / f"candidate-linear-shard{shard}.json"
    m = json.load(open(p, encoding="utf-8"))
    for result in m["results"]:
        rows.extend(result["case_scores"]["linear"])

print(f"total linear cases: {len(rows)}")


def show_group(items, key):
    groups: dict = {}
    for it in items:
        g = groups.setdefault(it[key], [])
        g.append(it["gain"])
    print()
    print(f"== by {key} ==")
    for k, vals in sorted(groups.items(), key=lambda kv: sum(kv[1]) / len(kv[1])):
        import statistics

        print(
            f"  {str(k):22s} n={len(vals):4d} mean_gain={sum(vals)/len(vals):.6f} "
            f"median={statistics.median(vals):.6f} min={min(vals):.6f}"
        )


import statistics

show_group(rows, "shape_bucket")
show_group(rows, "role")
show_group(rows, "role_family")

print()
print("== worst 15 cases ==")
for it in sorted(rows, key=lambda it: it["gain"])[:15]:
    print(
        f"  L{it['layer']:2d} {it['role']:8s} {it['shape_bucket']:20s} "
        f"w{it['input_width']:5d}x{it['output_width']:5d} "
        f"len={it['test_length']:5d} split={it['test_split']:5s} gain={it['gain']:.6f}"
    )

print()
print(f"overall mean_gain = {sum(r['gain'] for r in rows)/len(rows):.6f}")
print(f"min/median = {min(r['gain'] for r in rows):.6f}/{statistics.median(r['gain'] for r in rows):.6f}")
print(
    f"share of cases with gain<0.7: {sum(1 for r in rows if r['gain']<0.7)}"
    f" ({100*sum(1 for r in rows if r['gain']<0.7)/len(rows):.1f}%)"
)
print(
    f"share with gain<0.8: {100*sum(1 for r in rows if r['gain']<0.8)/len(rows):.1f}%"
)