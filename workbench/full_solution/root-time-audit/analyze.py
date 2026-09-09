"""Bucket per-line trace data from results.json into component tables."""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent

# (label, first_line, last_line) inclusive; line attributed by statement start
LINEAR_BASE_BUCKETS = [
    ("misc_setup/arg-checks", 7809, 7833),
    ("calib_loop (NVFP4 decode + stats + cov + sampling)", 7834, 7855),
    ("smooth candidate gen + weight sampling", 7856, 7903),
    ("baseline metrics eval", 7904, 7910),
    ("smooth x perm candidate search", 7911, 7937),
    ("permutation bases search", 7938, 7965),
    ("block-swap optimize", 7966, 7994),
    ("post-perm smooth re-search", 7995, 8020),
    ("block-Hadamard combos search", 8021, 8074),
    ("joint smooth/perm/block search (small layers)", 8075, 8146),
    ("E2E verify of selected transform", 8147, 8173),
    ("final pair transform + 2nd moment", 8174, 8189),
    ("quadratic gram transform", 8190, 8203),
    ("residual probes (base codec, folds)", 8204, 8242),
    ("rank1/rank2 nested defs (no-op)", 8243, 8390),
    ("v166 rank-1 residual fit (power iters)", 8391, 8416),
    ("L-R2 rank-2 residual fit + fused update", 8417, 8471),
    ("weight encode (GPTQ / dense_to_hif4)", 8472, 8508),
    ("weight E2E refine", 8509, 8516),
    ("importance + state prep", 8517, 8533),
    ("v202 sample-energy block order", 8534, 8546),
    ("activation ratio capture", 8547, 8562),
    ("activation gram", 8563, 8580),
    ("activation h_inv + adaptive-reg cholesky loop", 8581, 8666),
    ("adaptive-offsets search", 8667, 8732),
    ("state build + return", 8733, 8762),
]

ATTN_V189_BUCKETS = [
    ("arg checks", 9258, 9279),
    ("SAC K-center solve (C41)", 9280, 9293),
    ("stats loop (decode + sampling + moments)", 9294, 9398),
    ("A1 identity reference outputs", 9399, 9435),
    ("V importance (+A3 candidates)", 9436, 9472),
    ("moment finalize + identity perms", 9473, 9497),
    ("dual-track selection (whole Q/K sweep, x2 tracks)", 9498, 9932),
    ("_build_v_state def+call", 9933, 9964),
    ("_build_qk_states def", 9965, 10171),
    ("build winner Q/K states", 10172, 10185),
    ("A1 final gate (deployed MSE)", 10186, 10254),
    ("Fisher importance (C76.2)", 10255, 10385),
    ("A2 fixed H64 rotation", 10386, 10458),
    ("C76.4 variable H16/H32 rotations", 10459, 10541),
    ("A3 V importance candidates", 10542, 10602),
    ("v158 pair-matrix smooth", 10603, 10676),
    ("logit gain fit", 10677, 10716),
    ("return", 10717, 10720),
]

ATTN_WRAPPER_BUCKETS = [
    ("v189 stack call", 11537, 11544),
    ("window prep (NVFP4 decode)", 11545, 11554),
    ("R3 _a2_train_rotation (32 steps)", 11555, 11557),
    ("R3 gate losses + state update", 11558, 11588),
    ("fallback handler", 11589, 11595),
]

LINEAR_WRAPPER_BUCKETS = [
    ("base calibration call", 11650, 11652),
    ("sample-energy order (fallback path)", 11653, 11665),
]


def bucket(lines: dict[str, dict[str, float]], qual: str, buckets):
    table = lines.get(qual, {})
    rows = []
    for label, lo, hi in buckets:
        t = sum(v for ln, v in table.items() if lo <= int(ln) <= hi)
        rows.append((label, t))
    covered = sum(t for _, t in rows)
    total = sum(table.values())
    rows.append(("(sum of buckets)", covered))
    rows.append(("(traced function total)", total))
    return rows


def fmt_rows(rows, wall):
    out = ["| component | seconds | % of wall |", "|---|---:|---:|"]
    for label, t in rows:
        pct = (100.0 * t / wall) if wall else 0.0
        out.append(f"| {label} | {t:.3f} | {pct:.1f}% |")
    return "\n".join(out)


def main():
    primary = OUT / "results-nosync.json"
    if primary.is_file():
        data = json.loads(primary.read_text(encoding="utf-8"))
    else:
        data = json.loads((OUT / "results.json").read_text(encoding="utf-8"))
        linear_fix = OUT / "results-linear.json"
        if linear_fix.is_file():
            fixed = json.loads(linear_fix.read_text(encoding="utf-8"))
            cases = [c for c in fixed["cases"] if c["kind"] == "linear"]
            cases += [c for c in data["cases"] if c["kind"] != "linear"]
            data["cases"] = cases
    parts = []
    parts.append("## Config flags\n")
    parts.append("```json")
    parts.append(json.dumps(data["config"], indent=1, default=str))
    parts.append("```\n")

    for case in data["cases"]:
        wall = case["wall_seconds"]
        lines = case["lines"]
        if case["kind"] == "linear":
            title = f"Linear layer={case['layer']} role={case['role']} shape={case['shape']} wall={wall:.3f}s"
            body = "### linear_base (7802 body, per-statement buckets)\n"
            body += fmt_rows(bucket(lines, "linear_base", LINEAR_BASE_BUCKETS), wall)
            body += "\n\n### linear_wrapper (11645)\n"
            body += fmt_rows(bucket(lines, "linear_wrapper", LINEAR_WRAPPER_BUCKETS), wall)
            nested = [k for k in lines if k.startswith("linear_base.")]
            if nested:
                body += "\n\n### nested linear frames (drill-down)\n"
                body += "| frame | seconds | % of wall |\n|---|---:|---:|"
                for k in sorted(nested):
                    t = sum(lines[k].values())
                    body += f"\n| {k} | {t:.3f} | {100.0 * t / wall:.1f}% |"
        else:
            title = f"Attention layer={case['layer']} wall={wall:.3f}s"
            body = "### attn_wrapper (11529, R3)\n"
            body += fmt_rows(bucket(lines, "attn_wrapper", ATTN_WRAPPER_BUCKETS), wall)
            body += "\n\n### attn_v189 (9250 body)\n"
            body += fmt_rows(bucket(lines, "attn_v189", ATTN_V189_BUCKETS), wall)
            nested = [k for k in lines if k.startswith("attn_v189.")]
            if nested:
                body += "\n\n### nested attention frames (drill-down)\n"
                body += "| frame | seconds | % of wall |\n|---|---:|---:|"
                for k in sorted(nested):
                    t = sum(lines[k].values())
                    body += f"\n| {k} | {t:.3f} | {100.0 * t / wall:.1f}% |"
        body += "\n\n### wrapped function totals (cross-callsite, overlapping)\n"
        body += "| function | seconds | calls |\n|---|---:|---:|"
        for name, rec in sorted(case["functions"].items(), key=lambda kv: -kv[1]["seconds"]):
            body += f"\n| {name} | {rec['seconds']:.3f} | {rec['calls']} |"
        parts.append(f"## {title}\n\n{body}\n")

    (OUT / "tables.md").write_text("\n".join(parts), encoding="utf-8")
    print("wrote tables.md")


if __name__ == "__main__":
    main()
