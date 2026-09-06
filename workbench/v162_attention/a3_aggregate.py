"""Aggregate A3 six-shard ID and OOD runs for the deployed gate arm.

Reports the 48-case ID panel stats (deltas vs v162 zero), per-layer deployed
status, split means, L1 gates, and the OOD supplementary gate
|Delta(gain_in - gain_ood)| with parent = v162 (both gains zero).
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "artifacts/proxy_v3/v162-independent/attention"
SHARDS = (0, 1, 2, 3, 4, 5)


def load_cases(run: str, kind: str) -> dict[tuple, float]:
    cases: dict[tuple, float] = {}
    for shard in SHARDS:
        path = BASE / run / run / f"{kind}-attention-shard{shard}.json"
        if not path.exists():
            path = BASE / run / run / f"{kind}-ood-attention-shard{shard}.json"
        if not path.exists():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        for result in payload["results"]:
            for case in result["case_scores"]["attention"]:
                key = (int(case["layer"]), int(case["test_window"]), str(case["test_split"]), int(case["test_length"]))
                cases[key] = float(case["gain"])
    return cases


def describe(values: list[float]) -> dict:
    ordered = sorted(values)
    n = len(ordered)
    return {
        "n": n,
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "q25": ordered[int(0.25 * (n - 1))],
        "q75": ordered[int(0.75 * (n - 1))],
        "worst_quartile_mean": statistics.fmean(ordered[: max(1, n // 4)]),
        "positive": sum(1 for v in values if v > 0),
        "zero": sum(1 for v in values if v == 0),
        "negative": sum(1 for v in values if v < 0),
    }


def main() -> int:
    baseline_id = load_cases("a3-id", "baseline")
    candidate_id = load_cases("a3-id", "candidate")
    candidate_ood = load_cases("a3-ood", "candidate")
    baseline_ood = load_cases("a3-ood", "baseline")

    id_deltas = {k: candidate_id[k] - baseline_id[k] for k in candidate_id}
    layers = sorted({k[0] for k in id_deltas})
    splits = sorted({k[2] for k in id_deltas})

    report = {
        "id": {
            "overall": describe(list(id_deltas.values())),
            "per_layer_mean": {
                str(layer): round(statistics.fmean([v for k, v in id_deltas.items() if k[0] == layer]), 6)
                for layer in layers
            },
            "per_split_mean": {
                split: round(statistics.fmean([v for k, v in id_deltas.items() if k[2] == split]), 6)
                for split in splits
            },
            "l1_total": round(statistics.fmean([abs(v) for v in id_deltas.values()]), 6),
            "l1_negative": round(statistics.fmean([max(-v, 0.0) for v in id_deltas.values()]), 6),
        },
    }

    # OOD pairing by (layer, split, length) position within shard order is not
    # stable across domains; pair by case order inside each shard instead.
    def ordered_gains(run: str, kind: str) -> list[float]:
        values: list[float] = []
        for shard in SHARDS:
            path = BASE / run / run / f"{kind}-attention-shard{shard}.json"
            if not path.exists():
                path = BASE / run / run / f"{kind}-ood-attention-shard{shard}.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            for result in payload["results"]:
                for case in sorted(result["case_scores"]["attention"], key=lambda c: int(c["case_id"])):
                    values.append(float(case["gain"]))
        return values

    cand_id_seq = ordered_gains("a3-id", "candidate")
    cand_ood_seq = ordered_gains("a3-ood", "candidate")
    base_id_seq = ordered_gains("a3-id", "baseline")
    base_ood_seq = ordered_gains("a3-ood", "baseline")
    if not (len(cand_id_seq) == len(cand_ood_seq) == len(base_id_seq) == len(base_ood_seq)):
        raise ValueError("ID/OOD case count mismatch")
    gaps = [
        (c_id - c_ood) - (b_id - b_ood)
        for c_id, c_ood, b_id, b_ood in zip(cand_id_seq, cand_ood_seq, base_id_seq, base_ood_seq)
    ]
    report["ood"] = {
        "candidate_gain_in_mean": statistics.fmean(cand_id_seq),
        "candidate_gain_ood_mean": statistics.fmean(cand_ood_seq),
        "parent_gap_mean": statistics.fmean([b_id - b_ood for b_id, b_ood in zip(base_id_seq, base_ood_seq)]),
        "delta_gap_mean": statistics.fmean(gaps),
        "delta_gap_abs_mean": statistics.fmean([abs(g) for g in gaps]),
        "delta_gap_max_abs": max(abs(g) for g in gaps),
        "gate_threshold": 0.01,
        "blocked": abs(statistics.fmean(gaps)) > 0.01,
    }

    out = Path(__file__).resolve().parent / "research" / "a3_id_ood_summary.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    ov = report["id"]["overall"]
    print(
        f"ID 48-case: mean {ov['mean']:+.6f} median {ov['median']:+.6f} "
        f"+/0/- {ov['positive']}/{ov['zero']}/{ov['negative']} "
        f"L1_neg {report['id']['l1_negative']:.6f}"
    )
    print(f"per-layer mean: {report['id']['per_layer_mean']}")
    print(f"per-split mean: {report['id']['per_split_mean']}")
    o = report["ood"]
    print(
        f"OOD: in {o['candidate_gain_in_mean']:+.6f} ood {o['candidate_gain_ood_mean']:+.6f} "
        f"delta_gap_mean {o['delta_gap_mean']:+.6f} max|gap| {o['delta_gap_max_abs']:.6f} "
        f"blocked={o['blocked']}"
    )
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
