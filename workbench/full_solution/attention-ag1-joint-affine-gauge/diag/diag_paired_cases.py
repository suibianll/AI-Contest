"""Q2: per-case paired delta between v227 candidate and root baseline, six attention shards.

Groups deltas by layer (= shard), test split, and test length. Cross-references
the per-layer A2 arm from diag_calibration_audit.json (run that first or this
script recomputes arm from the cache).
"""
import glob
import json
import math
import os

import torch

RUN_DIR = r"artifacts/proxy_v3/attention-ag1-sixshard-full-20260910/candidate"
CACHE_DIR = r"artifacts/official_eval/cache/proxy-v3-calibration"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diag_paired_cases.json")

CAND_PREFIX = "165e1a6bc50abd41"
LAYER_TO_SHARD = {0: 0, 1: 1, 8: 2, 15: 3, 22: 4, 5: 5}
PANEL_TO_MODEL = {0: 3, 1: 7, 5: 11, 8: 19, 15: 23, 22: 31}


def load_arms():
    arms = {}
    for path in glob.glob(os.path.join(CACHE_DIR, CAND_PREFIX + "-attention-*.pt")):
        pack = torch.load(path, map_location="cpu", weights_only=False)
        for item in pack["attention_states"]:
            qs = item["states"]["q_state"]
            arms[int(item["layer"])] = {
                "arm": qs.get("a2_arm"),
                "gate_rel_improvement": (
                    (qs["a2_gate_loss_identity"] - qs["a2_gate_loss_rotation"])
                    / qs["a2_gate_loss_identity"]
                    if qs.get("a2_gate_loss_identity") else None
                ),
            }
    return arms


def load_cases(kind, shard):
    path = os.path.join(RUN_DIR, f"{kind}-attention-shard{shard}.json")
    doc = json.load(open(path, encoding="utf-8"))
    return doc["results"][0]["case_scores"]["attention"]


def main():
    arms = load_arms()
    all_rows = []
    for shard in range(6):
        cand = load_cases("candidate", shard)
        base = load_cases("baseline", shard)
        assert len(cand) == len(base), (shard, len(cand), len(base))
        for c, b in zip(cand, base):
            assert c["case_id"] == b["case_id"] and c["layer"] == b["layer"]
            assert c["test_window"] == b["test_window"]
            row = {
                "shard": shard,
                "layer": c["layer"],
                "model_layer": PANEL_TO_MODEL.get(c["layer"]),
                "case_id": c["case_id"],
                "test_window": c["test_window"],
                "test_split": c["test_split"],
                "test_length": c["test_length"],
                "gain_baseline": b["gain"],
                "gain_candidate": c["gain"],
                "delta": c["gain"] - b["gain"],
                "rel_mse_baseline": b["relative_player_mse"],
                "rel_mse_candidate": c["relative_player_mse"],
            }
            row.update({f"arm_{k}": v for k, v in arms.get(c["layer"], {}).items()})
            all_rows.append(row)

    # aggregates
    def agg(rows):
        n = len(rows)
        pos = sum(1 for r in rows if r["delta"] > 1e-12)
        neg = sum(1 for r in rows if r["delta"] < -1e-12)
        zero = n - pos - neg
        mean = sum(r["delta"] for r in rows) / n if n else 0.0
        return {"n": n, "mean_delta": mean, "pos": pos, "neg": neg, "zero": zero,
                "min": min((r["delta"] for r in rows), default=0.0),
                "max": max((r["delta"] for r in rows), default=0.0)}

    by_layer, by_split, by_length = {}, {}, {}
    for r in all_rows:
        by_layer.setdefault(r["layer"], []).append(r)
        by_split.setdefault(r["test_split"], []).append(r)
        by_length.setdefault(r["test_length"], []).append(r)

    summary = {
        "overall": agg(all_rows),
        "by_layer": {str(k): {**agg(v), "arm": arms.get(k, {}).get("arm"),
                              "gate_rel_improvement": arms.get(k, {}).get("gate_rel_improvement")}
                     for k, v in sorted(by_layer.items())},
        "by_split": {str(k): agg(v) for k, v in sorted(by_split.items())},
        "by_length": {str(k): agg(v) for k, v in sorted(by_length.items())},
    }
    out = {"summary": summary, "rows": all_rows}
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)

    print("overall:", summary["overall"])
    print("\nby layer (layer = shard's only FA layer):")
    for layer, s in summary["by_layer"].items():
        print(f"  layer {layer:>3} (shard {LAYER_TO_SHARD[int(layer)]}, model {PANEL_TO_MODEL[int(layer)]}): "
              f"arm={s['arm']:<14} mean={s['mean_delta']:+.6f} "
              f"pos/neg/zero={s['pos']}/{s['neg']}/{s['zero']} "
              f"min={s['min']:+.6f} max={s['max']:+.6f} "
              f"gate_rel_impr={s['gate_rel_improvement']}")
    print("\nby split:")
    for k, s in summary["by_split"].items():
        print(f"  {k:<12} mean={s['mean_delta']:+.6f} pos/neg/zero={s['pos']}/{s['neg']}/{s['zero']}")
    print("\nby length:")
    for k, s in summary["by_length"].items():
        print(f"  len {k:>5}: mean={s['mean_delta']:+.6f} pos/neg/zero={s['pos']}/{s['neg']}/{s['zero']}")


if __name__ == "__main__":
    main()
