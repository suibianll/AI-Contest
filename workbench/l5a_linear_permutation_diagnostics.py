"""Record L5a mode-3 admission diagnostics for every Qwen layer/role.

This script does not score a candidate.  It reuses the mode-3 workbench and
captures the two-fold calibration product proxy that decides whether a
permutation is stored.  The resulting table can be joined by layer/role with
the default-panel paired report to study false-positive admissions.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "workbench" / "l5a_linear_permutation_mode3_probe.py"
spec = importlib.util.spec_from_file_location("l5a_mode3_diag", SOURCE)
if spec is None or spec.loader is None:
    raise ImportError(f"cannot load source: {SOURCE}")
candidate = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = candidate
spec.loader.exec_module(candidate)
parent = candidate.base.parent

EVAL_SPEC = importlib.util.spec_from_file_location(
    "l5a_mode3_diag_eval", ROOT / "evaluator" / "official_eval.py"
)
if EVAL_SPEC is None or EVAL_SPEC.loader is None:
    raise ImportError("cannot load evaluator")
evaluator = importlib.util.module_from_spec(EVAL_SPEC)
sys.modules[EVAL_SPEC.name] = evaluator
EVAL_SPEC.loader.exec_module(evaluator)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--cache", type=Path,
        default=ROOT / "artifacts" / "official_eval" / "cache" / "qwen2.5-0.5b-proxy-v2.pt",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "artifacts" / "official_eval" / "l5a-linear-perm-mode3-diagnostics.json",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    raw = evaluator.load_pack(args.cache.resolve())
    device = torch.device(args.device)
    started = time.perf_counter()
    rows: list[dict[str, object]] = []
    for layer in range(int(raw.layers)):
        for role in raw.roles:
            weight = evaluator._pair(raw.weights[layer][role])
            weight = (weight[0].to(device), weight[1].to(device))
            calibration = []
            for fold in range(2):
                pair = evaluator._pair(raw.calibration_activations[role][fold][layer])
                calibration.append((pair[0].to(device), pair[1].to(device)))
            candidate.hif4_calibration_and_quantize_weight(
                weight[0], weight[1], calibration
            )
            diagnostic = dict(candidate.base._LAST_DIAGNOSTIC)
            rows.append({
                "layer": layer,
                "role": role,
                "role_family": "fc" if role in {"fc_gate", "fc_up"} else "qkv" if role in {"q", "k", "v"} else role,
                "calibration_lengths": [len(item.input_ids) for item in raw.calibration_windows[:2]],
                "diagnostic": diagnostic,
            })
            print(f"[diagnostic] layer={layer} role={role} accepted={diagnostic.get('accepted')}", flush=True)
    output = {
        "protocol": "proxy-v2",
        "diagnostic": "l5a-linear-permutation-mode3-admission-v1",
        "scope": "research-oracle",
        "status": "ok",
        "cache": str(args.cache.resolve()),
        "device": str(device),
        "wall_seconds": float(time.perf_counter() - started),
        "rows": rows,
        "note": "Admission diagnostics only; join with paired default/effect JSON for held-out behavior.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
