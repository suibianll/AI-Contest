"""Frozen final holdout for the 26000 plan §4.5.

Rules (frozen, evaluator-side only):
1. The holdout text below has never been used by any development candidate
   (the dev track uses ``real_data_eval.TEXT``); it is never modified after
   the freeze.
2. The frozen config produces 4 test token windows (>= 4 required) plus 2
   calibration windows, all deterministically derived from the frozen text.
3. The ledger file records only the seed/hash of the frozen text + config
   and, per run, the solution content hash and aggregate scores — never
   per-layer or per-component data.
4. Each candidate (identified by the sha256 of solution.py) may run the
   final holdout at most once; the total budget is 3 runs (project
   constraint).  Both rules are enforced before evaluation starts.
5. The holdout is final-acceptance only: it must not be used to search
   seeds, thresholds, or coverage.  Every run requires an explicit
   ``--reason`` recorded in the ledger.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from real_data_eval import (  # noqa: E402
    collect_real_data,
    instrument_solution,
    load_solution,
    score_attention,
    score_linear,
)
from nvfp4_sim import nvfp4_encode  # noqa: E402


HOLDOUT_TEXT = (
    "Autumn arrived early in the river valley, and the morning fog "
    "settled over the wooden bridge. An old cartographer measured the "
    "water level with a brass instrument his grandfather had built. "
    "Far away, a satellite recorded the same river from orbit at a "
    "different resolution. Both measurements describe one world. "
    "In the library, students copied maps by hand before printing "
    "presses made reproduction cheap and exact. Trade routes carried "
    "not only silk and salt but also arithmetic and paper. A merchant "
    "in Samarkand once computed a table of exchange rates for twelve "
    "cities, double checking every entry. Precision mattered then as "
    "it matters now in floating point circuits. Engineers debate how "
    "many bits a number truly needs to represent a physical quantity. "
    "Four bits seem too few, yet structured scaling recovers most of "
    "the lost accuracy. The proof is written in matrices, not words. "
    "Meanwhile the river kept flowing under the bridge, indifferent "
    "to rounding errors. A fisherman counted his catch by twos and "
    "rounded down out of habit. Somewhere between the satellite and "
    "the fisherman lies the whole history of measurement. This "
    "paragraph exists only to hold out against overfitting, and its "
    "sentences have never been seen by any calibration window. When "
    "a final candidate is ready, it will be scored here exactly once, "
    "and the number that comes back will be believed."
)

# Frozen evaluation configuration (never changed after freeze).
HOLDOUT_CONFIG: dict = {
    "layers": 12,
    "seq": 128,
    "calib": 2,
    "test": 4,  # >= 4 token windows (plan §4.5)
    "mode": "amax6",
    "attn_mask": "causal",
    "token_offset": 0,
    "kv_heads": None,
    "model": "models/gpt2",
}

HOLDOUT_BUDGET = 3  # total final-holdout runs allowed (project constraint)

DEFAULT_LEDGER = Path(__file__).resolve().parent / "holdout_ledger.json"


def holdout_seed_hash() -> str:
    """sha256 over the frozen text + frozen config (seed/hash record)."""

    payload = json.dumps(
        {"text": HOLDOUT_TEXT, "config": HOLDOUT_CONFIG},
        sort_keys=True,
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def solution_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_ledger(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("seed_hash") != holdout_seed_hash():
        raise RuntimeError(
            "ledger seed_hash does not match the frozen holdout; "
            "the holdout must never be modified after the freeze"
        )
    return data


def check_run_allowed(ledger: dict, sha: str) -> None:
    """Enforce §4.5 rules 4: once per candidate, budget 3 total."""

    runs = ledger.get("runs", [])
    for run in runs:
        if run.get("solution_sha256") == sha:
            raise RuntimeError(
                "candidate already consumed its single final-holdout run "
                f"(recorded {run.get('date', '?')} for reason "
                f"{run.get('reason', '?')})"
            )
    if len(runs) >= HOLDOUT_BUDGET:
        raise RuntimeError(
            f"holdout budget exhausted ({len(runs)}/{HOLDOUT_BUDGET} runs)"
        )


def freeze(path: Path) -> None:
    """Write the frozen ledger with the seed/hash; refuse to overwrite."""

    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("seed_hash") == holdout_seed_hash():
            print(f"holdout ledger already frozen: {path}")
            return
        raise RuntimeError(
            "refusing to overwrite a ledger frozen with a different seed"
        )
    ledger = {
        "seed_hash": holdout_seed_hash(),
        "frozen_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "budget": HOLDOUT_BUDGET,
        "runs": [],
    }
    path.write_text(
        json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"holdout frozen: {path}")
    print(f"seed_hash: {ledger['seed_hash']}")


def run_holdout(args: argparse.Namespace) -> None:
    ledger_path = Path(args.ledger)
    if not ledger_path.exists():
        raise RuntimeError(
            "holdout ledger is not frozen yet; run --freeze first"
        )
    ledger = load_ledger(ledger_path)

    solution_path = Path(args.solution)
    sha = solution_sha256(solution_path)
    check_run_allowed(ledger, sha)

    solution = load_solution(solution_path)
    stats = instrument_solution(solution)

    config = HOLDOUT_CONFIG
    model, weights, calibration, tests, q_heads, head_dim = collect_real_data(
        config["model"],
        config["layers"],
        config["seq"],
        config["calib"],
        config["test"],
        device=args.device,
        token_offset=config["token_offset"],
        text=HOLDOUT_TEXT,
    )
    kv_heads = q_heads if config["kv_heads"] is None else config["kv_heads"]
    layer_count = len(weights)
    hidden = int(model.config.n_embd)

    linear_scores: list[float] = []
    attention_scores: list[float] = []
    for layer_index in range(layer_count):
        for name in ("q", "k", "v", "o", "fc", "proj"):
            weight_pair = nvfp4_encode(
                weights[layer_index][name], config["mode"]
            )
            calibration_pairs = [
                nvfp4_encode(
                    calibration["act"][name][
                        batch * layer_count + layer_index
                    ],
                    config["mode"],
                )
                for batch in range(config["calib"])
            ]
            calibrated = solution.hif4_calibration_and_quantize_weight(
                *weight_pair, calibration_pairs
            )
            test_pairs = [
                nvfp4_encode(
                    tests["act"][name][batch * layer_count + layer_index],
                    config["mode"],
                )
                for batch in range(config["test"])
            ]
            linear_scores.append(
                score_linear(
                    solution,
                    weight_pair,
                    test_pairs,
                    calibrated["activation_state"],
                    calibrated["weight_params"],
                )
            )

        qkv_calibration = []
        for batch in range(config["calib"]):
            dense = calibration["qkv"][
                batch * layer_count + layer_index
            ].reshape(-1, 3 * hidden)
            q_dense, k_dense, v_dense = dense.chunk(3, dim=-1)
            qkv_calibration.append(
                {
                    "q": nvfp4_encode(q_dense, config["mode"]),
                    "k": nvfp4_encode(k_dense, config["mode"]),
                    "v": nvfp4_encode(v_dense, config["mode"]),
                }
            )
        states = solution.hif4_calibration_attention(
            qkv_calibration, q_heads, kv_heads, head_dim
        )
        qkv_tests = []
        for batch in range(config["test"]):
            dense = tests["qkv"][
                batch * layer_count + layer_index
            ].reshape(-1, 3 * hidden)
            q_dense, k_dense, v_dense = dense.chunk(3, dim=-1)
            qkv_tests.append(
                (
                    nvfp4_encode(q_dense, config["mode"]),
                    nvfp4_encode(k_dense, config["mode"]),
                    nvfp4_encode(v_dense, config["mode"]),
                )
            )
        per_layer = score_attention(
            solution,
            qkv_tests,
            states["q_state"],
            states["k_state"],
            states["v_state"],
            q_heads,
            kv_heads,
            head_dim,
            masks=(config["attn_mask"],),
        )
        attention_scores.extend(per_layer.values())

    linear_mean = sum(linear_scores) / len(linear_scores)
    attention_mean = sum(attention_scores) / len(attention_scores)
    stage = (
        stats["last_end"] - stats["first_start"]
        if stats["first_start"] is not None
        else float("nan")
    )

    run = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "solution_sha256": sha,
        "reason": args.reason,
        "linear_mean": linear_mean,
        "attention_mean": attention_mean,
        "algorithm_stage_seconds": stage,
    }
    ledger.setdefault("runs", []).append(run)
    ledger_path.write_text(
        json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # Aggregate output only (§4.5 rule 3): no per-layer, no per-component.
    print(f"Holdout solution_sha256: {sha}")
    print(f"linear_mean: {linear_mean:.6f}")
    print(f"attention_mean: {attention_mean:.6f}")
    print(
        f"algorithm_stage_seconds: {stage:.2f}s "
        f"calibration={stats['calibration']:.2f}s "
        f"dynamic={stats['dynamic']:.2f}s"
    )
    runs = ledger["runs"]
    print(
        f"holdout_runs_used: {len(runs)} "
        f"remaining: {HOLDOUT_BUDGET - len(runs)}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--solution",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "solution.py",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--ledger",
        default=str(DEFAULT_LEDGER),
        help="frozen holdout ledger path",
    )
    parser.add_argument(
        "--reason",
        help="why this final holdout is consumed (recorded in the ledger)",
    )
    parser.add_argument(
        "--freeze",
        action="store_true",
        help="freeze the ledger with the seed/hash (no evaluation)",
    )
    args = parser.parse_args(argv)
    if args.freeze:
        freeze(Path(args.ledger))
        return 0
    if not args.reason or not args.reason.strip():
        parser.error(
            "--reason is required: the holdout is final-acceptance only "
            "and must not be used for seed/threshold/coverage search"
        )
    run_holdout(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
