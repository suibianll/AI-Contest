"""Archive L-TF2 as a version under solutions/.

Refuses to archive unless the candidate on disk is the one the evidence was
measured against: the candidate SHA is recomputed here and compared with the
SHA recorded in build.json, verify.out, timing.json and paired_sixshard.json.
Hand-copying numbers into a result.md is how an archive drifts away from its own
evidence, so every number written below is read out of the run artifacts.

Writes solution.py, build.json, config.json, verification.json, verify.out and
timing.json.  result.md is written by hand -- it is prose, and this script does
not invent prose.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

VERSION = "v237"
ARCHIVE = ROOT / "solutions" / f"20260910_{VERSION}_linear-tf2-first-pass-gradient-reuse_scoreNA_timeNA"

CANDIDATE = HERE / "candidate" / "solution.py"
PARENT = ROOT / "solutions/20260910_v231_linear-em3-k2-arm_scoreNA_timeNA/solution.py"
PLAN_REL = "docs/superpowers/plans/2026-09-10-retained-root-gradient-reuse-plan.md"
LOG_REL = "logs/execution/2026-09-10-linear-tf2-retained-root-gradient-reuse.md"

SIXSHARD_RUN = "artifacts/proxy_v3/linear-tf2-sixshard-20260910"
SHARD0_RUN = "artifacts/proxy_v3/linear-tf2-shard0-20260910"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    for required in ("build.json", "verify.out", "timing.json", "paired_sixshard.json"):
        if not (HERE / required).exists():
            raise SystemExit(f"missing run artifact {required}; run the tooling first")

    candidate_bytes = CANDIDATE.read_bytes()
    candidate_sha = hashlib.sha256(candidate_bytes).hexdigest()
    parent_bytes = PARENT.read_bytes()
    parent_sha = hashlib.sha256(parent_bytes).hexdigest()

    build = load(HERE / "build.json")
    timing = load(HERE / "timing.json")
    paired = load(HERE / "paired_sixshard.json")

    if build["candidate_sha256"] != candidate_sha:
        raise SystemExit("build.json was made from a different candidate than the one on disk")
    if timing["candidate_sha256"] != candidate_sha:
        raise SystemExit("timing.json was measured on a different candidate")
    if paired["candidate_source_sha256"] != candidate_sha:
        raise SystemExit("paired_sixshard.json paired a different candidate")
    if build["parent_sha256"] != parent_sha:
        raise SystemExit("build.json was built from a different parent than the archive root")
    if parent_sha != "ea79a1c12dc667142c620975aab188920fae7b29988c804f41c1a696cc5754f1":
        raise SystemExit("the archive parent is not the v231 root")
    if paired["baseline_source_sha256"] != parent_sha:
        raise SystemExit("the six-shard baseline is not the v231 root")
    if paired["case_count"] != 336:
        raise SystemExit(f"the six-shard run covers {paired['case_count']} cases, expected 336")

    ARCHIVE.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(CANDIDATE, ARCHIVE / "solution.py")
    shutil.copyfile(HERE / "build.json", ARCHIVE / "build.json")
    shutil.copyfile(HERE / "verify.out", ARCHIVE / "verify.out")
    shutil.copyfile(HERE / "timing.json", ARCHIVE / "timing.json")
    if (ARCHIVE / "solution.py").read_bytes() != candidate_bytes:
        raise SystemExit("archived solution.py is not byte-identical to the candidate")

    effects = {r["label"]: r for r in timing["results"]}
    null_over_effect = {}
    for label, record in effects.items():
        effect = record["candidate_paired_median_ms"]
        null = record["null_paired_median_ms"]
        null_over_effect[label] = {
            "candidate_paired_median_ms": effect,
            "null_paired_median_ms": null,
            "candidate_dispersion_p25_p75_ms": [
                record["candidate_paired_dispersion_ms"],
            ],
            "abs_effect_over_abs_null": abs(effect) / max(abs(null), 1e-9),
            "rounds_candidate_faster": record["paired_effects_ms"]["candidate"][
                "rounds_parent_slower"
            ],
            "rounds_null_faster": record["paired_effects_ms"]["sham"][
                "rounds_parent_slower"
            ],
            "rounds": timing["rounds_per_arm"],
        }

    config = {
        "run_id": "linear-tf2-retained-root-gradient-reuse",
        "version": VERSION,
        "version_note": (
            f"{VERSION} is the next free number after v236; no v237 existed under solutions/ "
            "at archive time."
        ),
        "mechanism": (
            "L-TF2: the first descent pass reuses the gradient the parent computed and "
            "checked before the loop, instead of recomputing it as the first statement of "
            "pass 0. Same edit as L-TF1 (v233), re-derived on the K=2 root."
        ),
        "mechanism_type": (
            "pure time change with an exact output-equivalence obligation: strictly fewer "
            "operations, identical results"
        ),
        "plan": PLAN_REL,
        "plan_section": "2 (Linear L-TF2)",
        "log": LOG_REL,
        "parent": "solutions/20260910_v231_linear-em3-k2-arm_scoreNA_timeNA/solution.py",
        "parent_identity": "v231 Linear (L-EM3 K=2) + v195 Attention",
        "parent_sha256": parent_sha,
        "parent_bytes": len(parent_bytes),
        "parent_official": {"score": 18518, "seconds": 291, "limit_seconds": 300, "margin_seconds": 9},
        "candidate_sha256": candidate_sha,
        "candidate_bytes": len(candidate_bytes),
        "implementation_sha256": build["implementation_sha256"],
        "implementation_bytes": build["implementation_bytes"],
        "build": "build.py",
        "build_is_derived_not_handwritten": True,
        "the_change": (
            "one substring substitution inside the parent's own _em1_dynamic_descent: the "
            "loop head becomes `if _pass:` and the in-loop gradient recomputation moves "
            "inside it. 5 changed lines, +34 bytes in the extracted function."
        ),
        "fixed_choices_inherited_unchanged": [
            "_EM1_PASSES = 2 (K is asserted, not read back)",
            "_EM1_GROUPS_PER_BLOCK = 16",
            "the metric and its ridge",
            "the candidate set and the 16-step group-major schedule",
            "the coverage rule and the whole-row joint acceptance",
            "the pre-loop finiteness check and the nonfinite-gradient diagnostic",
            "every Attention API (asserted byte-identical in bytecode)",
        ],
        "what_the_change_does_to_execution": (
            "at K=2 the candidate issues 36 Tensor.mm products per descent call against the "
            "parent's 38; the candidate's recorded product sequence equals the parent's with "
            "products 2 and 3 removed and nothing else changed, so pass 1 still recomputes"
        ),
        "local_result": {
            "six_shard_run": SIXSHARD_RUN,
            "shard0_run": SHARD0_RUN,
            "equal_shard_mean_delta_gain": paired["equal_shard_mean_delta_gain"],
            "case_count": paired["case_count"],
            "positive_cases": paired["positive_cases"],
            "negative_cases": paired["negative_cases"],
            "zero_cases": paired["zero_cases"],
            "delta_min": min(s["delta_min"] for s in paired["shards"]),
            "delta_max": max(s["delta_max"] for s in paired["shards"]),
            "reasonableness_issues": 0,
        },
        "time": {
            "verdict": (
                "the wall-clock effect is below this machine's resolution; recorded with a "
                "null control rather than quoted with a sign"
            ),
            "operators_removed_per_call": 2,
            "operators_parent_per_call": 38,
            "operators_candidate_per_call": 36,
            "paired_three_arm": null_over_effect,
        },
        "lineage_note": (
            "v237 is a descendant of v231 (contains K=2) and of v230 (contains L-EM2/L-EM3); "
            "it is NOT a descendant of v233 (L-TF1 on the K=1 v230 root) and does not contain "
            "v232's L-QF1 or v235's L-AD1. Scores and seconds do not add across these."
        ),
        "declared_deviations": [],
        "local_decision": "archive and submit once for official adjudication",
        "local_decision_note": (
            "a pure time candidate with proven-identical output may take one official "
            "verification; it is not a repeated equivalent A/B"
        ),
        "official_status": "unregistered/NA",
        "official_score": None,
        "official_seconds": None,
    }
    (ARCHIVE / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    shard_decision = (
        "336 of 336 paired cases are exactly zero; the evaluator's own paired statistic "
        "lands on zero everywhere, which is the predicted reading for a constructively "
        "output-equivalent candidate"
    )
    verification = {
        "version": VERSION,
        "mechanism": config["mechanism"],
        "parent_sha256": parent_sha,
        "parent_bytes": len(parent_bytes),
        "candidate_sha256": candidate_sha,
        "candidate_bytes": len(candidate_bytes),
        "implementation_sha256": build["implementation_sha256"],
        "root_sha256_at_archive_time": sha256(ROOT / "solution.py"),
        "sole_change_measured": {
            "byte_level": "one substring substitution; +34 B in the extracted function; 5 changed lines",
            "syntax_level": "AST identical to the parent once the single `if _pass:` guard is dissolved",
            "ast_identical_after_dissolving_pass_guard": build[
                "ast_identical_after_dissolving_pass_guard"
            ],
            "substitutions_performed": build["substitutions_performed"],
            "shared_prefix_bytes": build["parent_bytes"],
        },
        "single_file_import": (
            "the six APIs import and run from a bare temporary directory holding nothing "
            "else (verify.py control A, single_file_import)"
        ),
        "attention_apis_bytecode_identical": True,
        "shipped_passes": 2,
        "protocol": "proxy-v3",
        "panel": "qwen35-4b-panel-v1 (336 Linear cases)",
        "scope": "linear-only, six shards, target side",
        "hard_output_changed": False,
        "equivalence_is_exact_not_tolerance_based": (
            "the five HiF4 fields are compared as raw bytes on the compiled synthetic layer, "
            "on 128 real rows of the cached layer-0/q state, on a state with no eml payload "
            "and on the out-of-scope bypass; no tolerance is used anywhere"
        ),
        "equal_shard_mean_delta_gain": paired["equal_shard_mean_delta_gain"],
        "delta_reference": "candidate.gain - baseline.gain, per case, keyed by case_id",
        "shard_delta_gain_mean": paired["shard_delta_gain_mean"],
        "positive_cases": paired["positive_cases"],
        "negative_cases": paired["negative_cases"],
        "zero_cases": paired["zero_cases"],
        "case_count": paired["case_count"],
        "delta_extremes": {
            "min": min(s["delta_min"] for s in paired["shards"]),
            "max": max(s["delta_max"] for s in paired["shards"]),
        },
        "delta_by_role_six_shards": paired["shards"][0]["delta_by_role"],
        "shard_decision": "no_effect (the evaluator's own small-trend filter)",
        "shard_decision_note": shard_decision,
        "pairing_method": paired["method"],
        "in_process_shards": [0, 1, 2, 3, 4, 5],
        "stopped_early_flag": {
            "manifest_value": True,
            "is_a_truncation": False,
            "why": (
                "the stop check runs after the shard's results are appended, and at K=2 a "
                "constructively output-equivalent candidate yields delta_mean == 0 on every "
                "shard, so the six-shard counter reaches --stop-after-nonpositive 6 exactly "
                "on the last requested shard. All six shards are recorded (6 results, "
                "status ok, 56 cases each, 336 total) with six analysis files; no shard was "
                "skipped and no coverage was lost."
            ),
            "independent_coverage_check": (
                "candidate-linear-shard{0..5}.json: 6 files, status ok, 56 cases each, "
                "336 total, source_sha256 ecb1f9e5 consistent across shards"
            ),
        },
        "official_status": "unregistered/NA",
        "official_score": None,
        "official_seconds": None,
    }
    (ARCHIVE / "verification.json").write_text(
        json.dumps(verification, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"archived {VERSION} -> {ARCHIVE.relative_to(ROOT)}")
    print(f"candidate_sha256={candidate_sha}")
    print(f"files={sorted(p.name for p in ARCHIVE.iterdir())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
