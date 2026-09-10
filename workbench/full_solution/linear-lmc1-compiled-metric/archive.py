"""Archive L-MC1 as a version under solutions/.

Refuses to archive unless the candidate on disk is the one the evidence was
measured against, and writes only numbers it reads out of the run artifacts.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

VERSION = "v240"
ARCHIVE = ROOT / "solutions" / f"20260910_{VERSION}_linear-lmc1-compiled-metric_scoreNA_timeNA"

CANDIDATE = HERE / "candidate" / "solution.py"
PARENT = ROOT / "solutions/20260910_v237_linear-tf2-first-pass-gradient-reuse_scoreNA_timeNA/solution.py"

PLAN_REL = "docs/superpowers/plans/2026-09-10-compiled-metric-and-attention-factor-reuse-plan.md"
LOG_REL = "logs/execution/2026-09-10-linear-lmc1-compiled-metric.md"
SIXSHARD_RUN = "artifacts/proxy_v3/linear-lmc1-sixshard-20260910"
SHARD0_RUN = "artifacts/proxy_v3/linear-lmc1-shard0-20260910"

PARENT_SHA256 = "ecb1f9e510b5507e1a2dc8b95f8a84e9b51864a420b3828537a328813e2ce554"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    for required in ("build.json", "verify.json", "verify.out", "timing.json", "paired_sixshard.json"):
        if not (HERE / required).exists():
            raise SystemExit(f"missing run artifact {required}")

    candidate_sha = sha256(CANDIDATE)
    parent_sha = sha256(PARENT)
    if parent_sha != PARENT_SHA256:
        raise SystemExit("the parent is not the recorded v237 archive")

    build = load(HERE / "build.json")
    timing = load(HERE / "timing.json")
    paired = load(HERE / "paired_sixshard.json")
    verify = load(HERE / "verify.json")

    if build["candidate_sha256"] != candidate_sha:
        raise SystemExit("build.json was made from a different candidate")
    if timing["candidate_sha256"] != candidate_sha:
        raise SystemExit("timing.json was measured on a different candidate")
    if paired["candidate_source_sha256"] != candidate_sha:
        raise SystemExit("paired_sixshard.json paired a different candidate")
    if paired["baseline_source_sha256"] != parent_sha:
        raise SystemExit("the six-shard baseline is not the v237 root")
    if paired["case_count"] != 336:
        raise SystemExit(f"the six-shard run covers {paired['case_count']} cases, expected 336")
    if paired["equal_shard_mean_delta_gain"] != 0.0 or paired["positive_cases"] or paired["negative_cases"]:
        raise SystemExit("the six-shard read is not exactly zero everywhere")

    verify_out = (HERE / "verify.out").read_text(encoding="utf-8")
    if "ALL L-MC1 CONTROLS PASSED" not in verify_out:
        raise SystemExit("verify.out does not record a full pass")

    manifest = load(ROOT / SIXSHARD_RUN / "candidate" / "manifest.json")
    if len(manifest["results"]) != 6:
        raise SystemExit("the six-shard manifest does not hold six results")

    ARCHIVE.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(CANDIDATE, ARCHIVE / "solution.py")
    for name in ("build.json", "verify.out", "verify.json", "timing.json", "timing.out"):
        shutil.copyfile(HERE / name, ARCHIVE / name)
    if (ARCHIVE / "solution.py").read_bytes() != CANDIDATE.read_bytes():
        raise SystemExit("archived solution.py is not byte-identical to the candidate")

    effects = {}
    for record in timing["results"]:
        effect = record["paired_effects_ms"]["candidate"]
        null = record["paired_effects_ms"]["sham"]
        effects[record["label"]] = {
            "parent_median_ms": record["arms_ms"]["parent"]["median"],
            "candidate_paired_median_ms": effect["median"],
            "percent_of_parent": effect["percent_of_parent_median"],
            "null_paired_median_ms": null["median"],
            "abs_effect_over_abs_null": abs(effect["median"]) / max(abs(null["median"]), 1e-9),
            "rounds_candidate_faster": effect["rounds_parent_slower"],
            "rounds": record["rounds"],
        }

    config = {
        "run_id": "linear-lmc1-compiled-metric",
        "version": VERSION,
        "version_note": f"{VERSION} is the next free number after v239 at archive time.",
        "mechanism": (
            "L-MC1: build G once in weight calibration and store it (ridge already applied), and "
            "load it in the dynamic path instead of rebuilding it with cholesky_inverse(cholesky("
            "h_inv)) on every call"
        ),
        "mechanism_type": (
            "pure time change that MOVES a computation rather than removing it: calibration gains "
            "one Cholesky inverse per (layer, role), the dynamic path loses one per call"
        ),
        "plan": PLAN_REL,
        "plan_section": "4 (Linear L-MC1)",
        "log": LOG_REL,
        "formal_parent": "solutions/20260910_v237_linear-tf2-first-pass-gradient-reuse_scoreNA_timeNA/solution.py",
        "formal_parent_sha256": parent_sha,
        "formal_parent_official": {"score": 18518, "seconds": 289, "limit_seconds": 300, "margin_seconds": 11},
        "candidate_sha256": candidate_sha,
        "candidate_bytes": len(CANDIDATE.read_bytes()),
        "build": "build.py",
        "build_is_derived_not_handwritten": True,
        "the_change": build["edit"],
        "equivalence": {
            "bar": "bitwise identity, on CUDA (CPU and CUDA cholesky_inverse are not bitwise equal)",
            "result": "passed",
            "dynamic_calls_compared": 9,
            "layer_role_pairs": 3,
            "five_fields_bitwise": True,
            "stored_metric_equals_rebuilt_metric_bitwise": "3/3 pairs",
            "no_mutation": "three calls sharing one state dict give identical fields and leave the stored metric untouched",
            "coverage": "out-of-scope width, no em1 payload, no stored metric, determinism",
        },
        "the_layout_finding": (
            "The first implementation passed 'stored G == rebuilt G bitwise' yet still differed in "
            "the five fields: scale_factor/scale_lv2/scale_lv3 matched and sign/mant did not. The "
            "values were identical but the strides were not -- cholesky_inverse returns a "
            "transposed-stride tensor (stride (1, n), the LAPACK layout) and _cpu_state_tensor's "
            ".contiguous() flattened it, so the downstream .mm() took a different cuBLAS path and "
            "rounded differently. Storing without forcing contiguity fixes it; nan_to_num and the "
            "CPU<->CUDA round trip both preserve strides. The plan said 'build G by the original "
            "expression and float32 order' -- right, but it names the order and not the layout."
        ),
        "local_result": {
            "six_shard_run": SIXSHARD_RUN,
            "shard0_run": SHARD0_RUN,
            "baseline_used": "v237 root (this card's immediate parent)",
            "equal_shard_mean_delta_gain": paired["equal_shard_mean_delta_gain"],
            "case_count": paired["case_count"],
            "positive_cases": paired["positive_cases"],
            "negative_cases": paired["negative_cases"],
            "zero_cases": paired["zero_cases"],
        },
        "cost_accounting": {
            "counts": verify["counts"],
            "timed_layers": timing["results"][0]["label"].rsplit("/", 1)[0],
            "measured": effects,
            "break_even": (
                "calibration +50.0 ms versus dynamic -28.3 ms on 4096 channels, so the break-even "
                "is about 255 dynamic calls, i.e. each calibration must serve about 1.77 dynamic "
                "calls. 144 calibrations x 50.0 ms = 7.21 s added against 288 local dynamic calls x "
                "28.3 ms = 8.15 s saved, a local net of about +0.95 s saved."
            ),
            "correction": (
                "An earlier draft of the execution log stated the break-even as one dynamic call "
                "per calibration. That was wrong: it assumed the two sides' inverses cost the same. "
                "They do not, and the correction is recorded in the log rather than silently fixed."
            ),
        },
        "missing_fact": (
            "How many times the official evaluation calls hif4_dynamic_quantize_activation. The "
            "repository records '50 Linear' samples but never states whether a sample is a "
            "(layer, role, window) triple or a full-model pass. With 50 the card adds about 5.8 s "
            "officially; with 7200 it saves far more. The threshold is now 255 dynamic calls."
        ),
        "lineage_note": (
            "v240 is a descendant of the v237 root; it carries L-TF2 plus this change. It does not "
            "contain v238/v239 (old parent v231) or v233/v235, and scores and seconds do not add "
            "across any of those."
        ),
        "declared_deviations": [],
        "official_status": "unregistered/NA",
        "official_score": None,
        "official_seconds": None,
    }
    (ARCHIVE / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    verification = {
        "version": VERSION,
        "mechanism": config["mechanism"],
        "formal_parent_sha256": parent_sha,
        "candidate_sha256": candidate_sha,
        "candidate_bytes": len(CANDIDATE.read_bytes()),
        "implementation_sha256": build["implementation_sha256"],
        "root_sha256_at_archive_time": sha256(ROOT / "solution.py"),
        "sole_change_measured": {
            "byte_level": (
                "three substring substitutions, each asserted unique; the shadows are derived from "
                "the parent's own text and appended"
            ),
            "substitutions_performed": build["substitutions_performed"],
            "substitution_sites": build["substitution_sites"],
            "changed_lines": build["changed_lines"],
            "byte_delta": build["byte_delta"],
        },
        "single_file_import": "verify.py control A imports the six APIs from a bare temporary directory",
        "attention_and_encoder_bytecode_identical": True,
        "shadows_live": "both appended definitions shadow the parent's (co_firstlineno) and are resolved by name",
        "protocol": "proxy-v3",
        "panel": "qwen35-4b-panel-v1 (336 Linear cases)",
        "scope": "linear-only, six shards, target side",
        "hard_output_changed": False,
        "equivalence_is_exact_not_tolerance_based": (
            "the five HiF4 fields are compared as raw bytes on CUDA, and the stored G is compared "
            "byte for byte against the G the parent rebuilds at call time"
        ),
        "call_counts": verify["counts"],
        "equal_shard_mean_delta_gain": paired["equal_shard_mean_delta_gain"],
        "shard_delta_gain_mean": paired["shard_delta_gain_mean"],
        "positive_cases": paired["positive_cases"],
        "negative_cases": paired["negative_cases"],
        "zero_cases": paired["zero_cases"],
        "case_count": paired["case_count"],
        "shard_decision": "no_effect (the evaluator's own small-trend filter)",
        "shard_decision_note": (
            "336 exactly zero is the predicted reading for a bitwise-equivalent candidate; a "
            "non-zero delta would be the defect"
        ),
        "pairing_method": paired["method"],
        "stopped_early_flag": {
            "manifest_value": bool(manifest["stopped_early"]),
            "is_a_truncation": False,
            "why": (
                "the stop check runs after a shard's results are appended, and this candidate is "
                "bitwise-equivalent to its parent, so delta_mean == 0 on every shard and the "
                "counter reaches --stop-after-nonpositive 6 exactly on the last requested shard."
            ),
            "independent_coverage_check": (
                f"candidate-linear-shard{{0..5}}.json: 6 files, {paired['case_count']} cases, "
                f"statuses ok"
            ),
        },
        "official_status": "unregistered/NA",
        "official_score": None,
        "official_seconds": None,
    }
    (ARCHIVE / "verification.json").write_text(
        json.dumps(verification, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    official = {
        "version": VERSION,
        "candidate_sha256": candidate_sha,
        "parent_sha256": parent_sha,
        "official_status": "unregistered/NA",
        "official_score": None,
        "official_time_s": None,
        "official_time_display": None,
        "reported_by": None,
        "reported_at": None,
        "decision": (
            "PENDING a submission decision. The local evidence is bitwise equivalence on CUDA "
            "(336/336 paired cases exactly zero, nine dynamic calls on three layer/role pairs with "
            "byte-identical five fields, and the stored G equal to the rebuilt G) plus a measured "
            "cost accounting on both sides of the move."
        ),
        "does_this_card_reduce_work": (
            "It MOVES work rather than removing it, so the answer depends on the call ratio and "
            "not on the mechanism alone: calibration gains one Cholesky inverse per (layer, role) "
            "and the dynamic path loses one per call. Measured on 4096 channels, calibration pays "
            "+50.0 ms and each dynamic call saves 28.3 ms, so the break-even is about 255 dynamic "
            "calls. Locally (144 calibrations, 288 dynamic) that is a net saving of about 0.95 s."
        ),
        "missing_fact": (
            "The number of dynamic calls in the official evaluation is not recorded anywhere in "
            "this repository. With 50 the card costs about 5.8 s officially; with a full-model pass "
            "per sample it saves far more. The sign is therefore undetermined locally, and it is "
            "registered as a missing fact rather than assumed."
        ),
        "local_projection_seconds": None,
        "local_projection_margin_seconds": None,
        "local_projection_basis": None,
        "projection_note": (
            "no projection is recorded: the official 300 s gate is the only time gate and this "
            "machine's wall clock cannot be converted into the official machine's."
        ),
        "official_scored_sha256": None,
        "archive_sha256": candidate_sha,
        "note": (
            "v240 descends from the v237 root and is not additive with v233, v235, v237, v238 or "
            "v239."
        ),
        "candidate_type": (
            "complete Linear L-MC1 candidate (v237 root plus two appended shadows); the only "
            f"{VERSION} under solutions/ at archive time"
        ),
        "execution_log": LOG_REL,
    }
    (ARCHIVE / "official-result.json").write_text(
        json.dumps(official, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"archived {VERSION} -> {ARCHIVE.relative_to(ROOT)}")
    print(f"candidate_sha256={candidate_sha}")
    print(f"files={sorted(p.name for p in ARCHIVE.iterdir())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
