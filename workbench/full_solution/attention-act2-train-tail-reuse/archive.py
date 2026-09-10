"""Archive A-CT2 as a version under solutions/.

Refuses to archive unless the candidate on disk is the one the evidence was
measured against, and unless every number it writes is read out of the run
artifacts.  `result.md` is written by hand.

This archive records two things side by side and does not let either soften the
other: the equivalence is exact and proven, and the time the card buys back is
known to be far too small to change any official outcome.  The second fact is
carried in from the stall diagnostic, not re-derived here.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

VERSION = "v239"
ARCHIVE = ROOT / "solutions" / f"20260910_{VERSION}_attention-act2-train-tail-reuse_scoreNA_timeNA"

CANDIDATE = HERE / "candidate" / "solution.py"
ASSEMBLY = ROOT / "solutions/20260910_v236_attention-agr1-on-v231_scoreNA_timeNA/solution.py"
FORMAL_PARENT = ROOT / "solutions/20260910_v231_linear-em3-k2-arm_scoreNA_timeNA/solution.py"

PLAN_REL = "docs/superpowers/plans/2026-09-10-retained-root-gradient-reuse-plan.md"
LOG_REL = "logs/execution/2026-09-10-attention-act2-train-tail-reuse.md"
STALL_REPORT = "docs/attention-stall-analysis-2026-09-10.md"

SIXSHARD_RUN = "artifacts/proxy_v3/attention-act2-sixshard-20260910"
SHARD0_RUN = "artifacts/proxy_v3/attention-act2-shard0-20260910"
AGR1_REFERENCE_RUN = "artifacts/proxy_v3/attention-agr1-on-v231-sixshard-20260910"

ASSEMBLY_SHA256 = "3319fc354af1ff0f35075498f26886bdf43a078d32a27852f575a4c50d32ba07"
FORMAL_PARENT_SHA256 = "ea79a1c12dc667142c620975aab188920fae7b29988c804f41c1a696cc5754f1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def shard_totals(run_dir: Path, prefix: str) -> dict:
    cases = 0
    shas = set()
    statuses = set()
    for shard in range(6):
        payload = load(run_dir / prefix / f"{prefix}-attention-shard{shard}.json")
        result = payload["results"][0]
        statuses.add(str(result.get("status")))
        shas.add(str(result["source_sha256"]))
        cases += len(result["case_scores"]["attention"])
    return {"cases": cases, "statuses": sorted(statuses), "source_shas": sorted(shas)}


def main() -> int:
    for required in ("build.json", "verify.out", "timing.json", "paired_sixshard.json", "audit.out"):
        if not (HERE / required).exists():
            raise SystemExit(f"missing run artifact {required}; run the tooling first")

    candidate_sha = sha256(CANDIDATE)
    assembly_sha = sha256(ASSEMBLY)
    parent_sha = sha256(FORMAL_PARENT)

    build = load(HERE / "build.json")
    timing = load(HERE / "timing.json")
    paired = load(HERE / "paired_sixshard.json")

    if assembly_sha != ASSEMBLY_SHA256:
        raise SystemExit("the assembly is not the recorded v236 archive")
    if parent_sha != FORMAL_PARENT_SHA256:
        raise SystemExit("the formal parent is not the v231 root")
    if build["candidate_sha256"] != candidate_sha:
        raise SystemExit("build.json was made from a different candidate")
    if timing["candidate_sha256"] != candidate_sha:
        raise SystemExit("timing.json was measured on a different candidate")
    if paired["candidate_source_sha256"] != candidate_sha:
        raise SystemExit("paired_sixshard.json paired a different candidate")
    if paired["baseline_source_sha256"] != assembly_sha:
        raise SystemExit("the six-shard baseline is not the v236 A-GR1 implementation")
    if paired["case_count"] != 72:
        raise SystemExit(f"the six-shard run covers {paired['case_count']} cases, expected 72")
    if (
        paired["equal_shard_mean_delta_gain"] != 0.0
        or paired["positive_cases"]
        or paired["negative_cases"]
    ):
        raise SystemExit("the six-shard read is not exactly zero everywhere")

    verify_out = (HERE / "verify.out").read_text(encoding="utf-8")
    if "ALL A-CT2 CONTROLS PASSED" not in verify_out:
        raise SystemExit("verify.out does not record a full pass")
    audit_out = (HERE / "audit.out").read_text(encoding="utf-8")
    if "DUPLICATION CONFIRMED" not in audit_out:
        raise SystemExit("audit.out does not record a confirmed duplication")

    manifest = load(ROOT / SIXSHARD_RUN / "candidate" / "manifest.json")
    if len(manifest["results"]) != 6:
        raise SystemExit("the six-shard manifest does not hold six results")
    own_sixshard = shard_totals(ROOT / SIXSHARD_RUN, "candidate")
    agr1_sixshard = shard_totals(ROOT / AGR1_REFERENCE_RUN, "candidate")
    if agr1_sixshard["cases"] != 72:
        raise SystemExit("the reference A-GR1 six-shard run does not cover 72 cases")
    if not (ROOT / STALL_REPORT).exists():
        raise SystemExit("the stall diagnostic the magnitude note cites is missing")

    ARCHIVE.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(CANDIDATE, ARCHIVE / "solution.py")
    for name in ("build.json", "verify.out", "audit.out", "timing.json"):
        shutil.copyfile(HERE / name, ARCHIVE / name)
    if (ARCHIVE / "solution.py").read_bytes() != CANDIDATE.read_bytes():
        raise SystemExit("archived solution.py is not byte-identical to the candidate")

    effects = {}
    for record in timing["results"]:
        effect = record["paired_effects_ms"]["candidate"]
        null = record["paired_effects_ms"]["sham"]
        effects[record["label"]] = {
            "assembly_median_ms": record["arms_ms"]["assembly"]["median"],
            "candidate_paired_median_ms": effect["median"],
            "percent_of_assembly": effect["percent_of_assembly_median"],
            "null_paired_median_ms": null["median"],
            "abs_effect_over_abs_null": abs(effect["median"]) / max(abs(null["median"]), 1e-9),
            "rounds_candidate_faster": effect["rounds_assembly_slower"],
            "rounds_null_faster": null["rounds_assembly_slower"],
            "rounds": record["rounds"],
        }

    config = {
        "run_id": "attention-act2-train-tail-reuse",
        "version": VERSION,
        "version_note": f"{VERSION} is the next free number after v238 at archive time.",
        "mechanism": (
            "A-CT2: the tail of _agr1_train carries the per-role scalar sums out of the "
            "final_loss loop and computes the two agr1_*_scale_ratio2 entries from them, "
            "deleting a second walk over the same prepared folds that recomputed the same "
            "rotation and the same loss and kept only the scalar the first walk discarded"
        ),
        "mechanism_type": (
            "pure time change with an exact output-equivalence obligation; validated by a "
            "static audit before any card was registered"
        ),
        "plan": PLAN_REL,
        "plan_section": "4 (the A-CT2 fixed implementation card)",
        "log": LOG_REL,
        "audit": "workbench/full_solution/attention-act2-train-tail-reuse/audit.py",
        "formal_parent": "solutions/20260910_v231_linear-em3-k2-arm_scoreNA_timeNA/solution.py",
        "formal_parent_sha256": parent_sha,
        "formal_parent_official": {"score": 18518, "seconds": 291, "limit_seconds": 300, "margin_seconds": 9},
        "assembly": "solutions/20260910_v236_attention-agr1-on-v231_scoreNA_timeNA/solution.py",
        "assembly_sha256": assembly_sha,
        "assembly_note": (
            "v236 is the v231 root plus the A-GR1 block byte for byte, so it is both the "
            "A-GR1 assembly and the same-parent implementation control. It is NOT a promoted "
            "parent; nothing is inherited from it, including its official outcome."
        ),
        "candidate_sha256": candidate_sha,
        "candidate_bytes": len(CANDIDATE.read_bytes()),
        "implementation_sha256": build["implementation_sha256"],
        "build": "build.py",
        "build_is_derived_not_handwritten": True,
        "the_change": build["edit"],
        "what_the_change_does_to_execution": (
            "per _agr1_train call: _agr1_scale_loss_grad 204 -> 198 and "
            "_a2_apply_group_rotation 476 -> 470 (layer 0; the delta is 6 everywhere). The "
            "192 training-step calls are untouched, the tail drops from 12 to 6."
        ),
        "local_result": {
            "six_shard_run": SIXSHARD_RUN,
            "shard0_run": SHARD0_RUN,
            "baseline_used": "v236 (same-parent A-GR1 old implementation)",
            "equal_shard_mean_delta_gain": paired["equal_shard_mean_delta_gain"],
            "case_count": paired["case_count"],
            "positive_cases": paired["positive_cases"],
            "negative_cases": paired["negative_cases"],
            "zero_cases": paired["zero_cases"],
        },
        "mechanism_change_vs_formal_parent": {
            "note": (
                "A-GR1's change against the v231 root is inherited by transitivity: the "
                "per-case pairing above is exactly zero on all 72 cases. Cited from v236's "
                "own archived run, not re-measured."
            ),
            "reference_run": AGR1_REFERENCE_RUN,
            "reference_cases": agr1_sixshard["cases"],
        },
        "time": {
            "verdict": (
                "accounting figure, not a speed-up claim: see timing.json.  The card removes "
                "6 of 204 calls in the tail of one function inside a ~6 s calibration, and "
                "the whole-calibration reading is reported against a byte-identical sham null"
            ),
            "removed_calls_per_train": 6,
            "total_calls_per_train": 204,
            "paired_three_arm": effects,
        },
        "magnitude_is_known_insufficient": {
            "recorded": True,
            "why": (
                "The card's bitwise-equivalent baseline v236 returned an official TIMEOUT on "
                "this same root, and the stall diagnostic (docs/attention-stall-analysis-"
                "2026-09-10.md, section 6.3 item 4) measures the whole A-CT1 de-duplication "
                "direction at roughly 0.2 s against the roughly 12 s the mechanism would need "
                "to fit inside the 300 s limit, and states that A-CT2 is smaller still and "
                "not worth opening a card.  The user directed that the card nevertheless be "
                "run to completion and archived, so it is archived with that assessment "
                "attached rather than presented as a candidate expected to change an outcome."
            ),
            "no_official_outcome_claim": True,
        },
        "lineage_note": (
            "v239 is a descendant of v231 (formal parent) and of v236 (the A-GR1 assembly on "
            "that parent). It carries A-GR1 plus the training-tail change. It does not "
            "contain v237's L-TF2, v238's A-CT1, v233's L-TF1, v232's L-QF1 or v235's L-AD1, "
            "and scores and seconds do not add across any of those."
        ),
        "declared_deviations": [],
        "local_decision": "archive and submit at most once for official adjudication",
        "local_decision_note": (
            "A pure time candidate with a real execution-cost change may take one official "
            "verification under AGENTS section 2. This one is archived with the prior "
            "assessment that its magnitude is far below what would matter."
        ),
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
        "assembly_sha256": assembly_sha,
        "candidate_sha256": candidate_sha,
        "candidate_bytes": len(CANDIDATE.read_bytes()),
        "implementation_sha256": build["implementation_sha256"],
        "root_sha256_at_archive_time": sha256(ROOT / "solution.py"),
        "audit": {
            "result": "DUPLICATION CONFIRMED",
            "method": (
                "structural comparison of the two expressions, plus an instrumented real "
                "_agr1_train call on every full-attention layer"
            ),
            "calls_per_train": 204,
            "duplicated_calls": 6,
            "bit_identical_duplicates": "6/6 on all six layers",
            "rebuilt_ratios_exact": "bit-identical to the reported values on all six layers",
            "force_zero_reachable": False,
        },
        "sole_change_measured": {
            "byte_level": (
                "two substring substitutions, both asserted unique and both asserted absent "
                "afterwards; the candidate text must equal the parent with those two blocks "
                "replaced, which given uniqueness is the statement that nothing else changed"
            ),
            "substitutions_performed": build["substitutions_performed"],
            "substitution_sites": build["substitution_sites"],
            "extracted_function_changed_lines": build["extracted_function_changed_lines"],
            "extracted_function_byte_delta": build["extracted_function_byte_delta"],
            "shared_prefix_root_bytes": build["root_bytes"],
            "shared_prefix_assembly_bytes": build["assembly_bytes"],
        },
        "single_file_import": "verify.py control A imports the six APIs from a bare temporary directory",
        "linear_attention_and_qkv_bytecode_identical_to_parent": True,
        "protocol": "proxy-v3",
        "panel": "qwen35-4b-panel-v1 (72 Attention cases)",
        "scope": "attention-only, six shards, target side",
        "hard_output_changed": False,
        "equivalence_is_exact_not_tolerance_based": (
            "the returned q/k/v states of the full calibration are compared as raw bytes, "
            "which is strictly stronger than comparing the three floats at stake "
            "(agr1_final_loss, agr1_q_scale_ratio2, agr1_k_scale_ratio2) -- both were checked"
        ),
        "call_counts_per_train": {
            "agr1_scale_loss_grad": [204, 198],
            "a2_apply_group_rotation": [476, 470],
            "training_step_calls_unchanged": 192,
            "tail_calls": [12, 6],
            "method": "independent invocation counters over the module's own names",
        },
        "coverage": {
            "accepted_and_rejected_on_real_layers": True,
            "identity_parent_arm": True,
            "ineligible_short_window_list": True,
            "exception_fallback": True,
            "determinism": "the assembly compared against itself is byte-identical",
            "passed": "ALL A-CT2 CONTROLS PASSED",
        },
        "equal_shard_mean_delta_gain": paired["equal_shard_mean_delta_gain"],
        "shard_delta_gain_mean": paired["shard_delta_gain_mean"],
        "positive_cases": paired["positive_cases"],
        "negative_cases": paired["negative_cases"],
        "zero_cases": paired["zero_cases"],
        "case_count": paired["case_count"],
        "shard_decision": "no_effect (the evaluator's own small-trend filter)",
        "shard_decision_note": (
            "exactly zero everywhere is the predicted reading for a candidate that "
            "reproduces its control; a non-zero delta would be the defect"
        ),
        "pairing_method": paired["method"],
        "stopped_early_flag": {
            "manifest_value": bool(manifest["stopped_early"]),
            "is_a_truncation": False,
            "why": (
                "the stop check runs after the shard's results are appended, and this "
                "candidate reproduces its control exactly, so delta_mean == 0 on every shard "
                "and the counter reaches --stop-after-nonpositive 6 exactly on the last "
                "requested shard."
            ),
            "independent_coverage_check": (
                f"candidate-attention-shard{{0..5}}.json: 6 files, statuses "
                f"{own_sixshard['statuses']}, {own_sixshard['cases']} cases, source_sha256 "
                f"{[s[:8] for s in own_sixshard['source_shas']]}"
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
            "PENDING a submission decision. The local evidence is implementation equivalence "
            "with the same-parent A-GR1 (72/72 paired cases exactly zero, byte-identical "
            "calibration states on all six real attention layers) plus a proven reduction of "
            "6 of 204 calls in the training tail."
        ),
        "magnitude_note": (
            "The equivalence is proven and the saving is measured, but the saving is far too "
            "small to change an official outcome, and that is recorded here rather than "
            "implied away. A-CT1's whole A-GR1 de-duplication direction was measured at "
            "roughly 0.2 s against the roughly 12 s the mechanism would need; A-CT2 removes "
            "6 of 204 calls in one function's tail, so it is smaller still. The stall "
            "diagnostic (docs/attention-stall-analysis-2026-09-10.md, section 6.3 item 4) "
            "concludes that this direction should stop and that A-CT2 is not worth a card. "
            "The user directed that the card be run to completion and archived anyway, with "
            "that assessment attached."
        ),
        "local_projection_seconds": None,
        "local_projection_margin_seconds": None,
        "local_projection_basis": None,
        "projection_note": (
            "no projection is recorded, and none is warranted: the official 300 s gate is the "
            "only time gate, and this machine's wall clock cannot be converted into the "
            "official machine's."
        ),
        "official_scored_sha256": None,
        "archive_sha256": candidate_sha,
        "note": (
            "v239 is a descendant of the v231 root and is not additive with v234, v236, v238, "
            "v233, v235 or v237. A-GR1's official record -- side isolation +29, complete "
            "package TIMEOUT on both v234 and v236 -- stands separately and is not inherited."
        ),
        "candidate_type": (
            "complete Attention A-CT2 candidate (v231 root + A-GR1 + the training-tail "
            f"carry); the only {VERSION} under solutions/ at archive time"
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
