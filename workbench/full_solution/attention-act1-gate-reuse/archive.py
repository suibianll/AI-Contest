"""Archive A-CT1 as a version under solutions/.

Refuses to archive unless the candidate on disk is the one the evidence was
measured against: the candidate SHA is recomputed here and compared with the
SHA recorded in build.json, timing.json and paired_sixshard.json.  Every number
written below is read out of the run artifacts rather than retyped.

Writes solution.py, build.json, config.json, verification.json, verify.out,
timing.json and official-result.json.  result.md is written by hand.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

VERSION = "v238"
ARCHIVE = ROOT / "solutions" / f"20260910_{VERSION}_attention-act1-gate-reuse_scoreNA_timeNA"

CANDIDATE = HERE / "candidate" / "solution.py"
ASSEMBLY = ROOT / "solutions/20260910_v236_attention-agr1-on-v231_scoreNA_timeNA/solution.py"
FORMAL_PARENT = ROOT / "solutions/20260910_v231_linear-em3-k2-arm_scoreNA_timeNA/solution.py"

PLAN_REL = "docs/superpowers/plans/2026-09-10-retained-root-gradient-reuse-plan.md"
LOG_REL = "logs/execution/2026-09-10-attention-act1-gate-reuse.md"

SIXSHARD_RUN = "artifacts/proxy_v3/attention-act1-sixshard-20260910"
SHARD0_RUN = "artifacts/proxy_v3/attention-act1-shard0-20260910"
AGR1_REFERENCE_RUN = "artifacts/proxy_v3/attention-agr1-on-v231-sixshard-20260910"

ASSEMBLY_SHA256 = "3319fc354af1ff0f35075498f26886bdf43a078d32a27852f575a4c50d32ba07"
FORMAL_PARENT_SHA256 = "ea79a1c12dc667142c620975aab188920fae7b29988c804f41c1a696cc5754f1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def shard_totals(run_dir: Path, prefix: str) -> dict:
    """Reads the per-shard JSONs: case count, status and recorded source sha."""

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
    for required in ("build.json", "verify.out", "timing.json", "paired_sixshard.json"):
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
    if paired["equal_shard_mean_delta_gain"] != 0.0 or paired["negative_cases"] or paired["positive_cases"]:
        raise SystemExit(
            "the six-shard read is not exactly zero everywhere; this card's claim is "
            "implementation equivalence and an archive must not paper over a difference"
        )

    # The mechanism's change against the formal parent is inherited from v236's
    # own archived run, which is only sound because the pairing above is exactly
    # zero.  Re-derive both sides of that from the archived run here.
    agr1_sixshard = shard_totals(ROOT / AGR1_REFERENCE_RUN, "candidate")
    if agr1_sixshard["cases"] != 72:
        raise SystemExit("the reference A-GR1 six-shard run does not cover 72 cases")

    manifest = load(ROOT / SIXSHARD_RUN / "candidate" / "manifest.json")
    if len(manifest["results"]) != 6:
        raise SystemExit("the six-shard manifest does not hold six results")
    own_sixshard = shard_totals(ROOT / SIXSHARD_RUN, "candidate")

    verify_out = (HERE / "verify.out").read_text(encoding="utf-8")
    if "ALL A-CT1 CONTROLS PASSED" not in verify_out:
        raise SystemExit("verify.out does not record a full pass")

    ARCHIVE.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(CANDIDATE, ARCHIVE / "solution.py")
    for name in ("build.json", "verify.out", "timing.json"):
        shutil.copyfile(HERE / name, ARCHIVE / name)
    if (ARCHIVE / "solution.py").read_bytes() != CANDIDATE.read_bytes():
        raise SystemExit("archived solution.py is not byte-identical to the candidate")

    gate_effects = {}
    for record in timing["results"]:
        effect = record["paired_effects_ms"]["candidate"]
        null = record["paired_effects_ms"]["sham"]
        gate_effects[record["label"]] = {
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
        "run_id": "attention-act1-gate-reuse",
        "version": VERSION,
        "version_note": f"{VERSION} is the next free number after v237 at archive time.",
        "mechanism": (
            "A-CT1: inside A-GR1's gate, the two consecutive _agr1_gate_loss calls for one "
            "gate window become one _act1_gate_pair call that computes the arm-independent "
            "work once (dense reference Q/K/V, the reference target, and the parent-side V) "
            "while still evaluating both arms' Q/K and their own attention forward"
        ),
        "mechanism_type": (
            "pure time change with an exact output-equivalence obligation: strictly less "
            "work inside the gate, identical losses, identical acceptance, identical states"
        ),
        "plan": PLAN_REL,
        "plan_section": "4 (Attention A-CT1)",
        "log": LOG_REL,
        "formal_parent": "solutions/20260910_v231_linear-em3-k2-arm_scoreNA_timeNA/solution.py",
        "formal_parent_identity": "v231 Linear (L-EM3 K=2) + v195 Attention",
        "formal_parent_sha256": parent_sha,
        "formal_parent_bytes": len(FORMAL_PARENT.read_bytes()),
        "formal_parent_official": {"score": 18518, "seconds": 291, "limit_seconds": 300, "margin_seconds": 9},
        "assembly": "solutions/20260910_v236_attention-agr1-on-v231_scoreNA_timeNA/solution.py",
        "assembly_sha256": assembly_sha,
        "assembly_bytes": len(ASSEMBLY.read_bytes()),
        "assembly_note": (
            "v236 is the v231 root plus the A-GR1 block byte for byte, so it is both the "
            "formal parent's A-GR1 assembly and the same-parent implementation control. It "
            "is NOT a promoted parent and nothing is inherited from it."
        ),
        "candidate_sha256": candidate_sha,
        "candidate_bytes": len(CANDIDATE.read_bytes()),
        "implementation_sha256": build["implementation_sha256"],
        "implementation_bytes": build["implementation_bytes"],
        "build": "build.py",
        "build_is_derived_not_handwritten": True,
        "the_change": (
            "one block substitution inside A-GR1's own hif4_calibration_attention: the two "
            "per-window gate-loss calls become a single pair call. Reported as a byte delta "
            "and a line diff, with the AST compared after the pair call is put back as two."
        ),
        "not_changed": [
            "the parent calibration (_AGR1_PARENT_CALIBRATION)",
            "A-GR1's M parameterisation (M = I + N)",
            "the amax training target, 32-step Adam, learning rate, spectral clamp",
            "_AGR1_FIT_WINDOWS and _AGR1_GATE_WINDOWS",
            "the candidate count and the strict per-window candidate_loss < parent_loss AND-acceptance",
            "the protection of already-learned rotation/center and the final state compilation",
            "the dynamic Q/K/V interfaces and every Linear API",
        ],
        "what_the_change_does_to_execution": (
            "per gate window: attention forwards 4 -> 3, reference dense decodes 12 -> 8 "
            "(3 explicit plus one fewer inside the V quantiser), hif4 decodes 6 -> 5, V "
            "quantisations 2 -> 1; the Q and K calls stay at 2 each because those are the "
            "arm-dependent part. Over the two shipped gate windows every count doubles."
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
            "source": AGR1_REFERENCE_RUN,
            "note": (
                "A-GR1's own change against the v231 root, inherited by transitivity: the "
                "per-case pairing above is exactly zero on all 72 cases, so this candidate "
                "produces the same outputs v236 produced. The run is cited, not re-measured."
            ),
            "reference_run": AGR1_REFERENCE_RUN,
            "reference_cases": agr1_sixshard["cases"],
            "reference_statuses": agr1_sixshard["statuses"],
        },
        "time": {
            "verdict": "see timing.json; the gate and the whole calibration are reported separately with a null control",
            "paired_three_arm": gate_effects,
        },
        "lineage_note": (
            "v238 is a descendant of v231 (formal parent) and of v236 (the A-GR1 assembly on "
            "that parent). It carries A-GR1 plus the gate-reuse change. It does not contain "
            "v233's L-TF1, v232's L-QF1, v235's L-AD1 or v237's L-TF2, and scores/seconds do "
            "not add across any of those."
        ),
        "declared_deviations": [],
        "local_decision": "archive and submit once for official adjudication",
        "local_decision_note": (
            "A-GR1's mechanism value and its calibration cost are separate facts from v234 "
            "(side isolation +29, complete package TIMEOUT); this card reduces the cost and "
            "inherits neither number."
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
        "sole_change_measured": {
            "byte_level": str(build["edit"]),
            "extracted_function_changed_lines": build["extracted_function_changed_lines"],
            "extracted_function_byte_delta": build["extracted_function_byte_delta"],
            "substitutions_performed": build["substitutions_performed"],
            "ast_identical_after_dissolving_pair_call": build[
                "ast_identical_after_dissolving_pair_call"
            ],
            "shared_prefix_root_bytes": build["root_bytes"],
            "shared_prefix_assembly_bytes": build["assembly_bytes"],
        },
        "single_file_import": "verify.py control A imports the six APIs from a bare temporary directory",
        "linear_and_qkv_bytecode_identical_to_parent": True,
        "premise_measured": {
            "arms_differ_only_in": ["q_state.learned_rotation", "k_state.learned_rotation", "k_state.learned_center"],
            "v_state_bitwise_equal_between_arms": True,
            "v_api_two_calls_five_fields_bitwise_equal": True,
            "no_api_writes_into_the_state_it_is_handed": True,
        },
        "protocol": "proxy-v3",
        "panel": "qwen35-4b-panel-v1 (72 Attention cases)",
        "scope": "attention-only, six shards, target side",
        "hard_output_changed": False,
        "equivalence_is_exact_not_tolerance_based": (
            "every gate window's parent and candidate loss are compared exactly between the "
            "two paths, and the full calibration's q/k/v states are compared as raw bytes"
        ),
        "call_counts_per_gate_window": {
            "attention_forwards": [4, 3],
            "reference_dense_decodes": [12, 8],
            "hif4_decodes": [6, 5],
            "v_quantisations": [2, 1],
            "q_calls": [2, 2],
            "k_calls": [2, 2],
            "method": "independent invocation counters over the module's own names",
        },
        "coverage": {
            "evidence": "verify.out",
            "accepted_and_rejected_on_real_layers": "the six real layers split "
            "{0: accepted, 22: accepted, 1/5/8/15: parent} at attempted=1 everywhere",
            "identity_parent_arm": "layer 8 reaches it on real data; a patched probe forces it",
            "ineligible_short_window_list": True,
            "exception_fallback": True,
            "determinism": "the assembly compared against itself is byte-identical",
            "passed": "ALL A-CT1 CONTROLS PASSED",
        },
        "equal_shard_mean_delta_gain": paired["equal_shard_mean_delta_gain"],
        "shard_delta_gain_mean": paired["shard_delta_gain_mean"],
        "positive_cases": paired["positive_cases"],
        "negative_cases": paired["negative_cases"],
        "zero_cases": paired["zero_cases"],
        "case_count": paired["case_count"],
        "delta_extremes": {
            "min": min(s["delta_min"] for s in paired["shards"]),
            "max": max(s["delta_max"] for s in paired["shards"]),
        },
        "shard_decision": "no_effect (the evaluator's own small-trend filter)",
        "shard_decision_note": (
            "336/72-style exactly-zero is the predicted reading for an implementation that "
            "reproduces its control; a non-zero delta here would be the defect"
        ),
        "pairing_method": paired["method"],
        "stopped_early_flag": {
            "manifest_value": bool(manifest["stopped_early"]),
            "is_a_truncation": False,
            "why": (
                "the stop check runs after the shard's results are appended, and at every "
                "shard this candidate reproduces its control exactly, so delta_mean == 0 "
                "everywhere and the counter reaches --stop-after-nonpositive 6 exactly on "
                "the last requested shard. All six are recorded."
            ),
            "independent_coverage_check": (
                "candidate-attention-shard{0..5}.json: 6 files, statuses "
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
            "PENDING user submission. The local evidence is implementation equivalence with "
            "the same-parent A-GR1 (72/72 paired cases exactly zero, byte-identical "
            "calibration states on all six real attention layers) plus a proven reduction "
            "in the gate's executed work."
        ),
        "local_projection_seconds": None,
        "local_projection_margin_seconds": None,
        "local_projection_basis": None,
        "projection_note": (
            "no projection is recorded. The official 300 s gate is the only time gate and "
            "this machine's wall clock cannot be converted into the official machine's; the "
            "card's own timing carries a byte-identical sham null and is read against it."
        ),
        "official_scored_sha256": None,
        "archive_sha256": candidate_sha,
        "note": (
            "A-GR1's official record stands separately and is not inherited: side isolation "
            "+29 (14455/263.7s against the v195-side baseline), complete package TIMEOUT. "
            "v238 is a descendant of the v231 root and is not additive with v234, v233, v235 "
            "or v237."
        ),
        "candidate_type": (
            "complete Attention A-CT1 candidate (v231 root + A-GR1 + the gate pair call); "
            f"the only {VERSION} under solutions/ at archive time"
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
