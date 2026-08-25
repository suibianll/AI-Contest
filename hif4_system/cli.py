"""Command-line lifecycle for the Torch-only HiF4 evaluation system."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .campaign import Campaign, HoldoutBudgetExhausted, SeedReservation
from .compliance import check_static
from .config import ConfigError, EvaluationConfig, load_config
from .models import CaseResult, TimingResult
from .registry import HashMismatch, PromotionRejected, Registry, RegistryError
from .runner import TrackReport, run_accuracy_track, run_cpu_track
from .solution_loader import sha256_file
from .statistics import Comparison, Decision, Summary, compare, decide, summarize


EXIT_OK = 0
EXIT_ARGUMENT = 2
EXIT_COMPLIANCE = 3
EXIT_EVALUATION = 4
EXIT_PROMOTION = 5


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _config_hash(config_path: Path | None = None) -> str:
    source = config_path or (Path(__file__).resolve().parents[1] / "config" / "default.json")
    return _sha256_bytes(source.resolve().read_bytes())


def _environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _campaign_for(root: Path) -> Campaign:
    directory = root / "campaigns" / "default"
    if (directory / "campaign.json").exists() or (directory / ".holdout_secret").exists():
        return Campaign.open(directory)
    return Campaign.create(directory)


def _seed_name(split: str, device: str) -> str:
    return f"{split}_{device}"


def _summary_or_none(cases: Sequence[CaseResult], config: EvaluationConfig, seed: int) -> Summary | None:
    return summarize(cases, config.bootstrap_rounds, seed) if cases else None


def _decision_for_track(
    track: TrackReport,
    config: EvaluationConfig,
    summary: Summary | None,
    incumbent_summary: Summary | None,
    comparison: Comparison | None,
    incumbent_timing: TimingResult | None,
) -> Decision:
    if track.status != "passed" or summary is None:
        return Decision(False, {"track_passed": False})
    if incumbent_summary is None:
        return Decision(True, {"track_passed": True, "incumbent_available": False})
    return decide(
        summary,
        incumbent_summary,
        comparison,
        track.timing,
        incumbent_timing,
        config.thresholds,
        track.authoritative_timing,
    )


def _track_payload(
    *,
    candidate_sha256: str,
    split: str,
    device: str,
    tier: str,
    commitment: str,
    track: TrackReport,
    config: EvaluationConfig,
    comparison: Comparison | None = None,
    decision: Decision | None = None,
) -> dict[str, Any]:
    summary = _summary_or_none(track.cases, config, 20260825)
    if decision is None:
        decision = _decision_for_track(track, config, summary, None, comparison, None)
    return {
        "status": track.status,
        "candidate_sha256": candidate_sha256,
        "split": split,
        "tier": tier,
        "device_type": device,
        "authoritative_timing": track.authoritative_timing,
        "seed_commitment": commitment,
        "cases": [
            {
                "seed_id": row.seed_id,
                "kind": row.kind,
                "scenario": row.scenario,
                "test_index": row.test_index,
                "causal": row.causal,
                "compute_dtype": row.compute_dtype,
                "mse_std": row.mse_std,
                "mse_player": row.mse_player,
                "score": row.score,
            }
            for row in track.cases
        ],
        "summary": summary.to_dict() if summary is not None else {"case_count": 0},
        "comparison": comparison.to_dict() if comparison is not None else None,
        "decision": decision.to_dict(),
        "timing": {
            "player_quant_seconds": track.timing.player_quant_seconds,
            "wall_seconds": track.timing.wall_seconds,
            "peak_rss_bytes": track.timing.peak_rss_bytes,
        },
        "errors": list(track.errors),
        "metadata": {
            "candidate_sha256": candidate_sha256,
            "device_type": device,
            "authoritative_timing": track.authoritative_timing,
            "environment": _environment(),
            "runner": track.metadata or {},
            "config_sha256": _config_hash(),
        },
    }


def _run_track(
    candidate: Path,
    seeds: Sequence[int],
    tier: str,
    device: str,
    config: EvaluationConfig,
) -> TrackReport:
    timeout = config.timeouts[tier]
    dtypes = ("fp32",)
    causal = (False, True)
    if device == "cpu":
        return run_cpu_track(candidate, seeds, tier, dtypes, causal, timeout)
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
        return run_accuracy_track(candidate, seeds, tier, "cuda", dtypes, causal, timeout)
    raise ConfigError(f"unsupported device: {device}")


def _write_report(root: Path, label: str, payload: Mapping[str, Any]) -> Path:
    sha = str(payload.get("candidate_sha256", "unknown"))[:12]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = root / "reports" / f"{stamp}-{label}-{sha}.json"
    _atomic_json(path, payload)
    return path.resolve()


def _evaluate_once(
    candidate: Path,
    root: Path,
    tier: str,
    device: str,
    split: str,
    config: EvaluationConfig,
) -> tuple[int, Path | None, dict[str, Any], SeedReservation | None]:
    source = candidate.resolve()
    candidate_sha = sha256_file(source)
    campaign = _campaign_for(root)
    reservation: SeedReservation | None = None
    report: dict[str, Any]
    status = "failed"
    report_path: Path | None = None
    try:
        reservation = campaign.reserve(split, tier)
        compliance = check_static(source)
        if not compliance.passed:
            report = {
                "status": "rejected",
                "candidate_sha256": candidate_sha,
                "split": split,
                "tier": tier,
                "device_type": device,
                "authoritative_timing": device == "cpu",
                "seed_commitment": reservation.commitment,
                "violations": list(compliance.violations),
                "warnings": list(compliance.warnings),
            }
            status = "rejected"
            report_path = _write_report(root, _seed_name(split, device), report)
            return EXIT_COMPLIANCE, report_path, report, reservation
        try:
            track = _run_track(source, reservation.seeds, tier, device, config)
            report = _track_payload(
                candidate_sha256=candidate_sha,
                split=split,
                device=device,
                tier=tier,
                commitment=reservation.commitment,
                track=track,
                config=config,
            )
            status = track.status
            report_path = _write_report(root, _seed_name(split, device), report)
            return (EXIT_OK if track.status == "passed" else EXIT_EVALUATION), report_path, report, reservation
        except Exception as error:
            report = {
                "status": "crashed",
                "candidate_sha256": candidate_sha,
                "split": split,
                "tier": tier,
                "device_type": device,
                "authoritative_timing": device == "cpu",
                "seed_commitment": reservation.commitment,
                "errors": [f"{type(error).__name__}: {error}"],
            }
            status = "crashed"
            report_path = _write_report(root, _seed_name(split, device), report)
            return EXIT_EVALUATION, report_path, report, reservation
    finally:
        if reservation is not None:
            campaign.finish(reservation, status=status, report=str(report_path) if report_path else None)


def _print_report(code: int, path: Path | None, payload: Mapping[str, Any]) -> int:
    print(json.dumps({"status": payload.get("status"), "report": str(path) if path else None}, ensure_ascii=False))
    if path is not None:
        print(path)
    return code


def _cmd_init(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    registry = Registry(root / "registry")
    record = registry.initialize(Path(args.champion), args.expected_sha256 or sha256_file(Path(args.champion).resolve()))
    print(f"Champion initialized: {record.sha256}")
    print(record.solution_path.resolve())
    return EXIT_OK


def _cmd_evaluate(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    config = load_config(Path(args.config) if args.config else None)
    code, path, payload, _ = _evaluate_once(Path(args.candidate), root, args.tier, args.device, args.split, config)
    return _print_report(code, path, payload)


def _cmd_validate(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    config = load_config(Path(args.config) if args.config else None)
    source = Path(args.candidate).resolve()
    registry = Registry(root / "registry")
    champion = registry.champion()
    candidate_sha = sha256_file(source)
    campaign = _campaign_for(root)
    all_reports: dict[str, Any] = {}
    failures: list[str] = []

    dev_reservation = campaign.reserve("dev", args.tier)
    dev_status = "failed"
    try:
        try:
            gpu_track = _run_track(source, dev_reservation.seeds, args.tier, "cuda", config)
            gpu_payload = _track_payload(
                candidate_sha256=candidate_sha,
                split="dev",
                device="cuda",
                tier=args.tier,
                commitment=dev_reservation.commitment,
                track=gpu_track,
                config=config,
            )
            all_reports["gpu_dev"] = gpu_payload
            if gpu_track.status != "passed":
                failures.append("gpu dev evaluation failed")
        except Exception as error:
            all_reports["gpu_dev"] = {"status": "crashed", "candidate_sha256": candidate_sha, "error": str(error)}
            failures.append(f"gpu dev: {error}")

        try:
            cpu_track = _run_track(source, dev_reservation.seeds, args.tier, "cpu", config)
            incumbent_track = _run_track(champion.solution_path, dev_reservation.seeds, args.tier, "cpu", config)
            candidate_summary = _summary_or_none(cpu_track.cases, config, 20260826)
            incumbent_summary = _summary_or_none(incumbent_track.cases, config, 20260826)
            comparison = compare(cpu_track.cases, incumbent_track.cases, config.bootstrap_rounds, 20260827) if candidate_summary and incumbent_summary else None
            decision = _decision_for_track(cpu_track, config, candidate_summary, incumbent_summary, comparison, incumbent_track.timing)
            cpu_payload = _track_payload(
                candidate_sha256=candidate_sha,
                split="dev",
                device="cpu",
                tier=args.tier,
                commitment=dev_reservation.commitment,
                track=cpu_track,
                config=config,
                comparison=comparison,
                decision=decision,
            )
            all_reports["cpu_dev"] = cpu_payload
            if not decision.promote:
                failures.append("cpu dev promotion gates failed")
        except Exception as error:
            all_reports["cpu_dev"] = {"status": "crashed", "candidate_sha256": candidate_sha, "error": str(error)}
            failures.append(f"cpu dev: {error}")
        dev_status = "passed" if not failures else "failed"
    finally:
        campaign.finish(dev_reservation, dev_status, None)

    if failures and "gpu dev" in " ".join(failures):
        combined = {"status": "failed", "candidate_sha256": candidate_sha, "reports": all_reports, "errors": failures}
        path = _write_report(root, "validate", combined)
        return _print_report(EXIT_EVALUATION, path, combined)

    holdout_reservation = campaign.reserve("holdout", args.tier)
    holdout_status = "failed"
    try:
        try:
            holdout_track = _run_track(source, holdout_reservation.seeds, args.tier, "cpu", config)
            incumbent_holdout = _run_track(champion.solution_path, holdout_reservation.seeds, args.tier, "cpu", config)
            candidate_summary = _summary_or_none(holdout_track.cases, config, 20260828)
            incumbent_summary = _summary_or_none(incumbent_holdout.cases, config, 20260828)
            comparison = compare(holdout_track.cases, incumbent_holdout.cases, config.bootstrap_rounds, 20260829) if candidate_summary and incumbent_summary else None
            decision = _decision_for_track(holdout_track, config, candidate_summary, incumbent_summary, comparison, incumbent_holdout.timing)
            all_reports["holdout"] = _track_payload(
                candidate_sha256=candidate_sha,
                split="holdout",
                device="cpu",
                tier=args.tier,
                commitment=holdout_reservation.commitment,
                track=holdout_track,
                config=config,
                comparison=comparison,
                decision=decision,
            )
            if not decision.promote:
                failures.append("holdout promotion gates failed")
            holdout_status = "passed" if holdout_track.status == "passed" and decision.promote else "failed"
        except Exception as error:
            all_reports["holdout"] = {"status": "crashed", "candidate_sha256": candidate_sha, "error": str(error)}
            failures.append(f"holdout: {error}")
    finally:
        campaign.finish(holdout_reservation, holdout_status, None)

    combined = {
        "status": "passed" if not failures else "failed",
        "candidate_sha256": candidate_sha,
        "reports": all_reports,
        "errors": failures,
        "environment": _environment(),
        "config_sha256": _config_hash(),
    }
    path = _write_report(root, "validate", combined)
    if not failures:
        candidate = registry.register_candidate(source, all_reports)
        combined["candidate_id"] = candidate.id
        _atomic_json(path, combined)
        print(f"Candidate registered: {candidate.id}")
        return _print_report(EXIT_OK, path, combined)
    return _print_report(EXIT_PROMOTION, path, combined)


def _cmd_promote(args: argparse.Namespace) -> int:
    registry = Registry(Path(args.root).resolve() / "registry")
    candidate = next((item for item in registry.pending() if item.id == args.candidate_id), None)
    if candidate is None:
        print(f"unknown pending candidate: {args.candidate_id}", file=sys.stderr)
        return EXIT_ARGUMENT
    try:
        record = registry.promote(candidate.id, args.sha256 or candidate.sha256)
    except (HashMismatch, PromotionRejected, RegistryError) as error:
        print(str(error), file=sys.stderr)
        return EXIT_PROMOTION
    print(f"Champion promoted: {record.id} {record.sha256}")
    return EXIT_OK


def _cmd_history(args: argparse.Namespace) -> int:
    registry = Registry(Path(args.root).resolve() / "registry")
    print(json.dumps(registry.history(), ensure_ascii=False, indent=2))
    return EXIT_OK


def _cmd_rollback(args: argparse.Namespace) -> int:
    registry = Registry(Path(args.root).resolve() / "registry")
    try:
        record = registry.rollback(args.version_id)
    except RegistryError as error:
        print(str(error), file=sys.stderr)
        return EXIT_ARGUMENT
    print(f"Champion rolled back: {record.id} {record.sha256}")
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hif4-eval")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init")
    init.add_argument("--champion", required=True)
    init.add_argument("--expected-sha256")
    init.add_argument("--root", default=".")
    init.set_defaults(handler=_cmd_init)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("candidate")
    evaluate.add_argument("--tier", choices=("smoke", "standard", "soak"), default="smoke")
    evaluate.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    evaluate.add_argument("--split", choices=("dev", "holdout"), default="dev")
    evaluate.add_argument("--config")
    evaluate.add_argument("--root", default=".")
    evaluate.set_defaults(handler=_cmd_evaluate)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--candidate", required=True)
    validate.add_argument("--tier", choices=("smoke", "standard", "soak"), default="standard")
    validate.add_argument("--config")
    validate.add_argument("--root", default=".")
    validate.set_defaults(handler=_cmd_validate)

    promote = subparsers.add_parser("promote")
    promote.add_argument("--candidate-id", required=True)
    promote.add_argument("--sha256")
    promote.add_argument("--root", default=".")
    promote.set_defaults(handler=_cmd_promote)

    history = subparsers.add_parser("history")
    history.add_argument("--root", default=".")
    history.set_defaults(handler=_cmd_history)

    rollback = subparsers.add_parser("rollback")
    rollback.add_argument("--version-id")
    rollback.add_argument("--root", default=".")
    rollback.set_defaults(handler=_cmd_rollback)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
        return int(args.handler(args))
    except (ConfigError, FileNotFoundError, HoldoutBudgetExhausted, ValueError, OSError) as error:
        print(str(error), file=sys.stderr)
        return EXIT_ARGUMENT



