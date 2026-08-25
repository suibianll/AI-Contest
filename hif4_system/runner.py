"""Process-isolated evaluation tracks.

The CPU track is authoritative for timing.  The accuracy track can run on a
requested device (including CUDA when installed) but is explicitly marked
non-authoritative for timing so hardware differences cannot silently change a
promotion decision.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch

from .models import CaseResult, TimingResult


@dataclass(frozen=True)
class WorkerRequest:
    candidate: str
    seed: int
    tier: str
    device: str
    compute_dtypes: tuple[str, ...]
    causal_modes: tuple[bool, ...]
    config_path: str | None = None

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate,
            "seed": int(self.seed),
            "tier": self.tier,
            "device": self.device,
            "compute_dtypes": list(self.compute_dtypes),
            "causal_modes": list(self.causal_modes),
            "config_path": self.config_path,
        }


@dataclass(frozen=True)
class WorkerResponse:
    status: str
    cases: tuple[CaseResult, ...] = ()
    timing: TimingResult | None = None
    metadata: dict[str, Any] | None = None
    error: str = ""


@dataclass(frozen=True)
class TrackReport:
    status: str
    device_type: str
    authoritative_timing: bool
    cases: tuple[CaseResult, ...]
    timing: TimingResult
    errors: tuple[str, ...] = ()
    metadata: dict[str, Any] | None = None


def _parse_response(payload: dict[str, Any]) -> WorkerResponse:
    cases = tuple(CaseResult(**item) for item in payload.get("cases", []))
    timing_payload = payload.get("timing")
    timing = TimingResult(**timing_payload) if timing_payload is not None else None
    metadata = payload.get("metadata")
    return WorkerResponse(
        status=str(payload.get("status", "crashed")),
        cases=cases,
        timing=timing,
        metadata=dict(metadata) if isinstance(metadata, dict) else {},
        error=str(payload.get("error", "")),
    )


def _runner_root() -> Path:
    return Path(__file__).resolve().parents[1]


def run_isolated(request: WorkerRequest, timeout_seconds: float) -> WorkerResponse:
    """Run a single seed in a fresh Python process and capture its outcome."""
    root = _runner_root()
    temp_root = root / ".runner_tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    run_dir = temp_root / uuid.uuid4().hex
    run_dir.mkdir()
    request_path = run_dir / "request.json"
    response_path = run_dir / "response.json"
    request_path.write_text(json.dumps(request.to_jsonable(), ensure_ascii=False), encoding="utf-8")
    command = [
        sys.executable,
        "-m",
        "hif4_system.worker",
        "--request",
        str(request_path),
        "--response",
        str(response_path),
    ]
    environment = os.environ.copy()
    environment.setdefault("PYTHONUNBUFFERED", "1")
    process = subprocess.Popen(
        command,
        cwd=str(root),
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stdout = ""
    stderr = ""
    try:
        try:
            stdout, stderr = process.communicate(timeout=float(timeout_seconds))
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
            return WorkerResponse("timeout", error=f"worker timed out after {timeout_seconds:.3f}s")
        try:
            if response_path.is_file():
                payload = json.loads(response_path.read_text(encoding="utf-8"))
                response = _parse_response(payload)
                if process.returncode not in (0, None) and response.status == "passed":
                    return WorkerResponse("crashed", error=f"worker exited with code {process.returncode}")
                if response.status == "crashed" and not response.error:
                    detail = stderr.strip() or stdout.strip()
                    response = WorkerResponse(
                        response.status, response.cases, response.timing, response.metadata, detail
                    )
                return response
            detail = stderr.strip() or stdout.strip() or f"worker exited with code {process.returncode}"
            return WorkerResponse("crashed", error=detail)
        except (OSError, TypeError, ValueError, KeyError) as error:
            detail = stderr.strip() or stdout.strip()
            return WorkerResponse("crashed", error=f"invalid worker response: {error}; {detail}")
    finally:
        for path in run_dir.iterdir():
            path.unlink(missing_ok=True)
        run_dir.rmdir()


def _relabel_cases(cases: Sequence[CaseResult], seed_index: int) -> tuple[CaseResult, ...]:
    return tuple(
        CaseResult(
            seed_id=f"seed-{seed_index:02d}",
            kind=row.kind,
            scenario=row.scenario,
            test_index=row.test_index,
            causal=row.causal,
            compute_dtype=row.compute_dtype,
            mse_std=row.mse_std,
            mse_player=row.mse_player,
            score=row.score,
        )
        for row in cases
    )


def _run_track(
    candidate: Path,
    seeds: Sequence[int],
    tier: str,
    device: str,
    compute_dtypes: Sequence[str],
    causal_modes: Sequence[bool],
    timeout_seconds: float,
    authoritative_timing: bool,
    config_path: Path | None = None,
) -> TrackReport:
    torch_device = torch.device(device)
    all_cases: list[CaseResult] = []
    errors: list[str] = []
    player_seconds = 0.0
    wall_seconds = 0.0
    metadata: dict[str, Any] = {"candidate": str(candidate.resolve()), "seed_count": len(seeds)}
    for seed_index, seed in enumerate(seeds):
        request = WorkerRequest(
            candidate=str(candidate.resolve()),
            seed=int(seed),
            tier=tier,
            device=str(torch_device),
            compute_dtypes=tuple(compute_dtypes),
            causal_modes=tuple(bool(value) for value in causal_modes),
            config_path=str(config_path.resolve()) if config_path is not None else None,
        )
        response = run_isolated(request, timeout_seconds)
        if response.status != "passed":
            errors.append(f"seed {seed}: {response.status}: {response.error}".strip())
            continue
        all_cases.extend(_relabel_cases(response.cases, seed_index))
        if response.timing is not None:
            player_seconds += response.timing.player_quant_seconds
            wall_seconds += response.timing.wall_seconds
        if response.metadata:
            metadata[f"seed-{seed_index:02d}"] = response.metadata
    status = "passed" if not errors else "failed"
    return TrackReport(
        status=status,
        device_type=torch_device.type,
        authoritative_timing=authoritative_timing,
        cases=tuple(all_cases),
        timing=TimingResult(player_seconds, wall_seconds),
        errors=tuple(errors),
        metadata=metadata,
    )


def run_cpu_track(
    candidate: Path,
    seeds: Sequence[int],
    tier: str,
    compute_dtypes: Sequence[str],
    causal_modes: Sequence[bool],
    timeout_seconds: float,
    config_path: Path | None = None,
) -> TrackReport:
    return _run_track(
        Path(candidate), seeds, tier, "cpu", compute_dtypes, causal_modes, timeout_seconds, True, config_path
    )


def run_accuracy_track(
    candidate: Path,
    seeds: Sequence[int],
    tier: str,
    device: str,
    compute_dtypes: Sequence[str],
    causal_modes: Sequence[bool],
    timeout_seconds: float,
    config_path: Path | None = None,
) -> TrackReport:
    return _run_track(
        Path(candidate), seeds, tier, device, compute_dtypes, causal_modes, timeout_seconds, False, config_path
    )


