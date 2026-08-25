from __future__ import annotations

from pathlib import Path

from hif4_system.runner import WorkerRequest, run_accuracy_track, run_cpu_track, run_isolated


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"


def _request(path: Path) -> WorkerRequest:
    return WorkerRequest(
        candidate=str(path.resolve()),
        seed=101,
        tier="smoke",
        device="cpu",
        compute_dtypes=("fp32",),
        causal_modes=(False, True),
    )


def test_worker_captures_candidate_crash() -> None:
    response = run_isolated(_request(FIXTURES / "crashing_solution.py"), timeout_seconds=10)

    assert response.status == "crashed"
    assert "fixture crash" in response.error


def test_worker_captures_timeout() -> None:
    response = run_isolated(_request(FIXTURES / "hanging_solution.py"), timeout_seconds=0.5)

    assert response.status == "timeout"
    temporary_root = ROOT / ".runner_tmp"
    assert not (list(temporary_root.iterdir()) if temporary_root.exists() else [])


def test_cpu_track_is_authoritative() -> None:
    report = run_cpu_track(FIXTURES / "minimal_solution.py", [101], "smoke", ("fp32",), (False, True), 30)

    assert report.status == "passed"
    assert report.device_type == "cpu"
    assert report.authoritative_timing is True
    assert report.cases


def test_accuracy_track_is_not_authoritative_when_run_on_cpu() -> None:
    report = run_accuracy_track(FIXTURES / "minimal_solution.py", [101], "smoke", "cpu", ("fp32",), (False,), 30)

    assert report.status == "passed"
    assert report.authoritative_timing is False


