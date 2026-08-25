from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_V9_SHA256 = "a6b8b858156164333d1d3ca25c6233b4845061f40a16d4cf74695ecdbb9041f7"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_cli(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "cli.py", *args, "--root", str(root)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_v9_cpu_smoke_report_is_auditable(tmp_path: Path) -> None:
    initialized = _run_cli(tmp_path, "init", "--champion", str(ROOT / "solution.py"))
    assert initialized.returncode == 0, initialized.stderr

    evaluated = _run_cli(tmp_path, "evaluate", str(ROOT / "solution.py"), "--tier", "smoke", "--device", "cpu")
    assert evaluated.returncode == 0, evaluated.stderr
    reports = sorted((tmp_path / "reports").glob("*.json"))
    assert reports
    report = json.loads(reports[-1].read_text(encoding="utf-8"))

    assert report["metadata"]["candidate_sha256"] == EXPECTED_V9_SHA256
    assert report["metadata"]["device_type"] == "cpu"
    assert report["metadata"]["authoritative_timing"] is True
    assert report["summary"]["case_count"] > 0
    assert report["seed_commitment"]
    assert _sha256(next((tmp_path / "registry" / "versions").glob("*/solution.py"))) == EXPECTED_V9_SHA256


def test_holdout_report_does_not_write_raw_seeds(tmp_path: Path) -> None:
    evaluated = _run_cli(tmp_path, "evaluate", str(ROOT / "solution.py"), "--tier", "smoke", "--device", "cpu", "--split", "holdout")
    assert evaluated.returncode == 0, evaluated.stderr
    campaign = json.loads((tmp_path / "campaigns" / "default" / "campaign.json").read_text(encoding="utf-8"))
    assert campaign["holdout_uses"] == 1
    assert all("seeds" not in run and "holdout_seeds" not in run for run in campaign["runs"])
    report_text = " ".join(path.read_text(encoding="utf-8") for path in (tmp_path / "reports").glob("*.json"))
    assert "seed_commitment" in report_text
