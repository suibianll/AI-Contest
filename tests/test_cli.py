from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_V9_SHA256 = "a6b8b858156164333d1d3ca25c6233b4845061f40a16d4cf74695ecdbb9041f7"


def _cli_runner(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "cli.py", *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_init_cli_freezes_v9(tmp_path: Path) -> None:
    result = _cli_runner("init", "--champion", str(ROOT / "solution_v9_champion.py"), "--root", str(tmp_path))

    assert result.returncode == 0, result.stderr
    assert EXPECTED_V9_SHA256 in result.stdout
    pointer = json.loads((tmp_path / "registry" / "champion.json").read_text(encoding="utf-8"))
    assert pointer["sha256"] == EXPECTED_V9_SHA256


def test_init_cli_rejects_wrong_hash_with_stable_code(tmp_path: Path) -> None:
    result = _cli_runner(
        "init",
        "--champion",
        str(ROOT / "solution_v9_champion.py"),
        "--expected-sha256",
        "0" * 64,
        "--root",
        str(tmp_path),
    )

    assert result.returncode == 2
    assert "hash mismatch" in (result.stdout + result.stderr).lower()


def test_legacy_wrapper_rejects_non_torch_backend() -> None:
    result = subprocess.run(
        [sys.executable, "hif4_generalization_eval.py", "--backend", "numpy"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "torch" in (result.stdout + result.stderr).lower()


