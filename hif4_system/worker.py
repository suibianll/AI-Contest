"""Subprocess entry point for one isolated candidate evaluation.

The worker deliberately communicates only JSON-compatible result models.  No
Torch tensors or imported candidate modules cross the process boundary, which
keeps a crashed candidate (or a candidate that leaks state) from poisoning the
parent evaluator.
"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any, Mapping

import torch

from .config import load_config
from .evaluator import evaluate_solution
from .models import to_jsonable
from .solution_loader import load_solution
from .suites import build_suite


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _evaluate(request: Mapping[str, Any]) -> dict[str, Any]:
    config_value = request.get("config_path")
    config = load_config(Path(str(config_value)) if config_value else None)
    tier_name = str(request["tier"])
    tier = config.tier(tier_name)
    device = torch.device(str(request["device"]))
    suite = build_suite(int(request["seed"]), tier, device)
    api = load_solution(Path(str(request["candidate"])))
    result = evaluate_solution(
        api,
        suite,
        device,
        tuple(str(value) for value in request.get("compute_dtypes", ["fp32"])),
        tuple(bool(value) for value in request.get("causal_modes", [False, True])),
    )
    return {
        "status": "passed",
        "cases": to_jsonable(result.cases),
        "timing": to_jsonable(result.timing),
        "metadata": to_jsonable(result.metadata),
        "error": "",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one isolated HiF4 candidate evaluation")
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--response", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        request = json.loads(args.request.read_text(encoding="utf-8"))
        response = _evaluate(request)
    except BaseException as error:  # candidate code is untrusted; always report a response
        response = {
            "status": "crashed",
            "cases": [],
            "timing": None,
            "metadata": {},
            "error": f"{type(error).__name__}: {error}\n{traceback.format_exc()}",
        }
    _write_json_atomic(args.response, response)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through subprocesses
    raise SystemExit(main())
