"""Stable command-line entrypoint for the replacement eval-v3 system."""

from __future__ import annotations

try:
    from eval_system import build_parser, run
except ModuleNotFoundError as exc:  # pragma: no cover - package import path
    if exc.name != "eval_system":
        raise
    from .eval_system import build_parser, run


if __name__ == "__main__":
    result = run(build_parser().parse_args())
    print(
        {
            "protocol": result.get("protocol"),
            "records": len(result.get("records", result.get("results", []))),
            "reasonableness_issues": len(result.get("reasonableness", {}).get("issues", [])),
        }
    )
