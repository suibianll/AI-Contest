"""Compatibility entry point for the Torch-only evaluator.

New automation should call ``cli.py`` directly.  A small legacy adapter keeps
the old ``--backend torch`` spelling useful while rejecting unsupported
backends instead of silently selecting a simulation.
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from hif4_system.cli import main as lifecycle_main


def main(argv: Sequence[str] | None = None) -> int:
    values = list(argv) if argv is not None else sys.argv[1:]
    if "--backend" in values:
        index = values.index("--backend")
        try:
            backend = values[index + 1]
        except IndexError:
            print("--backend requires torch", file=sys.stderr)
            return 2
        if backend != "torch":
            print("only the torch backend is supported", file=sys.stderr)
            return 2
        del values[index : index + 2]
    if values and values[0] in {"init", "evaluate", "validate", "promote", "history", "rollback"}:
        return lifecycle_main(values)
    parser = argparse.ArgumentParser(prog="hif4_generalization_eval.py")
    parser.add_argument("--candidate", "--solution", dest="solution", required=True)
    parser.add_argument("--tier", choices=("smoke", "standard", "soak"), default="smoke")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--split", choices=("dev", "holdout"), default="dev")
    parser.add_argument("--root", default=".")
    parser.add_argument("--campaign-dir")
    parser.add_argument("--incumbent")
    parser.add_argument("--output")
    parser.add_argument("--include-cases", action="store_true")
    parser.add_argument("--attention-mask", choices=("both", "causal", "noncausal"), default="both")
    parser.add_argument("--compute-dtypes", default="fp32")
    parser.add_argument("--max-holdout-uses", type=int, default=3)
    parser.add_argument("--config")
    args = parser.parse_args(values)
    return lifecycle_main(
        [
            "evaluate",
            args.solution,
            "--tier",
            args.tier,
            "--device",
            args.device,
            "--split",
            args.split,
            "--root",
            args.root,
        ] + ([ "--config", args.config ] if args.config else [])
    )


if __name__ == "__main__":
    raise SystemExit(main())


