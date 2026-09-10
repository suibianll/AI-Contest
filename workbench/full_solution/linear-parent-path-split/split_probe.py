"""One probe: split the parent activation path into dense / block-order / GPTQ solve.

The L-TF1 work priced a whole dynamic call and found that the parent's own
dynamic activation path -- everything that runs with the ``em1`` payload removed
-- is **90-93%** of it, against 7-10% for the L-EM2 descent and 3.2-3.5% for the
metric build (``linear-tf1-gradient-reuse/attribute.py``).

That 90-93% is a single bucket and naming it is not a localization.  The plan's
section 4 branch 2 ("precompute the original block-to-block compensation
coefficients and have the dynamic stage consume them by the block order the
current input produces") rests on the assumption that the **GPTQ block solve**
dominates.  Nothing measured so far proves that, and taking the bucket for the
solve would be treating a guess as a localization.  So this script asks exactly
one question and stops: inside that bucket, how much is the dense construction
(nvfp4 decode plus the smooth and permutation multiplies), how much is the
block-order selection, and how much is the reordered GPTQ solve?  What is left
is the hook, the state copy, the fallback path and CPU gaps.

Method: the parent's helpers are resolved from module globals at call time, so
replacing the module attribute measures the real shipped path without editing a
byte of it.  The wrappers are pass-through -- they call the original and return
its result unchanged; a wrapper that did anything else would be measuring a
different program.  Each segment is bracketed by a pair of ``torch.cuda.Event``s
and a ``perf_counter`` pair, and the events are read only after the round's
single sync, so no segment timing perturbs another.

Both clocks are reported because they answer different questions.  Event time is
GPU-stream time; perf_counter is wall time including CPU-side Python.  A segment
large on the wall clock and small on the event clock is CPU-bound, and that
distinction decides whether branch 2 -- a GPU-side recomputation -- can reach it
at all.

Three parts, in the order the parent runs them:

  dense       ``_static_actorder_dense_from_state`` -- nvfp4 decode plus the
              smooth and permutation multiplies
  order       ``_combined_dynamic_sample_energy_block_order_fast`` -- which
              block order this input selects
  gptq        ``_combined_dynamic_fast_reordered_gptq`` -- the sequential
              block solve in that order

This is the one probe the plan allows.  No parameter, seed, threshold or
granularity sweep, and no second probe: it either selects branch 2 or it does
not, and the number decides.

Read-only with respect to the repo: writes ``split_probe.json`` next to itself.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CACHE_DIR = ROOT / "artifacts/official_eval/cache/proxy-v3-calibration"
PACK = ROOT / "artifacts/official_eval/cache/qwen3.5-4b-proxy-v2.pt"

PARENT_SHA_PREFIX = "0f1af6dbc207ff32"

# (layer, role) targets.  q is 2560 channels and o is 4096; both are inside the
# metric's scope, so both reach the descent and both are real shapes the
# official run pays for.  proj is 9216 and never enters the descent, so it would
# measure a path the plan cannot act on.
TARGETS = ((0, "q"), (0, "o"))
ROW_COUNTS = (128, 512)

PARTS = ("dense", "order", "gptq")
SYMBOLS = {
    "dense": "_static_actorder_dense_from_state",
    "order": "_combined_dynamic_sample_energy_block_order_fast",
    "gptq": "_combined_dynamic_fast_reordered_gptq",
}

# The first pass of this probe found the GPTQ call dominant (80-95%), but that
# call contains two very different things: the per-block encoder loop
# (_dense_to_hif4, run once per 64-channel block) and the sequential
# compensation solve (torch.linalg.solve plus its matmul).  Branch 2 of the plan
# only pays if the *compensation* is the cost; if the encoder is, the answer is
# branch 3 instead.  So the same probe brackets these two as well.  This is a
# refinement of the one split, not a second probe.
INNER_SYMBOLS = {
    "encoder": "_dense_to_hif4",
}
INNER_TORCH = {
    "solve": "linalg.solve",
}
INNER_PARTS = ("encoder", "solve")

ROUNDS = 15
WARMUP = 3


def load_solution(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def find_state(layer: int, role: str) -> dict:
    for path in sorted(CACHE_DIR.glob(f"{PARENT_SHA_PREFIX}-linear-*.pt")):
        payload = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
        for entry in payload["weight_states"]:
            if int(entry["layer"]) == layer and str(entry["role"]) == role:
                del payload
                return dict(entry["state"])
    raise SystemExit(f"no cached state for layer {layer} / role {role}")


class SegmentRecorder:
    """Brackets every call of the wrapped helpers with both clocks.

    Events are recorded inside the round and resolved afterwards, once the round
    has been synchronised -- an ``elapsed_time`` read on an event that has not
    completed is not a measurement.
    """

    def __init__(self, module, symbols, inner_symbols=None, inner_torch=None):
        self.module = module
        self.symbols = symbols
        self.inner_symbols = inner_symbols or {}
        self.inner_torch = inner_torch or {}
        self.originals: dict[str, object] = {}
        self.torch_originals: dict[str, object] = {}
        self.pending: list[dict] = []
        self.active = False

    def __enter__(self):
        for part, symbol in self.symbols.items():
            original = getattr(self.module, symbol, None)
            if original is None or not callable(original):
                raise SystemExit(f"module has no callable {symbol}")
            self.originals[part] = original
            setattr(self.module, symbol, self._wrap(part, original))
        for part, symbol in self.inner_symbols.items():
            original = getattr(self.module, symbol, None)
            if original is None or not callable(original):
                raise SystemExit(f"module has no callable {symbol}")
            self.originals[part] = original
            setattr(self.module, symbol, self._wrap(part, original))
        for part, dotted in self.inner_torch.items():
            holder = torch
            *path, leaf = dotted.split(".")
            for step in path:
                holder = getattr(holder, step)
            original = getattr(holder, leaf, None)
            if original is None or not callable(original):
                raise SystemExit(f"torch has no callable {dotted}")
            self.torch_originals[part] = (holder, leaf, original)
            setattr(holder, leaf, self._wrap(part, original))
        self.active = True
        return self

    def _wrap(self, part, original):
        recorder = self

        def wrapper(*args, **kwargs):
            if not recorder.active:
                return original(*args, **kwargs)
            begin_event = torch.cuda.Event(enable_timing=True)
            finish_event = torch.cuda.Event(enable_timing=True)
            begin_event.record()
            begin = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                recorder.pending.append(
                    {
                        "part": part,
                        "wall_ms": (time.perf_counter() - begin) * 1000.0,
                        "begin_event": begin_event,
                        "finish_event": finish_event,
                    }
                )
                finish_event.record()

        return wrapper

    def __exit__(self, *exc_info):
        self.active = False
        for part, original in self.originals.items():
            symbol = self.symbols.get(part) or self.inner_symbols[part]
            setattr(self.module, symbol, original)
        for part, (holder, leaf, original) in self.torch_originals.items():
            setattr(holder, leaf, original)
        return False

    def take(self):
        """Resolves the round's segments; call only after a device sync."""
        segments = self.pending
        self.pending = []
        for segment in segments:
            segment["gpu_ms"] = segment["begin_event"].elapsed_time(
                segment["finish_event"]
            )
        return segments


def measure(module, quant, scale, state, rounds):
    """Times the whole call and, within each call, its three named parts."""

    def call():
        return module.hif4_dynamic_quantize_activation(quant, scale, dict(state))

    for _ in range(WARMUP):
        call()
    torch.cuda.synchronize()

    tracked = list(PARTS) + list(INNER_PARTS)
    totals = []
    per_part_gpu: dict[str, list[float]] = {part: [] for part in tracked}
    per_part_wall: dict[str, list[float]] = {part: [] for part in tracked}
    per_part_calls: dict[str, list[int]] = {part: [] for part in tracked}
    per_round_inner: dict[str, list[float]] = {part: [] for part in INNER_PARTS}

    with SegmentRecorder(
        module, SYMBOLS, INNER_SYMBOLS, INNER_TORCH
    ) as recorder:
        for _ in range(rounds):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            call()
            end.record()
            torch.cuda.synchronize()
            totals.append(start.elapsed_time(end))

            counts = {part: 0 for part in tracked}
            round_inner = {part: 0.0 for part in INNER_PARTS}
            for segment in recorder.take():
                part = segment["part"]
                counts[part] += 1
                per_part_gpu[part].append(segment["gpu_ms"])
                per_part_wall[part].append(segment["wall_ms"])
                if part in round_inner:
                    round_inner[part] += segment["gpu_ms"]
            for part in tracked:
                per_part_calls[part].append(counts[part])
            for part in INNER_PARTS:
                per_round_inner[part].append(round_inner[part])

    def summary(values):
        if not values:
            return None
        ordered = sorted(values)
        return {
            "median_ms": statistics.median(ordered),
            "min_ms": ordered[0],
            "max_ms": ordered[-1],
        }

    return {
        "rounds": rounds,
        "full_ms": statistics.median(totals),
        "full_unordered_ms": {
            "min": min(totals),
            "max": max(totals),
            "stdev": statistics.stdev(totals),
        },
        "calls_per_round": {
            part: statistics.median(per_part_calls[part]) for part in tracked
        },
        "part_gpu": {part: summary(per_part_gpu[part]) for part in tracked},
        "part_wall": {part: summary(per_part_wall[part]) for part in tracked},
        "encoder_calls_per_round": statistics.median(per_part_calls["encoder"]),
        # The inner parts are summed within each round before the median is
        # taken, because they fire many times per call and the per-call total is
        # the quantity the branch decision needs.
        "inner_gpu_ms": {
            part: statistics.median(per_round_inner[part]) for part in INNER_PARTS
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rounds", type=int, default=ROUNDS)
    parser.add_argument("--output", default=str(HERE / "split_probe.json"))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    device = torch.device("cuda")

    sys.path.insert(0, str(ROOT / "evaluator"))
    import official_eval as evaluator  # noqa: PLC0415

    parent = load_solution(ROOT / "solution.py", "split_parent")
    for symbol in SYMBOLS.values():
        if not callable(getattr(parent, symbol, None)):
            raise SystemExit(f"the shipped root has no callable {symbol}")

    pack = torch.load(PACK, map_location="cpu", mmap=True, weights_only=False)
    raw = pack["test_activations"]

    report = {
        "device": torch.cuda.get_device_name(0),
        "parent_sha256_prefix": PARENT_SHA_PREFIX,
        "rounds": args.rounds,
        "purpose": "one split of the 90-93% bucket; not a timing claim about any candidate",
        "parts": list(PARTS),
        "states": [],
    }

    for layer, role in TARGETS:
        state = find_state(layer, role)
        base_quant, base_scale = evaluator._pair(raw[role][1][layer].to(torch.float32))
        for rows in ROW_COUNTS:
            repeat = -(-rows // base_quant.shape[0])
            quant = base_quant.repeat(repeat, 1)[:rows].contiguous().to(device)
            scale = base_scale.repeat(repeat, 1)[:rows].contiguous().to(device)

            record = measure(parent, quant, scale, state, args.rounds)
            record.update(
                {
                    "layer": layer,
                    "role": role,
                    "rows": rows,
                    "channels": int(state["in_features"]),
                }
            )

            named = 0.0
            for part in PARTS:
                summary = record["part_gpu"][part]
                named += summary["median_ms"] if summary else 0.0
            record["named_gpu_ms"] = named
            gptq_ms = record["part_gpu"]["gptq"]["median_ms"]
            inner = record["inner_gpu_ms"]
            record["gptq_inner_share_percent"] = {
                part: 100.0 * inner[part] / gptq_ms for part in INNER_PARTS
            }
            record["gptq_remainder_gpu_ms"] = (
                gptq_ms - sum(inner[part] for part in INNER_PARTS)
            )
            record["gptq_remainder_percent_of_call"] = (
                100.0 * record["gptq_remainder_gpu_ms"] / record["full_ms"]
            )
            record["residual_gpu_ms"] = record["full_ms"] - named
            record["residual_gpu_percent"] = (
                100.0 * record["residual_gpu_ms"] / record["full_ms"]
            )
            report["states"].append(record)

            print(
                f"\n=== layer{layer}/{role}  rows={rows}  "
                f"channels={record['channels']} ==="
            )
            print(f"  {'full call':>10}  gpu {record['full_ms']:9.3f} ms")
            for part in PARTS:
                summary = record["part_gpu"][part]
                wall = record["part_wall"][part]
                count = record["calls_per_round"][part]
                if summary is None:
                    print(f"  {part:>10}  not called")
                    continue
                print(
                    f"  {part:>10}  gpu {summary['median_ms']:8.3f} ms "
                    f"({100.0 * summary['median_ms'] / record['full_ms']:5.1f}%)   "
                    f"wall {wall['median_ms']:8.3f} ms   "
                    f"calls/round {count:g}"
                )
            print(
                f"  {'residual':>10}  gpu {record['residual_gpu_ms']:8.3f} ms "
                f"({record['residual_gpu_percent']:5.1f}%)   "
                f"= hook + state copy + fallback + CPU gaps"
            )
            print(
                f"    inside gptq ({gptq_ms:.1f} ms, "
                f"{100.0 * gptq_ms / record['full_ms']:.1f}% of the call):"
            )
            for part in INNER_PARTS:
                value = inner[part]
                count = record["calls_per_round"][part]
                print(
                    f"      {part:>8}  gpu {value:8.3f} ms "
                    f"({record['gptq_inner_share_percent'][part]:5.1f}% of gptq, "
                    f"{100.0 * value / record['full_ms']:5.1f}% of the call)   "
                    f"calls/round {count:g}"
                )
            print(
                f"      {'rest':>8}  gpu {record['gptq_remainder_gpu_ms']:8.3f} ms "
                f"({100.0 * record['gptq_remainder_gpu_ms'] / gptq_ms:5.1f}% of gptq, "
                f"{record['gptq_remainder_percent_of_call']:5.1f}% of the call)   "
                f"= index_select gathers + the compensation matmul"
            )

    Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
