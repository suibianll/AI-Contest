"""Last localization layer: inside ``_dense_to_hif4``, the exact candidate solve
versus everything else.

``split_probe.py`` localized the dynamic call to ``_dense_to_hif4`` (70-91% of a
call, run once per 64-channel block) and found that its per-block cost is nearly
row-independent: 4096 channels cost the same at 128 and at 512 rows, and 2560
channels fit 8.93 ms fixed per block plus 0.0224 ms per row.

Plan section 4.2 names the next question and stops there: of that per-block
figure, how much is ``_solve_exact_hierarchy`` evaluating the K candidate scale
codes, and how much is the remaining tensor movement?  The answer decides where
the branch-3 card may act -- but only where it may act without touching the
candidate set, the coverage, the rounding candidates or the refinement rounds.

So this brackets, inside the encoder window:

  batched   ``_solve_exact_hierarchy`` on a 5-D input -- the single call that
            evaluates all candidate codes at once (``x_expanded`` is
            ``[K, N, 8, 2, 4]``)
  edge      ``_solve_exact_hierarchy`` on a 4-D input -- the per-step calls the
            edge extension makes, one Python call and one device sync per step
  adaround  ``_adaround_mantissa`` -- the 16-pattern enumeration the Gram path
            uses in place of ``torch.round``; absent when the state has no gram

Two facts about the window are read as well, because they decide whether a
"fewer, larger operations" change can reach this cost at all.  ``wall`` and the
CUDA-event span both include GPU idle time, so their agreement says only that no
CPU work happens outside the window; ``thread_time`` is the main thread's own
CPU time, so a thread time near the wall time means the cost is being paid on
the CPU side (dispatch, allocation, synchronization) rather than in FLOPs.

The counts are reported for the same reason: a torch-level call count is not a
kernel count -- method-form calls such as ``x.amax(...)`` are not interceptable
here and are excluded -- but it bounds the launch traffic from below, and it is
the quantity a batching change would reduce.

Read-only with respect to the repo: writes ``encoder_probe.json`` next to itself.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from torch.utils._python_dispatch import TorchDispatchMode


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CACHE_DIR = ROOT / "artifacts/official_eval/cache/proxy-v3-calibration"
PACK = ROOT / "artifacts/official_eval/cache/qwen3.5-4b-proxy-v2.pt"

PARENT_SHA_PREFIX = "0f1af6dbc207ff32"

TARGETS = ((0, "q"), (0, "o"))
ROW_COUNTS = (128, 512)

# The torch-level calls the encoder makes in module form.  Method-form calls
# (``.amax``, ``.reshape``, ``.index_select`` ...) cannot be intercepted without
# patching ``torch.Tensor``, which is not exposed for assignment; they are
# therefore absent from the counts, and the counts are a lower bound.
TORCH_OPS = (
    "round",
    "clamp",
    "einsum",
    "minimum",
    "where",
    "stack",
    "gather",
    "floor",
    "ceil",
    "arange",
    "nonzero",
    "topk",
    "index_select",
    "cat",
    "tensor",
    "zeros",
    "full",
    "sign",
    "nan_to_num",
    "quantile",
)

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


class EncoderRecorder:
    """Brackets every ``_dense_to_hif4`` call and whatever it calls inside."""

    def __init__(self, module):
        self.module = module
        self.originals: dict[str, object] = {}
        self.torch_originals: dict[str, object] = {}
        self.pending: list[dict] = []
        self.current: dict | None = None
        self.window_open = False

    def __enter__(self):
        self.originals["encoder"] = self.module._dense_to_hif4
        self.originals["solve"] = self.module._solve_exact_hierarchy
        self.originals["adaround"] = self.module._adaround_mantissa
        self.module._dense_to_hif4 = self._wrap_encoder(self.originals["encoder"])
        self.module._solve_exact_hierarchy = self._wrap_inner(
            "solve", self.originals["solve"]
        )
        self.module._adaround_mantissa = self._wrap_inner(
            "adaround", self.originals["adaround"]
        )
        for name in TORCH_OPS:
            original = getattr(torch, name, None)
            if original is None or not callable(original):
                raise SystemExit(f"torch has no callable {name}")
            self.torch_originals[name] = original
            setattr(torch, name, self._wrap_torch(name, original))
        return self

    def __exit__(self, *exc_info):
        self.module._dense_to_hif4 = self.originals["encoder"]
        self.module._solve_exact_hierarchy = self.originals["solve"]
        self.module._adaround_mantissa = self.originals["adaround"]
        for name, original in self.torch_originals.items():
            setattr(torch, name, original)
        return False

    def _wrap_encoder(self, original):
        recorder = self

        def wrapper(*args, **kwargs):
            record = {
                "begin_event": torch.cuda.Event(enable_timing=True),
                "finish_event": torch.cuda.Event(enable_timing=True),
                "begin_wall": time.perf_counter(),
                "begin_thread": time.thread_time(),
                "inner": [],
                "ops": Counter(),
                "shape": tuple(int(v) for v in args[0].shape),
            }
            previous = recorder.current
            recorder.current = record
            recorder.window_open = True
            record["begin_event"].record()
            try:
                return original(*args, **kwargs)
            finally:
                recorder.window_open = False
                recorder.current = previous
                record["finish_event"].record()
                record["wall_ms"] = (time.perf_counter() - record["begin_wall"]) * 1000.0
                record["thread_ms"] = (
                    time.thread_time() - record["begin_thread"]
                ) * 1000.0
                recorder.pending.append(record)

        return wrapper

    def _wrap_inner(self, part, original):
        recorder = self

        def wrapper(*args, **kwargs):
            record = recorder.current
            if record is None:
                return original(*args, **kwargs)
            entry = {
                "part": part,
                "ndim": int(args[0].ndim),
                "shape": tuple(int(v) for v in args[0].shape),
                "begin_event": torch.cuda.Event(enable_timing=True),
                "finish_event": torch.cuda.Event(enable_timing=True),
                "begin_wall": time.perf_counter(),
            }
            entry["begin_event"].record()
            try:
                return original(*args, **kwargs)
            finally:
                entry["finish_event"].record()
                entry["wall_ms"] = (time.perf_counter() - entry["begin_wall"]) * 1000.0
                record["inner"].append(entry)

        return wrapper

    def _wrap_torch(self, name, original):
        recorder = self

        def wrapper(*args, **kwargs):
            if recorder.window_open and recorder.current is not None:
                recorder.current["ops"][name] += 1
            return original(*args, **kwargs)

        return wrapper

    def take(self):
        """Resolves the round's encoder calls; call only after a device sync."""
        calls = self.pending
        self.pending = []
        for record in calls:
            record["gpu_ms"] = record["begin_event"].elapsed_time(
                record["finish_event"]
            )
            for entry in record["inner"]:
                entry["gpu_ms"] = entry["begin_event"].elapsed_time(
                    entry["finish_event"]
                )
        return calls


class OpCounter(TorchDispatchMode):
    """Counts every aten op dispatched while an encoder window is open.

    This is a count, not a measurement: the mode adds a Python call to every
    dispatch, so a timing taken underneath it is inflated and none is taken.
    What it answers is whether the encoder's constant per-call cost is a long
    chain of cheap ops or a few expensive ones -- and it counts method-form
    calls (``x.amax``, ``.index_select``) that a torch-module patch cannot see.
    """

    def __init__(self, recorder):
        super().__init__()
        self.recorder = recorder
        self.counts: Counter = Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if self.recorder.window_open:
            self.counts[str(func)] += 1
        return func(*args, **(kwargs if kwargs is not None else {}))


def count_ops(module, quant, scale, state, rounds):
    """Counts aten ops inside the encoder window; reports counts only."""

    def call():
        return module.hif4_dynamic_quantize_activation(quant, scale, dict(state))

    per_round: list[dict] = []
    with EncoderRecorder(module) as recorder:
        with OpCounter(recorder) as counter:
            for _ in range(rounds):
                counter.counts.clear()
                call()
                torch.cuda.synchronize()
                calls = recorder.take()
                per_round.append(
                    {
                        "encoder_calls": len(calls),
                        "counts": dict(counter.counts),
                        "total": sum(counter.counts.values()),
                    }
                )

    encoder_calls = statistics.median(r["encoder_calls"] for r in per_round)
    totals: Counter = Counter()
    for record in per_round:
        totals.update(record["counts"])
    return {
        "rounds": rounds,
        "encoder_calls_per_round": encoder_calls,
        "ops_per_round": statistics.median(r["total"] for r in per_round),
        "ops_per_encoder_call": (
            statistics.median(r["total"] for r in per_round) / encoder_calls
            if encoder_calls
            else None
        ),
        "top_ops_per_round": [
            {"op": op, "count": count} for op, count in totals.most_common(20)
        ],
    }


def measure(module, quant, scale, state, rounds):
    def call():
        return module.hif4_dynamic_quantize_activation(quant, scale, dict(state))

    for _ in range(WARMUP):
        call()
    torch.cuda.synchronize()

    per_round: list[dict] = []
    shape_seen: dict[str, object] = {}

    with EncoderRecorder(module) as recorder:
        for _ in range(rounds):
            call()
            torch.cuda.synchronize()
            calls = recorder.take()

            aggregate = {
                "calls": len(calls),
                "gpu_ms": sum(c["gpu_ms"] for c in calls),
                "wall_ms": sum(c["wall_ms"] for c in calls),
                "thread_ms": sum(c["thread_ms"] for c in calls),
                "batched": {"count": 0, "gpu_ms": 0.0},
                "edge": {"count": 0, "gpu_ms": 0.0},
                "adaround": {"count": 0, "gpu_ms": 0.0},
                "ops": Counter(),
            }
            for record in calls:
                aggregate["ops"].update(record["ops"])
                for entry in record["inner"]:
                    if entry["part"] == "adaround":
                        bucket = aggregate["adaround"]
                    else:
                        bucket = (
                            aggregate["batched"]
                            if entry["ndim"] == 5
                            else aggregate["edge"]
                        )
                    bucket["count"] += 1
                    bucket["gpu_ms"] += entry["gpu_ms"]
                    # The batched input shape carries the candidate count K and
                    # the refined-block count N: the quantity a "fewer, larger
                    # operations" change would be measured against.
                    seen = shape_seen.setdefault(
                        f"{entry['part']}_ndim{entry['ndim']}", {"shapes": []}
                    )
                    if entry["shape"] not in seen["shapes"]:
                        seen["shapes"].append(entry["shape"])
            per_round.append(aggregate)

    def median_of(key_path):
        values = []
        for aggregate in per_round:
            node = aggregate
            for key in key_path:
                node = node[key]
            values.append(node)
        return statistics.median(values)

    def bucket_summary(name):
        return {
            "calls_per_round": median_of([name, "count"]),
            "gpu_ms_per_round": median_of([name, "gpu_ms"]),
        }

    encoder_gpu = median_of(["gpu_ms"])
    encoder_wall = median_of(["wall_ms"])
    encoder_thread = median_of(["thread_ms"])
    batched = bucket_summary("batched")
    edge = bucket_summary("edge")
    adaround = bucket_summary("adaround")

    op_counts = {}
    for name in TORCH_OPS:
        values = [aggregate["ops"].get(name, 0) for aggregate in per_round]
        op_counts[name] = statistics.median(values)

    return {
        "rounds": rounds,
        "encoder_calls_per_round": median_of(["calls"]),
        "encoder_gpu_ms": encoder_gpu,
        "encoder_wall_ms": encoder_wall,
        "encoder_thread_ms": encoder_thread,
        "batched_solve": batched,
        "edge_solve": edge,
        "adaround": adaround,
        "rest_ms": encoder_gpu
        - batched["gpu_ms_per_round"]
        - edge["gpu_ms_per_round"]
        - adaround["gpu_ms_per_round"],
        "torch_op_counts_per_round": op_counts,
        "shapes": shape_seen,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rounds", type=int, default=ROUNDS)
    parser.add_argument("--count-rounds", type=int, default=2)
    parser.add_argument("--output", default=str(HERE / "encoder_probe.json"))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    device = torch.device("cuda")

    sys.path.insert(0, str(ROOT / "evaluator"))
    import official_eval as evaluator  # noqa: PLC0415

    parent = load_solution(ROOT / "solution.py", "encoder_parent")

    pack = torch.load(PACK, map_location="cpu", mmap=True, weights_only=False)
    raw = pack["test_activations"]

    report = {
        "device": torch.cuda.get_device_name(0),
        "parent_sha256_prefix": PARENT_SHA_PREFIX,
        "rounds": args.rounds,
        "purpose": (
            "last localization layer: inside _dense_to_hif4, the candidate exact "
            "solve versus the rest, as plan section 4.2 requires"
        ),
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
                    "has_gram": state.get("gram") is not None,
                    "max_refine_ratio": float(state["max_refine_ratio"]),
                    "max_refine_blocks": int(state["max_refine_blocks"]),
                    "num_offsets": int(len(state["offsets"])),
                }
            )
            record["aten_ops"] = count_ops(
                parent, quant, scale, state, args.count_rounds
            )
            report["states"].append(record)

            calls = record["encoder_calls_per_round"]
            gpu = record["encoder_gpu_ms"]
            print(
                f"\n=== layer{layer}/{role}  rows={rows}  "
                f"channels={record['channels']}  gram={record['has_gram']} ==="
            )
            print(
                f"  encoder  {calls:g} calls/round, {gpu:8.3f} ms/round "
                f"= {gpu / calls:7.3f} ms/call"
            )
            print(
                f"    wall {record['encoder_wall_ms']:8.3f} ms   "
                f"thread {record['encoder_thread_ms']:8.3f} ms "
                f"({100.0 * record['encoder_thread_ms'] / record['encoder_wall_ms']:.0f}% of wall)"
            )
            for name, label in (
                ("batched_solve", "solve batched (ndim 5)"),
                ("edge_solve", "solve edge   (ndim 4)"),
                ("adaround", "adaround_mantissa"),
            ):
                bucket = record[name]
                per_call = (
                    bucket["gpu_ms_per_round"] / calls if calls else 0.0
                )
                share = (
                    100.0 * bucket["gpu_ms_per_round"] / gpu if gpu else 0.0
                )
                print(
                    f"    {label:22}  {bucket['calls_per_round']:6g} calls/round  "
                    f"{per_call:7.3f} ms/call  ({share:5.1f}% of encoder)"
                )
            print(
                f"    {'rest':22}  {record['rest_ms'] / calls:7.3f} ms/call  "
                f"({100.0 * record['rest_ms'] / gpu:5.1f}% of encoder)"
            )
            counted = sum(record["torch_op_counts_per_round"].values())
            print(
                f"    torch-module calls/round: {counted:g} "
                f"({counted / calls:.0f} per encoder call; method-form excluded)"
            )
            aten = record["aten_ops"]
            print(
                f"    aten ops/encoder call: {aten['ops_per_encoder_call']:.0f} "
                f"(counts only; no timing taken under the dispatch mode)"
            )
            for row in aten["top_ops_per_round"][:8]:
                per_call = row["count"] / aten["encoder_calls_per_round"]
                print(f"        {row['op']:38} {per_call:7.2f} / encoder call")

    Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
