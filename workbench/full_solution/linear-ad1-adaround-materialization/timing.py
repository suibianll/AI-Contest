"""L-AD1 timing, with the null control that makes the number mean something.

The card removes memory traffic and four operator launches from a function that
is 51-68% of the Gram-layer encoder, so it is a small effect on a whole
activation call and it must be reported at the size it actually is.  Two
measurements are taken, because they answer different questions:

  helper   ``_adaround_mantissa`` alone, on the real 5-D batched shape the
           encoder passes (``[6, N, 8, 2, 4]``, K=6 candidates) and on a 4-D
           edge-extension shape.  This is where the change acts; if it does
           nothing here it does nothing anywhere, and no whole-call figure can
           rescue it.
  call     the real ``hif4_dynamic_quantize_activation`` on real cached states.
           This is the quantity that could ever reach the official clock.

Three arms run in every round, on the same state and the same activation:

  parent      the shipped root
  candidate   the root plus L-AD1
  sham        the shipped root again, loaded a second time from the same file

The sham arm is the point.  It is byte-identical to the parent, so the paired
difference between it and the parent is what this measurement reports when there
is nothing to measure -- an empirical null for the estimator itself, on this
machine, at this hour, with this much thermal drift.  An effect that is not
clearly larger than the null is not a measurement, and saying so is more useful
than quoting a median with a sign on it.

Order rotates each round so a monotone drift in clocks is spread over all three
arms rather than charged to whichever always ran last.  Timing is
``torch.cuda.Event`` with an explicit synchronise; no profiler is attached, no
kernel is replayed, nothing is instrumented.  Each arm gets untimed warm-up
calls, because a first call on a given state pays for cuBLAS handles and the
inverse.

Nothing here is converted to official seconds: the official 300 s gate is the
only time gate, and a local GPU measurement cannot produce an official figure.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import statistics
import sys
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PARENT = ROOT / "solution.py"
CANDIDATE = HERE / "implementation.generated.py"
CACHE_DIR = ROOT / "artifacts/official_eval/cache/proxy-v3-calibration"
PACK = ROOT / "artifacts/official_eval/cache/qwen3.5-4b-proxy-v2.pt"

# Derived from the parent file, not transcribed: the calibration cache is keyed
# by the root's sha256, so a rebase that left this behind would time the new
# candidate against the *previous* root's calibrated states.
PARENT_SHA_PREFIX = hashlib.sha256(PARENT.read_bytes()).hexdigest()[:16]

TARGETS = ((0, "q"), (0, "o"))
ROW_COUNTS = (128, 512)
# The batched solve inputs the encoder probe observed: K = 6 candidates, N = the
# row count, and the [8, 2, 4] block payload.
HELPER_SHAPES = ((6, 128, 8, 2, 4), (6, 509, 8, 2, 4), (1, 128, 8, 2, 4))
ROUNDS = 31
WARMUP = 3
HELPER_ITERATIONS = 200


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


def summarise(values) -> dict:
    ordered = sorted(values)
    return {
        "median": statistics.median(ordered),
        "min": ordered[0],
        "max": ordered[-1],
        "p25": ordered[len(ordered) // 4],
        "p75": ordered[(3 * len(ordered)) // 4],
        "stdev": statistics.stdev(ordered) if len(ordered) > 1 else 0.0,
    }


def paired_effect(samples, name, parent_median, rounds) -> dict:
    paired = sorted(p - c for p, c in zip(samples["parent"], samples[name]))
    return {
        **summarise(paired),
        "percent_of_parent_median": 100.0 * statistics.median(paired) / parent_median,
        "rounds_parent_slower": sum(1 for value in paired if value > 0),
        "rounds": rounds,
    }


def run_arms(arms, timed, rounds):
    names = [name for name, _ in arms]
    for _ in range(WARMUP):
        for _, module in arms:
            timed(module)
    samples = {name: [] for name in names}
    for index in range(rounds):
        offset = index % len(arms)
        order = arms[offset:] + arms[:offset]
        for name, module in order:
            samples[name].append(timed(module))
    return names, samples


def report(label, arms, timed, rounds, extra=None) -> dict:
    names, samples = run_arms(arms, timed, rounds)
    parent_median = statistics.median(samples["parent"])
    effects = {
        name: paired_effect(samples, name, parent_median, rounds)
        for name in names
        if name != "parent"
    }
    record = {
        "label": label,
        "rounds": rounds,
        "arms_ms": {name: summarise(values) for name, values in samples.items()},
        "paired_effects_ms": effects,
        "null_paired_median_ms": effects["sham"]["median"],
        "null_paired_dispersion_ms": effects["sham"]["p75"] - effects["sham"]["p25"],
        "candidate_paired_median_ms": effects["candidate"]["median"],
        "candidate_paired_dispersion_ms": effects["candidate"]["p75"]
        - effects["candidate"]["p25"],
    }
    if extra:
        record.update(extra)
    return record


def print_record(record, unit="ms") -> None:
    effect = record["paired_effects_ms"]["candidate"]
    null = record["paired_effects_ms"]["sham"]
    print(
        f"[timing] {record['label']:>28}  parent "
        f"{record['arms_ms']['parent']['median']:9.4f} {unit}  "
        f"effect {effect['median']:+9.4f} {unit} "
        f"({effect['percent_of_parent_median']:+.3f}%)  "
        f"null {null['median']:+9.4f} {unit}  "
        f"|effect|/|null| {abs(effect['median']) / max(abs(null['median']), 1e-9):5.2f}x  "
        f"candidate faster in {effect['rounds_parent_slower']}/{record['rounds']}  "
        f"null faster in {null['rounds_parent_slower']}/{record['rounds']}",
        flush=True,
    )


def helper_measure(arms, shape, device) -> dict:
    """Times the helper alone, on a shape the encoder really passes it.

    ``HELPER_ITERATIONS`` calls are issued between one pair of events and the
    total is divided by the count: a single call is a few hundred microseconds,
    which is close to the CUDA event resolution, and averaging inside one event
    pair removes the per-launch resolution error without removing the work.
    """
    x_abs = torch.rand(shape, dtype=torch.float32, device=device) + 0.5
    local_scale = torch.full(shape, 1.31, dtype=torch.float32, device=device)
    sign = torch.ones(shape, dtype=torch.float32, device=device)
    gram = torch.eye(4, dtype=torch.float32, device=device).expand(
        *shape[:-1], 4, 4
    ).contiguous()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    def timed(module):
        start.record()
        for _ in range(HELPER_ITERATIONS):
            module._adaround_mantissa(x_abs, local_scale, sign, gram)
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / HELPER_ITERATIONS

    return report(
        f"helper {tuple(shape)}",
        arms,
        timed,
        ROUNDS,
        extra={
            "shape": list(shape),
            "iterations_per_sample": HELPER_ITERATIONS,
            "element_count": int(x_abs.numel()),
        },
    )


def call_measure(arms, evaluator, pack, layer, role, state, rows, device) -> dict:
    raw = pack["test_activations"]
    base_quant, base_scale = evaluator._pair(raw[role][1][layer].to(torch.float32))
    repeat = -(-rows // base_quant.shape[0])
    quant = base_quant.repeat(repeat, 1)[:rows].contiguous().to(device)
    scale = base_scale.repeat(repeat, 1)[:rows].contiguous().to(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    def timed(module):
        start.record()
        module.hif4_dynamic_quantize_activation(quant, scale, dict(state))
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end)

    return report(
        f"call layer{layer}/{role} rows={rows}",
        arms,
        timed,
        ROUNDS,
        extra={
            "layer": layer,
            "role": role,
            "rows": rows,
            "channels": int(quant.shape[1]),
            "has_gram": state.get("gram") is not None,
        },
    )


def render(path: Path) -> int:
    """Re-prints a stored measurement under the same format, without measuring.

    The recorded samples are the evidence; re-running the measurement to
    produce a log file would replace them with slightly different ones and
    leave the write-up quoting numbers that are no longer on disk.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    print(
        f"device {payload['device']}  rounds/arm {payload['rounds_per_arm']}  "
        f"warmup {payload['warmup_calls_per_arm']}\nclock {payload['clock']}\n"
        f"sham: {payload['sham']}\n"
    )
    for record in payload["helper_records"] + payload["call_records"]:
        print_record(record, unit="ms")
    return 0


def main() -> int:
    global ROUNDS

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rounds", type=int, default=ROUNDS)
    parser.add_argument("--output", default=str(HERE / "timing.json"))
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="re-print a stored timing.json without touching the GPU",
    )
    args = parser.parse_args()
    ROUNDS = args.rounds

    if args.render_only:
        return render(Path(args.output))

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required: the card's claim is a GPU-time claim")
    device = torch.device("cuda")

    parent = load_solution(PARENT, "ad1_timing_parent")
    candidate = load_solution(CANDIDATE, "ad1_timing_candidate")
    sham = load_solution(PARENT, "ad1_timing_sham")
    arms = (("parent", parent), ("candidate", candidate), ("sham", sham))

    sys.path.insert(0, str(ROOT / "evaluator"))
    import official_eval as evaluator  # noqa: PLC0415

    pack = torch.load(PACK, map_location="cpu", mmap=True, weights_only=False)

    helper_records = [
        helper_measure(arms, shape, device) for shape in HELPER_SHAPES
    ]
    for record in helper_records:
        print_record(record, unit="ms")

    call_records = []
    for layer, role in TARGETS:
        state = find_state(layer, role)
        for rows in ROW_COUNTS:
            record = call_measure(
                arms, evaluator, pack, layer, role, state, rows, device
            )
            call_records.append(record)
            print_record(record, unit="ms")

    payload = {
        "device": torch.cuda.get_device_name(0),
        "rounds_per_arm": ROUNDS,
        "warmup_calls_per_arm": WARMUP,
        "ordering": "rotated each round so no arm is always last",
        "clock": "torch.cuda.Event, no profiler, no replay",
        "parent_sha256": hashlib.sha256(PARENT.read_bytes()).hexdigest(),
        "candidate_sha256": hashlib.sha256(CANDIDATE.read_bytes()).hexdigest(),
        "sham": "solution.py loaded a second time; byte-identical to the parent",
        "helper_records": helper_records,
        "call_records": call_records,
    }
    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
