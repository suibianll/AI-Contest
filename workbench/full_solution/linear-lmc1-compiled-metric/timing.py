"""L-MC1 timing, with the null control, measured on both sides of the move.

This card does not remove work, it moves it: the calibration gains one Cholesky
inverse per (layer, role) and the dynamic path loses one per call.  So both sides
are timed, because reporting only one would misstate the card:

  calibration   `hif4_calibration_and_quantize_weight` on real weights
  dynamic       `hif4_dynamic_quantize_activation` on real activations, using
                each arm's own calibrated state

Each is measured with three arms on the same inputs and device:

  parent      the v237 root
  candidate   the v237 root plus L-MC1
  sham        the parent again, loaded a second time from the same file

The sham is byte-identical to the parent, so its paired difference is the null --
what this estimator reports when nothing changed, on this machine, at this hour.
An effect that is not clearly larger than that is not a measurement, and the
plan's own rule is that a local wall clock neither gates a submission nor closes
a card; this is recorded as an accounting figure.

Note on the dynamic arm: the parent's cached calibration states carry no stored
metric, so timing the candidate against a *cached* state would measure the
deactivated arm rather than the change.  Each arm is therefore timed against the
state its own calibration produced.

Nothing here is converted into official seconds.
"""

from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import statistics
import sys

import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CANDIDATE = HERE / "candidate" / "solution.py"
PARENT = ROOT / "solutions/20260910_v237_linear-tf2-first-pass-gradient-reuse_scoreNA_timeNA/solution.py"
PACK = ROOT / "artifacts/official_eval/cache/qwen3.5-4b-proxy-v2.pt"

ROUNDS = 15
WARMUP = 2


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def summarise(values):
    ordered = sorted(values)
    return {
        "median": statistics.median(ordered),
        "min": ordered[0],
        "max": ordered[-1],
        "p25": ordered[len(ordered) // 4],
        "p75": ordered[(3 * len(ordered)) // 4],
        "stdev": statistics.stdev(ordered),
    }


def time_arms(label, arms, call, rounds):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    def timed(module):
        start.record()
        call(module)
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end)

    for _ in range(WARMUP):
        for _, module in arms:
            timed(module)

    samples = {name: [] for name, _ in arms}
    for index in range(rounds):
        offset = index % len(arms)
        order = arms[offset:] + arms[:offset]
        for name, module in order:
            samples[name].append(timed(module))

    summary = {name: summarise(values) for name, values in samples.items()}
    effects = {}
    for name in samples:
        if name == "parent":
            continue
        paired = sorted(p - c for p, c in zip(samples["parent"], samples[name]))
        effects[name] = {
            **summarise(paired),
            "percent_of_parent_median": 100.0
            * statistics.median(paired)
            / summary["parent"]["median"],
            "rounds_parent_slower": sum(1 for value in paired if value > 0),
            "rounds": rounds,
        }
    return {
        "label": label,
        "rounds": rounds,
        "arms_ms": summary,
        "paired_effects_ms": effects,
    }


def report(record, rounds):
    effect = record["paired_effects_ms"]["candidate"]
    null = record["paired_effects_ms"]["sham"]
    print(
        f"[timing] {record['label']:>28}  parent "
        f"{record['arms_ms']['parent']['median']:9.3f} ms  "
        f"effect {effect['median']:+8.3f} ms "
        f"({effect['percent_of_parent_median']:+.3f}%)  "
        f"null {null['median']:+8.3f} ms  |effect|/|null| "
        f"{abs(effect['median']) / max(abs(null['median']), 1e-9):.2f}x  "
        f"candidate faster in {effect['rounds_parent_slower']}/{rounds}",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rounds", type=int, default=ROUNDS)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--role", default="o")
    parser.add_argument("--output", default=str(HERE / "timing.json"))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required: the equivalence and the claim are device-bound")
    device = torch.device(args.device)

    sys.path.insert(0, str(ROOT / "evaluator"))
    import official_eval as v2  # noqa: PLC0415

    parent = load_module(PARENT, "lmc1_timing_parent")
    candidate = load_module(CANDIDATE, "lmc1_timing_candidate")
    sham = load_module(PARENT, "lmc1_timing_sham")
    arms = (("parent", parent), ("candidate", candidate), ("sham", sham))

    pack = torch.load(PACK, map_location="cpu", mmap=True, weights_only=False)
    layer, role = int(args.layer), str(args.role)

    weight_quant, weight_scale = v2._pair(
        pack["weights"][layer][role].to(torch.float32)
    )
    weight_quant = weight_quant.to(device)
    weight_scale = weight_scale.to(device)
    calibration = [
        tuple(t.to(device) for t in v2._pair(tensor.to(torch.float32)))
        for tensor in (
            pack["calibration_activations"][role][s][layer]
            for s in range(len(pack["calibration_activations"][role]))
            if pack["calibration_activations"][role][s][layer] is not None
        )
    ][:2]

    def calibrate(module):
        return module.hif4_calibration_and_quantize_weight(
            weight_quant, weight_scale, calibration
        )

    calibration_record = time_arms(
        f"layer{layer}/{role}/calibration", arms, calibrate, args.rounds
    )

    # Each arm's own state, so the dynamic arm is measured with the metric present.
    states = {name: dict(calibrate(module)["activation_state"]) for name, module in arms}
    activation = [
        tuple(t.to(device) for t in v2._pair(tensor.to(torch.float32)))
        for tensor in (
            pack["test_activations"][role][w][layer]
            for w in range(len(pack["test_activations"][role]))
            if pack["test_activations"][role][w][layer] is not None
        )
    ][:1][0]

    def dynamic(module):
        name = next(n for n, m in arms if m is module)
        return module.hif4_dynamic_quantize_activation(
            activation[0], activation[1], dict(states[name])
        )

    dynamic_record = time_arms(
        f"layer{layer}/{role}/dynamic", arms, dynamic, args.rounds
    )

    for record in (calibration_record, dynamic_record):
        report(record, args.rounds)

    payload = {
        "device": torch.cuda.get_device_name(0),
        "rounds_per_arm": args.rounds,
        "warmup_calls_per_arm": WARMUP,
        "ordering": "rotated each round so no arm is always last",
        "clock": "torch.cuda.Event, no profiler, no replay",
        "parent_sha256": hashlib.sha256(PARENT.read_bytes()).hexdigest(),
        "candidate_sha256": hashlib.sha256(CANDIDATE.read_bytes()).hexdigest(),
        "sham": "the parent archive loaded a second time; byte-identical to it",
        "layer": layer,
        "role": role,
        "note": (
            "the dynamic arm uses each arm's own calibrated state: the parent's cached "
            "states carry no stored metric, so timing against one would measure the "
            "deactivated arm rather than the change"
        ),
        "results": [calibration_record, dynamic_record],
    }
    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
