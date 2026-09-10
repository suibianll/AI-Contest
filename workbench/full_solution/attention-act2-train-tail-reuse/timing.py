"""A-CT2 timing, with the null control that makes the number mean something.

The plan requires the timing to separate parent calibration, training and gate.
This card's change lives in the tail of ``_agr1_train``, so two levels are
measured rather than one -- reporting only the whole calibration call would hide
the change, and reporting only the changed function would overstate it:

  whole-calibration   ``hif4_calibration_attention`` end to end, which is what
                      the run actually pays for
  training            ``_agr1_train`` on its own, which is where the six
                      removed calls live

Three arms run in every round, on the same windows, states and device:

  assembly    v236 (v231 root + A-GR1), the implementation control
  candidate   the same plus A-CT2
  sham        the assembly again, loaded a second time from the same file

The sham arm is the point.  It is *byte-identical* to the assembly, so the paired
difference between it and the assembly is what this measurement reports when
there is nothing to measure -- an empirical null for the estimator itself.  An
effect that is not clearly larger than the null is not a measurement.  That is
the expected outcome here: the card removes 6 of 204 calls in the tail of a
function that is itself a fraction of a 6-second calibration, so this is written
down as an accounting figure and not as a speed-up claim.

Order rotates each round; timing is ``torch.cuda.Event`` with an explicit
synchronise, no profiler and no replay.

Nothing here is converted to official seconds.
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
ASSEMBLY = ROOT / "solutions/20260910_v236_attention-agr1-on-v231_scoreNA_timeNA/solution.py"
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
        if name == "assembly":
            continue
        paired = sorted(p - c for p, c in zip(samples["assembly"], samples[name]))
        effects[name] = {
            **summarise(paired),
            "percent_of_assembly_median": 100.0
            * statistics.median(paired)
            / summary["assembly"]["median"],
            "rounds_assembly_slower": sum(1 for value in paired if value > 0),
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
        f"[timing] {record['label']:>28}  assembly "
        f"{record['arms_ms']['assembly']['median']:9.3f} ms  "
        f"effect {effect['median']:+8.3f} ms "
        f"({effect['percent_of_assembly_median']:+.3f}%)  "
        f"null {null['median']:+8.3f} ms  |effect|/|null| "
        f"{abs(effect['median']) / max(abs(null['median']), 1e-9):.2f}x  "
        f"candidate faster in {effect['rounds_assembly_slower']}/{rounds}",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rounds", type=int, default=ROUNDS)
    parser.add_argument("--layers", default="0,22")
    parser.add_argument("--output", default=str(HERE / "timing.json"))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required: the card's claim is a GPU-time claim")
    device = torch.device(args.device)

    sys.path.insert(0, str(ROOT / "evaluator"))
    import official_eval as v2  # noqa: PLC0415

    assembly = load_module(ASSEMBLY, "act2_timing_assembly")
    candidate = load_module(CANDIDATE, "act2_timing_candidate")
    sham = load_module(ASSEMBLY, "act2_timing_sham")
    arms = (("assembly", assembly), ("candidate", candidate), ("sham", sham))

    pack = torch.load(PACK, map_location="cpu", mmap=True, weights_only=False)
    q_heads = int(pack["q_heads"])
    kv_heads = int(pack["kv_heads"])
    head_dim = int(pack["head_dim"])

    results = []
    for layer in [int(part) for part in args.layers.split(",") if part.strip()]:
        windows = [
            {
                "q": tuple(t.to(device) for t in v2._pair(pack["calibration_qkv"][s][layer][0])),
                "k": tuple(t.to(device) for t in v2._pair(pack["calibration_qkv"][s][layer][1])),
                "v": tuple(t.to(device) for t in v2._pair(pack["calibration_qkv"][s][layer][2])),
            }
            for s in range(len(pack["calibration_qkv"]))
        ]

        def whole(module):
            return module.hif4_calibration_attention(windows, q_heads, kv_heads, head_dim)

        whole_record = time_arms(f"layer{layer}/whole-calibration", arms, whole, args.rounds)
        results.append(whole_record)

        # The training call alone: this card's change is in its tail.
        states = assembly._AGR1_PARENT_CALIBRATION(windows, q_heads, kv_heads, head_dim)
        fit_windows = windows[: len(windows) - len(assembly._AGR1_GATE_WINDOWS)]

        def train(module):
            return module._agr1_train(
                fit_windows, states, q_heads, kv_heads, head_dim, device
            )

        train_record = time_arms(f"layer{layer}/agr1-train", arms, train, args.rounds)
        results.append(train_record)

        for record in (whole_record, train_record):
            report(record, args.rounds)

    payload = {
        "device": torch.cuda.get_device_name(0),
        "rounds_per_arm": args.rounds,
        "warmup_calls_per_arm": WARMUP,
        "ordering": "rotated each round so no arm is always last",
        "clock": "torch.cuda.Event, no profiler, no replay",
        "assembly_sha256": hashlib.sha256(ASSEMBLY.read_bytes()).hexdigest(),
        "candidate_sha256": hashlib.sha256(CANDIDATE.read_bytes()).hexdigest(),
        "sham": "the assembly archive loaded a second time; byte-identical to it",
        "removed_calls_per_train": 6,
        "total_calls_per_train": 204,
        "results": results,
    }
    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
