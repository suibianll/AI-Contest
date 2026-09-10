"""A-CT1 timing, with the null control that makes the number mean something.

The plan requires the timing to separate parent calibration, training and gate,
and to keep a profiler-free wall clock.  The change this card makes lives only
in the gate, so the gate is measured on its own as well as inside the whole
calibration call -- reporting only the whole call would hide it, and reporting
only the gate would overstate it.

Three arms run in every round, on the same windows, states and device:

  assembly    v236 (v231 root + A-GR1), the implementation control
  candidate   the same plus A-CT1
  sham        the assembly again, loaded a second time from the same file

The sham arm is the point.  It is *byte-identical* to the assembly, so the
paired difference between it and the assembly is what this measurement reports
when there is nothing to measure -- an empirical null for the estimator itself,
on this machine, at this hour.  A candidate effect that is not clearly larger
than the null is not a measurement.

Order rotates each round so a monotonic clock drift is spread over all three
arms.  Timing is ``torch.cuda.Event`` with an explicit synchronise; no profiler
is attached and no kernel is replayed.  Warm-up calls run untimed first.

Peak allocated memory is recorded per arm, because the card's premise is that
less work is done: a change that traded time for memory would show up here.

Nothing here is converted to official seconds: the official 300 s gate is the
only time gate, and a local GPU measurement cannot produce an official figure.
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
    """Paired three-arm timing: ``call(module)`` once per arm per round."""

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
        # Rotate the execution order so no arm is systematically last.
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


def peak_bytes(device, module, call):
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    call(module)
    torch.cuda.synchronize(device)
    return int(torch.cuda.max_memory_allocated(device))


def report(record, rounds):
    effect = record["paired_effects_ms"]["candidate"]
    null = record["paired_effects_ms"]["sham"]
    print(
        f"[timing] {record['label']:>30}  assembly "
        f"{record['arms_ms']['assembly']['median']:8.3f} ms  "
        f"effect {effect['median']:+7.3f} ms "
        f"({effect['percent_of_assembly_median']:+.3f}%)  "
        f"null {null['median']:+7.3f} ms  |effect|/|null| "
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

    assembly = load_module(ASSEMBLY, "act1_timing_assembly")
    candidate = load_module(CANDIDATE, "act1_timing_candidate")
    sham = load_module(ASSEMBLY, "act1_timing_sham")
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

        whole_record = time_arms(
            f"layer{layer}/whole-calibration", arms, whole, args.rounds
        )
        whole_record["arm_peak_bytes"] = {
            name: peak_bytes(device, module, whole) for name, module in arms
        }
        results.append(whole_record)

        # The gate alone: the part this card changes.
        states = assembly._AGR1_PARENT_CALIBRATION(windows, q_heads, kv_heads, head_dim)
        trained = assembly._agr1_parent_copy(states)
        tq, tk, center, _ = assembly._agr1_train(
            windows[: len(windows) - len(assembly._AGR1_GATE_WINDOWS)],
            states, q_heads, kv_heads, head_dim, device,
        )
        trained["q_state"]["learned_rotation"] = tq
        trained["k_state"]["learned_rotation"] = tk
        if center is not None:
            trained["k_state"]["learned_center"] = center
        item = windows[assembly._AGR1_GATE_WINDOWS[0]]

        def old_gate(module):
            module._agr1_gate_loss(item, states, q_heads, kv_heads, head_dim)
            module._agr1_gate_loss(item, trained, q_heads, kv_heads, head_dim)

        def new_gate(module):
            module._act1_gate_pair(item, states, trained, q_heads, kv_heads, head_dim)

        def gate(module):
            # The candidate runs the new path; the assembly and the sham run the
            # old one, which is what makes the sham a null for this comparison.
            return new_gate(module) if module is candidate else old_gate(module)

        gate_record = time_arms(
            f"layer{layer}/gate-window-pair", arms, gate, args.rounds
        )
        gate_record["note"] = (
            "assembly and sham run A-GR1's two gate-loss calls; candidate runs one "
            "_act1_gate_pair call"
        )
        results.append(gate_record)

        for record in (whole_record, gate_record):
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
        "results": results,
    }
    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
