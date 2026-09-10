"""L-TF1 timing, with the null control that makes the number mean something.

The card's claim is a time claim, so it has to be measured rather than argued
from an operator count.  The plan requires a same-state, same-device,
profiler-free paired measurement reported as a median plus a dispersion, and
forbids converting the result into official seconds.

Three arms run in every round, on the same state and the same activation:

  parent      the shipped root
  candidate   the root plus L-TF1
  sham        the shipped root again, loaded a second time from the same file

The sham arm is the point.  It is *byte-identical* to the parent, so the paired
difference between it and the parent is what this measurement reports when
there is nothing to measure -- an empirical null for the estimator itself, on
this machine, at this hour, with this much thermal drift.  A candidate effect
that is not clearly larger than the null is not a measurement, and saying so is
more useful than quoting a median with a sign on it.

Order rotates each round so a monotone drift in clocks is spread over all three
arms rather than charged to whichever always ran last.  Timing is
``torch.cuda.Event`` with an explicit synchronise beforehand; CUDA events are
the GPU's own timestamps, so this measures the kernel stream at normal speed --
no profiler is attached, no kernel is replayed, nothing is instrumented.  Each
arm gets untimed warm-up calls, because a first call on a given state pays for
cuBLAS handles and the cholesky inverse.

Two row counts are measured because the two removed products scale with rows
while most of the call does not, and reporting only the larger would overstate
the card.

Nothing here is converted to official seconds: the official 300 s gate is the
only time gate, and a local GPU measurement cannot produce an official figure.
"""

from pathlib import Path
import argparse
import importlib.util
import json
import statistics
import sys

import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CANDIDATE = HERE / "candidate" / "solution.py"
CACHE_DIR = ROOT / "artifacts/official_eval/cache/proxy-v3-calibration"
PACK = ROOT / "artifacts/official_eval/cache/qwen3.5-4b-proxy-v2.pt"

PARENT_SHA_PREFIX = "0f1af6dbc207ff32"

# (layer, role) targets.  q is 2560 channels and o is 4096; both are inside the
# metric's scope.  proj is 9216 and never enters the descent, so it would only
# add time the card cannot reach.
TARGETS = ((0, "q"), (0, "o"))
ROW_COUNTS = (128, 512)
ROUNDS = 31
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


def measure(label, arms, quant, scale, state) -> dict:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    def timed(module):
        start.record()
        module.hif4_dynamic_quantize_activation(quant, scale, dict(state))
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end)

    names = [name for name, _ in arms]
    for _ in range(WARMUP):
        for _, module in arms:
            timed(module)

    samples = {name: [] for name in names}
    for index in range(ROUNDS):
        # Rotate the execution order so no arm is systematically last.
        offset = index % len(arms)
        order = arms[offset:] + arms[:offset]
        for name, module in order:
            samples[name].append(timed(module))

    rows = int(quant.shape[0])
    channels = int(quant.shape[1])
    parent_median = statistics.median(samples["parent"])
    summary = {name: summarise(values) for name, values in samples.items()}

    effects = {}
    for name in names:
        if name == "parent":
            continue
        paired = sorted(p - c for p, c in zip(samples["parent"], samples[name]))
        effects[name] = {
            **summarise(paired),
            "percent_of_parent_median": 100.0 * statistics.median(paired) / parent_median,
            "rounds_parent_slower": sum(1 for value in paired if value > 0),
            "rounds": ROUNDS,
        }

    return {
        "label": label,
        "rows": rows,
        "channels": channels,
        "rounds": ROUNDS,
        "arms_ms": summary,
        "paired_effects_ms": effects,
        # Two products removed, each rows x channels @ channels x channels, two
        # floating-point operations per multiply-add.
        "removed_flops": 2 * 2 * rows * channels * channels,
        "null_paired_median_ms": effects["sham"]["median"],
        "null_paired_dispersion_ms": effects["sham"]["p75"] - effects["sham"]["p25"],
        "candidate_paired_median_ms": effects["candidate"]["median"],
        "candidate_paired_dispersion_ms": effects["candidate"]["p75"]
        - effects["candidate"]["p25"],
    }


def main() -> int:
    global ROUNDS

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rounds", type=int, default=ROUNDS)
    parser.add_argument("--output", default=str(HERE / "timing.json"))
    args = parser.parse_args()
    ROUNDS = args.rounds

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required: the card's claim is a GPU-time claim")
    device = torch.device(args.device)

    parent = load_solution(ROOT / "solution.py", "tf1_timing_parent")
    candidate = load_solution(CANDIDATE, "tf1_timing_candidate")
    sham = load_solution(ROOT / "solution.py", "tf1_timing_sham")
    for name, module in (("parent", parent), ("candidate", candidate), ("sham", sham)):
        if int(module._EM1_PASSES) != 1:
            raise SystemExit(f"{name} must ship K = 1")

    sys.path.insert(0, str(ROOT / "evaluator"))
    import official_eval as evaluator  # noqa: PLC0415

    pack = torch.load(PACK, map_location="cpu", mmap=True, weights_only=False)
    raw = pack["test_activations"]
    # test_activations[role] is indexed [window][layer]; window 1 is the one the
    # verifier uses.
    encoded = {
        (layer, role): evaluator._pair(raw[role][1][layer].to(torch.float32))
        for layer, role in TARGETS
    }

    results = []
    for layer, role in TARGETS:
        base_quant, base_scale = encoded[(layer, role)]
        state = find_state(layer, role)
        for rows in ROW_COUNTS:
            repeat = -(-rows // base_quant.shape[0])
            quant = base_quant.repeat(repeat, 1)[:rows].contiguous().to(device)
            scale = base_scale.repeat(repeat, 1)[:rows].contiguous().to(device)
            record = measure(
                f"layer{layer}/{role}/{rows}",
                (("parent", parent), ("candidate", candidate), ("sham", sham)),
                quant,
                scale,
                state,
            )
            results.append(record)
            null = record["paired_effects_ms"]["sham"]
            effect = record["paired_effects_ms"]["candidate"]
            print(
                f"[timing] {record['label']:>18}  parent "
                f"{record['arms_ms']['parent']['median']:8.3f} ms  "
                f"effect {effect['median']:+7.3f} ms "
                f"({effect['percent_of_parent_median']:+.3f}%)  "
                f"null {null['median']:+7.3f} ms  "
                f"|effect|/|null| "
                f"{abs(effect['median']) / max(abs(null['median']), 1e-9):.2f}x  "
                f"candidate faster in {effect['rounds_parent_slower']}/{ROUNDS}  "
                f"null faster in {null['rounds_parent_slower']}/{ROUNDS}",
                flush=True,
            )

    payload = {
        "device": torch.cuda.get_device_name(0),
        "rounds_per_arm": ROUNDS,
        "warmup_calls_per_arm": WARMUP,
        "ordering": "rotated each round so no arm is always last",
        "clock": "torch.cuda.Event, no profiler, no replay",
        "parent_sha256_prefix": PARENT_SHA_PREFIX,
        "candidate_sha256": __import__("hashlib").sha256(CANDIDATE.read_bytes()).hexdigest(),
        "sham": "solution.py loaded a second time; byte-identical to the parent",
        "k": 1,
        "results": results,
    }
    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
