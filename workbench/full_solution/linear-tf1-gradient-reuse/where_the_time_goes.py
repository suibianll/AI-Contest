"""Where does one dynamic descent call actually spend its time?

L-TF1 removes two matrix products per call.  Whether that is worth anything
depends entirely on what share of the call those two products are, and the card
cannot answer that from an operator count.  This script prices the parts
separately, on the same device and the same shapes the real calls use.

It is a *replica*: it re-runs the descent's own operations in the descent's own
order rather than instrumenting the shipped function, so nothing here can
perturb the code under test.  Each part is timed with CUDA events, warm, and
reported as a median over several rounds.

Parts, in the order the descent performs them:

  metric        h_inv -> cholesky -> cholesky_inverse -> ridge subtraction
  reference     the nvfp4 decode of the activation
  gradient      the two mm products L-TF1 removes one of (x2 in the parent)
  group_gram    the (blocks,16,4,blocks,16,4) view followed by the advanced
                index that selects the within-group 4x4 blocks
  steps         the sixteen row_delta.mm(metric) products of the step loop

The group_gram line is the one worth looking at: the view is free, but the
advanced index broadcasts to (blocks,16,4,blocks,16,4) = 6.5M elements to
select 10240 of them.
"""

from pathlib import Path
import json
import statistics
import sys

import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CACHE_DIR = ROOT / "artifacts/official_eval/cache/proxy-v3-calibration"
PACK = ROOT / "artifacts/official_eval/cache/qwen3.5-4b-proxy-v2.pt"

PARENT_SHA_PREFIX = "0f1af6dbc207ff32"
BLOCK = 64
GROUPS = 16
ROUNDS = 21
WARMUP = 3


def find_state(layer: int, role: str) -> dict:
    for path in sorted(CACHE_DIR.glob(f"{PARENT_SHA_PREFIX}-linear-*.pt")):
        payload = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
        for entry in payload["weight_states"]:
            if int(entry["layer"]) == layer and str(entry["role"]) == role:
                del payload
                return dict(entry["state"])
    raise SystemExit(f"no cached state for layer {layer} / role {role}")


def timeit(device, rounds, function):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(WARMUP):
        function()
    torch.cuda.synchronize()
    samples = []
    for _ in range(rounds):
        start.record()
        function()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    samples.sort()
    return {
        "median_ms": statistics.median(samples),
        "min_ms": samples[0],
        "max_ms": samples[-1],
    }


def main() -> int:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    device = torch.device("cuda")

    sys.path.insert(0, str(ROOT / "evaluator"))
    import official_eval as evaluator  # noqa: PLC0415
    import importlib.util  # noqa: PLC0415

    spec = importlib.util.spec_from_file_location("wtt_solution", ROOT / "solution.py")
    solution = importlib.util.module_from_spec(spec)
    sys.modules["wtt_solution"] = solution
    spec.loader.exec_module(solution)

    pack = torch.load(PACK, map_location="cpu", mmap=True, weights_only=False)
    quant_cpu, scale_cpu = evaluator._pair(
        pack["test_activations"]["q"][1][0].to(torch.float32)
    )

    report = {"device": torch.cuda.get_device_name(0), "rounds": ROUNDS, "states": []}
    for layer, rows in ((0, 128), (0, 512)):
        state = find_state(layer, "q")
        h_inv = state["h_inv"].to(device=device, dtype=torch.float32)
        channels = int(state["in_features"])
        blocks = channels // BLOCK

        # The pack carries 128 rows per window; the larger case repeats them,
        # which changes the values the arithmetic sees but not how much there is.
        repeat = -(-rows // quant_cpu.shape[0])
        quant = quant_cpu.repeat(repeat, 1)[:rows].contiguous().to(device)
        scale = scale_cpu.repeat(repeat, 1)[:rows].contiguous().to(device)
        # The real decoded reference, so the products carry real values rather
        # than denormals or zeros that a GPU might shortcut.
        reference = solution._dequantize_nvfp4_float32(quant, scale).to(
            device=device, dtype=torch.float32
        )

        # The real metric, built once exactly as the descent builds it: the
        # inverse of h_inv with the ridge taken off the diagonal.
        metric = torch.cholesky_inverse(torch.linalg.cholesky(h_inv))
        ridge = float(metric.diagonal().mean()) - float(state["em1"]["gram_diag_mean"])
        metric.diagonal().sub_(ridge)

        parts = {}
        parts["metric_cholesky_inverse"] = timeit(
            device,
            ROUNDS,
            lambda: torch.cholesky_inverse(torch.linalg.cholesky(h_inv)),
        )

        def build_group_gram():
            metric6 = metric.reshape(blocks, GROUPS, 4, blocks, GROUPS, 4)
            block_index = torch.arange(blocks, device=device).reshape(blocks, 1, 1, 1)
            group_index = torch.arange(GROUPS, device=device).reshape(1, GROUPS, 1, 1)
            return metric6[block_index, group_index, :, block_index, group_index, :].reshape(
                blocks, GROUPS, 4, 4
            )

        parts["group_gram_index"] = timeit(device, ROUNDS, build_group_gram)

        deployed = torch.zeros(rows, channels, device=device)

        def gradient_once():
            return (deployed - reference).mm(metric) + reference.mm(metric)

        parts["gradient_two_mm"] = timeit(device, ROUNDS, gradient_once)

        row_delta = torch.zeros(rows, channels, device=device)

        def step_mm():
            return row_delta.mm(metric)

        parts["one_step_mm"] = timeit(device, ROUNDS, step_mm)

        def sixteen_steps():
            for _ in range(GROUPS):
                row_delta.mm(metric)

        parts["sixteen_step_mm"] = timeit(device, ROUNDS, sixteen_steps)

        record = {
            "layer": layer,
            "role": "q",
            "rows": rows,
            "channels": channels,
            "parts": parts,
            "share_of_steps_plus_gradient": {
                name: parts[name]["median_ms"]
                / (
                    parts["gradient_two_mm"]["median_ms"]
                    + parts["sixteen_step_mm"]["median_ms"]
                )
                for name in ("gradient_two_mm", "sixteen_step_mm")
            },
        }
        report["states"].append(record)
        print(f"\n=== layer{layer}/q  rows={rows}  channels={channels} ===")
        for name, values in parts.items():
            print(
                f"  {name:>26}  {values['median_ms']:9.4f} ms   "
                f"[{values['min_ms']:.4f}, {values['max_ms']:.4f}]"
            )

    (HERE / "where_the_time_goes.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print("\nwrote where_the_time_goes.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
