"""Isolate the L-EM2 calibration hook so its cost is measurable at all.

``time_calibration.py`` pairs whole calibration calls, and a parent call is
2-10 s long there, so a 10-20 ms hook is buried: its last run put the in=2560
delta at -52 ms and the in=4096 delta at +24 ms, i.e. one noise-dominated number
and one real one.  The hook's *own* cost is what the projection needs, so this
probe calls ``_em1_compile_metric`` directly on a real shard state -- same
weights, same calibration activations, same ``h_inv`` the parent produced --
with the parent excluded entirely.

Verified alongside each timing: the hook's stored payload still reproduces the
metric (control C's check, inlined) so this is timing the real thing.

Read-only probe on the real shard cache; prints only.
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "evaluator"))
sys.path.insert(0, str(HERE / "candidate"))

import official_eval as v2  # noqa: E402

CANDIDATE = HERE / "candidate" / "solution.py"
CACHE = (
    ROOT
    / "artifacts/official_eval/cache/proxy-v3-calibration"
    / "56dc805d6e5a3aef-linear-452e6fe4aa5747590535.pt"
)
PACK = ROOT / "artifacts/official_eval/cache/qwen3.5-4b-proxy-v2.pt"
CASES = ((0, "q"), (0, "o"))
REPEATS = 12
OFFICIAL_BY_CHANNELS = {2560: 120, 4096: 24}


def main() -> int:
    device = torch.device("cuda")
    candidate = v2.load_solution(CANDIDATE)
    payload = torch.load(CACHE, map_location="cpu", mmap=True, weights_only=False)
    pack = torch.load(PACK, map_location="cpu", mmap=True, weights_only=False)

    per_channels: dict[int, float] = {}
    for layer, role in CASES:
        entry = next(
            item
            for item in payload["weight_states"]
            if int(item["layer"]) == layer and str(item["role"]) == role
        )
        channels = int(entry["state"]["in_features"])
        weight_quant, weight_scale = v2._pair(
            pack["weights"][layer][role].to(torch.float32)
        )
        weight_quant = weight_quant.to(device)
        weight_scale = weight_scale.to(device)
        # Same shape the parent returns: the calibrated activation state, the
        # deployed HiF4 weight params, and the ridge scalar's raw material.
        result = {
            "activation_state": dict(entry["state"]),
            "weight_params": entry["params"],
        }

        def compile_once():
            result["activation_state"].pop("em1", None)
            diagnostics: dict = {"em1_arm": "unavailable"}
            candidate._em1_compile_metric(weight_quant, weight_scale, result, diagnostics)
            return diagnostics

        compile_once()
        torch.cuda.synchronize(device)
        samples = []
        for _ in range(REPEATS):
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            diagnostics = compile_once()
            torch.cuda.synchronize(device)
            samples.append(time.perf_counter() - started)
        # _em1_compile_metric reports through its diagnostics dict; only the
        # wrapper folds those into the state, and the wrapper is not under test.
        diagnostics = compile_once()
        compiled = result["activation_state"]
        arm = diagnostics.get("em1_arm")
        if arm != "compiled" or "em1" not in compiled:
            raise SystemExit(f"layer{layer}/{role}: arm is {arm!r}, no payload stored")

        # The hook is cheap only if it is still correct: rebuild G from the
        # payload exactly as the dynamic path does and compare to the Gram.
        h_inv = compiled["h_inv"].to(device=device, dtype=torch.float32)
        inverse = torch.cholesky_inverse(torch.linalg.cholesky(h_inv))
        ridge = float(inverse.diagonal().mean()) - float(compiled["em1"]["gram_diag_mean"])
        metric = inverse
        metric.diagonal().sub_(ridge)  # same in-place form as the dynamic path
        params = {key: value.to(device) for key, value in entry["params"].items()}
        logical_shape = v2.dequantize_nvfp4(weight_quant, weight_scale).shape
        deployed = v2.dequantize_hif4(params, logical_shape).to(torch.float32)
        gram = deployed.transpose(0, 1).mm(deployed)
        g_rel = float(
            (metric - gram).norm() / max(float(gram.norm()), 1e-30)
        )

        median = statistics.median(samples)
        per_channels[channels] = median
        print(
            f"layer{layer}/{role} in={channels} rows={deployed.shape[0]} "
            f"median={median * 1e3:.2f} ms min={min(samples) * 1e3:.2f} ms "
            f"max={max(samples) * 1e3:.2f} ms | G_rel={g_rel:.3e} "
            f"arm={arm}",
            flush=True,
        )
        if g_rel > 1e-3:
            raise SystemExit(f"layer{layer}/{role}: G recovery broke ({g_rel:.3e})")

    projected = sum(
        per_channels[channels] * count
        for channels, count in OFFICIAL_BY_CHANNELS.items()
    )
    print(
        "\n[compile hook] "
        + ", ".join(f"in={ch} {value * 1e3:.2f} ms/call" for ch, value in per_channels.items())
        + f"\n  projected added on {sum(OFFICIAL_BY_CHANNELS.values())} official "
        f"in-scope calibrations: {projected:+.2f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
