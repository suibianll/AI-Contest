"""Split the L-EM2 calibration hook's ~41 ms into its parts, on real states.

``profile_compile_hook.py`` established *that* the hook costs 41.31 ms (in=2560)
and 52.04 ms (in=4096), which over the official 144 in-scope calibrations is
+5.8 to +6.2 s -- the largest single remaining cost in the shipped v230 root,
and the one that caps how many groupstep passes K the 300 s gate can afford.

It does not say *where* those milliseconds go, and the obvious guesses do not
survive arithmetic: two 2560x2560 fp32 matmuls at rows=512 are ~13 GFLOP, a few
milliseconds at best on this GPU, and the n^2 traffic is ~130 MB.  So this probe
times every component of the deployed hook separately, plus the algebraic
rewrite that would replace two n^2 matmuls and an n^2 subtraction with one
matmul and two vectors:

    gram - cross = (W_hat - W)^T W_hat        (one matmul, no n^2 gram)
    mean(diag(gram)) = mean(W_hat^2 over rows)(no n^2 gram either)

Read-only CPU/GPU probe on the real shard cache; prints only.  It exists to
decide whether a metric-reclaim card is worth opening, so it deliberately does
not write any state or touch the candidate.
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

import official_eval as v2  # noqa: E402

CANDIDATE = (
    ROOT / "workbench/full_solution/linear-em3-k2-arm/candidate/solution.py"
)
CACHE = (
    ROOT
    / "artifacts/official_eval/cache/proxy-v3-calibration"
    / "56dc805d6e5a3aef-linear-452e6fe4aa5747590535.pt"
)
PACK = ROOT / "artifacts/official_eval/cache/qwen3.5-4b-proxy-v2.pt"
CASES = ((0, "q"), (0, "o"))
REPEATS = 12
OFFICIAL_BY_CHANNELS = {2560: 120, 4096: 24}


def bench(fn, device, repeats: int = REPEATS) -> float:
    fn()
    torch.cuda.synchronize(device)
    samples = []
    for _ in range(repeats):
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        fn()
        torch.cuda.synchronize(device)
        samples.append(time.perf_counter() - started)
    return statistics.median(samples)


diagnostics_box: dict = {}


def _lean_inline(candidate, weight_quant, weight_scale, result, diagnostics) -> None:
    """The rewritten hook: one matmul, a column-sum diagonal, no diagnostics."""

    state = result["activation_state"]
    state.pop("em1", None)
    params = result["weight_params"]
    dense = candidate._dequantize_nvfp4_float32(weight_quant, weight_scale).to(
        torch.float32
    )
    deployed = candidate._dequantize_hif4(params).to(
        device=dense.device, dtype=torch.float32
    )
    h_matrix = (deployed - dense).transpose(0, 1).mm(deployed)
    state["em1"] = {
        "h": candidate._cpu_state_tensor(h_matrix.contiguous()),
        "gram_diag_mean": float(deployed.square().sum(dim=0).mean()),
        "channels": int(deployed.shape[1]),
        "version": 2,
    }


def _no_diagnostics(candidate, weight_quant, weight_scale, result, diagnostics) -> None:
    """The deployed hook with its two n^2 diagnostic scans removed.

    ``_em1_compile_metric`` reports ``em1_h_norm`` via ``float(h.norm())`` and
    guards with ``torch.isfinite(h).all()`` -- both read the whole n^2 matrix
    and both force a device sync.  This variant patches them out by calling a
    copy of the body with those two lines dropped, so the difference is exactly
    what the diagnostics cost.
    """

    state = result["activation_state"]
    state.pop("em1", None)
    params = result["weight_params"]
    dense = candidate._dequantize_nvfp4_float32(weight_quant, weight_scale).to(
        torch.float32
    )
    deployed = candidate._dequantize_hif4(params).to(
        device=dense.device, dtype=torch.float32
    )
    gram = deployed.transpose(0, 1).mm(deployed)
    cross = dense.transpose(0, 1).mm(deployed)
    h_matrix = (gram - cross).to(torch.float32)
    state["em1"] = {
        "h": candidate._cpu_state_tensor(h_matrix.contiguous()),
        "gram_diag_mean": float(gram.diagonal().mean()),
        "channels": int(deployed.shape[1]),
        "version": 2,
    }


def main() -> int:
    device = torch.device("cuda")
    candidate = v2.load_solution(CANDIDATE)
    payload = torch.load(CACHE, map_location="cpu", mmap=True, weights_only=False)
    pack = torch.load(PACK, map_location="cpu", mmap=True, weights_only=False)

    totals: dict[int, dict[str, float]] = {}
    for layer, role in CASES:
        entry = next(
            item
            for item in payload["weight_states"]
            if int(item["layer"]) == layer and str(item["role"]) == role
        )
        weight_quant, weight_scale = v2._pair(
            pack["weights"][layer][role].to(torch.float32)
        )
        weight_quant = weight_quant.to(device)
        weight_scale = weight_scale.to(device)
        channels = int(entry["state"]["in_features"])
        params = {key: value.to(device) for key, value in entry["params"].items()}
        rows = int(params["mant"].shape[0])

        print(f"\n=== layer{layer}/{role}  in={channels} rows={rows} ===", flush=True)

        dense = candidate._dequantize_nvfp4_float32(weight_quant, weight_scale).to(
            torch.float32
        )
        deployed = candidate._dequantize_hif4(params).to(
            device=dense.device, dtype=torch.float32
        )
        gram = deployed.transpose(0, 1).mm(deployed)
        cross = dense.transpose(0, 1).mm(deployed)
        h_matrix = gram - cross

        parts = {
            "dequantize_nvfp4_float32": lambda: candidate._dequantize_nvfp4_float32(
                weight_quant, weight_scale
            ).to(torch.float32),
            "dequantize_hif4": lambda: candidate._dequantize_hif4(params).to(
                device=device, dtype=torch.float32
            ),
            "gram = D^T D  (n^2 matmul)": lambda: deployed.transpose(0, 1).mm(deployed),
            "cross = W^T D (n^2 matmul)": lambda: dense.transpose(0, 1).mm(deployed),
            "h = gram - cross (n^2 sub)": lambda: gram - cross,
            "isfinite(h).all() (n^2 scan)": lambda: bool(
                torch.isfinite(h_matrix).all()
            ),
            "float(h.norm()) (n^2 scan)": lambda: float(h_matrix.norm()),
            "gram.diagonal().mean()": lambda: float(gram.diagonal().mean()),
            "h.to(cpu) D2H (n^2 copy)": lambda: candidate._cpu_state_tensor(
                h_matrix.contiguous()
            ),
            "--- rewrite: diff = D - W (r x n)": lambda: deployed - dense,
            "--- rewrite: diff^T D (one matmul)": lambda: (deployed - dense)
            .transpose(0, 1)
            .mm(deployed),
            "--- rewrite: mean(D^2 colsum)": lambda: float(
                deployed.square().sum(dim=0).mean()
            ),
        }
        measured = {name: bench(fn, device) for name, fn in parts.items()}

        result = {
            "activation_state": dict(entry["state"]),
            "weight_params": entry["params"],
        }

        def current_hook():
            result["activation_state"].pop("em1", None)
            diagnostics: dict = {"em1_arm": "unavailable"}
            candidate._em1_compile_metric(
                weight_quant, weight_scale, result, diagnostics
            )

        full = bench(current_hook, device)

        # Ablations of the deployed hook.  Summing the parts above accounts for
        # only ~15 ms of the ~38 ms the whole hook takes, so the parts miss
        # something -- most plausibly fresh n^2 allocations and the syncs the
        # diagnostics force.  These variants subtract one stage at a time from
        # the real function to find it.  They change the result, which is fine:
        # this is a timing probe, not a correctness one.
        lean_inline = bench(
            lambda: _lean_inline(
                candidate, weight_quant, weight_scale, result, diagnostics_box
            ),
            device,
        )
        no_diag = bench(
            lambda: _no_diagnostics(
                candidate, weight_quant, weight_scale, result, diagnostics_box
            ),
            device,
        )
        print(f"  {'ABLATION lean (1 matmul + colsum, no diag)':<40} {lean_inline * 1e3:8.2f} ms"
              f" {lean_inline * 120 * 1e3:10.0f} ms")
        print(f"  {'ABLATION current minus n^2 diagnostics':<40} {no_diag * 1e3:8.2f} ms"
              f" {no_diag * 120 * 1e3:10.0f} ms")

        print(f"  {'component':<40} {'median':>10}  {'x144 official':>12}")
        for name, seconds in measured.items():
            print(
                f"  {name:<40} {seconds * 1e3:8.2f} ms {seconds * 120 * 1e3:10.0f} ms"
            )
        print(f"  {'FULL hook (deployed)':<40} {full * 1e3:8.2f} ms {full * 120 * 1e3:10.0f} ms")

        # Correctness of the rewrite: same metric up to fp32 reassociation.
        rewritten = (deployed - dense).transpose(0, 1).mm(deployed)
        rel = float((rewritten - h_matrix).norm() / max(float(h_matrix.norm()), 1e-30))
        rewritten_diag = float(deployed.square().sum(dim=0).mean())
        diag_rel = abs(rewritten_diag - float(gram.diagonal().mean())) / max(
            abs(float(gram.diagonal().mean())), 1e-30
        )
        print(f"  rewrite |dH|/|H| = {rel:.3e}   diag rel = {diag_rel:.3e}")
        totals[channels] = {
            "full": full,
            "dequant": measured["dequantize_nvfp4_float32"]
            + measured["dequantize_hif4"],
            "matmuls": measured["gram = D^T D  (n^2 matmul)"]
            + measured["cross = W^T D (n^2 matmul)"],
            "scans": measured["isfinite(h).all() (n^2 scan)"]
            + measured["float(h.norm()) (n^2 scan)"],
            "d2h": measured["h.to(cpu) D2H (n^2 copy)"],
        }

    print("\n=== weighted over the official 144 in-scope calibrations ===", flush=True)
    for key in ("full", "dequant", "matmuls", "scans", "d2h"):
        seconds = sum(
            totals[channels][key] * OFFICIAL_BY_CHANNELS[channels]
            for channels in totals
        )
        print(f"  {key:<10} {seconds:6.2f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
