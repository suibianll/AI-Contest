"""Level-0 S-grid oracle probe for C29 HAES.

Post-C21C optimization plan §4.6 "Level 0": before any coordinate-descent
search is implemented, answer the mechanism-ceiling question with a cheap
oracle — per-group independent enumeration of the S discrete grid
(z in {-4..4}, s = 2^(z/8)), each group taking the z that minimizes the
activation hard reconstruction error, with codes re-adapted by the
deployed dynamic quantization path (`_nvfp4_to_hif4`, only the folded
`multiplier` differs).  Single-sided (activation only), no Pareto gating,
no centering — a generous upper bound on what C29's constrained search
could achieve.

Compliance: evaluator-side probe; uses only the activation's own
reconstruction error.  Never builds a Linear output, never reads holdout.

Variant calls go through the deployed path one calibration sample at a
time (refinement ranking is global within a call, so variants must not be
flattened/batched).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import ModuleType

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nvfp4_sim import nvfp4_encode  # noqa: E402
from real_data_eval import collect_real_data  # noqa: E402

Z_LEVELS = tuple(range(-4, 5))
COMPONENTS = ("q", "k", "v", "o", "fc", "proj")


def load_solution(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("_hif4_probe_solution", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load solution: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _scatter_scale(
    s_permuted: torch.Tensor,
    permutation: torch.Tensor | None,
    channels: int,
) -> torch.Tensor:
    """Scatter a permuted-coordinate diagonal back to original coordinates."""

    if permutation is None:
        return s_permuted
    s_original = torch.ones(channels, dtype=torch.float32)
    s_original[permutation] = s_permuted
    return s_original


@torch.no_grad()
def probe_component(
    solution: ModuleType,
    weight_pair,
    calib_pairs,
    device: torch.device,
) -> dict:
    calibrated = solution.hif4_calibration_and_quantize_weight(
        weight_pair[0], weight_pair[1], calib_pairs
    )
    state = calibrated["activation_state"]
    channels = int(state["in_features"])
    smooth_inv = state["smooth_inv"]  # [C] cpu float32 or None
    permutation = state["permutation"]  # cpu int64 or None
    size = int(state["block_smooth_size"])
    seed = int(state["block_smooth_seed"])
    importance = state["importance"]
    gram = state.get("gram")
    gram8 = state.get("gram8")
    offsets = state["offsets"]
    error_threshold = float(state["error_threshold"])
    accept_margin = float(state["accept_margin"])
    max_refine_ratio = float(state["max_refine_ratio"])
    max_refine_blocks = int(state["max_refine_blocks"])

    identity = torch.ones(channels, dtype=torch.float32)
    base_inv = smooth_inv if smooth_inv is not None else identity

    def deployed(multiplier, pair):
        return solution._nvfp4_to_hif4(
            pair[0],
            pair[1],
            multiplier=multiplier,
            permutation=permutation,
            block_smooth_size=size,
            block_smooth_seed=seed,
            importance=importance,
            group_gram=gram,
            group_gram8=gram8,
            search_offsets=offsets,
            error_threshold=error_threshold,
            accept_margin=accept_margin,
            max_refine_ratio=max_refine_ratio,
            max_refine_blocks=max_refine_blocks,
        )

    # Dense activations, pre-R (after D and P) and post-R references.
    acts = [
        solution._dequantize_nvfp4_float32(*p).to(
            device=device, dtype=torch.float32
        )
        for p in calib_pairs
    ]
    d = base_inv.reciprocal().to(device)
    pres = []
    for a in acts:
        a1 = a * d.reshape(1, -1)
        if permutation is not None:
            a1 = a1.index_select(-1, permutation.to(device))
        pres.append(a1)

    def post_transform(pre: torch.Tensor) -> torch.Tensor:
        if size == 0:
            return pre
        return solution._block_hadamard_transform(pre, size, seed)

    posts = [post_transform(p) for p in pres]

    # Parent losses (all samples) + per-group errors on the search sample.
    parent_loss = 0.0
    search_group_err: dict[int, torch.Tensor] = {}
    for index, (pair, post) in enumerate(zip(calib_pairs, posts)):
        hat = solution._dequantize_hif4(deployed(base_inv, pair)).to(
            torch.float32
        )
        err = (post - hat.to(post.device)).square()
        parent_loss += float(err.sum())
        if index == 0:
            for s in (4, 8, 16):
                if channels % s == 0:
                    search_group_err[s] = err.reshape(
                        -1, channels // s, s
                    ).sum(dim=(0, 2)).cpu()

    def run_arm(s_size: int) -> dict:
        groups = channels // s_size
        best_z = torch.zeros(groups, dtype=torch.int64)
        predicted = search_group_err[s_size].clone()
        pair0 = calib_pairs[0]
        pre0 = pres[0]
        for g in range(groups):
            start = g * s_size
            best_err = float(search_group_err[s_size][g])
            bz = 0
            for z in Z_LEVELS:
                if z == 0:
                    continue
                s_permuted = torch.ones(channels, dtype=torch.float32)
                s_permuted[start : start + s_size] = 2.0 ** (z / 8.0)
                s_original = _scatter_scale(
                    s_permuted, permutation, channels
                )
                multiplier = base_inv / s_original
                params = deployed(multiplier, pair0)
                hat_post = solution._dequantize_hif4(params).to(torch.float32)
                # Map the reconstruction back to the parent pre-R frame:
                # R is self-inverse, then multiply by S to undo the candidate
                # scaling.  Measuring in the parent frame removes the pure
                # rescaling artifact (scaling a group by alpha shrinks its
                # S-frame error energy by alpha^2 without any real gain).
                hat_pre = post_transform(hat_post.to(pres[0].device))
                hat_parent = hat_pre * s_permuted.to(hat_pre.device).reshape(
                    1, -1
                )
                err = float(
                    (hat_parent[:, start : start + s_size] - pre0[:, start : start + s_size])
                    .square()
                    .sum()
                )
                if err < best_err:
                    best_err = err
                    bz = z
            best_z[g] = bz
            predicted[g] = best_err

        # Simultaneous application of all selected z values.
        s_permuted = torch.ones(channels, dtype=torch.float32)
        nonzero = 0
        for g in range(groups):
            z = int(best_z[g])
            if z != 0:
                start = g * s_size
                s_permuted[start : start + s_size] = 2.0 ** (z / 8.0)
                nonzero += 1
        s_original = _scatter_scale(s_permuted, permutation, channels)
        multiplier = base_inv / s_original
        sim_loss = 0.0
        # Parent-frame measurement (see run_arm loop note): map the
        # reconstruction back through R^-1 and S before comparing with the
        # parent pre-R activation, so pure group rescaling gains nothing.
        s_diag = s_permuted.to(device).reshape(1, -1)
        for pair, pre in zip(calib_pairs, pres):
            params = deployed(multiplier, pair)
            hat_post = solution._dequantize_hif4(params).to(torch.float32)
            hat_pre = post_transform(hat_post.to(pre.device))
            hat_parent = hat_pre * s_diag
            sim_loss += float((hat_parent - pre).square().sum())
        return {
            "s_size": s_size,
            "groups": groups,
            "nonzero_groups": nonzero,
            "predicted_loss": float(predicted.sum()),
            "simultaneous_loss": sim_loss,
            "reduction": 1.0 - sim_loss / max(parent_loss, 1e-30),
            "best_z": best_z.tolist(),
        }

    arms = {}
    bound_size = size if size != 0 else 4
    arms["bound"] = run_arm(bound_size)
    if bound_size != 4:
        arms["s4"] = run_arm(4)
    return {
        "block_smooth_size": size,
        "block_smooth_seed": seed,
        "channels": channels,
        "parent_loss": parent_loss,
        "arms": arms,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--solution",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "solution.py",
    )
    parser.add_argument(
        "--model",
        default=str(Path(__file__).resolve().parents[1] / "models" / "gpt2"),
    )
    parser.add_argument("--layers", default="0,5,11")
    parser.add_argument("--seq", type=int, default=128)
    parser.add_argument("--calib", type=int, default=2)
    parser.add_argument("--mode", default="amax6")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--components", default=",".join(COMPONENTS), help="subset to probe"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent
        / "hierarchy_scale_probe_results.json",
    )
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    solution = load_solution(args.solution)
    layer_indices = [int(x) for x in args.layers.split(",")]
    components = tuple(x.strip() for x in args.components.split(","))

    model, weights, calibration, _tests, _heads, _dim = collect_real_data(
        args.model,
        layers=max(layer_indices) + 1,
        sequence_length=args.seq,
        calibration_samples=args.calib,
        test_samples=1,
        device="cpu",
    )
    del model
    layer_count = max(layer_indices) + 1

    results = []
    started = time.perf_counter()
    for layer in layer_indices:
        for name in components:
            if name not in COMPONENTS:
                raise ValueError(f"unknown component: {name}")
            weight_pair = nvfp4_encode(weights[layer][name], args.mode)
            calib_pairs = [
                nvfp4_encode(
                    calibration["act"][name][batch * layer_count + layer],
                    args.mode,
                )
                for batch in range(args.calib)
            ]
            # Keep the deployed encoder on the probe device.
            weight_pair = tuple(t.to(device) for t in weight_pair)
            calib_pairs = [
                tuple(t.to(device) for t in p) for p in calib_pairs
            ]
            t0 = time.perf_counter()
            entry = probe_component(solution, weight_pair, calib_pairs, device)
            entry["layer"] = layer
            entry["component"] = name
            entry["seconds"] = time.perf_counter() - t0
            results.append(entry)
            bound = entry["arms"]["bound"]
            print(
                f"layer {layer:2d} {name:4s} block={entry['block_smooth_size']:2d}"
                f" parent={entry['parent_loss']:.4e}"
                f" bound(S{bound['s_size']}) red={bound['reduction']*100:6.2f}%"
                f" nz={bound['nonzero_groups']}/{bound['groups']}"
                f" [{entry['seconds']:.1f}s]",
                flush=True,
            )

    total_parent = sum(r["parent_loss"] for r in results)
    total_sim = sum(
        r["arms"]["bound"]["simultaneous_loss"] for r in results
    )
    summary = {
        "total_parent_loss": total_parent,
        "total_bound_simultaneous_loss": total_sim,
        "total_bound_reduction": 1.0 - total_sim / max(total_parent, 1e-30),
        "veto_gate": 0.05,
        "verdict": (
            "PASS (proceed to Level-1 probe)"
            if 1.0 - total_sim / max(total_parent, 1e-30) >= 0.05
            else "VETO (C29 main mechanism failed, go to C30)"
        ),
        "buckets": {},
    }
    for r in results:
        key = str(r["block_smooth_size"])
        bucket = summary["buckets"].setdefault(
            key,
            {"parent": 0.0, "sim": 0.0, "count": 0},
        )
        bucket["parent"] += r["parent_loss"]
        bucket["sim"] += r["arms"]["bound"]["simultaneous_loss"]
        bucket["count"] += 1
    for key, bucket in summary["buckets"].items():
        bucket["reduction"] = 1.0 - bucket["sim"] / max(bucket["parent"], 1e-30)

    payload = {"results": results, "summary": summary}
    args.output.write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(
        f"\nTOTAL bound-arm energy reduction: "
        f"{summary['total_bound_reduction']*100:.2f}% "
        f"(veto gate 5%) -> {summary['verdict']}"
    )
    for key in sorted(summary["buckets"]):
        bucket = summary["buckets"][key]
        print(
            f"  bucket block_size={key}: n={bucket['count']}"
            f" reduction={bucket['reduction']*100:.2f}%"
        )
    print(f"elapsed {time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
