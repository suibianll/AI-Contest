"""C30 Level-1 feasibility probe: two-fold generalization + Pareto acceptance.

Pre-registered in the execution log (C30 Level-1 entry).  The Level-0 probe
measured the mechanism ceiling in-sample (the permutation was fitted and
evaluated on the same activations).  This probe answers the deployment
questions:

1. how much of the oracle improvement survives out-of-fold (fit on fold A,
   evaluate on fold B, and vice versa, plan section 4.9);
2. the aggregate gain after per-component Pareto acceptance (section 8.3
   item 8: a component keeps the C30 permutation only if neither operand
   side degrades in either fold direction, otherwise it falls back to the
   parent permutation);
3. whether the section 8.4 gates (within-16 edge capture >= +20%, magnitude
   incompatibility penalty increase <= 5%) hold on the validation fold.

Protocol per component (calib=4: fold A = samples 0,1; fold B = 2,3; each
fold matches the deployed calibration size of 2 samples):
- for each direction (fit, eval) in [(A, B), (B, A)]:
  * parent calibration (d, permutation, block_smooth_size, seed) runs on
    the fit fold only — exactly the deployed pipeline;
  * the C30 permutation is built from the fit fold's edge utility
    edge(i, j) = |H_A^fit[i, j]| * sqrt(r_i * r_j)  (lambda = 0 per the
    Level-0 finding), where r is the parent weight-quantization residual
    per channel;
  * both arms are scored on the *validation* fold with the deployed
    quantizer harness (``_r64_operand_losses``): the second moment and the
    evaluation samples come from the eval fold, the weight sample is the
    fixed 256-row draw, losses are measured in the fixed parent frame.

Compliance: evaluator-side probe; operand-local statistics only; no Linear
output construction, no holdout access.
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
from c30_permutation_probe import (  # noqa: E402
    COMPONENTS,
    hierarchical_permutation,
    load_solution,
    within16_capture,
)
from nvfp4_sim import nvfp4_encode  # noqa: E402
from real_data_eval import collect_real_data  # noqa: E402

TOLERANCE = 0.0  # Pareto acceptance: neither side may degrade at all


def _parent_residual_channels(
    solution: ModuleType,
    weight_dense: torch.Tensor,
    d: torch.Tensor,
    parent_perm: torch.Tensor,
    size: int,
    seed: int,
) -> torch.Tensor:
    """Per-channel L2 norm of the parent weight-quantization residual."""

    weight_smooth = solution._linear_pair_transform(
        weight_dense, d, parent_perm, size, seed, weight_side=True
    )
    weight_hat = solution._dequantize_hif4(
        solution._dense_to_hif4(weight_smooth)
    )
    residual_permuted = weight_smooth - weight_hat
    residual = torch.zeros_like(residual_permuted)
    residual.index_copy_(1, parent_perm, residual_permuted)
    return residual.square().sum(dim=0).sqrt()


def _magnitude_penalty(
    act_mag: torch.Tensor, perm: torch.Tensor
) -> float:
    """Mean |log-magnitude| gap inside 16-channel windows of ``perm``."""

    channels = int(act_mag.numel())
    log_mag = act_mag.clamp_min(1e-12).log()
    ordered = log_mag.index_select(0, perm.to(log_mag.device))
    ordered = ordered.reshape(-1, 16)
    gaps = (
        ordered.unsqueeze(1) - ordered.unsqueeze(2)
    ).abs()
    n = ordered.shape[1]
    if n <= 1:
        return 0.0
    # Diagonal gaps are zero, so the full mean rescales to the off-diagonal
    # mean by n / (n - 1).
    full_mean = gaps.mean()
    return float(full_mean * n / (n - 1))


def probe_component(
    solution: ModuleType,
    weight_dense: torch.Tensor,
    fold_samples: dict[str, list[torch.Tensor]],
) -> dict:
    device = weight_dense.device
    channels = int(weight_dense.shape[1])

    weight_sample = solution._sample_rows(
        weight_dense, solution._LINEAR_WEIGHT_EVAL_ROWS
    )
    weight_pair = nvfp4_encode(weight_dense.cpu(), mode="amax6")
    weight_quant = weight_pair[0].to(device)
    weight_scale = weight_pair[1].to(device)

    directions = ("A->B", "B->A")
    report: dict = {"directions": {}, "fold_meta": {}}

    parent_state: dict[str, dict] = {}
    c30_state: dict[str, dict] = {}
    for direction in directions:
        fit_key = "A" if direction.startswith("A") else "B"
        eval_key = "B" if direction.startswith("A") else "A"
        fit_samples = fold_samples[fit_key]
        eval_samples = fold_samples[eval_key]

        calib_pairs = [
            tuple(t.to(device) for t in nvfp4_encode(a.cpu(), mode="amax6"))
            for a in fit_samples
        ]
        calibrated = solution.hif4_calibration_and_quantize_weight(
            weight_quant, weight_scale, calib_pairs
        )
        state = calibrated["activation_state"]
        d = (
            state["smooth_inv"].to(device=device, dtype=torch.float32)
            if state["smooth_inv"] is not None
            else torch.ones(channels, dtype=torch.float32, device=device)
        )
        parent_perm = (
            state["permutation"].to(device=device, dtype=torch.int64)
            if state["permutation"] is not None
            else torch.arange(channels, device=device)
        )
        size = int(state["block_smooth_size"])
        seed = int(state["block_smooth_seed"])

        # Fit-fold statistics.
        fit_rows = torch.cat(fit_samples, dim=0)
        eval_rows = torch.cat(eval_samples, dim=0)
        fit_second_moment = (
            fit_rows.square().sum(dim=0) / float(fit_rows.shape[0])
        )
        eval_second_moment = (
            eval_rows.square().sum(dim=0) / float(eval_rows.shape[0])
        )

        r = _parent_residual_channels(
            solution, weight_dense, d, parent_perm, size, seed
        )
        fit_gram = fit_rows.t() @ fit_rows / float(fit_rows.shape[0])
        edge_fit = fit_gram.abs() * torch.sqrt(
            torch.outer(r, r).clamp_min(0.0)
        )
        c30_perm = hierarchical_permutation(edge_fit, channels).to(device)

        eval_gram = eval_rows.t() @ eval_rows / float(eval_rows.shape[0])
        edge_eval = eval_gram.abs() * torch.sqrt(
            torch.outer(r, r).clamp_min(0.0)
        )

        act_mag = fit_rows.abs().amax(dim=0) / d
        penalty_parent = _magnitude_penalty(act_mag, parent_perm)
        penalty_c30 = _magnitude_penalty(act_mag, c30_perm)

        arms = {"parent": parent_perm, "c30": c30_perm}
        losses: dict[str, dict] = {}
        for name, perm in arms.items():
            weight_loss, act_losses = solution._r64_operand_losses(
                weight_sample,
                eval_second_moment,
                eval_samples,
                d,
                perm,
                size,
                seed,
            )
            losses[name] = {
                "weight_loss": float(weight_loss),
                "act_loss_mean": sum(act_losses) / len(act_losses),
                "within16_capture": within16_capture(edge_eval, perm),
            }

        improvements = {
            "weight": (
                losses["parent"]["weight_loss"]
                - losses["c30"]["weight_loss"]
            )
            / losses["parent"]["weight_loss"],
            "act": (
                losses["parent"]["act_loss_mean"]
                - losses["c30"]["act_loss_mean"]
            )
            / losses["parent"]["act_loss_mean"],
        }
        pareto = (
            losses["c30"]["weight_loss"]
            <= losses["parent"]["weight_loss"] + TOLERANCE
            and losses["c30"]["act_loss_mean"]
            <= losses["parent"]["act_loss_mean"] + TOLERANCE
        )
        report["directions"][direction] = {
            "losses": losses,
            "improvements": {k: float(v) for k, v in improvements.items()},
            "combined_improvement": float(
                improvements["weight"] + improvements["act"]
            ),
            "pareto_ok": bool(pareto),
            "capture_ratio": float(
                losses["c30"]["within16_capture"]
                / max(losses["parent"]["within16_capture"], 1e-12)
            ),
            "penalty_ratio": float(
                penalty_c30 / max(penalty_parent, 1e-12)
            ),
        }
        parent_state[direction] = {
            "block_smooth_size": size,
            "identity_parent": bool(
                torch.equal(
                    parent_perm, torch.arange(channels, device=device)
                )
            ),
        }
        c30_state[direction] = {
            "same_as_parent": bool(torch.equal(c30_perm, parent_perm)),
        }

    accepted = all(
        report["directions"][d]["pareto_ok"] for d in directions
    )
    deployed_combined = (
        sum(
            report["directions"][d]["combined_improvement"]
            for d in directions
        )
        / len(directions)
        if accepted
        else 0.0
    )
    report["fold_meta"] = {
        "parent": parent_state,
        "c30": c30_state,
    }
    report["pareto_accepted"] = bool(accepted)
    report["deployed_combined_improvement"] = float(deployed_combined)
    return report


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
    parser.add_argument("--calib", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "c30_level1_probe_results.json",
    )
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    solution = load_solution(args.solution)
    layer_indices = [int(x) for x in args.layers.split(",")]

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
        for name in COMPONENTS:
            weight_dense = solution._dequantize_nvfp4_float32(
                *nvfp4_encode(weights[layer][name], "amax6")
            ).to(device=device, dtype=torch.float32)
            act_samples = [
                solution._dequantize_nvfp4_float32(
                    *nvfp4_encode(
                        calibration["act"][name][batch * layer_count + layer],
                        "amax6",
                    )
                ).to(device=device, dtype=torch.float32)
                for batch in range(args.calib)
            ]
            half = args.calib // 2
            fold_samples = {
                "A": act_samples[:half],
                "B": act_samples[half:],
            }
            t0 = time.perf_counter()
            entry = probe_component(solution, weight_dense, fold_samples)
            entry["layer"] = layer
            entry["component"] = name
            entry["seconds"] = time.perf_counter() - t0
            results.append(entry)
            a2b = entry["directions"]["A->B"]
            b2a = entry["directions"]["B->A"]
            print(
                f"layer {layer:2d} {name:4s}"
                f" | A->B w {a2b['improvements']['weight']*100:+6.2f}%"
                f" a {a2b['improvements']['act']*100:+6.2f}%"
                f" cap x{a2b['capture_ratio']:.2f} pen x{a2b['penalty_ratio']:.2f}"
                f" | B->A w {b2a['improvements']['weight']*100:+6.2f}%"
                f" a {b2a['improvements']['act']*100:+6.2f}%"
                f" cap x{b2a['capture_ratio']:.2f} pen x{b2a['penalty_ratio']:.2f}"
                f" | accepted={entry['pareto_accepted']}"
                f" deployed {entry['deployed_combined_improvement']*100:+6.2f}%"
                f" [{entry['seconds']:.1f}s]",
                flush=True,
            )

    def mean(values: list[float]) -> float:
        return sum(values) / max(len(values), 1)

    def direction_improvements(direction: str) -> tuple[float, float]:
        w, a = [], []
        for r in results:
            d = r["directions"][direction]
            w.append(d["improvements"]["weight"])
            a.append(d["improvements"]["act"])
        return mean(w), mean(a)

    w_ab, a_ab = direction_improvements("A->B")
    w_ba, a_ba = direction_improvements("B->A")

    accepted = [r for r in results if r["pareto_accepted"]]
    accepted_combined = [
        r["deployed_combined_improvement"] for r in accepted
    ]
    # Sign consistency across fold directions, measured over ALL components
    # (the accepted-only version is vacuous: Pareto acceptance already
    # forces non-negative combined improvement in both directions).
    sign_consistent = 0
    for r in results:
        ab = r["directions"]["A->B"]["combined_improvement"]
        ba = r["directions"]["B->A"]["combined_improvement"]
        if (ab >= 0.0) == (ba >= 0.0):
            sign_consistent += 1

    capture_ratios = []
    penalty_ratios = []
    for r in results:
        for d in ("A->B", "B->A"):
            capture_ratios.append(r["directions"][d]["capture_ratio"])
            penalty_ratios.append(r["directions"][d]["penalty_ratio"])

    acceptance_rate = len(accepted) / max(len(results), 1)
    deployed_mean = mean(
        [r["deployed_combined_improvement"] for r in results]
    )
    accepted_mean = mean(accepted_combined)

    gates = {
        "both_directions_positive": bool(
            (w_ab + a_ab) > 0.0 and (w_ba + a_ba) > 0.0
        ),
        "acceptance_ge_75pct": bool(acceptance_rate >= 0.75),
        "accepted_mean_combined_ge_10pct": bool(accepted_mean >= 0.10),
        "capture_ratio_ge_1p20": bool(mean(capture_ratios) >= 1.20),
        "penalty_le_1p05": bool(mean(penalty_ratios) <= 1.05),
        "sign_consistency_ge_80pct": bool(
            (sign_consistent / max(len(results), 1)) >= 0.80
        ),
    }
    verdict = (
        "PASS (proceed to full implementation)"
        if all(gates.values())
        else "VETO (C30 rejected at Level-1)"
    )
    if all(gates.values()) and accepted_mean < 0.16:
        verdict = "WEAK PASS (accepted mean < 16%; ablation required)"

    summary = {
        "direction_means": {
            "A->B": {"weight": w_ab, "act": a_ab, "combined": w_ab + a_ab},
            "B->A": {"weight": w_ba, "act": a_ba, "combined": w_ba + a_ba},
        },
        "acceptance_rate": acceptance_rate,
        "accepted_count": len(accepted),
        "total_count": len(results),
        "accepted_mean_combined": accepted_mean,
        "deployed_mean_combined": deployed_mean,
        "mean_capture_ratio": mean(capture_ratios),
        "mean_penalty_ratio": mean(penalty_ratios),
        "sign_consistency": sign_consistent / max(len(results), 1),
        "gates": gates,
        "verdict": verdict,
    }

    payload = {"results": results, "summary": summary}
    args.output.write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print()
    print(f"direction means: A->B {w_ab*100:+.2f}%/{a_ab*100:+.2f}%"
          f"  B->A {w_ba*100:+.2f}%/{a_ba*100:+.2f}% (w/a)")
    print(
        f"acceptance {len(accepted)}/{len(results)}"
        f" ({acceptance_rate*100:.0f}%), accepted mean combined"
        f" {accepted_mean*100:+.2f}%, deployed mean {deployed_mean*100:+.2f}%"
    )
    print(
        f"capture x{mean(capture_ratios):.3f}  penalty x{mean(penalty_ratios):.3f}"
        f"  sign-consistency {summary['sign_consistency']*100:.0f}%"
    )
    for key, value in gates.items():
        print(f"  gate {key}: {'PASS' if value else 'FAIL'}")
    print(f"VERDICT: {verdict}")
    print(f"elapsed {time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
