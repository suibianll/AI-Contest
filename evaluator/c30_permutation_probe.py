"""C30 Level-0 oracle probe: Hessian-aware hierarchical permutation ceiling.

Pre-registered in the execution log (C30 entry): before implementing any
C30 code, measure the mechanism ceiling — an oracle permutation built by
greedy hierarchical grouping on the real edge utility

    edge(i, j) = |H_A[i, j]| * sqrt(r_i * r_j) - lam * penalty(i, j)

(H_A: full activation Gram over dense calibration rows; r: per-channel
energy of the parent's weight quantization residual; penalty: magnitude
incompatibility on both operand sides), evaluated with the parent's own
operand-local loss harness (``_r64_operand_losses`` — the same base
quantizer the parent calibration's candidate metric uses, so arms differ
only in the permutation argument).

Arms per component:
- parent      — the deployed calibration's selected permutation;
- oracle-l0 / oracle-l1 — hierarchical greedy 4->8->16->32->64 grouping
  with lambda = 0 / 1;
- random      — seeded random permutation (sensitivity control).

Veto gate (pre-registered): mean combined improvement (act% + weight%)
across components < 10%, or either side's mean improvement <= 0, or mean
within-16 edge capture < +20% relative to parent -> C30 rejected at
Level-0.

Compliance: evaluator-side probe; operand-local statistics only; no
Linear output, no holdout access.
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

COMPONENTS = ("q", "k", "v", "o", "fc", "proj")


def load_solution(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("_hif4_perm_probe", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load solution: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.members = [[i] for i in range(n)]

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if len(self.members[ra]) < len(self.members[rb]):
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.members[ra].extend(self.members[rb])
        self.members[rb] = []
        return True


def _group_aggregate(
    utility: torch.Tensor, groups: list[list[int]]
) -> torch.Tensor:
    """S[a, b] = sum of utility between members of groups a and b."""

    channels = int(utility.shape[0])
    gsize = len(groups)
    indicator = torch.zeros(
        channels, gsize, dtype=utility.dtype, device=utility.device
    )
    for gi, group in enumerate(groups):
        for ch in group:
            indicator[ch, gi] = 1.0
    return indicator.t() @ (utility @ indicator)


def _greedy_edge_groups(
    utility: torch.Tensor, cap: int
) -> list[list[int]]:
    """Seed groups by greedily joining the highest-utility edges."""

    channels = int(utility.shape[0])
    tri = torch.triu_indices(channels, channels, offset=1)
    values = utility[tri[0], tri[1]]
    order = torch.argsort(values, descending=True)
    limit = min(int(order.numel()), 16 * channels)
    top = order[:limit].cpu().tolist()
    rows = tri[0].cpu().tolist()
    cols = tri[1].cpu().tolist()

    uf = _UnionFind(channels)
    for idx in top:
        i, j = rows[idx], cols[idx]
        ri, rj = uf.find(i), uf.find(j)
        if ri == rj:
            continue
        if len(uf.members[ri]) + len(uf.members[rj]) > cap:
            continue
        uf.union(ri, rj)

    groups = [m for m in uf.members if m]
    groups.sort(key=lambda g: (len(g), min(g)))
    # Complete undersized groups: repeatedly merge the smallest group
    # with the smallest partner that fits the cap; groups that fit no
    # partner are left to merge at the next level up.
    while len(groups) > 1:
        groups.sort(key=lambda g: (len(g), min(g)))
        a = groups[0]
        if len(a) >= cap:
            break
        partner = None
        for idx in range(1, len(groups)):
            if len(a) + len(groups[idx]) <= cap:
                partner = idx
                break
        if partner is None:
            break
        b = groups.pop(partner)
        groups[0] = sorted(a + b)
    groups.sort(key=lambda g: min(g))
    return [sorted(g) for g in groups]


def _merge_groups(
    utility: torch.Tensor, groups: list[list[int]], cap: int
) -> list[list[int]]:
    """Merge groups pairwise by aggregated inter-group utility."""

    if len(groups) <= 1:
        return [list(g) for g in groups]
    agg = _group_aggregate(utility, groups)
    gsize = len(groups)
    tri = torch.triu_indices(gsize, gsize, offset=1)
    values = agg[tri[0], tri[1]]
    order = torch.argsort(values, descending=True).cpu().tolist()
    rows = tri[0].cpu().tolist()
    cols = tri[1].cpu().tolist()

    uf = _UnionFind(gsize)
    sizes = [len(groups[gi]) for gi in range(gsize)]
    for idx in order:
        a, b = rows[idx], cols[idx]
        ra, rb = uf.find(a), uf.find(b)
        if ra == rb:
            continue
        if sizes[ra] + sizes[rb] > cap:
            continue
        uf.union(ra, rb)
        sizes[uf.find(ra)] = sizes[ra] + sizes[rb]
    seen: set[int] = set()
    result: list[list[int]] = []
    for gi in range(gsize):
        root = uf.find(gi)
        if root in seen:
            continue
        seen.add(root)
        members: list[int] = []
        for gj in uf.members[root]:
            members.extend(groups[gj])
        result.append(sorted(members))
    result.sort(key=lambda g: min(g))
    return result


def hierarchical_permutation(
    utility: torch.Tensor, channels: int
) -> torch.Tensor:
    """Deterministic hierarchical 4->8->16->32->64 grouping permutation."""

    groups = _greedy_edge_groups(utility, cap=4)
    for cap in (8, 16, 32, 64):
        if len(groups) <= 1:
            break
        groups = _merge_groups(utility, groups, cap)
    order: list[int] = []
    for group in groups:
        order.extend(group)
    assert len(order) == channels and sorted(order) == list(range(channels))
    return torch.tensor(order, dtype=torch.int64)


def within16_capture(utility: torch.Tensor, perm: torch.Tensor) -> float:
    """Fraction of off-diagonal utility captured inside 16-channel windows."""

    channels = int(utility.shape[0])
    position = torch.empty(channels, dtype=torch.int64, device=perm.device)
    position[perm.to(perm.device)] = torch.arange(
        channels, device=perm.device
    )
    group = position // 16
    same = group.unsqueeze(0) == group.unsqueeze(1)
    same.fill_diagonal_(False)
    total = float(utility[~same].sum())
    if total <= 0.0:
        return 0.0
    return float(utility[same].sum()) / total


def probe_component(
    solution: ModuleType,
    weight_dense: torch.Tensor,
    act_samples: list[torch.Tensor],
) -> dict:
    device = weight_dense.device
    channels = int(weight_dense.shape[1])

    weight_pair = nvfp4_encode(weight_dense.cpu(), mode="amax6")
    calib_pairs = [
        nvfp4_encode(a.cpu(), mode="amax6") for a in act_samples
    ]
    calibrated = solution.hif4_calibration_and_quantize_weight(
        weight_pair[0].to(device),
        weight_pair[1].to(device),
        [tuple(t.to(device) for t in p) for p in calib_pairs],
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

    rows = torch.cat(act_samples, dim=0)
    second_moment = rows.square().sum(dim=0) / float(rows.shape[0])

    # Parent weight residual per channel (original coordinates).
    weight_smooth = solution._linear_pair_transform(
        weight_dense, d, parent_perm, size, seed, weight_side=True
    )
    weight_hat = solution._dequantize_hif4(
        solution._dense_to_hif4(weight_smooth)
    )
    residual_permuted = weight_smooth - weight_hat
    residual = torch.zeros_like(residual_permuted)
    residual.index_copy_(1, parent_perm, residual_permuted)
    r = residual.square().sum(dim=0).sqrt()

    # Edge utility and magnitude penalty.
    h_a = rows.t() @ rows / float(rows.shape[0])
    base_edge = h_a.abs() * torch.sqrt(torch.outer(r, r)).clamp_min(0.0)
    act_mag = rows.abs().amax(dim=0) / d
    weight_mag = weight_dense.abs().amax(dim=0) * d
    log_act = act_mag.clamp_min(1e-12).log()
    log_weight = weight_mag.clamp_min(1e-12).log()
    penalty = (log_act.unsqueeze(0) - log_act.unsqueeze(1)).abs() + (
        log_weight.unsqueeze(0) - log_weight.unsqueeze(1)
    ).abs()
    utility0 = base_edge
    utility1 = base_edge - penalty

    arms: dict[str, torch.Tensor] = {
        "parent": parent_perm,
        "oracle_l0": hierarchical_permutation(utility0, channels).to(device),
        "oracle_l1": hierarchical_permutation(utility1, channels).to(device),
        "random": torch.randperm(channels, generator=torch.Generator().manual_seed(1234)).to(device),
    }

    weight_sample = solution._sample_rows(
        weight_dense, solution._LINEAR_WEIGHT_EVAL_ROWS
    )
    report: dict = {
        "block_smooth_size": size,
        "block_smooth_seed": seed,
        "identity_parent": bool(
            torch.equal(parent_perm, torch.arange(channels, device=device))
        ),
        "arms": {},
    }
    for name, perm in arms.items():
        weight_loss, act_losses = solution._r64_operand_losses(
            weight_sample,
            second_moment,
            act_samples,
            d,
            perm,
            size,
            seed,
        )
        report["arms"][name] = {
            "weight_loss": float(weight_loss),
            "act_loss_mean": sum(act_losses) / len(act_losses),
            "within16_capture": within16_capture(base_edge, perm),
        }
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
    parser.add_argument("--calib", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent
        / "c30_permutation_probe_results.json",
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
            t0 = time.perf_counter()
            entry = probe_component(solution, weight_dense, act_samples)
            entry["layer"] = layer
            entry["component"] = name
            entry["seconds"] = time.perf_counter() - t0
            results.append(entry)
            parent = entry["arms"]["parent"]
            best = min(
                (
                    arm
                    for arm in ("oracle_l0", "oracle_l1")
                ),
                key=lambda a: entry["arms"][a]["weight_loss"]
                + entry["arms"][a]["act_loss_mean"],
            )
            arm = entry["arms"][best]
            imp_w = (
                parent["weight_loss"] - arm["weight_loss"]
            ) / parent["weight_loss"]
            imp_a = (
                parent["act_loss_mean"] - arm["act_loss_mean"]
            ) / parent["act_loss_mean"]
            rnd = entry["arms"]["random"]
            imp_rw = (
                parent["weight_loss"] - rnd["weight_loss"]
            ) / parent["weight_loss"]
            imp_ra = (
                parent["act_loss_mean"] - rnd["act_loss_mean"]
            ) / parent["act_loss_mean"]
            print(
                f"layer {layer:2d} {name:4s} block={entry['block_smooth_size']:2d}"
                f" idparent={entry['identity_parent']}"
                f" | oracle({best}) w {imp_w*100:+6.2f}% a {imp_a*100:+6.2f}%"
                f" cap16 {parent['within16_capture']*100:5.1f}%->"
                f"{arm['within16_capture']*100:5.1f}%"
                f" | random w {imp_rw*100:+6.2f}% a {imp_ra*100:+6.2f}%"
                f" [{entry['seconds']:.1f}s]",
                flush=True,
            )

    def mean(values: list[float]) -> float:
        return sum(values) / max(len(values), 1)

    def arm_improvements(arm: str) -> tuple[list[float], list[float]]:
        w, a = [], []
        for r in results:
            p = r["arms"]["parent"]
            m = r["arms"][arm]
            w.append((p["weight_loss"] - m["weight_loss"]) / p["weight_loss"])
            a.append(
                (p["act_loss_mean"] - m["act_loss_mean"])
                / p["act_loss_mean"]
            )
        return w, a

    summary = {}
    for arm in ("oracle_l0", "oracle_l1", "random"):
        w, a = arm_improvements(arm)
        caps = [
            r["arms"][arm]["within16_capture"] / max(
                r["arms"]["parent"]["within16_capture"], 1e-12
            )
            for r in results
        ]
        summary[arm] = {
            "mean_weight_improvement": mean(w),
            "mean_act_improvement": mean(a),
            "mean_combined": mean(w) + mean(a),
            "mean_capture_ratio": mean(caps),
        }
    best_arm = (
        "oracle_l0"
        if summary["oracle_l0"]["mean_combined"]
        >= summary["oracle_l1"]["mean_combined"]
        else "oracle_l1"
    )
    s = summary[best_arm]
    gate_combined = s["mean_combined"] >= 0.10
    gate_sides = (
        s["mean_weight_improvement"] > 0.0
        and s["mean_act_improvement"] > 0.0
    )
    gate_capture = s["mean_capture_ratio"] >= 1.20
    verdict = (
        "PASS (proceed to Level-1)"
        if (gate_combined and gate_sides and gate_capture)
        else "VETO (C30 rejected at Level-0)"
    )
    summary["best_arm"] = best_arm
    summary["gates"] = {
        "combined_ge_10pct": gate_combined,
        "both_sides_positive": gate_sides,
        "capture_ratio_ge_1p20": gate_capture,
    }
    summary["verdict"] = verdict

    payload = {"results": results, "summary": summary}
    args.output.write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(f"\nBEST ARM: {best_arm}")
    for arm in ("oracle_l0", "oracle_l1", "random"):
        v = summary[arm]
        print(
            f"  {arm:9s} w {v['mean_weight_improvement']*100:+6.2f}%"
            f" a {v['mean_act_improvement']*100:+6.2f}%"
            f" combined {v['mean_combined']*100:+6.2f}%"
            f" cap16 x{v['mean_capture_ratio']:.3f}"
        )
    print(f"VERDICT: {verdict}")
    print(f"elapsed {time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
