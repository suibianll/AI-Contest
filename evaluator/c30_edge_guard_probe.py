"""C30 §8.2 compliance pre-ruling probe: the edge(i,j) pattern under the guard.

The C30 candidate (Hessian-aware hierarchical permutation) builds its
channel graph as

    r_i       = sqrt(sum_rows(E_weight[row, i]^2))     # weight-local residual
    edge(i,j) = abs(H_A[i, j]) * sqrt(r_i * r_j)       # elementwise combination
    perm      = deterministic ordering derived from edge utility

before any implementation work, the post-C21C plan §8.2 requires this
pattern to be submitted to ``linear_compliance_guard`` for an explicit
ruling.  This probe runs three arms:

- ``edge_pattern``  — the canonical C30 form above (elementwise product of
  the activation Gram and weight-residual channel statistics, permutation
  into activation_state);
- ``gram_only``     — control: permutation from the Gram alone (no weight
  data);
- ``cross_contraction`` — boundary/negative: the same two operands combined
  by a *contraction* (activation residual x weight residual), which must
  stay a hard violation.

Ruling semantics recorded alongside the machine verdict:
- the whitelist ("分别计算 operand-local 指标，再以预注册规则组合候选排名；
  组合过程中不得出现两操作数的收缩乘积") permits elementwise combination
  of operand-local statistics and forbids contractions between the operands'
  data — the edge pattern is the former;
- edge(i,j) is a Cauchy-Schwarz bound on the coupling terms of
  trace(E_W H_A E_W^T), the same mathematical family as the whitelisted
  C23 Q(W) Hessian loss; no Linear output is constructed;
- the resulting permutation is a shared channel transform in the same
  category as the SmoothQuant scale D (also derived from both operands).

Boundary conditions attached to the ruling:
1. edge must remain elementwise — any contraction mixing activation-derived
   with weight-residual-derived tensors is a violation (cross_contraction
   arm proves the guard enforces this);
2. the [K, K] edge matrix itself must not enter activation_state (it is a
   fitting artifact; only the permutation does);
3. r must be channel-wise statistics of the weight residual; the raw
   residual tensor never enters activation_state.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from linear_compliance_guard import runtime_guard, static_guard  # noqa: E402


EDGE_PATTERN_SOURCE = '''
"""C30 edge-pattern fixture (compliance pre-ruling)."""
import torch


def _decode_pair(carrier, scale):
    shape = carrier.shape
    flat = carrier.reshape(*shape[:-1], -1, 16)
    dense = flat * scale.reshape(*scale.shape, 1)
    return dense.reshape(shape)


def hif4_calibration_and_quantize_weight(w_carrier, w_scale, calib_pairs):
    # Operand-local statistics, computed separately per side.
    acts = [_decode_pair(c, s) for (c, s) in calib_pairs]
    a = torch.cat(acts, dim=0)
    h_a = a.t() @ a / float(a.shape[0])            # activation Gram

    w = _decode_pair(w_carrier, w_scale)           # dense weight
    w_hat = torch.round(w * 8.0) / 8.0             # quantization stub
    e_w = w - w_hat                                # weight residual
    r = e_w.square().sum(dim=0).sqrt()             # [K] channel energy

    # C30 edge: elementwise combination of the two statistics (no
    # contraction between activation-derived and weight-derived data).
    edge = h_a.abs() * torch.sqrt(torch.outer(r, r))

    # Deterministic ordering derived from edge utility (grouping stub).
    score = edge.sum(dim=1)
    perm = torch.argsort(score, stable=True).to(torch.int64)

    return {
        "weight": w_hat,
        "activation_state": {"permutation": perm.cpu()},
    }
'''

GRAM_ONLY_SOURCE = '''
"""Control fixture: permutation from the activation Gram alone."""
import torch


def _decode_pair(carrier, scale):
    shape = carrier.shape
    flat = carrier.reshape(*shape[:-1], -1, 16)
    dense = flat * scale.reshape(*scale.shape, 1)
    return dense.reshape(shape)


def hif4_calibration_and_quantize_weight(w_carrier, w_scale, calib_pairs):
    acts = [_decode_pair(c, s) for (c, s) in calib_pairs]
    a = torch.cat(acts, dim=0)
    h_a = a.t() @ a / float(a.shape[0])
    score = h_a.abs().sum(dim=1)
    perm = torch.argsort(score, stable=True).to(torch.int64)
    return {
        "weight": _decode_pair(w_carrier, w_scale),
        "activation_state": {"permutation": perm.cpu()},
    }
'''

CROSS_CONTRACTION_SOURCE = '''
"""Boundary fixture: same operands combined by a contraction (must violate)."""
import torch


def _decode_pair(carrier, scale):
    shape = carrier.shape
    flat = carrier.reshape(*shape[:-1], -1, 16)
    dense = flat * scale.reshape(*scale.shape, 1)
    return dense.reshape(shape)


def hif4_calibration_and_quantize_weight(w_carrier, w_scale, calib_pairs):
    acts = [_decode_pair(c, s) for (c, s) in calib_pairs]
    a = torch.cat(acts, dim=0)
    h_a = a.t() @ a / float(a.shape[0])

    w = _decode_pair(w_carrier, w_scale)
    w_hat = torch.round(w * 8.0) / 8.0
    e_w = w - w_hat                                # weight residual
    r = e_w.square().sum(dim=0).sqrt()

    a_hat = torch.round(a * 8.0) / 8.0
    e_a = a - a_hat                                # activation residual

    edge = h_a.abs() * torch.sqrt(torch.outer(r, r))
    # Forbidden form: contraction of the activation residual with the
    # weight residual (the removed cross8 mechanism in a new shape).
    cross = torch.einsum("nk,mk->nm", e_a, e_w)
    score = edge.sum(dim=1) + cross.square().sum()
    perm = torch.argsort(score, stable=True).to(torch.int64)
    return {
        "weight": w_hat,
        "activation_state": {"permutation": perm.cpu()},
    }
'''


def make_module(source: str) -> ModuleType:
    module = ModuleType("c30_pre_ruling_fixture")
    exec(compile(source, "<c30_pre_ruling_fixture>", "exec"), module.__dict__)
    return module


def run_arm(name: str, source: str) -> dict:
    torch.manual_seed(305)
    tokens, out_features, channels = 53, 37, 64
    weight = torch.randn(out_features, channels) * 0.1
    activations = [torch.randn(tokens, channels) * 0.1 for _ in range(2)]
    report = runtime_guard(
        make_module(source),
        weight,
        activations,
        tokens=tokens,
        out_features=out_features,
    )
    return {
        "arm": name,
        "static_violations": static_guard(source),
        "violations": report["violations"],
        "review": report["review"],
        "contraction_count": report["contraction_count"],
    }


def main() -> int:
    ruling = {
        "date": "2026-08-28",
        "candidate": "C30",
        "pattern": "edge(i,j) = |H_A[i,j]| * sqrt(r_i * r_j); perm into state",
        "arms": [
            run_arm("edge_pattern", EDGE_PATTERN_SOURCE),
            run_arm("gram_only", GRAM_ONLY_SOURCE),
            run_arm("cross_contraction", CROSS_CONTRACTION_SOURCE),
        ],
    }

    edge = ruling["arms"][0]
    cross = ruling["arms"][2]
    edge_passes = not edge["static_violations"] and not edge["violations"]
    cross_caught = bool(cross["violations"])
    ruling["machine_verdict"] = {
        "edge_pattern_passes": edge_passes,
        "cross_contraction_caught": cross_caught,
        "edge_pattern_review_entries": edge["review"],
    }
    ruling["ruling"] = (
        "PASS"
        if edge_passes and cross_caught
        else "FAIL"
    )
    ruling["conditions"] = [
        "edge stays elementwise; no contraction mixing activation-derived "
        "with weight-residual-derived tensors (cross_contraction arm "
        "demonstrates the violating form)",
        "the [K, K] edge matrix must not enter activation_state; only the "
        "permutation does",
        "r must be channel-wise statistics of the weight residual; the raw "
        "residual tensor never enters activation_state",
    ]
    ruling["semantics"] = (
        "Whitelist combining clause: elementwise combination of "
        "operand-local statistics for candidate ranking is legal; edge(i,j) "
        "is a Cauchy-Schwarz bound on the coupling terms of "
        "trace(E_W H_A E_W^T), the same family as the whitelisted C23 Q(W) "
        "Hessian loss; the permutation is a shared channel transform in the "
        "same category as the SmoothQuant scale D. No Linear output is "
        "constructed."
    )

    output = Path(__file__).resolve().parent / "c30_edge_guard_ruling.json"
    output.write_text(json.dumps(ruling, indent=2), encoding="utf-8")
    for arm in ruling["arms"]:
        print(f"[{arm['arm']}] static={arm['static_violations']}")
        print(f"  violations={arm['violations']}")
        print(f"  review={arm['review']}")
    print(f"\nRULING: {ruling['ruling']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
