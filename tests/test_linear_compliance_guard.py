from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluator"))

from linear_compliance_guard import (  # noqa: E402
    guard_solution_file,
    runtime_guard,
    static_guard,
)


def load_solution() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "compliance_guard_solution", ROOT / "solution.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_fake_solution(source: str) -> ModuleType:
    module = ModuleType("fake_solution")
    exec(compile(source, "<fake_solution>", "exec"), module.__dict__)
    return module


FAKE_SUPERVISED_SOURCE = """
import torch


def hif4_calibration_and_quantize_weight(weight_quant, weight_scale, pairs):
    weight = weight_quant * weight_scale.sum()
    activation = pairs[0][0]
    # Forbidden: Linear output supervision on calibration data.
    supervised = activation @ weight.t()
    score = supervised.square().mean()
    return {
        "weight_params": {"scale_factor": weight.sum(dim=-1, keepdim=True)},
        "activation_state": {"importance": weight.sum(dim=0)},
    }
"""


FAKE_CROSS_RESIDUAL_SOURCE = """
import torch


def hif4_calibration_and_quantize_weight(weight_quant, weight_scale, pairs):
    activation = pairs[0][0]
    weight_hat = weight_quant * 0.5
    weight_err = weight_quant - weight_hat          # weight residual
    act_hat = activation * 0.5
    act_err = activation - act_hat                  # activation residual
    # Forbidden: cross residual operator (cross8 in disguise).
    cross = torch.einsum("nk,mk->nm", act_err, weight_err)
    state = {"smooth_inv": (cross.sum(dim=0)[:64])}
    return {"weight_params": {}, "activation_state": state}
"""


FAKE_LEGAL_SOURCE = """
import torch


def hif4_calibration_and_quantize_weight(weight_quant, weight_scale, pairs):
    activation = pairs[0][0]
    cov = activation.t() @ activation          # legal: A^T A gram
    gram = cov / activation.shape[0]
    weight = weight_quant * weight_scale.sum()
    weight_hat = weight * 0.5
    weight_err = weight - weight_hat           # weight residual
    # Legal: Q(W) Hessian loss (weight residual x activation gram).
    loss = torch.einsum("mk,kn,mn->", weight_err, gram, weight_err)
    smooth = (activation.abs().amax(dim=0) + weight.abs().amax(dim=0))
    return {
        "weight_params": {},
        "activation_state": {"smooth_inv": smooth, "gram": gram[:64, :64]},
    }
"""


def test_static_guard_accepts_current_solution() -> None:
    source = (ROOT / "solution.py").read_text(encoding="utf-8")
    assert static_guard(source) == []


def test_static_guard_rejects_forbidden_symbols() -> None:
    cases = {
        "_linear_output_candidate_metrics(a, b)": "forbidden symbol",
        "state = {'cross8': tensor}": "forbidden state key",
        "group_cross8 = compute()": "forbidden symbol",
        "_ACTIVATION_QUADRATIC8_CROSS_GAIN = True": "forbidden symbol",
        "result = {'output_residual': x}": "forbidden state key",
    }
    for snippet, expected in cases.items():
        violations = static_guard(snippet)
        assert any(expected in message for message in violations), snippet


def test_static_guard_flags_renamed_cross_contraction() -> None:
    source = """
def calibrate(activation_stream, weight_dense):
    supervised = activation_stream.mm(weight_dense.t())
    return supervised.mean()
"""
    violations = static_guard(source)
    assert any(
        "cross contraction" in message for message in violations
    )


def test_runtime_guard_accepts_current_solution() -> None:
    solution = load_solution()
    torch.manual_seed(301)
    tokens, out_features, channels = 53, 37, 64
    weight = torch.randn(out_features, channels) * 0.1
    activations = [torch.randn(tokens, channels) * 0.1 for _ in range(2)]
    report = runtime_guard(
        solution,
        weight,
        activations,
        tokens=tokens,
        out_features=out_features,
    )
    assert report["violations"] == []
    assert report["contraction_count"] > 0
    # Dual-taint channel statistics (the SmoothQuant scale) land in
    # review, never in violations.
    for message in report["review"]:
        assert "residual" not in message


def test_runtime_guard_rejects_linear_output_supervision() -> None:
    fake = make_fake_solution(FAKE_SUPERVISED_SOURCE)
    torch.manual_seed(302)
    tokens, out_features, channels = 53, 37, 64
    weight = torch.randn(out_features, channels) * 0.1
    activations = [torch.randn(tokens, channels) * 0.1]
    report = runtime_guard(
        fake,
        weight,
        activations,
        tokens=tokens,
        out_features=out_features,
    )
    assert any("leaked" in message for message in report["violations"])


def test_runtime_guard_rejects_cross_residual_state() -> None:
    fake = make_fake_solution(FAKE_CROSS_RESIDUAL_SOURCE)
    torch.manual_seed(303)
    tokens, out_features, channels = 53, 37, 64
    weight = torch.randn(out_features, channels) * 0.1
    activations = [torch.randn(tokens, channels) * 0.1]
    report = runtime_guard(
        fake,
        weight,
        activations,
        tokens=tokens,
        out_features=out_features,
    )
    assert any(
        "cross residual" in message or "residual" in message
        for message in report["violations"]
    )


def test_runtime_guard_allows_legal_hessian_and_grams() -> None:
    fake = make_fake_solution(FAKE_LEGAL_SOURCE)
    torch.manual_seed(304)
    tokens, out_features, channels = 53, 37, 64
    weight = torch.randn(out_features, channels) * 0.1
    activations = [torch.randn(tokens, channels) * 0.1]
    report = runtime_guard(
        fake,
        weight,
        activations,
        tokens=tokens,
        out_features=out_features,
    )
    assert report["violations"] == []
    assert report["contraction_count"] >= 2


FAKE_C30_EDGE_SOURCE = """
import torch


def hif4_calibration_and_quantize_weight(weight_quant, weight_scale, pairs):
    acts = [c * s.sum() for (c, s) in pairs]
    a = torch.cat(acts, dim=0)
    h_a = a.t() @ a / float(a.shape[0])            # activation gram
    weight = weight_quant * weight_scale.sum()
    weight_hat = torch.round(weight * 8.0) / 8.0
    e_w = weight - weight_hat                      # weight residual
    r = e_w.square().sum(dim=0).sqrt()             # channel energy
    # C30 edge: elementwise combination, no contraction between the
    # two operands' data (pre-ruled legal, evaluator/c30_edge_guard_probe.py).
    edge = h_a.abs() * torch.sqrt(torch.outer(r, r))
    perm = torch.argsort(edge.sum(dim=1), stable=True).to(torch.int64)
    return {"weight_params": {}, "activation_state": {"permutation": perm}}
"""


def test_runtime_guard_reviews_c30_edge_permutation() -> None:
    """C30 pre-ruling regression: elementwise edge pattern passes with an
    explicit review entry for the dual-side permutation (never silent,
    never a violation)."""

    fake = make_fake_solution(FAKE_C30_EDGE_SOURCE)
    torch.manual_seed(305)
    tokens, out_features, channels = 53, 37, 64
    weight = torch.randn(out_features, channels) * 0.1
    activations = [torch.randn(tokens, channels) * 0.1 for _ in range(2)]
    report = runtime_guard(
        fake,
        weight,
        activations,
        tokens=tokens,
        out_features=out_features,
    )
    assert report["violations"] == []
    # The permutation ([C] int64, {G, W} taints) lands in review.
    assert len(report["review"]) >= 1
    for message in report["review"]:
        assert "combines activation and weight residuals" not in message


def test_guard_solution_file_full_gate_passes() -> None:
    report = guard_solution_file(ROOT / "solution.py")
    assert report["violations"] == []
    assert report["contraction_count"] > 0
    assert report["state_tensor_count"] > 0
