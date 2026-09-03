"""Numeric and isolation checks for the unnumbered 21765-A workbench."""

from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluator"))

import reference_hif4  # noqa: E402
from nvfp4_sim import nvfp4_encode  # noqa: E402


PARENT = ROOT / "solutions/20260903_v160_v159-linear-l1batch_v158-attn-a2_scoreNA_timeNA/solution.py"
CANDIDATE = ROOT / "workbench/score21765_crossfold_softmax_fisher.py"
API_NAMES = (
    "hif4_calibration_and_quantize_weight",
    "hif4_dynamic_quantize_activation",
    "hif4_calibration_attention",
    "hif4_dynamic_quantize_q",
    "hif4_dynamic_quantize_k",
    "hif4_dynamic_quantize_v",
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def values_equal(left: Any, right: Any) -> bool:
    if torch.is_tensor(left) and torch.is_tensor(right):
        return left.dtype == right.dtype and left.shape == right.shape and bool(
            torch.equal(left, right)
        )
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(
            values_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (tuple, list)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(
            values_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def state_without_importance(state: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in state.items() if key != "importance"}


def main() -> None:
    torch.manual_seed(20260903)
    parent = load_module("score21765_parent", PARENT)
    candidate = load_module("score21765_candidate", CANDIDATE)
    for api in API_NAMES:
        assert hasattr(candidate, api), api
    assert candidate._ATTN_CROSSFOLD_FISHER
    assert not candidate._ATTN_FISHER_IMPORTANCE
    print("[A1] isolated import + six APIs: OK")

    for api in (
        "hif4_calibration_and_quantize_weight",
        "hif4_dynamic_quantize_activation",
        "hif4_dynamic_quantize_q",
        "hif4_dynamic_quantize_k",
        "hif4_dynamic_quantize_v",
    ):
        assert inspect.getsource(getattr(candidate, api)) == inspect.getsource(
            getattr(parent, api)
        ), api
    print("[A1] all dynamic API implementations unchanged: OK")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q_heads, kv_heads, head_dim = 14, 2, 64
    calibration = []
    for tokens in (10, 24, 40, 56, 72):
        q = torch.randn(tokens, q_heads * head_dim, device=device) * 0.18
        k = torch.randn(tokens, kv_heads * head_dim, device=device) * 0.16
        v = torch.randn(tokens, kv_heads * head_dim, device=device) * 0.20
        calibration.append(
            {
                "q": tuple(t.to(device) for t in nvfp4_encode(q)),
                "k": tuple(t.to(device) for t in nvfp4_encode(k)),
                "v": tuple(t.to(device) for t in nvfp4_encode(v)),
            }
        )

    parent_states = parent.hif4_calibration_attention(
        calibration, q_heads, kv_heads, head_dim
    )
    candidate_states = candidate.hif4_calibration_attention(
        calibration, q_heads, kv_heads, head_dim
    )
    assert set(candidate_states) == {"q_state", "k_state", "v_state"}
    for state in candidate_states.values():
        reference_hif4.validate_state(state)
    assert values_equal(parent_states["v_state"], candidate_states["v_state"])
    actual_changed = {}
    for name in ("q_state", "k_state"):
        assert values_equal(
            state_without_importance(parent_states[name]),
            state_without_importance(candidate_states[name]),
        ), name
        parent_importance = parent_states[name]["importance"].to(torch.float32)
        candidate_importance = candidate_states[name]["importance"].to(torch.float32)
        assert bool(torch.isfinite(candidate_importance).all())
        assert bool((candidate_importance > 0).all())
        actual_changed[name[0]] = int(
            ((parent_importance - candidate_importance).abs() > 1.0e-7).sum()
        )
        heads = q_heads if name == "q_state" else kv_heads
        parent_means = parent_importance.reshape(heads, head_dim).mean(dim=-1)
        candidate_means = candidate_importance.reshape(heads, head_dim).mean(dim=-1)
        assert torch.allclose(parent_means, candidate_means, rtol=2.0e-6, atol=2.0e-7)
    print("[A1] only Q/K importance may change; V and head means unchanged: OK")

    assert len(candidate._ATTN_CROSSFOLD_FISHER_AUDIT) == 1
    audit = candidate._ATTN_CROSSFOLD_FISHER_AUDIT[0]
    assert audit["folds"] == 5 and audit["observations"] == 10
    for name in ("q", "k"):
        arm = audit[name]
        assert 0.0 <= arm["rho_min"] <= arm["rho_max"] <= 1.0
        assert arm["changed_channels"] == actual_changed[name]
        assert arm["multiplier_min"] >= torch.exp(torch.tensor(-2.0)).item() - 1.0e-6
        assert arm["multiplier_max"] <= torch.exp(torch.tensor(2.0)).item() + 1.0e-6
    print("[A0] five-fold/two-mask audit statistics: OK")
    print(audit)

    test = torch.randn(32, q_heads * head_dim, device=device) * 0.19
    test_pair = tuple(t.to(device) for t in nvfp4_encode(test))
    q_params = candidate.hif4_dynamic_quantize_q(
        test_pair[0], test_pair[1], q_heads, head_dim, candidate_states["q_state"]
    )
    reference_hif4.validate_hif4_params(q_params, test.shape)
    test_kv = test[:, : kv_heads * head_dim]
    test_kv_pair = tuple(t.to(device) for t in nvfp4_encode(test_kv))
    for api, state_name in (
        (candidate.hif4_dynamic_quantize_k, "k_state"),
        (candidate.hif4_dynamic_quantize_v, "v_state"),
    ):
        params = api(
            test_kv_pair[0],
            test_kv_pair[1],
            kv_heads,
            head_dim,
            candidate_states[state_name],
        )
        reference_hif4.validate_hif4_params(params, test_kv.shape)
    print("[A1] Q/K/V output legality: OK")
    print("ALL 21765-A WORKBENCH CHECKS PASSED")


if __name__ == "__main__":
    main()
