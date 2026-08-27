from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluator"))

from nvfp4_sim import nvfp4_encode  # noqa: E402
from real_data_eval import TEXT, instrument_solution  # noqa: E402


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def assert_nested_equal(left, right) -> None:
    assert type(left) is type(right)
    if torch.is_tensor(left):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert set(left) == set(right)
        for key in left:
            assert_nested_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for l_value, r_value in zip(left, right):
            assert_nested_equal(l_value, r_value)
    else:
        assert left == right


def validate_state(value) -> tuple[int, int]:
    tensors = 0
    elements = 0

    def walk(item) -> None:
        nonlocal tensors, elements
        if torch.is_tensor(item):
            tensors += 1
            elements += int(item.numel())
            assert item.device.type == "cpu"
            assert item.layout == torch.strided
            assert item.is_contiguous()
            assert not item.requires_grad
            assert torch.isfinite(item.to(torch.float32)).all()
        elif isinstance(item, dict):
            for child in item.values():
                walk(child)
        elif isinstance(item, (tuple, list)):
            for child in item:
                walk(child)
        else:
            assert item is None or isinstance(item, (bool, int, float, str))

    walk(value)
    return tensors, elements


def test_release_flags_are_a1_only() -> None:
    solution = load_module("release_flags", ROOT / "solution.py")
    assert solution._ATTN_OUTPUT_SELECTOR is True
    assert solution._ATTN_H64 is False
    assert solution._V_IMPORTANCE_CANDIDATES is False
    assert solution._L1_DATA_DRIVEN_SCALE is False
    assert solution._WEIGHT_QUADRATIC8 is True
    assert solution._WEIGHT_QUADRATIC16 is True
    assert solution._ACTIVATION_QUADRATIC_MAX_FEATURES == 4096
    assert solution._ACTIVATION_QUADRATIC8 is True
    assert solution._ACTIVATION_QUADRATIC8_MIN_FEATURES == 1025
    assert solution._ACTIVATION_QUADRATIC8_MAX_RATIO == 0.02
    assert solution._ACTIVATION_QUADRATIC8_SWEEPS == 1


def test_submission_has_no_file_io_or_debug_output() -> None:
    source = (ROOT / "solution.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden_imports = {"os", "pathlib", "subprocess", "socket", "requests", "numpy"}
    forbidden_calls = {"open", "print", "exec", "eval", "compile", "input"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert not ({alias.name.split(".")[0] for alias in node.names} & forbidden_imports)
        elif isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] not in forbidden_imports
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in forbidden_calls


def test_feature_off_is_field_for_field_b0_equivalent() -> None:
    current = load_module("feature_off_current", ROOT / "solution.py")
    baseline = load_module("feature_off_b0", ROOT / "solution_b0_tmp.py")
    current._ATTN_OUTPUT_SELECTOR = False
    current._ATTN_H64 = False
    current._V_IMPORTANCE_CANDIDATES = False
    current._L1_DATA_DRIVEN_SCALE = False

    torch.manual_seed(901)
    q_heads, kv_heads, head_dim = 4, 2, 64
    samples = []
    for _ in range(2):
        q = torch.randn(32, q_heads * head_dim)
        k = torch.randn(32, kv_heads * head_dim)
        v = torch.randn(32, kv_heads * head_dim)
        samples.append(
            {"q": nvfp4_encode(q), "k": nvfp4_encode(k), "v": nvfp4_encode(v)}
        )

    actual = current.hif4_calibration_attention(
        samples, q_heads, kv_heads, head_dim
    )
    expected = baseline.hif4_calibration_attention(
        samples, q_heads, kv_heads, head_dim
    )
    assert_nested_equal(actual, expected)


def test_rotation_invariance_and_head_dim_128_state_legality() -> None:
    solution = load_module("rotation_128", ROOT / "solution.py")
    torch.manual_seed(128)
    q_heads, kv_heads, head_dim = 8, 2, 128
    q = torch.randn(24, q_heads * head_dim)
    k = torch.randn(24, kv_heads * head_dim)
    v = torch.randn(24, kv_heads * head_dim)
    signs = solution._attention_rotation_signs(kv_heads, head_dim, 0)
    q_rot = solution._apply_attention_rotation(q, q_heads, head_dim, signs)
    k_rot = solution._apply_attention_rotation(k, kv_heads, head_dim, signs)

    group = q_heads // kv_heads
    reference = torch.einsum(
        "thd,shd->tsh",
        q.reshape(24, q_heads, head_dim),
        k.reshape(24, kv_heads, head_dim).repeat_interleave(group, dim=1),
    )
    rotated = torch.einsum(
        "thd,shd->tsh",
        q_rot.reshape(24, q_heads, head_dim),
        k_rot.reshape(24, kv_heads, head_dim).repeat_interleave(group, dim=1),
    )
    torch.testing.assert_close(reference, rotated, rtol=3e-5, atol=8e-5)

    sample = {"q": nvfp4_encode(q), "k": nvfp4_encode(k), "v": nvfp4_encode(v)}
    states = solution.hif4_calibration_attention(
        [sample], q_heads, kv_heads, head_dim
    )
    tensor_count, _ = validate_state(states)
    assert tensor_count < 4096
    outputs = (
        solution.hif4_dynamic_quantize_q(
            *sample["q"], q_heads, head_dim, states["q_state"]
        ),
        solution.hif4_dynamic_quantize_k(
            *sample["k"], kv_heads, head_dim, states["k_state"]
        ),
        solution.hif4_dynamic_quantize_v(
            *sample["v"], kv_heads, head_dim, states["v_state"]
        ),
    )
    for params in outputs:
        assert torch.isfinite(solution._dequantize_hif4(params)).all()


def test_timing_wrapper_does_not_double_count_nested_calls() -> None:
    module = ModuleType("timing_fixture")

    def dynamic(value):
        return value + 1

    def calibration(value):
        return module.hif4_dynamic_quantize_q(value)

    for name in (
        "hif4_dynamic_quantize_activation",
        "hif4_dynamic_quantize_q",
        "hif4_dynamic_quantize_k",
        "hif4_dynamic_quantize_v",
    ):
        setattr(module, name, dynamic)
    module.hif4_calibration_and_quantize_weight = calibration
    module.hif4_calibration_attention = calibration

    stats = instrument_solution(module)
    assert module.hif4_calibration_attention(1) == 2
    assert stats["calls"]["hif4_calibration_attention"] == 1
    assert stats["calls"]["hif4_dynamic_quantize_q"] == 0
    assert stats["nested_calls"]["hif4_dynamic_quantize_q"] == 1
    assert stats["dynamic"] == 0.0

    assert module.hif4_dynamic_quantize_q(1) == 2
    assert stats["calls"]["hif4_dynamic_quantize_q"] == 1
    assert stats["dynamic"] > 0.0


def test_local_holdout_offsets_are_fixed_and_distinct() -> None:
    from transformers import GPT2Tokenizer

    tokenizer = GPT2Tokenizer.from_pretrained(ROOT / "models" / "gpt2")
    ids = tokenizer(TEXT, return_tensors="pt")["input_ids"][0]
    windows = []
    for requested in (0, 97, 193, 389):
        offset = requested % int(ids.numel())
        rotated = torch.cat((ids[offset:], ids[:offset])) if offset else ids
        windows.append(rotated[:128])
    assert all(window.numel() == 128 for window in windows)
    assert len({tuple(window.tolist()) for window in windows}) == len(windows)


def test_weight_quadratic8_refinement_is_non_increasing() -> None:
    solution = load_module("weight_quadratic8", ROOT / "solution.py")
    torch.manual_seed(808)
    dense = torch.randn(4, 64)
    params = solution._dense_to_hif4(
        dense,
        search_offsets=(-1, 1),
        max_refine_ratio=1.0,
    )
    factor = torch.randn(64, 64)
    covariance = factor.T @ factor + 0.1 * torch.eye(64)
    gram8 = solution._flat_group_gram8(covariance, 64)
    before = solution._dequantize_hif4(params)
    refined = solution._refine_weight_groups8(dense, params, gram8)
    after = solution._dequantize_hif4(refined)
    grams = gram8.unsqueeze(0).expand(4, -1, -1, -1).reshape(-1, 8, 8)
    before_error = (before - dense).reshape(-1, 8)
    after_error = (after - dense).reshape(-1, 8)
    before_loss = torch.einsum(
        "ni,nij,nj->", before_error, grams, before_error
    )
    after_loss = torch.einsum(
        "ni,nij,nj->", after_error, grams, after_error
    )
    assert after_loss <= before_loss + 1.0e-5
    validate_state(refined)


def test_weight_quadratic16_refinement_is_non_increasing() -> None:
    solution = load_module("weight_quadratic16", ROOT / "solution.py")
    torch.manual_seed(1616)
    dense = torch.randn(4, 64)
    params = solution._dense_to_hif4(
        dense,
        search_offsets=(-1, 1),
        max_refine_ratio=1.0,
    )
    factor = torch.randn(64, 64)
    covariance = factor.T @ factor + 0.1 * torch.eye(64)
    gram16 = solution._flat_group_gram16(covariance, 64)
    before = solution._dequantize_hif4(params)
    refined = solution._refine_weight_groups16(dense, params, gram16)
    after = solution._dequantize_hif4(refined)
    grams = gram16.unsqueeze(0).expand(4, -1, -1, -1).reshape(-1, 16, 16)
    before_error = (before - dense).reshape(-1, 16)
    after_error = (after - dense).reshape(-1, 16)
    before_loss = torch.einsum(
        "ni,nij,nj->", before_error, grams, before_error
    )
    after_loss = torch.einsum(
        "ni,nij,nj->", after_error, grams, after_error
    )
    assert after_loss <= before_loss + 1.0e-5
    validate_state(refined)


def test_wide_activation_grams_are_legal_state_tensors() -> None:
    solution = load_module("wide_activation_gram", ROOT / "solution.py")
    torch.manual_seed(4096)
    weight = torch.randn(4, 3072)
    gram = weight.T @ weight
    state = {
        "gram": solution._cpu_state_tensor(
            solution._flat_group_gram(gram, 3072)
        ),
        "gram8": solution._cpu_state_tensor(
            solution._flat_group_gram8(gram, 3072)
        ),
    }
    tensor_count, elements = validate_state(state)
    assert tensor_count == 2
    assert elements == 36_864
