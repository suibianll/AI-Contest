from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluator"))

from nvfp4_sim import nvfp4_encode  # noqa: E402
from real_data_eval import TEXT, instrument_solution  # noqa: E402
from synthetic_attention_eval import (  # noqa: E402
    check_dynamic_params,
    check_state_tree,
    encode_case,
    generate_case_data,
)


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
    assert solution._ACTIVATION_QUADRATIC8_MIN_FEATURES == 64
    assert solution._ACTIVATION_QUADRATIC8_MAX_RATIO == 0.60
    assert solution._ACTIVATION_QUADRATIC8_SWEEPS == 2
    assert solution._ACTIVATION_QUADRATIC8_CALIBRATION_GATE is False
    assert solution._ACTIVATION_QUADRATIC8_GATE_MAX_FEATURES == 1024
    assert solution._ACTIVATION_REFINE_MAX_RATIO == 1.0
    assert solution._WEIGHT_FULL64 is True
    assert solution._WEIGHT_FULL64_MAX_RATIO == 0.30
    assert solution._WEIGHT_FULL64_BEAM_KEEP == 2
    assert solution._WEIGHT_FULL64_MAX_RATIO_NARROW == 1.0
    assert solution._WEIGHT_FULL64_MAX_RATIO_WIDE == 0.25
    assert solution._WEIGHT_FULL64_SECOND_COORDINATE is False
    assert solution._LINEAR_R64 is False
    assert solution._HIERARCHY_PERMUTATION is False
    assert not hasattr(solution, "_ACTIVATION_QUADRATIC8_CROSS_TERM")
    assert not hasattr(solution, "_ACTIVATION_QUADRATIC8_CROSS_GAIN_SELECTION")
    assert not hasattr(solution, "_ACTIVATION_QUADRATIC8_CROSS_CALIBRATION_GATE")
    assert solution._LINEAR_R64 is False
    assert solution._LINEAR_R64_BLOCK == 64
    assert solution._LINEAR_R64_STAGE1_SEEDS == tuple(range(32))
    assert solution._LINEAR_R64_STAGE2_KEEP == 4
    assert solution._LINEAR_R64_MIN_IMPROVEMENT == 0.005
    assert solution._LINEAR_R64_WORST_TOLERANCE == 0.002


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


def test_activation8_refinement_gate_returns_a_decision() -> None:
    solution = load_module("activation8_gate", ROOT / "solution.py")
    torch.manual_seed(414)
    gram = torch.randn(64, 64)
    gram = gram.T @ gram + 0.1 * torch.eye(64)
    importance = solution._normalize_importance(
        torch.rand(64) + 0.5, 64
    )
    decision = solution._activation8_refinement_is_safe(
        [torch.randn(8, 64), torch.randn(8, 64)],
        torch.ones(64),
        torch.arange(64),
        0,
        0,
        importance,
        solution._flat_group_gram(gram, 64),
        solution._flat_group_gram8(gram, 64),
        1.0,
    )
    assert isinstance(decision, bool)


def test_activation8_refinement_gate_rejects_regression() -> None:
    solution = load_module("activation8_gate_reject", ROOT / "solution.py")
    torch.manual_seed(415)
    # A degenerate gram8 (zero) makes the refinement a no-op, so the
    # activation-local loss cannot improve and the gate must reject.
    gram = 1.0e-12 * torch.eye(64)
    importance = solution._normalize_importance(
        torch.rand(64) + 0.5, 64
    )
    decision = solution._activation8_refinement_is_safe(
        [torch.randn(8, 64)],
        torch.ones(64),
        torch.arange(64),
        0,
        0,
        importance,
        solution._flat_group_gram(gram, 64),
        solution._flat_group_gram8(gram, 64),
        1.0,
    )
    assert decision is False


# E1 冻结合成矩阵的代表子集：尾部债务场景（heavy_tail、官方踩坑场景
# saturated_logits、v_outlier）+ balanced 基准。全矩阵由
# evaluator/synthetic_attention_eval.py CLI 覆盖。
SYNTHETIC_SMOKE_CASES = (
    ("heavy_tail", 4, 2, 128, 128),
    ("heavy_tail", 4, 4, 128, 32),
    ("saturated_logits", 4, 2, 64, 32),
    ("v_outlier", 4, 4, 64, 128),
    ("balanced", 4, 4, 64, 32),
)


def test_synthetic_attention_states_and_params_are_legal() -> None:
    solution = load_module("synthetic_states", ROOT / "solution.py")
    for scenario, q_heads, kv_heads, head_dim, seq in SYNTHETIC_SMOKE_CASES:
        label = f"{scenario}_h{q_heads}_kv{kv_heads}_d{head_dim}_s{seq}"
        for mode in ("amax6", "amax4", "pow2"):
            calib, tests = generate_case_data(
                scenario, q_heads, kv_heads, head_dim, seq, 0
            )
            calib_pairs, test_pairs = encode_case(calib, tests, mode)
            states = solution.hif4_calibration_attention(
                calib_pairs, q_heads, kv_heads, head_dim
            )
            failures: list[str] = []
            for side in ("q_state", "k_state", "v_state"):
                tensors, _ = validate_state(states[side])
                assert tensors > 0
                check_state_tree(states[side], f"{label}/{side}", failures)
            check_dynamic_params(
                solution,
                test_pairs,
                states,
                q_heads,
                kv_heads,
                head_dim,
                f"{label}/mode={mode}",
                failures,
            )
            assert failures == []


def test_synthetic_case_generation_is_deterministic() -> None:
    first = generate_case_data("heavy_tail", 4, 2, 128, 32, 2)
    second = generate_case_data("heavy_tail", 4, 2, 128, 32, 2)
    for first_batch, second_batch in zip(first[0], second[0]):
        for side in ("q", "k", "v"):
            assert torch.equal(first_batch[side], second_batch[side])
    for first_case, second_case in zip(first[1], second[1]):
        for first_side, second_side in zip(first_case, second_case):
            assert torch.equal(first_side, second_side)


# ---------------------------------------------------------------------------
# Dormant Linear R64 incoherence-transform regression coverage.
# ---------------------------------------------------------------------------

_C21C_PARENT = (
    ROOT
    / "solutions"
    / "20260827_v025_c21c-compliance-baseline"
    / "solution.py"
)


def _r64_calibration_inputs(seed: int = 1234):
    torch.manual_seed(seed)
    weight = torch.randn(96, 128) * 0.05
    activations = [torch.randn(128, 128) * 0.5 for _ in range(2)]
    weight_pair = nvfp4_encode(weight, "amax6")
    calib_pairs = [nvfp4_encode(a, "amax6") for a in activations]
    return weight_pair, calib_pairs


def _r64_inverse(solution: ModuleType, y: torch.Tensor, seed: int):
    """Inverse of ``_apply_linear_r64``: FWHT then the same signs."""

    channels = int(y.shape[-1])
    signs = solution._linear_r64_signs(channels, seed, y.device, y.dtype)
    blocks = y.reshape(*y.shape[:-1], channels // 64, 64)
    z = solution._fwht_last_dim(blocks)
    return (z * signs.reshape(channels // 64, 64)).reshape(*y.shape)


def test_fwht64_matches_dense_hadamard() -> None:
    solution = load_module("fwht64", ROOT / "solution.py")
    torch.manual_seed(64)
    x = torch.randn(3, 5, 64)
    original = x.clone()
    dense = solution._hadamard_matrix_unchecked(64, x.device, x.dtype)
    reference = x @ dense
    fast = solution._fwht_last_dim(x)
    assert float((fast - reference).abs().max()) < 1.0e-5
    # The input is never destroyed in place.
    assert torch.equal(x, original)
    # bfloat16 runs through the same deterministic op sequence.
    bf16 = solution._fwht_last_dim(x.to(torch.bfloat16))
    assert torch.isfinite(bf16.to(torch.float32)).all()


def test_linear_r64_is_orthogonal() -> None:
    solution = load_module("r64_orthogonal", ROOT / "solution.py")
    eye = torch.eye(64)
    for seed in (0, 7, 31):
        r = solution._apply_linear_r64(eye, seed)
        err = (r.T @ r - torch.eye(64)).abs().max()
        assert float(err) < 1.0e-5


def test_linear_r64_activation_roundtrip() -> None:
    solution = load_module("r64_act_roundtrip", ROOT / "solution.py")
    torch.manual_seed(6401)
    x = torch.randn(16, 128)
    for seed in (0, 5, 31):
        y = solution._apply_linear_r64(x, seed)
        x_back = _r64_inverse(solution, y, seed)
        assert float((x_back - x).abs().max()) < 1.0e-5


def test_linear_r64_weight_roundtrip() -> None:
    solution = load_module("r64_w_roundtrip", ROOT / "solution.py")
    torch.manual_seed(6402)
    w = torch.randn(96, 128) * 0.05
    for seed in (1, 19):
        y = solution._apply_linear_r64(w, seed)
        w_back = _r64_inverse(solution, y, seed)
        assert float((w_back - w).abs().max()) < 1.0e-5


def test_linear_r64_state_is_seed_only() -> None:
    solution = load_module("r64_state", ROOT / "solution.py")
    solution._LINEAR_R64 = True
    weight_pair, calib_pairs = _r64_calibration_inputs()
    result = solution.hif4_calibration_and_quantize_weight(
        *weight_pair, calib_pairs
    )
    state = result["activation_state"]
    assert int(state["block_smooth_size"]) in (0, 4, 8, 16, 64)
    assert isinstance(state["block_smooth_seed"], int)
    validate_state(state)

    def walk(value) -> None:
        if torch.is_tensor(value):
            assert tuple(value.shape) != (64, 64)
        elif isinstance(value, dict):
            for child in value.values():
                walk(child)
        elif isinstance(value, (tuple, list)):
            for child in value:
                walk(child)

    walk(state)


def test_linear_r64_disabled_keeps_r64_path_dormant() -> None:
    solution = load_module("r64_disabled", ROOT / "solution.py")
    assert solution._LINEAR_R64 is False
    weight_pair, calib_pairs = _r64_calibration_inputs()
    result = solution.hif4_calibration_and_quantize_weight(
        *weight_pair, calib_pairs
    )
    # R64 is off: the returned dynamic state must stay seed-only (no
    # 64x64 mixing matrix in the state tree).
    state = result["activation_state"]

    def walk(value) -> None:
        if torch.is_tensor(value):
            assert tuple(value.shape) != (64, 64)
        elif isinstance(value, dict):
            for child in value.values():
                walk(child)
        elif isinstance(value, (tuple, list)):
            for child in value:
                walk(child)

    walk(state)


def test_linear_r64_candidate_falls_back_on_regression() -> None:
    solution = load_module("r64_fallback", ROOT / "solution.py")
    solution._LINEAR_R64 = True
    torch.manual_seed(6403)
    weight = torch.randn(64, 128) * 0.05
    activations = [torch.randn(32, 128) * 0.5 for _ in range(2)]
    d = torch.ones(128)
    permutation = torch.arange(128)
    second_moment = activations[0].square().mean(dim=0)
    seed, metrics = solution._select_r64_candidate(
        weight,
        activations,
        second_moment,
        d,
        permutation,
        0,
        0,
        (0.0, (0.0, 0.0)),
    )
    # A perfect baseline cannot be improved: the selection must keep the
    # parent transform (-1) instead of forcing an R64 seed.
    assert seed == -1
    assert metrics == (0.0, (0.0, 0.0))


def test_linear_r64_is_deterministic_cpu_cuda() -> None:
    solution = load_module("r64_determinism", ROOT / "solution.py")
    torch.manual_seed(6404)
    x = torch.randn(9, 128)
    first = solution._apply_linear_r64(x, 11)
    second = solution._apply_linear_r64(x, 11)
    assert torch.equal(first, second)
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available on this host")
    cuda_out = solution._apply_linear_r64(x.to("cuda"), 11)
    assert float((cuda_out.cpu() - first).abs().max()) < 1.0e-5
    cuda_again = solution._apply_linear_r64(x.to("cuda"), 11)
    assert torch.equal(cuda_out, cuda_again)
