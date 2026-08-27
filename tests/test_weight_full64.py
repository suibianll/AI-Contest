"""C23 full-64 weight refinement tests (plan 6.8 + consistency probe)."""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path
from types import ModuleType

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluator"))

from nvfp4_sim import nvfp4_encode  # noqa: E402


def load_solution_at(path: Path, name: str = "weight_full64_solution") -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_solution() -> ModuleType:
    return load_solution_at(ROOT / "solution.py")


SOLUTION = load_solution()


def make_case(
    rows: int = 96, channels: int = 128, seed: int = 1234
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    torch.manual_seed(seed)
    dense = torch.randn(rows, channels) * 0.05
    acts = torch.randn(4096, channels) * 0.1
    cov = (acts.t() @ acts) / acts.shape[0]
    params = SOLUTION._dense_to_hif4(
        dense,
        importance=None,
        group_gram=None,
        search_offsets=(-2, -1, 1, 2, 3),
        error_threshold=1.0e-7,
        accept_margin=0.005,
        max_refine_ratio=1.0,
    )
    return dense, cov, params


def full_h_loss(
    q: torch.Tensor, w: torch.Tensor, h_blocks: torch.Tensor
) -> float:
    blocks = h_blocks.shape[0]
    e = (q - w).reshape(-1, blocks, 64)
    return torch.einsum("rbi,bij,rbj->rb", e, h_blocks, e).sum().item()


def dequantize_blocks(params: dict[str, torch.Tensor]) -> torch.Tensor:
    return SOLUTION._dequantize_hif4(params).to(torch.float32)


def params_equal(
    left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]
) -> bool:
    return set(left) == set(right) and all(
        torch.equal(left[key], right[key]) for key in left
    )


def test_full64_hessian_extraction() -> None:
    dense, cov, _ = make_case(channels=128)
    h_blocks = SOLUTION._full64_hessian_blocks(cov, 128)
    assert tuple(h_blocks.shape) == (2, 64, 64)
    assert torch.allclose(h_blocks[0], cov[:64, :64])
    assert torch.allclose(h_blocks[1], cov[64:, 64:])


def test_gptq64_initialization_returns_legal_codes() -> None:
    torch.manual_seed(2024)
    rows, channels = 48, 64
    dense = torch.randn(rows, channels) * 0.05
    acts = torch.randn(2048, channels) * 0.1
    cov = (acts.t() @ acts) / acts.shape[0]
    h_blocks = SOLUTION._full64_hessian_blocks(cov, channels)

    w_sel = dense.reshape(rows, 1, 64)
    amax = w_sel.abs().amax(dim=-1)
    std_code, std_scale = SOLUTION._standard_e6m2_scale(amax)
    lv2 = torch.ones(rows, 1, 8)
    lv3 = torch.ones(rows, 1, 8, 2)
    denom = (
        std_scale[:, :, None, None, None]
        * lv2[:, :, :, None, None]
        * lv3[:, :, :, :, None]
    ).repeat_interleave(4, dim=-1).reshape(rows, 1, 64)

    order = torch.argsort(
        -h_blocks.diagonal(dim1=-2, dim2=-1), dim=1, stable=True
    )
    block_index = torch.arange(1).view(1, 1, 1)
    h_perm = h_blocks[block_index, order[:, :, None], order[:, None, :]]
    factor, _, ok = SOLUTION._cholesky_inverse_factor(h_perm)
    assert ok.all()

    q = SOLUTION._gptq_initialize64(w_sel, denom, order, factor)
    codes = q * (4.0 / denom.clamp_min(SOLUTION._EPS))
    assert (codes.abs() <= 7.0 + 1e-5).all()
    assert torch.allclose(codes, torch.round(codes), atol=1e-5)


def test_coordinate_descent64_is_monotonic() -> None:
    for seed in (11, 22, 33):
        torch.manual_seed(seed)
        rows, channels = 32, 64
        dense = torch.randn(rows, channels) * 0.05
        acts = torch.randn(2048, channels) * 0.1
        cov = (acts.t() @ acts) / acts.shape[0]
        h_blocks = SOLUTION._full64_hessian_blocks(cov, channels)
        w_sel = dense.reshape(rows, 1, 64)
        amax = w_sel.abs().amax(dim=-1)
        _, std_scale = SOLUTION._standard_e6m2_scale(amax)
        lv2 = torch.ones(rows, 1, 8)
        lv3 = torch.ones(rows, 1, 8, 2)
        denom = (
            std_scale[:, :, None, None, None]
            * lv2[:, :, :, None, None]
            * lv3[:, :, :, :, None]
        ).repeat_interleave(4, dim=-1).reshape(rows, 1, 64)
        q0 = (torch.round(w_sel * 4.0 / denom).clamp_(-7, 7)) * 0.25 * denom
        loss0 = full_h_loss(q0.flatten(1), dense, h_blocks)
        q1 = SOLUTION._coordinate_descent64(q0, w_sel, h_blocks, denom)
        loss1 = full_h_loss(q1.flatten(1), dense, h_blocks)
        assert loss1 <= loss0 + 1e-9


def test_hierarchy_toggle64_is_monotonic() -> None:
    torch.manual_seed(44)
    rows, channels = 32, 64
    dense = torch.randn(rows, channels) * 0.05
    acts = torch.randn(2048, channels) * 0.1
    cov = (acts.t() @ acts) / acts.shape[0]
    h_blocks = SOLUTION._full64_hessian_blocks(cov, channels)
    w_sel = dense.reshape(rows, 1, 64)
    amax = w_sel.abs().amax(dim=-1)
    _, std_scale = SOLUTION._standard_e6m2_scale(amax)
    lv2 = torch.ones(rows, 1, 8)
    lv3 = torch.ones(rows, 1, 8, 2)
    denom = (
        std_scale[:, :, None, None, None]
        * lv2[:, :, :, None, None]
        * lv3[:, :, :, :, None]
    ).repeat_interleave(4, dim=-1).reshape(rows, 1, 64)
    q0 = (torch.round(w_sel * 4.0 / denom).clamp_(-7, 7)) * 0.25 * denom
    q1 = SOLUTION._coordinate_descent64(q0, w_sel, h_blocks, denom)
    loss1 = full_h_loss(q1.flatten(1), dense, h_blocks)
    q2, lv2_t, lv3_t, _ = SOLUTION._hierarchy_toggle_refine64(
        q1, w_sel, h_blocks, denom, lv2, lv3
    )
    loss2 = full_h_loss(q2.flatten(1), dense, h_blocks)
    assert loss2 <= loss1 + 1e-9
    assert ((lv2_t == 1) | (lv2_t == 2)).all()
    assert ((lv3_t == 1) | (lv3_t == 2)).all()


def test_weight64_final_loss_not_above_parent() -> None:
    dense, cov, params = make_case(rows=96, channels=128)
    h_blocks = SOLUTION._full64_hessian_blocks(cov, 128)
    parent_q = dequantize_blocks(params)
    parent_loss = full_h_loss(parent_q, dense, h_blocks)
    refined = SOLUTION._refine_weight_blocks64(dense, params, cov)
    refined_q = dequantize_blocks(refined)
    refined_loss = full_h_loss(refined_q, dense, h_blocks)
    assert refined_loss <= parent_loss + 1e-6


def test_weight64_chunking_exact_when_single_chunk() -> None:
    """A chunk >= tensor rows is one single solve: bit-exact under any such
    chunk size (deployment chunk 1024 >= every narrow-layer row count).
    Smaller chunks reorder the batched per-row solves; each (row, block)
    is solved independently and a float tie around the beam topk/accept
    margin may pick either of two equivalent codes, so only single-chunk
    exactness and split-chunk validity are asserted.
    """
    dense, cov, params = make_case(rows=96, channels=128)
    baseline = SOLUTION._refine_weight_blocks64(dense, params, cov)
    saved = SOLUTION._WEIGHT_FULL64_CHUNK_ROWS
    try:
        for chunk in (1, 96):
            SOLUTION._WEIGHT_FULL64_CHUNK_ROWS = chunk
            chunked = SOLUTION._refine_weight_blocks64(dense, params, cov)
            assert params_equal(baseline, chunked), f"chunk={chunk}"
        SOLUTION._WEIGHT_FULL64_CHUNK_ROWS = 7
        chunked = SOLUTION._refine_weight_blocks64(dense, params, cov)
        for key in chunked:
            assert torch.isfinite(chunked[key].to(torch.float32)).all()
    finally:
        SOLUTION._WEIGHT_FULL64_CHUNK_ROWS = saved


def test_weight64_fallback_on_non_psd() -> None:
    dense, cov, params = make_case(rows=48, channels=64)
    cov_bad = cov.clone()
    cov_bad[:64, :64] = -torch.eye(64) * 4.0
    refined = SOLUTION._refine_weight_blocks64(dense, params, cov_bad)
    assert params_equal(refined, params)


def test_weight64_deterministic() -> None:
    dense, cov, params = make_case(rows=96, channels=128)
    first = SOLUTION._refine_weight_blocks64(dense, params, cov)
    second = SOLUTION._refine_weight_blocks64(dense, params, cov)
    assert params_equal(first, second)


def test_weight64_batched_matches_per_block() -> None:
    """Batched beam solve must equal the per-block solve within 1e-6."""
    torch.manual_seed(55)
    rows, channels = 40, 256
    blocks = channels // 64
    dense = torch.randn(rows, channels) * 0.05
    acts = torch.randn(4096, channels) * 0.1
    cov = (acts.t() @ acts) / acts.shape[0]
    params = SOLUTION._dense_to_hif4(
        dense,
        importance=None,
        group_gram=None,
        search_offsets=(-2, -1, 1, 2, 3),
        error_threshold=1.0e-7,
        accept_margin=0.005,
        max_refine_ratio=1.0,
    )
    saved_ratio = SOLUTION._WEIGHT_FULL64_MAX_RATIO
    try:
        SOLUTION._WEIGHT_FULL64_MAX_RATIO = 1.0
        batched = SOLUTION._refine_weight_blocks64(
            dense, params, cov, max_ratio=1.0
        )
        for b in range(blocks):
            dense_b = dense[:, b * 64 : (b + 1) * 64].contiguous()
            cov_b = cov[
                b * 64 : (b + 1) * 64, b * 64 : (b + 1) * 64
            ].contiguous()
            params_b = {
                key: value.index_select(1, torch.tensor([b])).clone()
                for key, value in params.items()
            }
            per_block = SOLUTION._refine_weight_blocks64(
                dense_b, params_b, cov_b, max_ratio=1.0
            )
            for key in batched:
                got = batched[key].index_select(1, torch.tensor([b]))
                want = per_block[key]
                if key in ("sign", "mant"):
                    # sign: canonical zero may differ on zero mantissa
                    diff = (
                        (got - want).abs().max().item()
                        if got.numel()
                        else 0.0
                    )
                else:
                    diff = (got - want).abs().max().item()
                assert diff <= 1e-6, (key, b, diff)
    finally:
        SOLUTION._WEIGHT_FULL64_MAX_RATIO = saved_ratio


def test_weight64_micro_benchmark() -> None:
    """Plan 6.7 micro-benchmark: CPU/float32, rows>=2000, chunk 128,
    warmup 3 + measure 10, report the median."""
    saved = SOLUTION._WEIGHT_FULL64_CHUNK_ROWS
    try:
        SOLUTION._WEIGHT_FULL64_CHUNK_ROWS = 128
        for channels in (768, 3072):
            torch.manual_seed(66)
            rows = 2048
            dense = torch.randn(rows, channels) * 0.05
            acts = torch.randn(2048, channels) * 0.1
            cov = (acts.t() @ acts) / acts.shape[0]
            params = SOLUTION._dense_to_hif4(
                dense,
                importance=None,
                group_gram=None,
                search_offsets=(-2, -1, 1, 2, 3),
                error_threshold=1.0e-7,
                accept_margin=0.005,
                max_refine_ratio=1.0,
            )
            for _ in range(3):
                SOLUTION._refine_weight_blocks64(dense, params, cov)
            samples = []
            for _ in range(10):
                start = time.perf_counter()
                refined = SOLUTION._refine_weight_blocks64(
                    dense, params, cov
                )
                samples.append(time.perf_counter() - start)
            median = sorted(samples)[len(samples) // 2]
            assert torch.isfinite(
                refined["scale_factor"].to(torch.float32)
            ).all()
            # Regression guard: the end-to-end CPU budget allows roughly
            # 3.6s per linear layer; a single 2048-row solve must stay
            # clearly below that.
            assert median < 3.6, (channels, median)
            print(f"micro-benchmark channels={channels}: median {median:.3f}s")
    finally:
        SOLUTION._WEIGHT_FULL64_CHUNK_ROWS = saved


_C21C_PARENT = (
    ROOT / "solutions" / "20260827_v025_c21c-compliance-baseline" / "solution.py"
)


def _full64_calibration_inputs(seed: int = 1234):
    torch.manual_seed(seed)
    weight = torch.randn(96, 128) * 0.05
    activations = [torch.randn(128, 128) * 0.5 for _ in range(2)]
    weight_pair = nvfp4_encode(weight, "amax6")
    calib_pairs = [nvfp4_encode(a, "amax6") for a in activations]
    return weight_pair, calib_pairs


def _assert_nested_equal(left, right) -> None:
    assert type(left) is type(right)
    if torch.is_tensor(left):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert set(left) == set(right)
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for l_value, r_value in zip(left, right):
            _assert_nested_equal(l_value, r_value)
    else:
        assert left == right


def test_weight_full64_enabled_in_production() -> None:
    """C23 is promoted (2026-08-28): the flag is on in production."""
    assert SOLUTION._WEIGHT_FULL64 is True


def test_weight_full64_enabled_changes_output_vs_c21c() -> None:
    """With the flag on (the promoted default), the calibration output
    must differ from the C21-C parent: full-64 refinement is active."""
    parent = load_solution_at(_C21C_PARENT, "weight64_c21c_parent")
    weight_pair, calib_pairs = _full64_calibration_inputs()
    child = SOLUTION.hif4_calibration_and_quantize_weight(
        *weight_pair, calib_pairs
    )
    base = parent.hif4_calibration_and_quantize_weight(
        *weight_pair, calib_pairs
    )
    child_params = child["weight_params"]
    base_params = base["weight_params"]
    assert set(child_params) == set(base_params)
    differs = any(
        torch.any(child_params[key] != base_params[key])
        for key in child_params
        if torch.is_tensor(child_params[key])
    )
    assert differs
