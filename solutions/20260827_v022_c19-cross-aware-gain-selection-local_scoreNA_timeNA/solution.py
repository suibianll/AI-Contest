"""HiF4 solution for the 2026 Huawei algorithm competition.

The implementation keeps the official HiF4 conversion as an explicit fallback,
selects calibration-gated equivalent scaling/reordering/block-matrix transforms,
and applies bounded scale/hierarchy refinement to difficult blocks. All
calibration states are plain CPU data.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Optional, Sequence, Union

import torch


_NVFP4_BLOCK_SIZE = 16
_HIF4_BLOCK_SIZE = 64
_E6M2_MIN = 2.0**-48
_E6M2_MAX = 49152.0
_HIF4_MAX_INNER = 7.0
_BF16_ONE_SEVENTH = 0.142578125
_EPS = 1.0e-12

_LINEAR_STATS_TOKENS = 4096
_LINEAR_EVAL_TOKENS = 128
_LINEAR_WEIGHT_EVAL_ROWS = 256
_ATTN_STATS_TOKENS = 4096
_ATTN_EVAL_TOKENS = 128

# Tunable calibration/refinement knobs.  Values are deliberately
# conservative so worst-case calibration and dynamic-quantization time stay
# bounded on large models; the sweep harness overrides them to find a better
# accuracy/runtime trade-off.
_WEIGHT_SMOOTH_ALPHAS = (0.25, 0.50, 0.75)
# 宽层（FFN 的 fc/proj，输入或输出 ≥ 2048）用更细的 alpha 网格：
# 通道多、候选统计更稳，细网格实测 +fc 0.0012 / +proj 0.0044；
# 窄层（q/k/v/o，768）细网格反而过拟合（-0.0003~-0.0021），保持 3 档。
_WEIGHT_SMOOTH_ALPHAS_WIDE = (0.25, 0.375, 0.50, 0.625, 0.75)
_WIDE_LAYER_MIN_DIM = 2048
# Matrix SmoothQuant candidates.  After the diagonal scale and optional
# permutation, apply the same orthogonal block transform to X and W.  In the
# usual X @ W convention this is a block-diagonal S on X and S^{-1} on W;
# orthogonality makes S^{-1}=S^T, so the row-major weight carrier can use the
# same right transform.  Only the winning block size enters dynamic state.
_BLOCK_SMOOTH_ALLOWED_SIZES = (4, 8, 16)
_BLOCK_SMOOTH_SIZES = _BLOCK_SMOOTH_ALLOWED_SIZES
_BLOCK_SMOOTH_SEEDS = (0, 1, 2, 3)
# Evaluation-only override used by the sweep harness.  Zero keeps the guarded
# production behavior; 4/8/16 forces that size while still choosing its best
# deterministic sign seed on calibration data.
_BLOCK_SMOOTH_FORCE_SIZE = 0
_BLOCK_SMOOTH_MIN_IMPROVEMENT = 0.005
_BLOCK_SMOOTH_WORST_TOLERANCE = 0.005
_WEIGHT_REFINE_ERROR_THRESHOLD = 1.0e-7
_WEIGHT_REFINE_ACCEPT_MARGIN = 0.005
_WEIGHT_REFINE_MAX_RATIO_SMALL = 1.0
_WEIGHT_REFINE_MAX_RATIO_LARGE = 1.0
_WEIGHT_REFINE_MAX_BLOCKS = 65_536

_ACTIVATION_REFINE_ERROR_THRESHOLD = 1.0e-7
_ACTIVATION_REFINE_ACCEPT_MARGIN = 0.02
_ACTIVATION_REFINE_MAX_RATIO = 0.70
_ACTIVATION_REFINE_MAX_BLOCKS = 32_768

_QK_SMOOTH_ALPHAS = (0.25, 0.50)
_WEIGHT_SMOOTH_RMS = True
_QK_SMOOTH_RMS = True
_ATTN_CENTER_MODES = (0, 2)
# A1: score attention calibration candidates by the real attention output
# error (causal primary, non-causal safety) instead of the Q/K reconstruction
# proxy.  Calibration-only cost; the dynamic path is unchanged.
# A1 is the current local Champion: it has a stable aggregate improvement
# across MHA/GQA, masks, token windows, and NVFP4 scale modes.  Individual
# tail regressions are tracked as the next optimization target instead of
# discarding the aggregate gain.
_ATTN_OUTPUT_SELECTOR = True
_ATTN_A1_MAX_TOKENS = 256
# A1 终验门：A1 候选排序基于朴素 HiF4 代理，部署路径（offset 搜索 +
# importance 困难块精修）与隐藏 test 分布的排序都可能错位。终验门用完整
# 部署路径重算真实 attention 输出误差，A1 winner 相对 B0 proxy winner
# （当前 Champion 的选择）需满足：causal 均值至少相对改善
# _A1_GATE_MIN_IMPROVEMENT，且逐样本与 non-causal 安全轨均值退化不超过
# _A1_GATE_WORST_TOLERANCE；否则逐层回退 B0 选择，保证不低于 Champion。
_A1_GATE_MIN_IMPROVEMENT = 0.005
_A1_GATE_WORST_TOLERANCE = 0.02
# A2 固定 H64：对 Q/K winner 施加组对齐 signed Hadamard 旋转（每个连续
# 64 维 head block 独立旋转，GQA 中同组 Q heads 与 K head 共享同一旋转，
# 保证 Q·K 点积不变）。首版只比较 2 个确定性 sign seed，旋转候选须通过
# 与 A1 相同的真实 attention 输出门控，否则保持无旋转 winner。配对复核
# 发现 MHA 单层和 GQA non-causal 尾部超过发布门槛；根候选回退为 A1-only，
# A2 实现保留供后续单机制官方配对。
_ATTN_H64 = False
_ATTN_H64_SEEDS = (0, 1)
_ATTN_H64_BLOCK = 64
# A3 V importance 候选：当前 E[A^2]（二阶矩对角项）之外，比较
#  - 一阶矩 E[A]（attention 概率推导的 head 权重，捕捉输出偏差项）；
#  - E[A^2] + E[A]^2（对角项 + 均值交叉项，均值误差抑制）。
# 候选仅改变 head 级 importance 向量（接口不变、动态路径零成本），
# 须经真实 attention 输出门控，否则保持当前 importance。
# 2026-08-26 配对评测：MHA causal -0.06pp / GQA 持平，未达 +0.2pp 晋级
# 门槛（且 MHA L5 -0.8pp），按 spec §10 不晋级，默认关闭（实现保留）。
_V_IMPORTANCE_CANDIDATES = False
# L1 数据驱动 scale（spec §6.1）：困难块在固定 offset 网格之外，额外生成
#  - 加权 LS 连续 scale：以当前五字段 winner 的离散解为锚点（mantissa
#    视为固定）求闭式 LS scale，再经 ±1 相邻 E6M2 code 与精确层级求解
#    交替更新（spec 所述“一至两轮交替”由相邻 code + exact solve 覆盖）；
#  - 截尾分位数 scale：块内 |x| 的 _L1_TRIM_QUANTILES 分位映射到顶层
#    mantissa（有意截断块内离群值，降低截尾 LS 误差）。
# 所有候选仍为合法 E6M2 code，与当前完整五字段 winner 逐块比较，未降低
# 损失时逐块回退（复用 offset 搜索的 improve-mask 机制）。
# 2026-08-26 诊断（artifacts/diag_l1_weights.py）与配对评测：weights 触发
# 15–119 块/矩阵但 plain MSE 仅改善 0.01–0.03%（分数 4 位小数不可见，
# 未达 +0.2pp 门槛）；attention 逐块损失下降但真实输出退化（Q/K 经
# softmax 非线性放大，proxy 错位，MHA -0.6pp）；dynamic 时间 +1.8s。
# 判定不晋级，默认关闭（实现保留，触发路径见 _dense_to_hif4 困难块段）。
_L1_DATA_DRIVEN_SCALE = False
_L1_TRIM_QUANTILES = (0.90, 0.95)
_L1_ADJACENT_CODE_DELTAS = (-1, 0, 1)
# Rank refinement by absolute block error (True) or normalized error (False).
_REFINE_RANK_BY_ABSOLUTE = True
_ATTN_REFINE_ERROR_THRESHOLD = 1.0e-7
_Q_REFINE_ACCEPT_MARGIN = 0.03
_Q_REFINE_MAX_RATIO = 0.60
_Q_REFINE_MAX_BLOCKS = 16_384
_K_REFINE_ACCEPT_MARGIN = 0.03
_K_REFINE_MAX_RATIO = 0.70
_K_REFINE_MAX_BLOCKS = 24_576
_V_REFINE_ACCEPT_MARGIN = 0.01
_V_REFINE_MAX_RATIO = 0.60
_V_REFINE_MAX_BLOCKS = 24_576

_SMOOTH_SCALE_MIN = 1.0 / 8.0
_SMOOTH_SCALE_MAX = 8.0
_QK_SMOOTH_MIN = 1.0 / 16.0
_QK_SMOOTH_MAX = 16.0

# Importance weights are mean-normalized; a floor keeps every channel at least
# this fraction of the mean.  Without it, calibrated importance can be ~0 for
# whole blocks (e.g. outlier-heavy Q/K data), their weighted losses vanish, and
# the scale search drifts on numerical noise while degrading the unweighted
# reconstruction of those blocks.
_IMPORTANCE_FLOOR = 0.05

# E6M2 code offsets.  Offset +2 is roughly the E6M2 analogue of the
# alternative 1.5x scale mode seen in microscaling scale search.
#
# Exact-solve analysis: the standard amax/7 code frequently rounds DOWN
# (clipping the block peak), and all other 4-element subgroups sit on
# arbitrary log phases, so the useful neighborhood is +1..+3 (~1.25x..2x
# scale).  Negative offsets shrink the representable range and almost never
# win on the common NVFP4 input regimes; -1 is kept as a fallback for
# overshooting codes.  -2 is kept for weights (calibration-only cost) as
# insurance for finer-grained input encodings where it wins up to a few
# percent of blocks, but dropped from the per-sample dynamic path.
_DYNAMIC_OFFSETS = (-1, 1, 2, 3)
_WEIGHT_OFFSETS = (-2, -1, 1, 2, 3)

# The per-block scale error over E6M2 codes is locally unimodal, so if the
# best fixed-window offset lands on a window edge, the true optimum may lie
# outside the window.  Extend the search beyond the winning edge (only for
# blocks that actually hit the edge) by up to this many extra codes.
_REFINE_EDGE_EXTENSION = True
_REFINE_EDGE_EXTEND_STEPS = 2

# Data-driven per-layer refine budgets: instead of a global hand-tuned ratio,
# calibration estimates the block-loss distribution and stores the smallest
# refine fraction that captures a target share of the total weighted loss.
_DATA_DRIVEN_RATIO = True
# 时间预算允许放宽后，把损失覆盖目标从 0.99 提到 0.999：实际 refine
# 比例从 ~0.95 提到 ~0.99（几乎全量），8 批测试上 7 类得分全部为正
# （attn +0.0018，其余 +0.0001~0.0003），动态耗时 +约 5%。
_RATIO_CAPTURE_TARGET = 0.999
_RATIO_MIN = 0.10

# Weight quantization can use the full per-block activation covariance as a
# quadratic loss (true output-MSE weighting) instead of the diagonal
# per-channel importance.  This is calibration-only: the Gram/covariance
# never enters a dynamic state, so the 4096-node state limit is unaffected.
_WEIGHT_QUADRATIC = True
_WEIGHT_QUADRATIC_MAX_FEATURES = 4096
# C3: after the existing exact 4x4 hierarchy solve, refine only the highest
# quadratic-loss 8-channel groups with coordinate updates driven by H*e.
# This is weight-calibration-only and adds nothing to dynamic state.
_WEIGHT_QUADRATIC8 = True
_WEIGHT_QUADRATIC8_MAX_RATIO = 0.05
_WEIGHT_QUADRATIC8_MAX_GROUPS = 8192
_WEIGHT_QUADRATIC8_SWEEPS = 2
_WEIGHT_QUADRATIC8_ACCEPT_MARGIN = 1.0e-5
_WEIGHT_QUADRATIC16 = True
_WEIGHT_QUADRATIC16_MAX_RATIO = 0.02
_WEIGHT_QUADRATIC16_MAX_GROUPS = 4096
_WEIGHT_QUADRATIC16_SWEEPS = 1
_WEIGHT_QUADRATIC16_ACCEPT_MARGIN = 1.0e-5

# Dynamic activation quantization can also use the full weight Gram (W^T W)
# as quadratic error weights.  Unlike Q/K covariances (estimated from a few
# calibration tokens), the weight Gram is static and well conditioned, so the
# same machinery that helped weight quantization should transfer.  Only the
# per-4-group 4x4 blocks are stored in the state (~4*channels elements).
_ACTIVATION_QUADRATIC = True
# Gram state is ~4*channels elements; cap so the single stored tensor stays
# within 4096 elements even under a strict element-count reading of the state
# node limit.  Wide layers (e.g. FFN down-projection, 3072) fall back to the
# diagonal importance automatically.
_ACTIVATION_QUADRATIC_MAX_FEATURES = 4096
_ACTIVATION_QUADRATIC8 = True
_ACTIVATION_QUADRATIC8_MIN_FEATURES = 64
_ACTIVATION_QUADRATIC8_MAX_RATIO = 0.08
_ACTIVATION_QUADRATIC8_MAX_GROUPS = 4096
_ACTIVATION_QUADRATIC8_SWEEPS = 1
_ACTIVATION_QUADRATIC8_ACCEPT_MARGIN = 1.0e-5
_ACTIVATION_QUADRATIC8_CALIBRATION_GATE = True
_ACTIVATION_QUADRATIC8_GATE_MAX_FEATURES = 1024
_ACTIVATION_QUADRATIC8_GATE_MIN_IMPROVEMENT = 5.0e-4
_ACTIVATION_QUADRATIC8_GATE_WORST_TOLERANCE = 1.0e-3
_ACTIVATION_QUADRATIC8_CROSS_TERM = True
_ACTIVATION_QUADRATIC8_CROSS_GAIN_SELECTION = True

# Permutation search bases.  The initial hierarchy-aware ordering combines the
# paired operands via max(log range); real-data diagnostics show the operand
# with the larger quantization burden (usually the weight/K side) often yields
# a better single-sided ordering.  Each basis is evaluated with the exact
# paired metric and accepted only when it clears the same safety gate as the
# smoothing candidates.
_PERMUTATION_BASES = True

# V 量化目前没有任何重要性：V 的误差进入 softmax 输出时被注意力权重
# 放大，而校准期可以静态估计每 KV head 的平均平方注意力质量
# E[A^2]（softmax 概率，因果掩码下按 token 位置平均）。head_dim=64 时
# 一个 HiF4 64 块恰好是一个 head 的一条位置切片，按 head 加权可以直接
# 作用到块上。与 Q/K 协方差不同，这是静态概率统计，不会过拟合少量 token。
_V_ATTENTION_IMPORTANCE = True
# 向均匀权重收缩的比例（0.0 = 均匀，1.0 = 完全按 E[A^2] 加权），
# 实测完全加权最优（attn +0.0010），收缩反而稀释收益。
_V_ATTENTION_IMPORTANCE_SHRINK = 1.0


def dequantize_nvfp4(
    quant_float: torch.Tensor,
    scale_float: torch.Tensor,
    blk_size: int = _NVFP4_BLOCK_SIZE,
) -> torch.Tensor:
    """Dequantize an NVFP4 carrier/scale pair to BF16."""

    if not torch.is_tensor(quant_float) or not torch.is_tensor(scale_float):
        raise TypeError("quant_float and scale_float must be torch.Tensor")
    if quant_float.ndim < 1:
        raise ValueError("quant_float must have at least one dimension")
    c = int(quant_float.shape[-1])
    if c % blk_size != 0:
        raise ValueError(
            f"Last dim {c} is not divisible by NVFP4 block size {blk_size}"
        )
    expected_scale_shape = tuple(quant_float.shape[:-1]) + (c // blk_size,)
    if tuple(scale_float.shape) != expected_scale_shape:
        raise ValueError(
            f"scale_float shape {tuple(scale_float.shape)} does not match "
            f"expected {expected_scale_shape}"
        )

    x = quant_float.unflatten(-1, (-1, blk_size))
    result = x * scale_float.unsqueeze(-1)
    return result.flatten(-2, -1).to(torch.bfloat16)


def _dequantize_nvfp4_float32(
    quant_float: torch.Tensor,
    scale_float: torch.Tensor,
) -> torch.Tensor:
    """Match the supplied BF16 dequantizer, then use FP32 for optimization."""

    return dequantize_nvfp4(quant_float, scale_float).to(torch.float32)


def _sample_rows(x: torch.Tensor, limit: int) -> torch.Tensor:
    """Deterministically sample at most ``limit`` rows without random state."""

    rows = int(x.shape[0])
    if rows <= limit:
        return x
    step = max(1, (rows + limit - 1) // limit)
    return x[::step][:limit]


def _safe_positive_vector(x: torch.Tensor, length: int) -> torch.Tensor:
    """Return a finite, positive FP32 vector of the requested length."""

    y = x.detach().to(dtype=torch.float32).reshape(-1)
    if int(y.numel()) != length:
        raise ValueError(f"Expected vector of length {length}, got {y.numel()}")
    return torch.nan_to_num(
        y, nan=1.0, posinf=1.0, neginf=1.0
    ).clamp_min(_EPS)


def _normalize_importance(
    importance: Optional[torch.Tensor],
    length: int,
) -> Optional[torch.Tensor]:
    if importance is None:
        return None
    w = importance.detach().to(dtype=torch.float32).reshape(-1)
    if int(w.numel()) != length:
        raise ValueError(
            f"Expected importance of length {length}, got {w.numel()}"
        )
    w = torch.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    mean = w.mean()
    if float(mean) <= _EPS:
        return torch.ones_like(w)
    return (w / mean).clamp_min(_IMPORTANCE_FLOOR)


def _standard_block_losses(
    dense: torch.Tensor,
    importance: Optional[torch.Tensor],
) -> torch.Tensor:
    """Per-block importance-weighted squared error of standard HiF4."""

    prefix = tuple(int(v) for v in dense.shape[:-1])
    channels = int(dense.shape[-1])
    if channels % _HIF4_BLOCK_SIZE != 0:
        raise ValueError(
            f"Last dim {channels} is not divisible by HiF4 block size 64"
        )
    blocks = channels // _HIF4_BLOCK_SIZE

    x = torch.nan_to_num(
        dense.detach().to(torch.float32),
        nan=0.0,
        posinf=_E6M2_MAX * _HIF4_MAX_INNER,
        neginf=-_E6M2_MAX * _HIF4_MAX_INNER,
    )
    x_grouped = x.reshape(*prefix, blocks, 8, 2, 4)
    x_abs = x_grouped.abs()
    max4 = x_abs.amax(dim=-1)
    max8 = max4.amax(dim=-1)
    amax = max8.amax(dim=-1)
    _, standard_scale = _standard_e6m2_scale(amax)

    e2 = max8 >= (4.0 * standard_scale[..., None])
    scale_lv2 = 1.0 + e2.to(torch.float32)
    e3 = max4 >= (
        2.0 * standard_scale[..., None, None] * scale_lv2[..., None]
    )
    scale_lv3 = 1.0 + e3.to(torch.float32)
    denominator = (
        standard_scale[..., None, None, None]
        * scale_lv2[..., None, None]
        * scale_lv3[..., None]
    )
    mantissa = (
        torch.round(x_abs * (4.0 / denominator)).clamp_(0.0, 7.0) * 0.25
    )

    channel_importance = _normalize_importance(importance, channels)
    if channel_importance is None:
        weighted_error = (x_abs - mantissa * denominator).square()
    else:
        weighted_error = (
            (x_abs - mantissa * denominator).square()
            * channel_importance.reshape(*([1] * len(prefix)), blocks, 8, 2, 4)
        )
    return weighted_error.sum(dim=(-1, -2, -3)).reshape(-1)


def _loss_capture_ratio(
    losses: torch.Tensor,
    *,
    target: float,
    ratio_min: float,
) -> float:
    """Smallest fraction of the largest-loss blocks covering ``target`` of the
    total loss.  This converts the per-block loss tail into a refine budget."""

    losses = losses.detach().to(torch.float32).reshape(-1)
    total = float(losses.sum())
    if total <= _EPS:
        return float(ratio_min)
    sorted_descending = torch.sort(losses, descending=True).values
    cumulative = torch.cumsum(sorted_descending, dim=0)
    k = int((cumulative < float(target) * total).sum()) + 1
    return float(
        min(1.0, max(float(ratio_min), k / max(1, int(losses.numel()))))
    )


def _flat_group_gram(cov: torch.Tensor, channels: int) -> torch.Tensor:
    """Extract per-4-group 4x4 block-diagonal quadratic weights as a flat
    ``[channels // 4, 4, 4]`` tensor (the only part the solver needs)."""

    blocks = channels // _HIF4_BLOCK_SIZE
    g = cov.reshape(blocks, 64, blocks, 64)
    g = torch.diagonal(g, dim1=0, dim2=2).permute(2, 0, 1)
    g = g.reshape(blocks, 16, 4, 16, 4)
    g = torch.diagonal(g, dim1=1, dim2=3).permute(0, 3, 1, 2)
    return g.reshape(blocks * 16, 4, 4)


def _flat_group_gram8(cov: torch.Tensor, channels: int) -> torch.Tensor:
    """Extract contiguous per-8-channel covariance blocks."""

    indices = torch.arange(channels, device=cov.device).reshape(-1, 8)
    return cov[indices[:, :, None], indices[:, None, :]]


def _flat_group_gram16(cov: torch.Tensor, channels: int) -> torch.Tensor:
    """Extract contiguous per-16-channel covariance blocks."""

    indices = torch.arange(channels, device=cov.device).reshape(-1, 16)
    return cov[indices[:, :, None], indices[:, None, :]]


def _refine_weight_groups8(
    dense: torch.Tensor,
    params: dict[str, torch.Tensor],
    group_gram8: torch.Tensor,
    *,
    group_cross8: Optional[torch.Tensor] = None,
    max_ratio: float = _WEIGHT_QUADRATIC8_MAX_RATIO,
    max_groups: int = _WEIGHT_QUADRATIC8_MAX_GROUPS,
    sweeps: int = _WEIGHT_QUADRATIC8_SWEEPS,
    accept_margin: float = _WEIGHT_QUADRATIC8_ACCEPT_MARGIN,
) -> dict[str, torch.Tensor]:
    """Coordinate-refine top-loss 8-channel groups using incremental H*e."""

    if dense.ndim != 2:
        return params
    rows, channels = map(int, dense.shape)
    if channels % _HIF4_BLOCK_SIZE != 0 or channels % 8 != 0:
        return params
    blocks = channels // _HIF4_BLOCK_SIZE
    expected_grams = blocks * 8
    if tuple(group_gram8.shape) != (expected_grams, 8, 8):
        return params
    if group_cross8 is not None and tuple(group_cross8.shape) != (
        expected_grams,
        8,
        8,
    ):
        return params

    dense8 = dense.reshape(rows, blocks, 8, 8).reshape(-1, 8)
    quantized8 = _dequantize_hif4(params).to(torch.float32).reshape(
        rows, blocks, 8, 8
    ).reshape(-1, 8)
    grams = group_gram8.unsqueeze(0).expand(rows, -1, -1, -1).reshape(
        -1, 8, 8
    )
    crosses = None
    if group_cross8 is not None:
        crosses = group_cross8.unsqueeze(0).expand(
            rows, -1, -1, -1
        ).reshape(-1, 8, 8)
    error = quantized8 - dense8
    losses = torch.einsum("ni,nij,nj->n", error, grams, error)
    ranking_scores = losses
    linear_all = None
    if (
        crosses is not None
        and _ACTIVATION_QUADRATIC8_CROSS_GAIN_SELECTION
    ):
        linear_all = torch.einsum("ni,nij->nj", dense8, crosses)
        gradient = torch.einsum("nij,nj->ni", grams, error) + linear_all
        diagonal = torch.diagonal(grams, dim1=-2, dim2=-1).clamp_min(_EPS)
        ranking_scores = (gradient.square() / diagonal).amax(dim=1)
    finite = torch.isfinite(ranking_scores) & (ranking_scores > _EPS)
    candidates = torch.nonzero(finite, as_tuple=False).reshape(-1)
    if int(candidates.numel()) == 0:
        return params
    cap = max(
        1,
        int(
            math.ceil(
                int(losses.numel()) * float(max_ratio)
            )
        ),
    )
    cap = min(cap, int(max_groups), int(candidates.numel()))
    if int(candidates.numel()) > cap:
        order = torch.topk(
            ranking_scores.index_select(0, candidates), k=cap, largest=True
        ).indices
        candidates = candidates.index_select(0, order)

    x_selected = dense8.index_select(0, candidates)
    q_selected = quantized8.index_select(0, candidates).clone()
    gram_selected = grams.index_select(0, candidates)
    cross_selected = (
        None if crosses is None else crosses.index_select(0, candidates)
    )
    error_selected = q_selected - x_selected
    he = torch.einsum("nij,nj->ni", gram_selected, error_selected)
    initial_loss = torch.einsum(
        "ni,nij,nj->n", error_selected, gram_selected, error_selected
    )
    linear = None
    if cross_selected is not None:
        linear = (
            linear_all.index_select(0, candidates)
            if linear_all is not None
            else torch.einsum("ni,nij->nj", x_selected, cross_selected)
        )
        initial_loss = initial_loss + 2.0 * (
            error_selected * linear
        ).sum(dim=1)

    scale = params["scale_factor"].reshape(rows, blocks, 1).expand(
        rows, blocks, 8
    )
    lv2 = params["scale_lv2"].reshape(rows, blocks, 8)
    lv3 = params["scale_lv3"].reshape(rows, blocks, 8, 2)
    denominator = (
        scale[..., None]
        * lv2[..., None]
        * lv3.repeat_interleave(4, dim=-1)
    ).reshape(-1, 8).index_select(0, candidates)
    signed_codes = torch.arange(
        -7, 8, dtype=torch.float32, device=dense.device
    ) * 0.25

    for _ in range(int(sweeps)):
        for coordinate in range(8):
            possible = denominator[:, coordinate, None] * signed_codes[None, :]
            delta = possible - q_selected[:, coordinate, None]
            diagonal = gram_selected[:, coordinate, coordinate].clamp_min(_EPS)
            change = (
                2.0
                * delta
                * (
                    he[:, coordinate, None]
                    if linear is None
                    else he[:, coordinate, None]
                    + linear[:, coordinate, None]
                )
                + delta.square() * diagonal[:, None]
            )
            best = change.argmin(dim=1)
            row_ids = torch.arange(int(candidates.numel()), device=dense.device)
            best_delta = delta[row_ids, best]
            improve = change[row_ids, best] < -_EPS
            best_delta = torch.where(
                improve, best_delta, torch.zeros_like(best_delta)
            )
            q_selected[:, coordinate] += best_delta
            error_selected[:, coordinate] += best_delta
            he += best_delta[:, None] * gram_selected[:, :, coordinate]

    final_loss = torch.einsum(
        "ni,nij,nj->n", error_selected, gram_selected, error_selected
    )
    if linear is not None:
        final_loss = final_loss + 2.0 * (
            error_selected * linear
        ).sum(dim=1)
        improve = final_loss < initial_loss - float(accept_margin) * (
            initial_loss.abs().clamp_min(_EPS)
        )
    else:
        improve = final_loss < initial_loss * (
            1.0 - float(accept_margin)
        )
    improved_indices = candidates[improve]
    if int(improved_indices.numel()) == 0:
        return params
    improved_q = q_selected[improve]
    improved_denominator = denominator[improve]
    improved_codes = torch.round(
        improved_q / improved_denominator.clamp_min(_EPS) * 4.0
    ).clamp(-7.0, 7.0)

    refined = dict(params)
    sign8 = params["sign"].clone().reshape(-1, 8)
    mant8 = params["mant"].clone().reshape(-1, 8)
    sign8.index_copy_(0, improved_indices, torch.sign(improved_codes))
    mant8.index_copy_(0, improved_indices, improved_codes.abs() * 0.25)
    refined["sign"] = sign8.reshape_as(params["sign"])
    refined["mant"] = mant8.reshape_as(params["mant"])
    return refined


def _refine_weight_groups16(
    dense: torch.Tensor,
    params: dict[str, torch.Tensor],
    group_gram16: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Refine top-loss 16-channel groups across adjacent 8-channel groups."""

    if dense.ndim != 2:
        return params
    rows, channels = map(int, dense.shape)
    if channels % _HIF4_BLOCK_SIZE != 0 or channels % 16 != 0:
        return params
    blocks = channels // _HIF4_BLOCK_SIZE
    expected_grams = blocks * 4
    if tuple(group_gram16.shape) != (expected_grams, 16, 16):
        return params

    dense16 = dense.reshape(rows, blocks, 4, 16).reshape(-1, 16)
    quantized16 = _dequantize_hif4(params).to(torch.float32).reshape(
        rows, blocks, 4, 16
    ).reshape(-1, 16)
    grams = group_gram16.unsqueeze(0).expand(rows, -1, -1, -1).reshape(
        -1, 16, 16
    )
    error = quantized16 - dense16
    losses = torch.einsum("ni,nij,nj->n", error, grams, error)
    candidates = torch.nonzero(
        torch.isfinite(losses) & (losses > _EPS), as_tuple=False
    ).reshape(-1)
    if int(candidates.numel()) == 0:
        return params
    cap = max(
        1,
        int(
            math.ceil(
                int(losses.numel()) * _WEIGHT_QUADRATIC16_MAX_RATIO
            )
        ),
    )
    cap = min(cap, _WEIGHT_QUADRATIC16_MAX_GROUPS, int(candidates.numel()))
    if int(candidates.numel()) > cap:
        order = torch.topk(
            losses.index_select(0, candidates), k=cap, largest=True
        ).indices
        candidates = candidates.index_select(0, order)

    x_selected = dense16.index_select(0, candidates)
    q_selected = quantized16.index_select(0, candidates).clone()
    gram_selected = grams.index_select(0, candidates)
    error_selected = q_selected - x_selected
    he = torch.einsum("nij,nj->ni", gram_selected, error_selected)
    initial_loss = torch.einsum(
        "ni,nij,nj->n", error_selected, gram_selected, error_selected
    )

    scale = params["scale_factor"].reshape(rows, blocks, 1).expand(
        rows, blocks, 8
    )
    lv2 = params["scale_lv2"].reshape(rows, blocks, 8)
    lv3 = params["scale_lv3"].reshape(rows, blocks, 8, 2)
    denominator8 = (
        scale[..., None]
        * lv2[..., None]
        * lv3.repeat_interleave(4, dim=-1)
    )
    denominator = denominator8.reshape(rows, blocks, 4, 16).reshape(
        -1, 16
    ).index_select(0, candidates)
    signed_codes = torch.arange(
        -7, 8, dtype=torch.float32, device=dense.device
    ) * 0.25

    for _ in range(_WEIGHT_QUADRATIC16_SWEEPS):
        for coordinate in range(16):
            possible = denominator[:, coordinate, None] * signed_codes[None, :]
            delta = possible - q_selected[:, coordinate, None]
            diagonal = gram_selected[:, coordinate, coordinate].clamp_min(_EPS)
            change = (
                2.0 * delta * he[:, coordinate, None]
                + delta.square() * diagonal[:, None]
            )
            best = change.argmin(dim=1)
            row_ids = torch.arange(int(candidates.numel()), device=dense.device)
            best_delta = delta[row_ids, best]
            improve = change[row_ids, best] < -_EPS
            best_delta = torch.where(
                improve, best_delta, torch.zeros_like(best_delta)
            )
            q_selected[:, coordinate] += best_delta
            error_selected[:, coordinate] += best_delta
            he += best_delta[:, None] * gram_selected[:, :, coordinate]

    final_loss = torch.einsum(
        "ni,nij,nj->n", error_selected, gram_selected, error_selected
    )
    improve = final_loss < initial_loss * (
        1.0 - _WEIGHT_QUADRATIC16_ACCEPT_MARGIN
    )
    improved_indices = candidates[improve]
    if int(improved_indices.numel()) == 0:
        return params
    improved_q = q_selected[improve]
    improved_denominator = denominator[improve]
    improved_codes = torch.round(
        improved_q / improved_denominator.clamp_min(_EPS) * 4.0
    ).clamp(-7.0, 7.0)

    refined = dict(params)
    sign16 = params["sign"].clone().reshape(-1, 16)
    mant16 = params["mant"].clone().reshape(-1, 16)
    sign16.index_copy_(0, improved_indices, torch.sign(improved_codes))
    mant16.index_copy_(0, improved_indices, improved_codes.abs() * 0.25)
    refined["sign"] = sign16.reshape_as(params["sign"])
    refined["mant"] = mant16.reshape_as(params["mant"])
    return refined


def _identity_permutation(length: int, device: torch.device) -> torch.Tensor:
    return torch.arange(length, dtype=torch.int64, device=device)


def _hierarchy_aware_permutation(
    first_range: torch.Tensor,
    second_range: torch.Tensor,
) -> torch.Tensor:
    """Cluster similarly scaled channels for the 64/8/4 HiF4 hierarchy.

    The two ranges describe the paired operands of an exactly equivalent
    transform (X/W or Q/K). Log-domain median normalization makes the
    ordering insensitive to the operands' unrelated global units.
    """

    if tuple(first_range.shape) != tuple(second_range.shape):
        raise ValueError("Paired channel ranges must have identical shapes")
    log_first = torch.log2(first_range.to(torch.float32).clamp_min(_EPS))
    log_second = torch.log2(second_range.to(torch.float32).clamp_min(_EPS))
    log_first = log_first - torch.median(log_first)
    log_second = log_second - torch.median(log_second)
    pressure = torch.maximum(log_first, log_second).reshape(-1)
    if int(pressure.numel()) == 0:
        return torch.empty(0, dtype=torch.int64, device=pressure.device)
    if float(pressure.max() - pressure.min()) < 0.25:
        return _identity_permutation(int(pressure.numel()), pressure.device)
    return torch.argsort(pressure, descending=True)


def _range_permutation(ranges: torch.Tensor) -> torch.Tensor:
    """1D argsort of log ranges; identity when the log spread is negligible."""

    log_r = torch.log2(ranges.to(torch.float32).clamp_min(_EPS))
    log_r = log_r - torch.median(log_r)
    flat = log_r.reshape(-1)
    if float(flat.max() - flat.min()) < 0.25:
        return _identity_permutation(int(flat.numel()), flat.device)
    return torch.argsort(flat, descending=True)


def _headwise_range_permutation(ranges: torch.Tensor) -> torch.Tensor:
    """Per-head argsort of log ranges (ranges: [heads, head_dim])."""

    log_r = torch.log2(ranges.to(torch.float32).clamp_min(_EPS))
    log_r = log_r - log_r.median(dim=-1, keepdim=True).values
    spread = log_r.amax(dim=-1) - log_r.amin(dim=-1)
    identity = torch.arange(
        int(ranges.shape[-1]), dtype=torch.int64, device=ranges.device
    ).expand_as(ranges)
    ordered = torch.argsort(log_r, dim=-1, descending=True)
    return torch.where(spread[:, None] >= 0.25, ordered, identity)


def _headwise_hierarchy_permutation(
    q_range: torch.Tensor,
    k_range: torch.Tensor,
) -> torch.Tensor:
    """Return a local feature permutation for each paired Q/KV head."""

    if q_range.ndim != 2 or tuple(q_range.shape) != tuple(k_range.shape):
        raise ValueError("Headwise Q/K ranges must have shape [heads, head_dim]")
    q_log = torch.log2(q_range.to(torch.float32).clamp_min(_EPS))
    k_log = torch.log2(k_range.to(torch.float32).clamp_min(_EPS))
    q_log = q_log - q_log.median(dim=-1, keepdim=True).values
    k_log = k_log - k_log.median(dim=-1, keepdim=True).values
    pressure = torch.maximum(q_log, k_log)
    permutation = torch.argsort(pressure, dim=-1, descending=True)

    spread = pressure.amax(dim=-1) - pressure.amin(dim=-1)
    identity = torch.arange(
        int(pressure.shape[-1]), dtype=torch.int64, device=pressure.device
    ).expand_as(permutation)
    return torch.where(spread[:, None] >= 0.25, permutation, identity)


def _flatten_head_permutation(local_permutation: torch.Tensor) -> torch.Tensor:
    heads, head_dim = map(int, local_permutation.shape)
    base = torch.arange(
        heads, dtype=torch.int64, device=local_permutation.device
    )[:, None] * head_dim
    return (local_permutation.to(torch.int64) + base).reshape(-1)


def _candidate_is_safe(
    candidate: tuple[float, tuple[float, ...]],
    baseline: tuple[float, tuple[float, ...]],
    *,
    min_mean_improvement: float,
    worst_tolerance: float,
) -> bool:
    candidate_mean, candidate_cases = candidate
    baseline_mean, baseline_cases = baseline
    if not math.isfinite(candidate_mean):
        return False
    if candidate_mean > baseline_mean * (1.0 - min_mean_improvement):
        return False
    if len(candidate_cases) != len(baseline_cases):
        return False
    for current, reference in zip(candidate_cases, baseline_cases):
        if current > reference * (1.0 + worst_tolerance) + 1.0e-8:
            return False
    return True


def _center_attention_k(
    dense: torch.Tensor,
    num_heads: int,
    head_dim: int,
    center_mode: int,
) -> torch.Tensor:
    """Apply a token-invariant K shift; softmax(QK^T) is unchanged."""

    mode = int(center_mode)
    if mode == 0:
        return dense
    if dense.ndim != 2 or int(dense.shape[0]) <= 0:
        raise ValueError("Attention centering expects a non-empty 2D tensor")
    if int(dense.shape[1]) != int(num_heads) * int(head_dim):
        raise ValueError("Invalid dimensions for attention centering")
    grouped = dense.reshape(-1, int(num_heads), int(head_dim))
    if mode == 2:
        center = 0.5 * (
            grouped.amax(dim=0, keepdim=True)
            + grouped.amin(dim=0, keepdim=True)
        )
    else:
        raise ValueError("Unsupported attention center mode")
    return (grouped - center).reshape_as(dense)


def _attention_forward(
    q_dense: torch.Tensor,
    k_dense: torch.Tensor,
    v_dense: torch.Tensor,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
    causal: bool,
) -> torch.Tensor:
    """Real attention output for (seq, channels) Q/K/V, evaluator-equivalent."""

    seq = int(q_dense.shape[0])
    group = q_num_heads // kv_num_heads
    q = q_dense.reshape(seq, q_num_heads, head_dim).transpose(0, 1)
    k = (
        k_dense.reshape(seq, kv_num_heads, head_dim)
        .transpose(0, 1)
        .repeat_interleave(group, dim=0)
    )
    v = (
        v_dense.reshape(seq, kv_num_heads, head_dim)
        .transpose(0, 1)
        .repeat_interleave(group, dim=0)
    )
    logits = q @ k.transpose(-1, -2) / math.sqrt(float(head_dim))
    if causal:
        logits = logits + torch.triu(
            torch.full((seq, seq), float("-inf"), device=logits.device), 1
        )
    probs = torch.softmax(logits, dim=-1)
    return (probs @ v).transpose(0, 1).reshape(seq, q_num_heads * head_dim)


def _attention_deployed_mse(
    q_pairs: list,
    k_pairs: list,
    v_hats: list,
    refs: list,
    q_state: dict,
    k_state: dict,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
) -> tuple:
    """Per-sample real attention output MSE (causal, non-causal) through the
    deployed dynamic quantization path, mirroring the evaluator scoring."""

    causal_scores: list = []
    safety_scores: list = []
    for (q_quant, q_scale), (k_quant, k_scale), v_hat, (ref_c, ref_n) in zip(
        q_pairs, k_pairs, v_hats, refs
    ):
        q_hat = _dequantize_hif4(
            hif4_dynamic_quantize_q(
                q_quant, q_scale, q_num_heads, head_dim, q_state
            )
        ).to(torch.float32)
        k_hat = _dequantize_hif4(
            hif4_dynamic_quantize_k(
                k_quant, k_scale, kv_num_heads, head_dim, k_state
            )
        ).to(torch.float32)
        out_c = _attention_forward(
            q_hat, k_hat, v_hat, q_num_heads, kv_num_heads, head_dim, True
        )
        out_n = _attention_forward(
            q_hat, k_hat, v_hat, q_num_heads, kv_num_heads, head_dim, False
        )
        causal_scores.append(float((out_c - ref_c).square().mean()))
        safety_scores.append(float((out_n - ref_n).square().mean()))
    return causal_scores, safety_scores


def _a1_gate_passes(
    winner_causal: list,
    winner_safety: list,
    reference_causal: list,
    reference_safety: list,
    safety_tolerance: Optional[float] = None,
) -> bool:
    """终验门判定：A1 winner 在部署路径上相对 B0 proxy winner 无退化。

    ``safety_tolerance`` 允许对 non-causal 安全轨均值采用更严格的容忍
    （默认取 _A1_GATE_WORST_TOLERANCE；旋转等纯方差均衡机制传 0.0 ——
    若机制真实有效则不应使安全轨均值变差）。
    """

    if (
        not winner_causal
        or not winner_safety
        or not reference_causal
        or not reference_safety
    ):
        return False
    if len(winner_causal) != len(reference_causal):
        return False
    for value in (
        winner_causal + winner_safety + reference_causal + reference_safety
    ):
        if not math.isfinite(value):
            return False
    winner_mean = sum(winner_causal) / len(winner_causal)
    reference_mean = sum(reference_causal) / len(reference_causal)
    if winner_mean > max(reference_mean, 1.0e-12) * (
        1.0 - _A1_GATE_MIN_IMPROVEMENT
    ):
        return False
    tolerance = 1.0 + _A1_GATE_WORST_TOLERANCE
    for value, reference in zip(winner_causal, reference_causal):
        if value > max(reference, 1.0e-12) * tolerance:
            return False
    for value, reference in zip(winner_safety, reference_safety):
        if value > max(reference, 1.0e-12) * tolerance:
            return False
    if safety_tolerance is None:
        safety_tolerance = _A1_GATE_WORST_TOLERANCE
    winner_safety_mean = sum(winner_safety) / len(winner_safety)
    reference_safety_mean = sum(reference_safety) / len(reference_safety)
    if winner_safety_mean > max(
        reference_safety_mean, 1.0e-12
    ) * (1.0 + safety_tolerance):
        return False
    return True


def _attention_head_square_mass(
    q: torch.Tensor,
    k: torch.Tensor,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
) -> tuple:
    """每个 KV head 的注意力质量（E[A] 与 E[A^2]）（因果 softmax）。

    V 输出误差被 A 加权：输出 MSE 的对角项由 ``E[A^2]`` 主导，块间
    偏差交叉项（均值误差）由 ``E[A]^2`` 主导。GQA 下每个 KV head 对应
    group 个 Q head，取组内平均。返回 ``(mean_mass, square_mass)``，
    形状均为 ``[kv_num_heads]``。
    """

    seq = int(q.shape[0])
    group = q_num_heads // kv_num_heads
    qh = q.reshape(seq, q_num_heads, head_dim)
    kh = k.reshape(seq, kv_num_heads, head_dim).repeat_interleave(group, dim=1)
    scores = torch.einsum("thd,shd->tsh", qh, kh) / math.sqrt(float(head_dim))
    mask = torch.triu(
        torch.full((seq, seq), float("-inf"), device=scores.device), 1
    ).unsqueeze(-1)
    probs = torch.softmax(scores + mask, dim=1)
    square_mass = probs.square().mean(dim=(0, 1))
    mean_mass = (
        probs.mean(dim=(0, 1)) if _V_IMPORTANCE_CANDIDATES else None
    )
    return (
        (
            None
            if mean_mass is None
            else mean_mass.reshape(kv_num_heads, group).mean(dim=1)
        ),
        square_mass.reshape(kv_num_heads, group).mean(dim=1),
    )


def _e6m2_encode_nearest(value: torch.Tensor) -> torch.Tensor:
    """Encode non-negative FP32 values into finite unsigned E6M2 codes.

    Codes 0..254 are finite and monotonic.  Code 255 is NaN and is never
    produced.  Round-to-nearest-even is inherited from ``torch.round``.
    """

    x = torch.nan_to_num(
        value.detach().to(torch.float32),
        nan=_E6M2_MIN,
        posinf=_E6M2_MAX,
        neginf=_E6M2_MIN,
    ).clamp(min=_E6M2_MIN, max=_E6M2_MAX)

    exponent = torch.floor(torch.log2(x))
    base = torch.pow(2.0, exponent)
    mantissa_field = torch.round((x / base - 1.0) * 4.0).to(torch.int64)

    carry = mantissa_field >= 4
    exponent = exponent + carry.to(exponent.dtype)
    mantissa_field = torch.where(
        carry, torch.zeros_like(mantissa_field), mantissa_field
    ).clamp(min=0, max=3)

    exponent_field = (exponent.to(torch.int64) + 48).clamp(min=0, max=63)
    code = exponent_field * 4 + mantissa_field
    return code.clamp(min=0, max=254).to(torch.int16)


def _e6m2_decode(code: torch.Tensor) -> torch.Tensor:
    c = code.to(torch.int64).clamp(min=0, max=254)
    exponent_field = torch.bitwise_right_shift(c, 2)
    mantissa_field = torch.bitwise_and(c, 3)
    exponent = exponent_field.to(torch.float32) - 48.0
    return torch.pow(2.0, exponent) * (
        1.0 + mantissa_field.to(torch.float32) * 0.25
    )


def _standard_e6m2_scale(amax: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the official amax/7 base scale with a BF16 intermediate."""

    high_precision_scale = (
        amax.to(torch.bfloat16) * _BF16_ONE_SEVENTH
    ).to(torch.float32)
    code = _e6m2_encode_nearest(high_precision_scale)
    return code, _e6m2_decode(code)


def _offsets_as_tuple(offsets: Optional[Iterable[int]]) -> tuple[int, ...]:
    ordered = [0]
    if offsets is None:
        return (0,)
    if torch.is_tensor(offsets):
        values = offsets.detach().to("cpu").reshape(-1).tolist()
    else:
        values = list(offsets)
    for raw in values:
        value = int(raw)
        if value not in ordered:
            ordered.append(value)
    return tuple(ordered)


def _solve_exact_hierarchy(
    x_abs: torch.Tensor,
    scale: torch.Tensor,
    importance: Optional[torch.Tensor],
    sign: Optional[torch.Tensor] = None,
    group_gram: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exactly solve lv2/lv3 for fixed scales using three loss tables.

    Args:
        x_abs: ``[num_blocks, 8, 2, 4]`` absolute values.
        scale: ``[num_blocks]`` finite E6M2 values.
        importance: optional tensor with the same shape as ``x_abs``.
        sign: ``[num_blocks, 8, 2, 4]`` signs (required with ``group_gram``).
        group_gram: ``[num_blocks, 8, 2, 4, 4]`` per-group quadratic weights;
            when given, the loss is the quadratic form ``delta^T G delta``
            instead of the diagonal per-channel weighted squares.
    """

    losses: list[torch.Tensor] = []
    mantissas: list[torch.Tensor] = []

    for total_exponent in (0, 1, 2):
        local_scale = scale[..., None, None, None] * float(1 << total_exponent)
        mant_code = torch.round(x_abs * (4.0 / local_scale)).clamp_(0.0, 7.0)
        mantissa = mant_code * 0.25
        if group_gram is not None:
            delta = sign * (x_abs - mantissa * local_scale)
            losses.append(
                torch.einsum(
                    "...abi,...abij,...abj->...ab", delta, group_gram, delta
                )
            )
        else:
            error = (x_abs - mantissa * local_scale).square()
            if importance is not None:
                error = error * importance
            losses.append(error.sum(dim=-1))
        mantissas.append(mantissa)

    loss_0, loss_1, loss_2 = losses
    choose_01 = loss_1 < loss_0
    choose_12 = loss_2 < loss_1

    cost_e2_0 = torch.minimum(loss_0, loss_1).sum(dim=-1)
    cost_e2_1 = torch.minimum(loss_1, loss_2).sum(dim=-1)
    e2 = cost_e2_1 < cost_e2_0
    e3 = torch.where(e2[..., None], choose_12, choose_01)

    block_loss = torch.where(e2, cost_e2_1, cost_e2_0).sum(dim=-1)
    total_exponent = e2.to(torch.int64)[..., None] + e3.to(torch.int64)

    # [..., 3, 4]，指数维固定在倒数第二维（批量时随输入维度自然后移）。
    mantissa_stack = torch.stack(mantissas, dim=-2)
    gather_index = total_exponent[..., None, None].expand(
        *total_exponent.shape, 1, 4
    )
    gather_dim = mantissa_stack.ndim - 2  # 指数维：4D 输入为 3，批量时随维度后移
    mantissa = torch.gather(
        mantissa_stack, gather_dim, gather_index
    ).squeeze(-2)

    scale_lv2 = 1.0 + e2.to(torch.float32)
    scale_lv3 = 1.0 + e3.to(torch.float32)
    return block_loss, scale_lv2, scale_lv3, mantissa


def _pack_hif4_params(
    prefix: tuple[int, ...],
    blocks: int,
    scale_factor: torch.Tensor,
    scale_lv2: torch.Tensor,
    scale_lv3: torch.Tensor,
    sign: torch.Tensor,
    mantissa: torch.Tensor,
) -> dict[str, torch.Tensor]:
    # Canonical zero: it is numerically irrelevant, but avoids relying on a
    # checker accepting sign=+/-1 when the final mantissa is zero.
    sign_out = sign.reshape(*prefix, blocks, 8, 2, 4)
    mantissa_out = mantissa.reshape(*prefix, blocks, 8, 2, 4)
    sign_out = torch.where(
        mantissa_out == 0.0, torch.zeros_like(sign_out), sign_out
    )
    return {
        "scale_factor": scale_factor.reshape(*prefix, blocks, 1, 1, 1),
        "scale_lv2": scale_lv2.reshape(*prefix, blocks, 8, 1, 1),
        "scale_lv3": scale_lv3.reshape(*prefix, blocks, 8, 2, 1),
        "sign": sign_out,
        "mant": mantissa_out,
    }


def _dense_to_hif4(
    dense: torch.Tensor,
    *,
    importance: Optional[torch.Tensor] = None,
    group_gram: Optional[torch.Tensor] = None,
    search_offsets: Optional[Union[Sequence[int], torch.Tensor]] = None,
    error_threshold: float = 0.0,
    accept_margin: float = 0.0,
    max_refine_ratio: float = 0.0,
    max_refine_blocks: Optional[int] = None,
) -> dict[str, torch.Tensor]:
    """Quantize a dense tensor into valid HiF4 parameters."""

    if group_gram is not None:
        expected_gram_shape = dense.shape[:-1] + (
            dense.shape[-1] // 64,
            8,
            2,
            4,
            4,
        )
        if tuple(group_gram.shape) != tuple(expected_gram_shape):
            raise ValueError(
                f"group_gram shape {tuple(group_gram.shape)} does not match "
                f"expected {expected_gram_shape}"
            )

    if dense.ndim < 1:
        raise ValueError("dense must have at least one dimension")
    prefix = tuple(int(v) for v in dense.shape[:-1])
    channels = int(dense.shape[-1])
    if channels % _HIF4_BLOCK_SIZE != 0:
        raise ValueError(
            f"Last dim {channels} is not divisible by HiF4 block size 64"
        )
    blocks = channels // _HIF4_BLOCK_SIZE

    x = torch.nan_to_num(
        dense.detach().to(torch.float32),
        nan=0.0,
        posinf=_E6M2_MAX * _HIF4_MAX_INNER,
        neginf=-_E6M2_MAX * _HIF4_MAX_INNER,
    )
    x_grouped = x.reshape(*prefix, blocks, 8, 2, 4)
    x_abs = x_grouped.abs()
    sign = torch.sign(x_grouped)

    max4 = x_abs.amax(dim=-1)
    max8 = max4.amax(dim=-1)
    amax = max8.amax(dim=-1)
    standard_code, standard_scale = _standard_e6m2_scale(amax)

    e2 = max8 >= (4.0 * standard_scale[..., None])
    scale_lv2 = 1.0 + e2.to(torch.float32)
    e3 = max4 >= (
        2.0 * standard_scale[..., None, None] * scale_lv2[..., None]
    )
    scale_lv3 = 1.0 + e3.to(torch.float32)

    denominator = (
        standard_scale[..., None, None, None]
        * scale_lv2[..., None, None]
        * scale_lv3[..., None]
    )
    mantissa = (
        torch.round(x_abs * (4.0 / denominator)).clamp_(0.0, 7.0) * 0.25
    )

    offsets = _offsets_as_tuple(search_offsets)
    refine_ratio = max(0.0, min(float(max_refine_ratio), 1.0))
    if refine_ratio <= 0.0 or len(offsets) == 0:
        return _pack_hif4_params(
            prefix,
            blocks,
            standard_scale,
            scale_lv2,
            scale_lv3,
            sign,
            mantissa,
        )

    channel_importance = _normalize_importance(importance, channels)
    if channel_importance is not None:
        channel_importance = channel_importance.to(x.device)
    if group_gram is not None:
        delta = sign * (x_abs - mantissa * denominator)
        weighted_error = torch.einsum(
            "...abi,...abij,...abj->...ab", delta, group_gram, delta
        )
        weighted_energy = x_abs.square()
        importance_view = None
    elif channel_importance is None:
        weighted_error = (x_abs - mantissa * denominator).square()
        weighted_energy = x_abs.square()
        importance_view = None
    else:
        importance_view = channel_importance.reshape(
            *([1] * len(prefix)), blocks, 8, 2, 4
        )
        weighted_error = (x_abs - mantissa * denominator).square() * importance_view
        weighted_energy = x_abs.square() * importance_view

    loss_reduce_dims = (-1, -2) if group_gram is not None else (-1, -2, -3)
    standard_loss = weighted_error.sum(dim=loss_reduce_dims)
    energy = weighted_energy.sum(dim=(-1, -2, -3))
    normalized_error = standard_loss / (energy + _EPS)

    flat_norm = normalized_error.reshape(-1)
    flat_loss = standard_loss.reshape(-1)
    hard_mask = flat_norm > float(error_threshold)
    hard_indices = torch.nonzero(hard_mask, as_tuple=False).reshape(-1)
    if int(hard_indices.numel()) == 0:
        return _pack_hif4_params(
            prefix,
            blocks,
            standard_scale,
            scale_lv2,
            scale_lv3,
            sign,
            mantissa,
        )

    total_blocks = int(flat_norm.numel())
    refine_cap = max(1, int(math.ceil(total_blocks * refine_ratio)))
    if max_refine_blocks is not None:
        refine_cap = min(refine_cap, max(1, int(max_refine_blocks)))
    if int(hard_indices.numel()) > refine_cap:
        if _REFINE_RANK_BY_ABSOLUTE:
            # Rank by the block's absolute (importance-weighted) reconstruction
            # error, i.e. its true contribution to the output MSE, instead of
            # the normalized error: under a fixed refinement budget this
            # greedily maximizes the total MSE reduction (and hence the
            # competition score).
            hard_indices = torch.topk(flat_loss, k=refine_cap, largest=True).indices
        else:
            hard_indices = torch.topk(flat_norm, k=refine_cap, largest=True).indices

    x_flat = x_abs.reshape(-1, 8, 2, 4)
    x_hard = x_flat.index_select(0, hard_indices)
    standard_loss_hard = standard_loss.reshape(-1).index_select(0, hard_indices)
    standard_code_hard = standard_code.reshape(-1).index_select(0, hard_indices)

    best_loss = standard_loss_hard.clone()
    best_scale = standard_scale.reshape(-1).index_select(0, hard_indices).clone()
    best_lv2 = scale_lv2.reshape(-1, 8).index_select(0, hard_indices).clone()
    best_lv3 = scale_lv3.reshape(-1, 8, 2).index_select(0, hard_indices).clone()
    best_mantissa = mantissa.reshape(-1, 8, 2, 4).index_select(
        0, hard_indices
    ).clone()
    sign_hard = sign.reshape(-1, 8, 2, 4).index_select(0, hard_indices)
    group_gram_hard = (
        None
        if group_gram is None
        else group_gram.reshape(-1, 8, 2, 4, 4).index_select(0, hard_indices)
    )
    best_offset = torch.zeros(
        int(hard_indices.numel()), dtype=torch.int64, device=x.device
    )

    if channel_importance is None:
        importance_hard = None
    else:
        block_importance = channel_importance.reshape(blocks, 8, 2, 4)
        channel_block_ids = torch.remainder(hard_indices, blocks)
        importance_hard = block_importance.index_select(0, channel_block_ids)

    # 全部 offset 一次性批量求解：把 [N] 块沿 offset 维展开成 [K, N]，
    # 一次精确求解后按块取 argmin。标准 code（offset 0）必须保留在候选里：
    # 阈值式 lv2/lv3 与精确解不等价（真实数据约半数块有更低损失），
    # offset 0 会把 hard 块的 lv2/lv3 升级为精确解。
    offset_values = torch.tensor(
        [int(o) for o in offsets], dtype=torch.int64, device=x.device
    )
    expanded_codes = (
        standard_code_hard.to(torch.int64).unsqueeze(0)
        + offset_values.unsqueeze(1)
    ).clamp(min=0, max=254)
    candidate_scales = _e6m2_decode(expanded_codes)
    num_offsets = int(offset_values.numel())
    x_expanded = x_hard.unsqueeze(0).expand(
        num_offsets, -1, -1, -1, -1
    )
    sign_expanded = sign_hard.unsqueeze(0).expand(
        num_offsets, -1, -1, -1, -1
    )
    importance_expanded = (
        None
        if importance_hard is None
        else importance_hard.unsqueeze(0).expand(
            num_offsets, -1, -1, -1, -1
        )
    )
    gram_expanded = (
        None
        if group_gram_hard is None
        else group_gram_hard.unsqueeze(0).expand(
            num_offsets, -1, -1, -1, -1, -1
        )
    )
    all_losses, all_lv2, all_lv3, all_mantissa = _solve_exact_hierarchy(
        x_expanded,
        candidate_scales,
        importance_expanded,
        sign_expanded,
        gram_expanded,
    )
    best_k = all_losses.argmin(dim=0)
    hard_arange = torch.arange(
        int(hard_indices.numel()), device=x.device
    )
    candidate_loss = all_losses[best_k, hard_arange]
    candidate_scale = candidate_scales[best_k, hard_arange]
    candidate_lv2 = all_lv2[best_k, hard_arange]
    candidate_lv3 = all_lv3[best_k, hard_arange]
    candidate_mantissa = all_mantissa[best_k, hard_arange]

    improve = candidate_loss < best_loss
    best_loss = torch.where(improve, candidate_loss, best_loss)
    best_scale = torch.where(improve, candidate_scale, best_scale)
    best_lv2 = torch.where(improve[:, None], candidate_lv2, best_lv2)
    best_lv3 = torch.where(improve[:, None, None], candidate_lv3, best_lv3)
    best_mantissa = torch.where(
        improve[:, None, None, None], candidate_mantissa, best_mantissa
    )
    best_offset = torch.where(
        improve, offset_values[best_k], best_offset
    )

    if _REFINE_EDGE_EXTENSION and len(offsets) > 1:
        lo_offset = int(offsets[0])
        hi_offset = int(offsets[-1])

        def extend_edge(edge: int, direction: int) -> None:
            nonlocal best_loss, best_scale
            nonlocal best_lv2, best_lv3, best_mantissa, best_offset
            mask = best_offset == edge
            for _ in range(_REFINE_EDGE_EXTEND_STEPS):
                if not bool(mask.any()):
                    return
                edge_indices = torch.nonzero(mask, as_tuple=False).reshape(-1)
                target = edge + direction
                if target < -254 or target > 254:
                    return
                edge_code = (
                    standard_code_hard.index_select(0, edge_indices).to(
                        torch.int64
                    )
                    + target
                ).clamp(min=0, max=254)
                edge_scale = _e6m2_decode(edge_code)
                edge_importance = (
                    None
                    if importance_hard is None
                    else importance_hard.index_select(0, edge_indices)
                )
                edge_loss, edge_lv2, edge_lv3, edge_mantissa = (
                    _solve_exact_hierarchy(
                        x_hard.index_select(0, edge_indices),
                        edge_scale,
                        edge_importance,
                        sign_hard.index_select(0, edge_indices),
                        (
                            None
                            if group_gram_hard is None
                            else group_gram_hard.index_select(0, edge_indices)
                        ),
                    )
                )
                improve = edge_loss < best_loss.index_select(0, edge_indices)
                improved = edge_indices[improve]
                if int(improved.numel()) == 0:
                    return
                best_loss.index_copy_(
                    0, improved, edge_loss[improve]
                )
                best_scale.index_copy_(0, improved, edge_scale[improve])
                best_lv2.index_copy_(0, improved, edge_lv2[improve])
                best_lv3.index_copy_(0, improved, edge_lv3[improve])
                best_mantissa.index_copy_(
                    0, improved, edge_mantissa[improve]
                )
                best_offset.index_copy_(
                    0,
                    improved,
                    torch.full_like(best_offset[improved], target),
                )
                edge = target
                mask = best_offset == target

        extend_edge(hi_offset, +1)
        extend_edge(lo_offset, -1)

    if _L1_DATA_DRIVEN_SCALE:
        # L1 数据驱动 scale 候选：锚定当前五字段 winner（best_* 来自
        # offset 搜索 + 边缘扩展），生成每块独立候选 code，全部经
        # _solve_exact_hierarchy 精确解后逐块 improve-mask 回退。
        if importance_hard is not None:
            l1_weights = importance_hard
        elif group_gram_hard is not None:
            l1_weights = torch.diagonal(
                group_gram_hard, dim1=-2, dim2=-1
            )
        else:
            l1_weights = None
        hierarchy = best_lv2[:, :, None, None] * best_lv3[:, :, :, None]
        model = hierarchy * best_mantissa
        if l1_weights is None:
            numerator = (model * x_hard).sum(dim=(1, 2, 3))
            denominator = (model * model).sum(dim=(1, 2, 3))
        else:
            numerator = (l1_weights * model * x_hard).sum(dim=(1, 2, 3))
            denominator = (l1_weights * model * model).sum(dim=(1, 2, 3))
        ls_scale = numerator / denominator.clamp_min(_EPS)
        flat_abs = x_hard.reshape(int(x_hard.shape[0]), -1)
        base_codes = [_e6m2_encode_nearest(ls_scale)]
        for trim_quantile in _L1_TRIM_QUANTILES:
            trim_scale = torch.quantile(
                flat_abs, float(trim_quantile), dim=1
            ) * (4.0 / 7.0)
            base_codes.append(_e6m2_encode_nearest(trim_scale))
        code_deltas = torch.tensor(
            _L1_ADJACENT_CODE_DELTAS, dtype=torch.int64, device=x.device
        )
        candidate_codes = torch.stack(
            [
                (
                    base_code.to(torch.int64).unsqueeze(0)
                    + code_deltas.unsqueeze(1)
                ).clamp(min=0, max=254)
                for base_code in base_codes
            ],
            dim=0,
        ).reshape(-1, int(x_hard.shape[0]))
        l1_scales = _e6m2_decode(candidate_codes)
        num_l1 = int(candidate_codes.shape[0])
        l1_losses, l1_lv2, l1_lv3, l1_mantissa = _solve_exact_hierarchy(
            x_hard.unsqueeze(0).expand(num_l1, -1, -1, -1, -1),
            l1_scales,
            (
                None
                if importance_hard is None
                else importance_hard.unsqueeze(0).expand(
                    num_l1, -1, -1, -1, -1
                )
            ),
            sign_hard.unsqueeze(0).expand(num_l1, -1, -1, -1, -1),
            (
                None
                if group_gram_hard is None
                else group_gram_hard.unsqueeze(0).expand(
                    num_l1, -1, -1, -1, -1, -1
                )
            ),
        )
        l1_best = l1_losses.argmin(dim=0)
        l1_loss = l1_losses[l1_best, hard_arange]
        l1_scale = l1_scales[l1_best, hard_arange]
        l1_lv2_best = l1_lv2[l1_best, hard_arange]
        l1_lv3_best = l1_lv3[l1_best, hard_arange]
        l1_mantissa_best = l1_mantissa[l1_best, hard_arange]
        improve_l1 = l1_loss < best_loss
        best_loss = torch.where(improve_l1, l1_loss, best_loss)
        best_scale = torch.where(improve_l1, l1_scale, best_scale)
        best_lv2 = torch.where(improve_l1[:, None], l1_lv2_best, best_lv2)
        best_lv3 = torch.where(
            improve_l1[:, None, None], l1_lv3_best, best_lv3
        )
        best_mantissa = torch.where(
            improve_l1[:, None, None, None], l1_mantissa_best, best_mantissa
        )

    margin = max(0.0, min(float(accept_margin), 0.99))
    accept = best_loss <= ((1.0 - margin) * standard_loss_hard)
    if not bool(torch.any(accept)):
        return _pack_hif4_params(
            prefix,
            blocks,
            standard_scale,
            scale_lv2,
            scale_lv3,
            sign,
            mantissa,
        )

    selected_indices = hard_indices[accept]
    out_scale = standard_scale.reshape(-1).clone()
    out_lv2 = scale_lv2.reshape(-1, 8).clone()
    out_lv3 = scale_lv3.reshape(-1, 8, 2).clone()
    out_mantissa = mantissa.reshape(-1, 8, 2, 4).clone()

    out_scale.index_copy_(0, selected_indices, best_scale[accept])
    out_lv2.index_copy_(0, selected_indices, best_lv2[accept])
    out_lv3.index_copy_(0, selected_indices, best_lv3[accept])
    out_mantissa.index_copy_(0, selected_indices, best_mantissa[accept])

    return _pack_hif4_params(
        prefix,
        blocks,
        out_scale,
        out_lv2,
        out_lv3,
        sign,
        out_mantissa,
    )


def _nvfp4_to_hif4(
    quant_float: torch.Tensor,
    scale_float: torch.Tensor,
    *,
    multiplier: Optional[torch.Tensor] = None,
    permutation: Optional[torch.Tensor] = None,
    block_smooth_size: int = 0,
    block_smooth_seed: int = 0,
    center_mode: int = 0,
    center_num_heads: Optional[int] = None,
    center_head_dim: Optional[int] = None,
    importance: Optional[torch.Tensor] = None,
    group_gram: Optional[torch.Tensor] = None,
    group_gram8: Optional[torch.Tensor] = None,
    group_cross8: Optional[torch.Tensor] = None,
    search_offsets: Optional[Union[Sequence[int], torch.Tensor]] = None,
    error_threshold: float = 0.0,
    accept_margin: float = 0.0,
    max_refine_ratio: float = 0.0,
    max_refine_blocks: Optional[int] = None,
    attention_rotation: Optional[torch.Tensor] = None,
    rotation_num_heads: Optional[int] = None,
) -> dict[str, torch.Tensor]:
    dense = _dequantize_nvfp4_float32(quant_float, scale_float)
    channels = int(dense.shape[-1])
    if int(center_mode) != 0:
        if center_num_heads is None or center_head_dim is None:
            raise ValueError("Attention centering requires head metadata")
        dense = _center_attention_k(
            dense,
            int(center_num_heads),
            int(center_head_dim),
            int(center_mode),
        )
    if multiplier is not None:
        scale = _safe_positive_vector(multiplier, channels).to(dense.device)
        dense.mul_(scale.reshape(*([1] * (dense.ndim - 1)), channels))
    if permutation is not None:
        order = permutation.detach().to(
            device=dense.device, dtype=torch.int64
        ).reshape(-1)
        if int(order.numel()) != channels:
            raise ValueError("Permutation width does not match tensor width")
        dense = dense.index_select(-1, order)
    if attention_rotation is not None:
        if rotation_num_heads is None:
            raise ValueError("Attention rotation requires head count")
        signs = attention_rotation.detach().to(device="cpu")
        head_dim = int(signs.shape[-1])
        if int(rotation_num_heads) * head_dim != channels:
            raise ValueError(
                "Attention rotation does not match tensor width"
            )
        if head_dim < 4 or (head_dim & (head_dim - 1)) != 0:
            raise ValueError("Attention rotation requires power-of-two head_dim")
        dense = _apply_attention_rotation(
            dense, int(rotation_num_heads), head_dim, signs
        )
    if int(block_smooth_size) != 0:
        dense = _block_hadamard_transform(
            dense, int(block_smooth_size), int(block_smooth_seed)
        )
    gram = None
    if group_gram is not None:
        gram = group_gram.detach().to(
            device=dense.device, dtype=torch.float32
        )
        expected = (channels // 4, 4, 4)
        if tuple(gram.shape) != expected:
            raise ValueError(
                f"group_gram shape {tuple(gram.shape)} does not match "
                f"expected {expected}"
            )
        blocks = channels // _HIF4_BLOCK_SIZE
        gram = gram.reshape(blocks, 8, 2, 4, 4).unsqueeze(0).expand(
            int(dense.shape[0]), blocks, 8, 2, 4, 4
        )
    params = _dense_to_hif4(
        dense,
        importance=importance,
        group_gram=gram,
        search_offsets=search_offsets,
        error_threshold=error_threshold,
        accept_margin=accept_margin,
        max_refine_ratio=max_refine_ratio,
        max_refine_blocks=max_refine_blocks,
    )
    if group_gram8 is not None:
        gram8 = group_gram8.detach().to(
            device=dense.device, dtype=torch.float32
        )
        params = _refine_weight_groups8(
            dense,
            params,
            gram8,
            group_cross8=(
                None
                if group_cross8 is None
                else group_cross8.detach().to(
                    device=dense.device, dtype=torch.float32
                )
            ),
            max_ratio=_ACTIVATION_QUADRATIC8_MAX_RATIO,
            max_groups=_ACTIVATION_QUADRATIC8_MAX_GROUPS,
            sweeps=_ACTIVATION_QUADRATIC8_SWEEPS,
            accept_margin=_ACTIVATION_QUADRATIC8_ACCEPT_MARGIN,
        )
    return params


def _dequantize_hif4(params: dict[str, torch.Tensor]) -> torch.Tensor:
    dense = (
        params["sign"]
        * params["mant"]
        * params["scale_lv3"]
        * params["scale_lv2"]
        * params["scale_factor"]
    )
    return dense.flatten(start_dim=-4, end_dim=-1)


def _smooth_scale(
    activation_amax: torch.Tensor,
    weight_amax: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    d = (activation_amax + _EPS).pow(alpha) / (
        weight_amax + _EPS
    ).pow(1.0 - alpha)
    d = torch.nan_to_num(
        d, nan=1.0, posinf=_SMOOTH_SCALE_MAX, neginf=_SMOOTH_SCALE_MIN
    )
    d = d.clamp(min=_SMOOTH_SCALE_MIN, max=_SMOOTH_SCALE_MAX)
    # A global normalization prevents an arbitrary overall scale drift while
    # retaining the relative channel smoothing.
    geometric_mean = torch.exp(torch.log(d).mean())
    return (d / geometric_mean).clamp(
        min=_SMOOTH_SCALE_MIN, max=_SMOOTH_SCALE_MAX
    )


def _hadamard_matrix(
    size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return a normalized Sylvester Hadamard matrix (size 4/8/16)."""

    n = int(size)
    if n not in _BLOCK_SMOOTH_ALLOWED_SIZES:
        raise ValueError(
            "block_smooth_size must be one of "
            f"{_BLOCK_SMOOTH_ALLOWED_SIZES}, got {n}"
        )
    return _hadamard_matrix_unchecked(n, device, dtype)


def _hadamard_matrix_unchecked(
    size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return a normalized Sylvester Hadamard matrix for power-of-two sizes."""

    n = int(size)
    if n < 1 or (n & (n - 1)) != 0:
        raise ValueError(f"Hadamard size must be a power of two, got {n}")
    h = torch.ones(1, 1, dtype=dtype, device=device)
    while int(h.shape[0]) < n:
        h = torch.cat(
            (torch.cat((h, h), dim=1), torch.cat((h, -h), dim=1)), dim=0
        )
    return h * (1.0 / math.sqrt(float(n)))


def _block_hadamard_transform(
    dense: torch.Tensor,
    block_size: int,
    seed: int = 0,
) -> torch.Tensor:
    """Apply a deterministic signed orthogonal transform to feature blocks.

    The signs avoid concentrating positively correlated channels in the DC
    Hadamard coefficient.  They are derived from the absolute feature index,
    so calibration and dynamic quantization only share ``block_size`` and a
    small integer ``seed``.
    """

    size = int(block_size)
    if size == 0:
        return dense
    channels = int(dense.shape[-1])
    if channels % size != 0:
        raise ValueError(
            f"Feature width {channels} is not divisible by block size {size}"
        )
    indices = torch.arange(channels, dtype=torch.int64, device=dense.device)
    bits = (
        indices * 1_103_515_245 + int(seed) * 214_013 + 12_345
    ).bitwise_and(1 << 30)
    signs = torch.where(bits == 0, 1.0, -1.0).to(dtype=dense.dtype)
    grouped = dense.reshape(*dense.shape[:-1], channels // size, size)
    grouped = grouped * signs.reshape(channels // size, size)
    h = _hadamard_matrix(size, dense.device, dense.dtype)
    return torch.matmul(grouped, h).reshape_as(dense)


def _attention_rotation_signs(
    kv_num_heads: int,
    head_dim: int,
    seed: int,
) -> torch.Tensor:
    """Deterministic per-(KV group, channel) signs shared by Q and K.

    Signs derive from the flat index within the [kv_num_heads, head_dim]
    layout using the same integer hash as the Linear block smoothing, so
    calibration and dynamic quantization agree without extra state.
    """

    index = torch.arange(
        kv_num_heads * head_dim, dtype=torch.int64, device="cpu"
    )
    bits = (
        index * 1_103_515_245 + int(seed) * 214_013 + 12_345
    ).bitwise_and(1 << 30)
    signs = torch.where(bits == 0, 1.0, -1.0)
    return signs.reshape(kv_num_heads, head_dim)


def _apply_attention_rotation(
    dense: torch.Tensor,
    rotation_num_heads: int,
    head_dim: int,
    signs: torch.Tensor,
) -> torch.Tensor:
    """Apply the group-aligned signed Hadamard rotation to head blocks.

    ``signs`` has shape [kv_num_heads, head_dim].  For K (or MHA) the heads
    map one-to-one; for Q the heads of the same KV group share their K
    rotation, which keeps Q·K dot products exactly invariant.
    """

    kv_num_heads = int(signs.shape[0])
    group_size = int(rotation_num_heads) // kv_num_heads
    if group_size * kv_num_heads != int(rotation_num_heads):
        raise ValueError("Rotation head count is not a GQA multiple")
    signs_f = signs.detach().to(
        device=dense.device, dtype=dense.dtype
    ).reshape(kv_num_heads, head_dim)
    if group_size > 1:
        signs_f = signs_f.repeat_interleave(group_size, dim=0)
    x = dense.reshape(*dense.shape[:-1], int(rotation_num_heads), head_dim)
    x = x * signs_f.reshape(int(rotation_num_heads), head_dim)
    block = _ATTN_H64_BLOCK if head_dim % _ATTN_H64_BLOCK == 0 else head_dim
    blocks = head_dim // block
    h = _hadamard_matrix_unchecked(block, dense.device, dense.dtype)
    x = x.reshape(*x.shape[:-2], int(rotation_num_heads), blocks, block)
    x = torch.matmul(x, h)
    return x.reshape(*dense.shape)


def _linear_pair_transform(
    dense: torch.Tensor,
    d: torch.Tensor,
    permutation: torch.Tensor,
    block_smooth_size: int,
    block_smooth_seed: int = 0,
    *,
    weight_side: bool,
) -> torch.Tensor:
    """Apply one side of the exactly equivalent Linear transform."""

    scale = d if weight_side else d.reciprocal()
    transformed = (dense * scale.unsqueeze(0)).index_select(-1, permutation)
    return _block_hadamard_transform(
        transformed, block_smooth_size, block_smooth_seed
    )


def _transformed_second_moment(
    second_moment: torch.Tensor,
    d: torch.Tensor,
    permutation: torch.Tensor,
    block_smooth_size: int,
    block_smooth_seed: int = 0,
) -> torch.Tensor:
    """Diagonal covariance after scale/permutation/block rotation.

    Without a full covariance the diagonal after a normalized Hadamard is the
    mean variance of each block.  The full covariance path below is still used
    for the quadratic weight solver once a candidate has been selected.
    """

    moment = (second_moment / d.square()).index_select(0, permutation)
    size = int(block_smooth_size)
    if size != 0:
        moment = moment.reshape(-1, size).mean(dim=-1, keepdim=True).expand(
            -1, size
        ).reshape(-1)
    return moment


def _transformed_covariance(
    covariance: torch.Tensor,
    d: torch.Tensor,
    permutation: torch.Tensor,
    block_smooth_size: int,
    block_smooth_seed: int = 0,
) -> torch.Tensor:
    """Full activation covariance after the equivalent transform."""

    scale = d.reciprocal().to(dtype=covariance.dtype)
    cov = covariance * scale.unsqueeze(0) * scale.unsqueeze(1)
    cov = cov.index_select(0, permutation).index_select(1, permutation)
    size = int(block_smooth_size)
    if size != 0:
        cov = _block_hadamard_transform(cov, size, block_smooth_seed)
        cov = _block_hadamard_transform(
            cov.t(), size, block_smooth_seed
        ).t()
    return cov


def _linear_candidate_metrics(
    weight: torch.Tensor,
    activation_second_moment: torch.Tensor,
    activation_samples: Sequence[torch.Tensor],
    d: torch.Tensor,
    permutation: torch.Tensor,
    block_smooth_size: int = 0,
    block_smooth_seed: int = 0,
) -> tuple[float, tuple[float, ...]]:
    """Score an equivalent Linear transform from operand-side statistics."""

    channels = int(weight.shape[1])
    order = permutation.to(device=weight.device, dtype=torch.int64).reshape(-1)
    if int(order.numel()) != channels:
        raise ValueError("Linear candidate permutation has an invalid width")

    weight_smooth = _linear_pair_transform(
        weight,
        d,
        order,
        block_smooth_size,
        block_smooth_seed,
        weight_side=True,
    )
    h_x = _transformed_second_moment(
        activation_second_moment, d, order, block_smooth_size
    )
    weight_params = _dense_to_hif4(weight_smooth)
    weight_hat = _dequantize_hif4(weight_params)

    weight_error = (
        (weight_smooth - weight_hat).square() * h_x.unsqueeze(0)
    ).sum()
    weight_energy = (weight_smooth.square() * h_x.unsqueeze(0)).sum()
    weight_score = weight_error / (weight_energy + _EPS)

    h_w = _normalize_importance(weight_hat.square().sum(dim=0), channels)
    if h_w is None:
        h_w = torch.ones(channels, dtype=torch.float32, device=weight.device)

    case_scores: list[float] = []
    for sample in activation_samples:
        smooth = _linear_pair_transform(
            sample,
            d,
            order,
            block_smooth_size,
            block_smooth_seed,
            weight_side=False,
        )
        params = _dense_to_hif4(smooth)
        reconstructed = _dequantize_hif4(params)
        error = ((smooth - reconstructed).square() * h_w.unsqueeze(0)).sum()
        energy = (smooth.square() * h_w.unsqueeze(0)).sum()
        score = torch.nan_to_num(
            weight_score + error / (energy + _EPS),
            nan=1.0e30,
            posinf=1.0e30,
            neginf=1.0e30,
        )
        case_scores.append(float(score))

    if not case_scores:
        case_scores.append(float(torch.nan_to_num(weight_score, nan=1.0e30)))
    mean_score = sum(case_scores) / float(len(case_scores))
    return mean_score, tuple(case_scores)


def _linear_output_candidate_metrics(
    weight: torch.Tensor,
    activation_samples: Sequence[torch.Tensor],
    d: torch.Tensor,
    permutation: torch.Tensor,
    block_smooth_size: int = 0,
    block_smooth_seed: int = 0,
) -> tuple[float, tuple[float, ...]]:
    """Score a transform by the actual sampled Linear output error.

    Operand-local reconstruction error is a useful cheap proxy for diagonal
    smoothing, but it misses cancellation between activation and weight errors
    after a non-diagonal transform.  Block-S candidates therefore use the
    end-to-end sampled objective that the competition ultimately measures.
    """

    order = permutation.to(device=weight.device, dtype=torch.int64).reshape(-1)
    weight_transformed = _linear_pair_transform(
        weight,
        d,
        order,
        block_smooth_size,
        block_smooth_seed,
        weight_side=True,
    )
    weight_hat = _dequantize_hif4(_dense_to_hif4(weight_transformed))
    case_scores: list[float] = []
    for sample in activation_samples:
        activation_transformed = _linear_pair_transform(
            sample,
            d,
            order,
            block_smooth_size,
            block_smooth_seed,
            weight_side=False,
        )
        activation_hat = _dequantize_hif4(
            _dense_to_hif4(activation_transformed)
        )
        reference = activation_transformed.mm(weight_transformed.t())
        reconstructed = activation_hat.mm(weight_hat.t())
        score = (reference - reconstructed).square().sum() / (
            reference.square().sum() + _EPS
        )
        case_scores.append(
            float(
                torch.nan_to_num(
                    score, nan=1.0e30, posinf=1.0e30, neginf=1.0e30
                )
            )
        )
    if not case_scores:
        return 1.0e30, (1.0e30,)
    return sum(case_scores) / float(len(case_scores)), tuple(case_scores)


def _activation_quadratic8_is_safe(
    weight_smooth: torch.Tensor,
    weight_hat: torch.Tensor,
    activation_samples: Sequence[torch.Tensor],
    d: torch.Tensor,
    permutation: torch.Tensor,
    block_smooth_size: int,
    block_smooth_seed: int,
    importance: torch.Tensor,
    group_gram4: torch.Tensor,
    group_gram8: torch.Tensor,
    group_cross8: Optional[torch.Tensor],
    activation_ratio: float,
) -> bool:
    """Gate an 8x8 activation residual on sampled final-output MSE."""

    channels = int(weight_smooth.shape[1])
    blocks = channels // _HIF4_BLOCK_SIZE
    base_losses: list[float] = []
    candidate_losses: list[float] = []
    for sample in activation_samples:
        transformed = _linear_pair_transform(
            sample,
            d,
            permutation,
            block_smooth_size,
            block_smooth_seed,
            weight_side=False,
        )
        gram4 = group_gram4.reshape(blocks, 8, 2, 4, 4).unsqueeze(0).expand(
            int(transformed.shape[0]), blocks, 8, 2, 4, 4
        )
        base_params = _dense_to_hif4(
            transformed,
            importance=importance,
            group_gram=gram4,
            search_offsets=_DYNAMIC_OFFSETS,
            error_threshold=_ACTIVATION_REFINE_ERROR_THRESHOLD,
            accept_margin=_ACTIVATION_REFINE_ACCEPT_MARGIN,
            max_refine_ratio=float(activation_ratio),
            max_refine_blocks=_ACTIVATION_REFINE_MAX_BLOCKS,
        )
        candidate_params = _refine_weight_groups8(
            transformed,
            base_params,
            group_gram8,
            group_cross8=group_cross8,
            max_ratio=_ACTIVATION_QUADRATIC8_MAX_RATIO,
            max_groups=_ACTIVATION_QUADRATIC8_MAX_GROUPS,
            sweeps=_ACTIVATION_QUADRATIC8_SWEEPS,
            accept_margin=_ACTIVATION_QUADRATIC8_ACCEPT_MARGIN,
        )
        reference = transformed.mm(weight_smooth.t())
        base_output = _dequantize_hif4(base_params).mm(weight_hat.t())
        candidate_output = _dequantize_hif4(candidate_params).mm(weight_hat.t())
        denominator = reference.square().sum() + _EPS
        base_losses.append(
            float((reference - base_output).square().sum() / denominator)
        )
        candidate_losses.append(
            float((reference - candidate_output).square().sum() / denominator)
        )

    if not base_losses:
        return False
    base_mean = sum(base_losses) / float(len(base_losses))
    candidate_mean = sum(candidate_losses) / float(len(candidate_losses))
    mean_safe = candidate_mean <= base_mean * (
        1.0 - _ACTIVATION_QUADRATIC8_GATE_MIN_IMPROVEMENT
    )
    worst_safe = all(
        candidate <= base * (
            1.0 + _ACTIVATION_QUADRATIC8_GATE_WORST_TOLERANCE
        )
        for base, candidate in zip(base_losses, candidate_losses)
    )
    return bool(mean_safe and worst_safe)


def _cpu_state_tensor(x: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(
        x.detach().to(device="cpu", dtype=torch.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).contiguous()


@torch.no_grad()
def hif4_calibration_and_quantize_weight(
    weight_quant: torch.Tensor,
    weight_scale: torch.Tensor,
    calib_activation_list: list,
) -> dict[str, Any]:
    """Calibrate SmoothQuant state and quantize a static Linear weight."""

    if not isinstance(calib_activation_list, list) or not calib_activation_list:
        raise ValueError("calib_activation_list must be a non-empty list")

    weight = _dequantize_nvfp4_float32(weight_quant, weight_scale)
    if weight.ndim != 2:
        raise ValueError("weight must be a 2D tensor [out_features, in_features]")
    out_features, in_features = map(int, weight.shape)
    if in_features % _HIF4_BLOCK_SIZE != 0:
        raise ValueError("in_features must be divisible by 64")

    sum_square = torch.zeros(in_features, dtype=torch.float32, device=weight.device)
    activation_amax = torch.zeros_like(sum_square)
    token_count = 0
    activation_samples: list[torch.Tensor] = []
    use_quadratic = (
        _WEIGHT_QUADRATIC
        and in_features <= _WEIGHT_QUADRATIC_MAX_FEATURES
    )
    if use_quadratic:
        cov_sum = torch.zeros(
            in_features, in_features, dtype=torch.float32, device=weight.device
        )

    for pair in calib_activation_list:
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError("Each calibration activation must be a (quant, scale) pair")
        activation = _dequantize_nvfp4_float32(pair[0], pair[1])
        if activation.ndim != 2 or int(activation.shape[1]) != in_features:
            raise ValueError("Calibration activation shape is incompatible with weight")
        stats_sample = _sample_rows(activation, _LINEAR_STATS_TOKENS)
        sum_square += stats_sample.square().sum(dim=0)
        if use_quadratic:
            cov_sum += stats_sample.t().mm(stats_sample)
        activation_amax = torch.maximum(
            activation_amax, stats_sample.abs().amax(dim=0)
        )
        token_count += int(stats_sample.shape[0])
        activation_samples.append(
            _sample_rows(activation, _LINEAR_EVAL_TOKENS).clone()
        )

    activation_second_moment = sum_square / float(max(token_count, 1))
    weight_amax = weight.abs().amax(dim=0)
    activation_rms = torch.sqrt(activation_second_moment.clamp_min(_EPS))
    weight_rms = torch.sqrt((weight * weight).mean(dim=0).clamp_min(_EPS))

    identity_d = torch.ones(
        in_features, dtype=torch.float32, device=weight.device
    )
    identity_perm = _identity_permutation(in_features, weight.device)
    smooth_candidates = [identity_d]
    smooth_alphas = (
        _WEIGHT_SMOOTH_ALPHAS_WIDE
        if (
            in_features >= _WIDE_LAYER_MIN_DIM
            or out_features >= _WIDE_LAYER_MIN_DIM
        )
        else _WEIGHT_SMOOTH_ALPHAS
    )
    for alpha in smooth_alphas:
        smooth_candidates.append(
            _smooth_scale(activation_amax, weight_amax, alpha)
        )
        if _WEIGHT_SMOOTH_RMS:
            smooth_candidates.append(
                _smooth_scale(activation_rms, weight_rms, alpha)
            )

    # Candidate search touches only sampled output rows.  The selected
    # transform is then applied to the full Weight exactly once.
    weight_sample = _sample_rows(weight, _LINEAR_WEIGHT_EVAL_ROWS)
    baseline_metrics = _linear_candidate_metrics(
        weight_sample,
        activation_second_moment,
        activation_samples,
        identity_d,
        identity_perm,
    )
    best_metrics = baseline_metrics
    best_d = identity_d
    best_perm = identity_perm
    best_block_smooth_size = 0
    best_block_smooth_seed = 0

    for candidate_index, candidate_d in enumerate(smooth_candidates):
        candidate_permutations = [identity_perm]
        sorted_perm = _hierarchy_aware_permutation(
            activation_amax / candidate_d,
            weight_amax * candidate_d,
        )
        if not torch.equal(sorted_perm, identity_perm):
            candidate_permutations.append(sorted_perm)

        for candidate_perm in candidate_permutations:
            if candidate_index == 0 and torch.equal(candidate_perm, identity_perm):
                continue
            metrics = _linear_candidate_metrics(
                weight_sample,
                activation_second_moment,
                activation_samples,
                candidate_d,
                candidate_perm,
            )
            uses_reordering = not torch.equal(candidate_perm, identity_perm)
            if (
                metrics[0] < best_metrics[0]
                and _candidate_is_safe(
                    metrics,
                    baseline_metrics,
                    min_mean_improvement=0.02 if uses_reordering else 0.01,
                    worst_tolerance=0.005 if uses_reordering else 0.02,
                )
            ):
                best_metrics = metrics
                best_d = candidate_d
                best_perm = candidate_perm

    # 置换基扩展：同一平滑 d 下比较 weight-only / activation-only 排序，
    # 诊断显示单侧排序常优于 max(log range) 组合排序。
    if _PERMUTATION_BASES:
        basis_ranges = {
            "w_amax": weight_amax * best_d,
            "x_amax": activation_amax / best_d,
            "w_rms": weight_rms * best_d,
            "x_rms": activation_rms / best_d,
        }
        seen = {tuple(best_perm.tolist())}
        for bname, b_range in basis_ranges.items():
            b_perm = _range_permutation(b_range)
            if torch.equal(b_perm, identity_perm):
                continue
            if tuple(b_perm.tolist()) in seen:
                continue
            seen.add(tuple(b_perm.tolist()))
            b_metrics = _linear_candidate_metrics(
                weight_sample,
                activation_second_moment,
                activation_samples,
                best_d,
                b_perm,
            )
            if (
                b_metrics[0] < best_metrics[0]
                and _candidate_is_safe(
                    b_metrics,
                    baseline_metrics,
                    min_mean_improvement=0.02,
                    worst_tolerance=0.005,
                )
            ):
                best_metrics = b_metrics
                best_perm = b_perm

    # Matrix SmoothQuant extension: within the channel groups selected above,
    # try non-diagonal block transforms of size 4/8/16.  The transform is a
    # deterministic signed Hadamard, hence exactly orthogonal and represented
    # in dynamic state by two small integers rather than a dense matrix.
    force_block_size = int(_BLOCK_SMOOTH_FORCE_SIZE)
    candidate_block_sizes = (
        (force_block_size,) if force_block_size else _BLOCK_SMOOTH_SIZES
    )
    forced_choice: Optional[tuple[tuple[float, tuple[float, ...]], int, int]] = None
    block_baseline_metrics = _linear_output_candidate_metrics(
        weight_sample,
        activation_samples,
        best_d,
        best_perm,
    )
    block_best_metrics = block_baseline_metrics
    if candidate_block_sizes:
        for candidate_size in candidate_block_sizes:
            size = int(candidate_size)
            if size <= 0 or in_features % size != 0:
                continue
            for candidate_seed in _BLOCK_SMOOTH_SEEDS:
                seed = int(candidate_seed)
                block_metrics = _linear_output_candidate_metrics(
                    weight_sample,
                    activation_samples,
                    best_d,
                    best_perm,
                    size,
                    seed,
                )
                if force_block_size:
                    if (
                        forced_choice is None
                        or block_metrics[0] < forced_choice[0][0]
                    ):
                        forced_choice = (block_metrics, size, seed)
                    continue
                if (
                    block_metrics[0] < block_best_metrics[0]
                    and _candidate_is_safe(
                        block_metrics,
                        block_baseline_metrics,
                        min_mean_improvement=_BLOCK_SMOOTH_MIN_IMPROVEMENT,
                        worst_tolerance=_BLOCK_SMOOTH_WORST_TOLERANCE,
                    )
                ):
                    block_best_metrics = block_metrics
                    best_block_smooth_size = size
                    best_block_smooth_seed = seed
    if forced_choice is not None:
        _, best_block_smooth_size, best_block_smooth_seed = forced_choice

    weight_smooth = _linear_pair_transform(
        weight,
        best_d,
        best_perm,
        best_block_smooth_size,
        best_block_smooth_seed,
        weight_side=True,
    )
    h_x_smooth = _transformed_second_moment(
        activation_second_moment,
        best_d,
        best_perm,
        best_block_smooth_size,
    )
    weight_group_gram = None
    weight_group_gram8 = None
    weight_group_gram16 = None
    if use_quadratic:
        gram = _transformed_covariance(
            cov_sum / float(max(token_count, 1)),
            best_d,
            best_perm,
            best_block_smooth_size,
            best_block_smooth_seed,
        )
        blocks = in_features // _HIF4_BLOCK_SIZE
        weight_group_gram = _flat_group_gram(gram, in_features).reshape(
            blocks, 8, 2, 4, 4
        ).unsqueeze(0).expand(
            int(weight.shape[0]), blocks, 8, 2, 4, 4
        )
        if _WEIGHT_QUADRATIC8:
            weight_group_gram8 = _flat_group_gram8(gram, in_features)
        if _WEIGHT_QUADRATIC16:
            weight_group_gram16 = _flat_group_gram16(gram, in_features)
    weight_params = _dense_to_hif4(
        weight_smooth,
        importance=h_x_smooth,
        group_gram=weight_group_gram,
        search_offsets=_WEIGHT_OFFSETS,
        error_threshold=_WEIGHT_REFINE_ERROR_THRESHOLD,
        accept_margin=_WEIGHT_REFINE_ACCEPT_MARGIN,
        max_refine_ratio=(
            _WEIGHT_REFINE_MAX_RATIO_SMALL
            if int(weight.numel()) <= 4_194_304
            else _WEIGHT_REFINE_MAX_RATIO_LARGE
        ),
        max_refine_blocks=_WEIGHT_REFINE_MAX_BLOCKS,
    )
    if weight_group_gram8 is not None:
        weight_params = _refine_weight_groups8(
            weight_smooth, weight_params, weight_group_gram8
        )
    if weight_group_gram16 is not None:
        weight_params = _refine_weight_groups16(
            weight_smooth, weight_params, weight_group_gram16
        )

    weight_hat = _dequantize_hif4(weight_params)
    activation_importance = _normalize_importance(
        weight_hat.square().sum(dim=0), in_features
    )
    if activation_importance is None:
        activation_importance = torch.ones_like(best_d)

    permutation_state = None
    if not torch.equal(best_perm, identity_perm):
        permutation_state = best_perm.detach().to(
            device="cpu", dtype=torch.int64
        ).contiguous()
    smooth_inv_state = None
    if not torch.equal(best_d, identity_d):
        smooth_inv_state = _cpu_state_tensor(best_d.reciprocal())

    if _DATA_DRIVEN_RATIO:
        loss_parts = []
        for sample in activation_samples:
            transformed = sample.to(dtype=torch.float32)
            if smooth_inv_state is not None:
                transformed = transformed * smooth_inv_state.to(
                    transformed.device
                ).reshape(1, -1)
            if permutation_state is not None:
                transformed = transformed.index_select(
                    -1, permutation_state.to(transformed.device)
                )
            if best_block_smooth_size != 0:
                transformed = _block_hadamard_transform(
                    transformed,
                    best_block_smooth_size,
                    best_block_smooth_seed,
                )
            loss_parts.append(
                _standard_block_losses(transformed, activation_importance)
            )
        activation_ratio = _loss_capture_ratio(
            torch.cat(loss_parts),
            target=_RATIO_CAPTURE_TARGET,
            ratio_min=_RATIO_MIN,
        )
    else:
        activation_ratio = _ACTIVATION_REFINE_MAX_RATIO

    activation_gram_state = None
    activation_gram8_state = None
    activation_cross8_state = None
    if (
        _ACTIVATION_QUADRATIC
        and in_features <= _ACTIVATION_QUADRATIC_MAX_FEATURES
    ):
        gram = weight_smooth.t().mm(weight_smooth)
        activation_gram_state = _cpu_state_tensor(
            _flat_group_gram(gram, in_features)
        )
        if (
            _ACTIVATION_QUADRATIC8
            and in_features >= _ACTIVATION_QUADRATIC8_MIN_FEATURES
        ):
            group_gram8 = _flat_group_gram8(gram, in_features)
            group_cross8 = None
            if _ACTIVATION_QUADRATIC8_CROSS_TERM:
                cross = (weight_hat - weight_smooth).t().mm(weight_hat)
                group_cross8 = _flat_group_gram8(cross, in_features)
            use_group8 = True
            if (
                _ACTIVATION_QUADRATIC8_CALIBRATION_GATE
                and in_features <= _ACTIVATION_QUADRATIC8_GATE_MAX_FEATURES
            ):
                use_group8 = _activation_quadratic8_is_safe(
                    weight_smooth,
                    weight_hat,
                    activation_samples,
                    best_d,
                    best_perm,
                    best_block_smooth_size,
                    best_block_smooth_seed,
                    activation_importance,
                    activation_gram_state.to(weight.device),
                    group_gram8,
                    group_cross8,
                    activation_ratio,
                )
            if use_group8:
                activation_gram8_state = _cpu_state_tensor(group_gram8)
                if group_cross8 is not None:
                    activation_cross8_state = _cpu_state_tensor(group_cross8)

    activation_state = {
        "smooth_inv": smooth_inv_state,
        "permutation": permutation_state,
        "block_smooth_size": int(best_block_smooth_size),
        "block_smooth_seed": int(best_block_smooth_seed),
        "importance": _cpu_state_tensor(activation_importance),
        "gram": activation_gram_state,
        "gram8": activation_gram8_state,
        "cross8": activation_cross8_state,
        "offsets": torch.tensor(_DYNAMIC_OFFSETS, dtype=torch.int8, device="cpu"),
        "error_threshold": _ACTIVATION_REFINE_ERROR_THRESHOLD,
        "accept_margin": _ACTIVATION_REFINE_ACCEPT_MARGIN,
        "max_refine_ratio": float(activation_ratio),
        "max_refine_blocks": _ACTIVATION_REFINE_MAX_BLOCKS,
        "in_features": int(in_features),
        "version": 3,
    }
    return {
        "weight_params": weight_params,
        "activation_state": activation_state,
    }


@torch.no_grad()
def hif4_dynamic_quantize_activation(
    activation_quant: torch.Tensor,
    activation_scale: torch.Tensor,
    activation_state: Any,
) -> dict[str, torch.Tensor]:
    if not isinstance(activation_state, dict):
        raise TypeError("activation_state must be a dict")
    channels = int(activation_quant.shape[-1])
    if channels != int(activation_state.get("in_features", -1)):
        raise ValueError("Activation hidden size does not match calibration state")
    return _nvfp4_to_hif4(
        activation_quant,
        activation_scale,
        multiplier=activation_state["smooth_inv"],
        permutation=activation_state["permutation"],
        block_smooth_size=int(activation_state.get("block_smooth_size", 0)),
        block_smooth_seed=int(activation_state.get("block_smooth_seed", 0)),
        importance=activation_state["importance"],
        group_gram=activation_state.get("gram"),
        group_gram8=activation_state.get("gram8"),
        group_cross8=activation_state.get("cross8"),
        search_offsets=activation_state["offsets"],
        error_threshold=float(activation_state["error_threshold"]),
        accept_margin=float(activation_state["accept_margin"]),
        max_refine_ratio=float(activation_state["max_refine_ratio"]),
        max_refine_blocks=int(activation_state["max_refine_blocks"]),
    )


def _smooth_qk_scale(
    q_peak: torch.Tensor,
    k_peak: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    d = (k_peak + _EPS).pow(alpha) / (q_peak + _EPS).pow(1.0 - alpha)
    return torch.nan_to_num(
        d, nan=1.0, posinf=_QK_SMOOTH_MAX, neginf=_QK_SMOOTH_MIN
    ).clamp(min=_QK_SMOOTH_MIN, max=_QK_SMOOTH_MAX)


def _attention_candidate_metrics(
    q_samples: Sequence[torch.Tensor],
    k_samples: Sequence[torch.Tensor],
    d_kv: torch.Tensor,
    q_second_moment: torch.Tensor,
    k_effective_second_moment: torch.Tensor,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
    q_permutation: torch.Tensor,
    k_permutation: torch.Tensor,
    center_mode: int,
    a1_context: Optional[dict] = None,
) -> tuple[float, tuple[float, ...]]:
    """Q/K quantization proxy with GQA-aligned equivalent transforms.

    With ``a1_context`` the score becomes the real attention output error
    (A1): causal MSE is the primary selector, non-causal MSE interleaves into
    the per-case tuple so ``_candidate_is_safe`` protects both masks.  The
    V quantization is held fixed across candidates, isolating the Q/K
    transform choice exactly like the proxy did.
    """

    group_size = q_num_heads // kv_num_heads
    d_q = d_kv.repeat_interleave(group_size, dim=0)
    d_k = d_kv.reciprocal()
    q_order = q_permutation.to(dtype=torch.int64, device=d_kv.device).reshape(-1)
    k_order = k_permutation.to(dtype=torch.int64, device=d_kv.device).reshape(-1)

    if a1_context is not None:
        causal_scores: list[float] = []
        safety_scores: list[float] = []
        identity_cases = a1_context["identity"]
        for index, (q_full, k_full, v_hat, (ref_c, ref_n)) in enumerate(
            zip(
                a1_context["q_full"],
                a1_context["k_full"],
                a1_context["v_hat"],
                a1_context["refs"],
            )
        ):
            q_smooth = (q_full * d_q.reshape(1, -1)).index_select(-1, q_order)
            k_centered = _center_attention_k(
                k_full, kv_num_heads, head_dim, center_mode
            )
            k_smooth = (k_centered * d_k.reshape(1, -1)).index_select(
                -1, k_order
            )
            q_hat = _dequantize_hif4(_dense_to_hif4(q_smooth))
            k_hat = _dequantize_hif4(_dense_to_hif4(k_smooth))
            out_c = _attention_forward(
                q_hat, k_hat, v_hat, q_num_heads, kv_num_heads, head_dim, True
            )
            out_n = _attention_forward(
                q_hat, k_hat, v_hat, q_num_heads, kv_num_heads, head_dim, False
            )
            err_c = float((out_c - ref_c).square().mean())
            err_n = float((out_n - ref_n).square().mean())
            id_c, id_n = identity_cases[index]
            ratio_c = err_c / max(id_c, 1.0e-12)
            ratio_n = err_n / max(id_n, 1.0e-12)
            if not (math.isfinite(ratio_c) and math.isfinite(ratio_n)):
                return 1.0e30, (1.0e30,)
            causal_scores.append(ratio_c)
            safety_scores.append(ratio_n)
        if not causal_scores:
            return 1.0e30, (1.0e30,)
        primary = sum(causal_scores) / len(causal_scores)
        if not math.isfinite(primary):
            return 1.0e30, (1.0e30,)
        cases = [
            value
            for pair in zip(causal_scores, safety_scores)
            for value in pair
        ]
        cases.append(sum(safety_scores) / len(safety_scores))
        return primary, tuple(cases)

    q_second_kv = q_second_moment.reshape(
        kv_num_heads, group_size, head_dim
    ).mean(dim=1)
    h_k = k_effective_second_moment * d_k.square()
    h_q = q_second_kv * d_kv.square()
    h_k_for_q = h_k.repeat_interleave(group_size, dim=0).reshape(-1)
    h_q_for_k = h_q.reshape(-1)
    h_k_for_q = h_k_for_q.index_select(0, q_order)
    h_q_for_k = h_q_for_k.index_select(0, k_order)
    h_k_for_q = _normalize_importance(h_k_for_q, q_num_heads * head_dim)
    h_q_for_k = _normalize_importance(h_q_for_k, kv_num_heads * head_dim)
    if h_k_for_q is None or h_q_for_k is None:
        raise RuntimeError("Attention importance construction failed")

    case_scores: list[float] = []
    for q_sample, k_sample in zip(q_samples, k_samples):
        q_smooth = (q_sample * d_q.reshape(1, -1)).index_select(
            -1, q_order
        )
        k_centered = _center_attention_k(
            k_sample, kv_num_heads, head_dim, center_mode
        )
        k_smooth = (k_centered * d_k.reshape(1, -1)).index_select(
            -1, k_order
        )
        q_hat = _dequantize_hif4(_dense_to_hif4(q_smooth))
        k_hat = _dequantize_hif4(_dense_to_hif4(k_smooth))

        q_error = (
            (q_smooth - q_hat).square() * h_k_for_q.reshape(1, -1)
        ).sum()
        q_energy = (q_smooth.square() * h_k_for_q.reshape(1, -1)).sum()
        k_error = (
            (k_smooth - k_hat).square() * h_q_for_k.reshape(1, -1)
        ).sum()
        k_energy = (k_smooth.square() * h_q_for_k.reshape(1, -1)).sum()
        score = torch.nan_to_num(
            q_error / (q_energy + _EPS) + k_error / (k_energy + _EPS),
            nan=1.0e30,
            posinf=1.0e30,
            neginf=1.0e30,
        )
        case_scores.append(float(score))

    if not case_scores:
        return 1.0e30, (1.0e30,)
    return sum(case_scores) / float(len(case_scores)), tuple(case_scores)


@torch.no_grad()
def hif4_calibration_attention(
    calib_qkv_list: list,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
) -> dict[str, Any]:
    """Calibrate static Smooth-QK and output-sensitive Q/K weights."""

    if not isinstance(calib_qkv_list, list) or not calib_qkv_list:
        raise ValueError("calib_qkv_list must be a non-empty list")
    if q_num_heads <= 0 or kv_num_heads <= 0 or head_dim <= 0:
        raise ValueError("head counts and head_dim must be positive")
    if q_num_heads % kv_num_heads != 0:
        raise ValueError("q_num_heads must be divisible by kv_num_heads")
    q_channels = q_num_heads * head_dim
    kv_channels = kv_num_heads * head_dim
    if q_channels % 64 != 0 or kv_channels % 64 != 0:
        raise ValueError("Flattened Q/K/V dimensions must be divisible by 64")

    q_sum_square = torch.zeros(q_num_heads, head_dim, dtype=torch.float32)
    k_sum_square = torch.zeros(kv_num_heads, head_dim, dtype=torch.float32)
    k_mid_sum_square = torch.zeros_like(k_sum_square)
    q_peak_square = torch.zeros_like(q_sum_square)
    k_peak_square = torch.zeros_like(k_sum_square)
    k_mid_peak_square = torch.zeros_like(k_sum_square)
    q_token_count = 0
    k_token_count = 0
    sample_count = 0
    v_head_mass = torch.zeros(kv_num_heads, dtype=torch.float32)
    v_head_mean_mass = torch.zeros(kv_num_heads, dtype=torch.float32)
    q_samples: list[torch.Tensor] = []
    k_samples: list[torch.Tensor] = []
    v_samples: list[torch.Tensor] = []
    a1_q: list[torch.Tensor] = []
    a1_k: list[torch.Tensor] = []
    a1_v: list[torch.Tensor] = []
    a1_q_pairs: list = []
    a1_k_pairs: list = []
    a1_v_pairs: list = []

    for sample in calib_qkv_list:
        if not isinstance(sample, dict) or set(sample.keys()) != {"q", "k", "v"}:
            raise ValueError("Each attention calibration sample must contain q/k/v")
        q = _dequantize_nvfp4_float32(*sample["q"])
        k = _dequantize_nvfp4_float32(*sample["k"])
        if not isinstance(sample["v"], (tuple, list)) or len(sample["v"]) != 2:
            raise ValueError("V calibration data must be an NVFP4 pair")
        v_quant, v_scale = sample["v"]
        if not torch.is_tensor(v_quant) or not torch.is_tensor(v_scale):
            raise TypeError("V calibration pair must contain tensors")
        if q.ndim != 2 or k.ndim != 2 or v_quant.ndim != 2:
            raise ValueError("Q/K/V calibration tensors must be 2D")
        if int(q.shape[1]) != q_channels:
            raise ValueError("Q calibration width does not match head metadata")
        if int(k.shape[1]) != kv_channels or int(v_quant.shape[1]) != kv_channels:
            raise ValueError("K/V calibration width does not match head metadata")
        expected_v_scale_shape = (int(v_quant.shape[0]), kv_channels // 16)
        if tuple(v_scale.shape) != expected_v_scale_shape:
            raise ValueError("V calibration scale shape is invalid")
        if int(q.shape[0]) != int(k.shape[0]) or int(k.shape[0]) != int(v_quant.shape[0]):
            raise ValueError("Q/K/V in a calibration sample must share seq_len")

        if q_sum_square.device != q.device:
            q_sum_square = q_sum_square.to(q.device)
            k_sum_square = k_sum_square.to(q.device)
            k_mid_sum_square = k_mid_sum_square.to(q.device)
            q_peak_square = q_peak_square.to(q.device)
            k_peak_square = k_peak_square.to(q.device)
            k_mid_peak_square = k_mid_peak_square.to(q.device)
            v_head_mass = v_head_mass.to(q.device)
            v_head_mean_mass = v_head_mean_mass.to(q.device)

        if _V_ATTENTION_IMPORTANCE:
            head_mean_mass, head_square_mass = _attention_head_square_mass(
                q, k, q_num_heads, kv_num_heads, head_dim
            )
            v_head_mass += head_square_mass
            if head_mean_mass is not None:
                v_head_mean_mass += head_mean_mass

        q_stats = _sample_rows(q, _ATTN_STATS_TOKENS).reshape(
            -1, q_num_heads, head_dim
        )
        k_stats = _sample_rows(k, _ATTN_STATS_TOKENS).reshape(
            -1, kv_num_heads, head_dim
        )
        k_mid_stats = _center_attention_k(
            k_stats.reshape(-1, kv_channels),
            kv_num_heads,
            head_dim,
            2,
        ).reshape(-1, kv_num_heads, head_dim)
        q_sum_square += q_stats.square().sum(dim=0)
        k_sum_square += k_stats.square().sum(dim=0)
        k_mid_sum_square += k_mid_stats.square().sum(dim=0)
        q_peak_square += q_stats.abs().amax(dim=0).square()
        k_peak_square += k_stats.abs().amax(dim=0).square()
        k_mid_peak_square += k_mid_stats.abs().amax(dim=0).square()
        q_token_count += int(q_stats.shape[0])
        k_token_count += int(k_stats.shape[0])
        sample_count += 1
        q_samples.append(_sample_rows(q, _ATTN_EVAL_TOKENS).clone())
        k_samples.append(_sample_rows(k, _ATTN_EVAL_TOKENS).clone())
        v_dense = _dequantize_nvfp4_float32(v_quant, v_scale)
        v_samples.append(_sample_rows(v_dense, _ATTN_EVAL_TOKENS).clone())
        if _ATTN_OUTPUT_SELECTOR:
            prefix = min(int(q.shape[0]), _ATTN_A1_MAX_TOKENS)
            a1_q.append(q[:prefix].clone())
            a1_k.append(k[:prefix].clone())
            a1_v.append(v_dense[:prefix].clone())
            a1_q_pairs.append(
                (sample["q"][0][:prefix], sample["q"][1][:prefix])
            )
            a1_k_pairs.append(
                (sample["k"][0][:prefix], sample["k"][1][:prefix])
            )
            a1_v_pairs.append((v_quant[:prefix], v_scale[:prefix]))

    a1_context = None
    if _ATTN_OUTPUT_SELECTOR and a1_q:
        v_hats = [_dequantize_hif4(_dense_to_hif4(v)) for v in a1_v]
        refs = []
        identity_cases = []
        for q, k, v, v_hat in zip(a1_q, a1_k, a1_v, v_hats):
            ref_c = _attention_forward(
                q, k, v, q_num_heads, kv_num_heads, head_dim, True
            )
            ref_n = _attention_forward(
                q, k, v, q_num_heads, kv_num_heads, head_dim, False
            )
            refs.append((ref_c, ref_n))
            q_hat = _dequantize_hif4(_dense_to_hif4(q))
            k_hat = _dequantize_hif4(_dense_to_hif4(k))
            id_c = _attention_forward(
                q_hat, k_hat, v_hat, q_num_heads, kv_num_heads, head_dim, True
            )
            id_n = _attention_forward(
                q_hat, k_hat, v_hat, q_num_heads, kv_num_heads, head_dim, False
            )
            identity_cases.append(
                (
                    float((id_c - ref_c).square().mean()),
                    float((id_n - ref_n).square().mean()),
                )
            )
        a1_context = {
            "q_full": a1_q,
            "k_full": a1_k,
            "v_hat": v_hats,
            "refs": refs,
            "identity": identity_cases,
        }

    v_importance = None
    v_importance_candidates: dict = {}
    if _V_ATTENTION_IMPORTANCE and sample_count > 0:
        head_importance = v_head_mass / float(max(sample_count, 1))
        head_mean_importance = v_head_mean_mass / float(
            max(sample_count, 1)
        )
        head_importance = head_importance / head_importance.mean().clamp_min(
            _EPS
        )
        head_mean_importance = head_mean_importance / (
            head_mean_importance.mean().clamp_min(_EPS)
        )
        if _V_ATTENTION_IMPORTANCE_SHRINK < 1.0:
            head_importance = 1.0 + _V_ATTENTION_IMPORTANCE_SHRINK * (
                head_importance - 1.0
            )
            head_mean_importance = 1.0 + _V_ATTENTION_IMPORTANCE_SHRINK * (
                head_mean_importance - 1.0
            )
        v_importance = _normalize_importance(
            head_importance.repeat_interleave(head_dim).reshape(-1),
            kv_channels,
        )
        if _V_IMPORTANCE_CANDIDATES and a1_context is not None:
            # A3 候选：一阶矩 E[A] 与 E[A^2] + E[A]^2（均值交叉项）。
            # 仅改变 head 级 importance 向量，V 坐标系不变。
            first_moment = _normalize_importance(
                head_mean_importance.repeat_interleave(head_dim).reshape(-1),
                kv_channels,
            )
            combined_head = head_importance + head_mean_importance.square()
            combined = _normalize_importance(
                combined_head.repeat_interleave(head_dim).reshape(-1),
                kv_channels,
            )
            v_importance_candidates["first_moment"] = first_moment
            v_importance_candidates["mean_cross"] = combined

    q_second_moment = q_sum_square / float(max(q_token_count, 1))
    k_second_moment = k_sum_square / float(max(k_token_count, 1))
    k_mid_second_moment = k_mid_sum_square / float(max(k_token_count, 1))
    q_peak = torch.sqrt(q_peak_square / float(max(sample_count, 1)))
    k_peak = torch.sqrt(k_peak_square / float(max(sample_count, 1)))
    k_mid_peak = torch.sqrt(k_mid_peak_square / float(max(sample_count, 1)))

    group_size = q_num_heads // kv_num_heads
    q_peak_kv = q_peak.reshape(kv_num_heads, group_size, head_dim).amax(dim=1)
    identity_d = torch.ones(
        kv_num_heads,
        head_dim,
        dtype=torch.float32,
        device=q_second_moment.device,
    )
    local_identity = torch.arange(
        head_dim, dtype=torch.int64, device=q_second_moment.device
    )[None, :].expand(kv_num_heads, -1)
    k_identity_perm = _flatten_head_permutation(local_identity)
    q_identity_perm = _flatten_head_permutation(
        local_identity.repeat_interleave(group_size, dim=0)
    )

    def _run_selection(use_a1: bool):
        context = a1_context if use_a1 else None
        baseline_metrics = _attention_candidate_metrics(
            q_samples,
            k_samples,
            identity_d,
            q_second_moment,
            k_second_moment,
            q_num_heads,
            kv_num_heads,
            head_dim,
            q_identity_perm,
            k_identity_perm,
            0,
            context,
        )
        best_metrics = baseline_metrics
        best_d = identity_d
        best_center_mode = 0
        best_q_perm = q_identity_perm
        best_k_perm = k_identity_perm

        # Midrange K-centering is an exact softmax invariance.  First select
        # the centering/smoothing pair with identity ordering, then test one
        # hierarchy-aware ordering for the selected pair to bound calibration
        # time.
        for center_mode in _ATTN_CENTER_MODES:
            if center_mode in (2, 3):
                effective_second = k_mid_second_moment
                effective_peak = k_mid_peak
            else:
                effective_second = k_second_moment
                effective_peak = k_peak
            q_rms_kv = torch.sqrt(
                q_second_moment.reshape(
                    kv_num_heads, group_size, head_dim
                ).mean(dim=1).clamp_min(_EPS)
            )
            k_rms = torch.sqrt(effective_second.clamp_min(_EPS))
            smooth_candidates = [identity_d]
            for alpha in _QK_SMOOTH_ALPHAS:
                smooth_candidates.append(
                    _smooth_qk_scale(q_peak_kv, effective_peak, alpha)
                )
                if _QK_SMOOTH_RMS:
                    smooth_candidates.append(
                        _smooth_qk_scale(q_rms_kv, k_rms, alpha)
                    )
            for candidate_index, candidate_d in enumerate(smooth_candidates):
                if center_mode == 0 and candidate_index == 0:
                    continue
                metrics = _attention_candidate_metrics(
                    q_samples,
                    k_samples,
                    candidate_d,
                    q_second_moment,
                    effective_second,
                    q_num_heads,
                    kv_num_heads,
                    head_dim,
                    q_identity_perm,
                    k_identity_perm,
                    center_mode,
                    context,
                )
                if (
                    metrics[0] < best_metrics[0]
                    and _candidate_is_safe(
                        metrics,
                        baseline_metrics,
                        min_mean_improvement=0.01,
                        worst_tolerance=0.02,
                    )
                ):
                    best_metrics = metrics
                    best_d = candidate_d
                    best_center_mode = center_mode

        selected_k_peak = (
            k_mid_peak if best_center_mode in (2, 3) else k_peak
        )
        selected_k_second = (
            k_mid_second_moment
            if best_center_mode in (2, 3)
            else k_second_moment
        )
        local_permutation = _headwise_hierarchy_permutation(
            q_peak_kv * best_d,
            selected_k_peak * best_d.reciprocal(),
        )
        candidate_k_perm = _flatten_head_permutation(local_permutation)
        candidate_q_perm = _flatten_head_permutation(
            local_permutation.repeat_interleave(group_size, dim=0)
        )
        if not torch.equal(candidate_k_perm, k_identity_perm):
            permutation_metrics = _attention_candidate_metrics(
                q_samples,
                k_samples,
                best_d,
                q_second_moment,
                selected_k_second,
                q_num_heads,
                kv_num_heads,
                head_dim,
                candidate_q_perm,
                candidate_k_perm,
                best_center_mode,
                context,
            )
            if (
                permutation_metrics[0] < best_metrics[0]
                and _candidate_is_safe(
                    permutation_metrics,
                    baseline_metrics,
                    min_mean_improvement=0.02,
                    worst_tolerance=0.005,
                )
            ):
                best_metrics = permutation_metrics
                best_q_perm = candidate_q_perm
                best_k_perm = candidate_k_perm

        # 置换基扩展：单侧排序（Q-only / K-only）常优于 max(log range) 组合。
        if _PERMUTATION_BASES:
            basis_ranges = {
                "q_amax": q_peak_kv * best_d,
                "k_amax": selected_k_peak * best_d.reciprocal(),
            }
            seen = {tuple(best_k_perm.tolist())}
            for bname, b_range in basis_ranges.items():
                b_local = _headwise_range_permutation(b_range)
                b_k_perm = _flatten_head_permutation(b_local)
                if torch.equal(b_k_perm, k_identity_perm):
                    continue
                if tuple(b_k_perm.tolist()) in seen:
                    continue
                seen.add(tuple(b_k_perm.tolist()))
                b_q_perm = _flatten_head_permutation(
                    b_local.repeat_interleave(group_size, dim=0)
                )
                b_metrics = _attention_candidate_metrics(
                    q_samples,
                    k_samples,
                    best_d,
                    q_second_moment,
                    selected_k_second,
                    q_num_heads,
                    kv_num_heads,
                    head_dim,
                    b_q_perm,
                    b_k_perm,
                    best_center_mode,
                    context,
                )
                if (
                    b_metrics[0] < best_metrics[0]
                    and _candidate_is_safe(
                        b_metrics,
                        baseline_metrics,
                        min_mean_improvement=0.02,
                        worst_tolerance=0.005,
                    )
                ):
                    best_metrics = b_metrics
                    best_q_perm = b_q_perm
                    best_k_perm = b_k_perm

        return best_d, best_center_mode, best_q_perm, best_k_perm

    # 双轨选择：A1 轨用真实 attention 输出误差（朴素 HiF4 代理量化），
    # proxy 轨复刻当前 Champion（B0）的 Q/K 重建 proxy 选择逻辑。终验门
    # 在部署路径上对比两个 winner，A1 无明确优势时回退 B0 选择。
    if a1_context is not None:
        a1_d, a1_center, a1_q_perm, a1_k_perm = _run_selection(True)
        proxy_d, proxy_center, proxy_q_perm, proxy_k_perm = _run_selection(
            False
        )
    else:
        (
            proxy_d,
            proxy_center,
            proxy_q_perm,
            proxy_k_perm,
        ) = _run_selection(False)
        a1_d, a1_center, a1_q_perm, a1_k_perm = (
            proxy_d,
            proxy_center,
            proxy_q_perm,
            proxy_k_perm,
        )

    def _build_v_state(importance) -> dict:
        if _DATA_DRIVEN_RATIO:
            v_ratio = _loss_capture_ratio(
                torch.cat(
                    [
                        _standard_block_losses(s, importance)
                        for s in v_samples
                    ]
                ),
                target=_RATIO_CAPTURE_TARGET,
                ratio_min=_RATIO_MIN,
            )
        else:
            v_ratio = _V_REFINE_MAX_RATIO
        return {
            "offsets": torch.tensor(
                _DYNAMIC_OFFSETS, dtype=torch.int8, device="cpu"
            ),
            "importance": (
                None if importance is None else _cpu_state_tensor(importance)
            ),
            "error_threshold": _ATTN_REFINE_ERROR_THRESHOLD,
            "accept_margin": _V_REFINE_ACCEPT_MARGIN,
            "max_refine_ratio": float(v_ratio),
            "max_refine_blocks": _V_REFINE_MAX_BLOCKS,
            "num_heads": int(kv_num_heads),
            "head_dim": int(head_dim),
            "version": 2,
        }

    v_state = _build_v_state(v_importance)

    def _build_qk_states(
        d: torch.Tensor,
        center_mode: int,
        q_perm: torch.Tensor,
        k_perm: torch.Tensor,
        rotation: Optional[torch.Tensor] = None,
    ) -> tuple:
        d_q = d.repeat_interleave(group_size, dim=0)
        d_k = d.reciprocal()
        q_second_kv = q_second_moment.reshape(
            kv_num_heads, group_size, head_dim
        ).mean(dim=1)
        eff_k_second = (
            k_mid_second_moment
            if int(center_mode) in (2, 3)
            else k_second_moment
        )
        h_k = eff_k_second * d_k.square()
        h_q = q_second_kv * d.square()
        h_k_for_q = h_k.repeat_interleave(group_size, dim=0).reshape(-1)
        h_q_for_k = h_q.reshape(-1)
        h_k_for_q = _normalize_importance(
            h_k_for_q.index_select(0, q_perm), q_channels
        )
        h_q_for_k = _normalize_importance(
            h_q_for_k.index_select(0, k_perm), kv_channels
        )
        if h_k_for_q is None:
            h_k_for_q = torch.ones(
                q_channels, dtype=torch.float32, device=d_q.device
            )
        if h_q_for_k is None:
            h_q_for_k = torch.ones(
                kv_channels, dtype=torch.float32, device=d_k.device
            )
        q_flat = d_q.reshape(-1)
        k_flat = d_k.reshape(-1)

        def q_transform(sample: torch.Tensor) -> torch.Tensor:
            transformed = (sample * q_flat.reshape(1, -1)).index_select(
                -1, q_perm
            )
            if rotation is not None:
                transformed = _apply_attention_rotation(
                    transformed, q_num_heads, head_dim, rotation
                )
            return transformed

        def k_transform(sample: torch.Tensor) -> torch.Tensor:
            transformed = (
                _center_attention_k(
                    sample, kv_num_heads, head_dim, int(center_mode)
                )
                * k_flat.reshape(1, -1)
            ).index_select(-1, k_perm)
            if rotation is not None:
                transformed = _apply_attention_rotation(
                    transformed, kv_num_heads, head_dim, rotation
                )
            return transformed

        if _DATA_DRIVEN_RATIO:
            q_ratio = _loss_capture_ratio(
                torch.cat(
                    [_standard_block_losses(q_transform(s), h_k_for_q)
                     for s in q_samples]
                ),
                target=_RATIO_CAPTURE_TARGET,
                ratio_min=_RATIO_MIN,
            )
            k_ratio = _loss_capture_ratio(
                torch.cat(
                    [_standard_block_losses(k_transform(s), h_q_for_k)
                     for s in k_samples]
                ),
                target=_RATIO_CAPTURE_TARGET,
                ratio_min=_RATIO_MIN,
            )
        else:
            q_ratio = _Q_REFINE_MAX_RATIO
            k_ratio = _K_REFINE_MAX_RATIO

        q_permutation_state = None
        k_permutation_state = None
        if not torch.equal(k_perm, k_identity_perm):
            q_permutation_state = q_perm.detach().to(
                device="cpu", dtype=torch.int64
            ).contiguous()
            k_permutation_state = k_perm.detach().to(
                device="cpu", dtype=torch.int64
            ).contiguous()
        q_multiplier_state = None
        k_multiplier_state = None
        if not torch.equal(d, identity_d):
            q_multiplier_state = _cpu_state_tensor(d_q.reshape(-1))
            k_multiplier_state = _cpu_state_tensor(d_k.reshape(-1))

        rotation_state = (
            None
            if rotation is None
            else rotation.detach().to(
                device="cpu", dtype=torch.float32
            ).contiguous()
        )
        q_state = {
            "multiplier": q_multiplier_state,
            "permutation": q_permutation_state,
            "importance": _cpu_state_tensor(h_k_for_q),
            "offsets": torch.tensor(_DYNAMIC_OFFSETS, dtype=torch.int8, device="cpu"),
            "error_threshold": _ATTN_REFINE_ERROR_THRESHOLD,
            "accept_margin": _Q_REFINE_ACCEPT_MARGIN,
            "max_refine_ratio": float(q_ratio),
            "max_refine_blocks": _Q_REFINE_MAX_BLOCKS,
            "num_heads": int(q_num_heads),
            "head_dim": int(head_dim),
            "version": 2,
        }
        k_state = {
            "multiplier": k_multiplier_state,
            "permutation": k_permutation_state,
            "center_mode": int(center_mode),
            "importance": _cpu_state_tensor(h_q_for_k),
            "offsets": torch.tensor(_DYNAMIC_OFFSETS, dtype=torch.int8, device="cpu"),
            "error_threshold": _ATTN_REFINE_ERROR_THRESHOLD,
            "accept_margin": _K_REFINE_ACCEPT_MARGIN,
            "max_refine_ratio": float(k_ratio),
            "max_refine_blocks": _K_REFINE_MAX_BLOCKS,
            "num_heads": int(kv_num_heads),
            "head_dim": int(head_dim),
            "version": 2,
        }
        if rotation_state is not None:
            q_state["rotation"] = rotation_state
            k_state["rotation"] = rotation_state
        return q_state, k_state

    q_state, k_state = _build_qk_states(
        a1_d, int(a1_center), a1_q_perm, a1_k_perm
    )
    final_d, final_center = a1_d, int(a1_center)
    final_q_perm, final_k_perm = a1_q_perm, a1_k_perm
    a1_v_hats = None
    base_causal = None
    base_safety = None

    # A1 终验门：A1 选择基于朴素 HiF4 代理，部署路径（offset 搜索 +
    # importance 精修）与隐藏 test 分布上的排序都可能错位。用完整
    # hif4_dynamic_quantize_q/k/v 路径在 calibration 前缀上重算真实
    # attention 输出误差（causal 主轨 + non-causal 安全轨，V 部署路径
    # 固定以隔离 Q/K 变换选择）；A1 winner 相对 B0 proxy winner（当前
    # Champion 的选择）无明确改善或安全轨退化时，回退 B0 选择。
    same_winner = (
        torch.equal(a1_d, proxy_d)
        and int(a1_center) == int(proxy_center)
        and torch.equal(a1_q_perm, proxy_q_perm)
        and torch.equal(a1_k_perm, proxy_k_perm)
    )
    if a1_context is not None and not same_winner:
        proxy_q_state, proxy_k_state = _build_qk_states(
            proxy_d, int(proxy_center), proxy_q_perm, proxy_k_perm
        )
        a1_v_hats = [
            _dequantize_hif4(
                hif4_dynamic_quantize_v(
                    v_quant, v_scale, kv_num_heads, head_dim, v_state
                )
            ).to(torch.float32)
            for v_quant, v_scale in a1_v_pairs
        ]
        winner_causal, winner_safety = _attention_deployed_mse(
            a1_q_pairs,
            a1_k_pairs,
            a1_v_hats,
            a1_context["refs"],
            q_state,
            k_state,
            q_num_heads,
            kv_num_heads,
            head_dim,
        )
        proxy_causal, proxy_safety = _attention_deployed_mse(
            a1_q_pairs,
            a1_k_pairs,
            a1_v_hats,
            a1_context["refs"],
            proxy_q_state,
            proxy_k_state,
            q_num_heads,
            kv_num_heads,
            head_dim,
        )
        if _a1_gate_passes(
            winner_causal, winner_safety, proxy_causal, proxy_safety
        ):
            base_causal, base_safety = winner_causal, winner_safety
        else:
            q_state, k_state = proxy_q_state, proxy_k_state
            final_d, final_center = proxy_d, int(proxy_center)
            final_q_perm, final_k_perm = proxy_q_perm, proxy_k_perm
            base_causal, base_safety = proxy_causal, proxy_safety

    # A2 固定 H64：对最终 Q/K winner 施加组对齐 signed Hadamard(64)
    # 旋转（同组 Q heads 与 K head 共享旋转，Q·K 点积严格不变），首版只
    # 比较 2 个确定性 sign seed。旋转候选须通过同一真实 attention 输出
    # 门控（causal 主轨 + non-causal 安全轨），否则保持无旋转 winner。
    if (
        _ATTN_H64
        and a1_context is not None
        and head_dim >= _ATTN_H64_BLOCK
        and head_dim % _ATTN_H64_BLOCK == 0
    ):
        if a1_v_hats is None:
            a1_v_hats = [
                _dequantize_hif4(
                    hif4_dynamic_quantize_v(
                        v_quant, v_scale, kv_num_heads, head_dim, v_state
                    )
                ).to(torch.float32)
                for v_quant, v_scale in a1_v_pairs
            ]
        if base_causal is None:
            base_causal, base_safety = _attention_deployed_mse(
                a1_q_pairs,
                a1_k_pairs,
                a1_v_hats,
                a1_context["refs"],
                q_state,
                k_state,
                q_num_heads,
                kv_num_heads,
                head_dim,
            )
        best_rotation_states = None
        best_rotation_mean = None
        for seed in _ATTN_H64_SEEDS:
            signs = _attention_rotation_signs(kv_num_heads, head_dim, seed)
            rotation_q_state, rotation_k_state = _build_qk_states(
                final_d,
                int(final_center),
                final_q_perm,
                final_k_perm,
                rotation=signs,
            )
            rotation_causal, rotation_safety = _attention_deployed_mse(
                a1_q_pairs,
                a1_k_pairs,
                a1_v_hats,
                a1_context["refs"],
                rotation_q_state,
                rotation_k_state,
                q_num_heads,
                kv_num_heads,
                head_dim,
            )
            if not _a1_gate_passes(
                rotation_causal,
                rotation_safety,
                base_causal,
                base_safety,
                safety_tolerance=0.0,
            ):
                continue
            rotation_mean = sum(rotation_causal) / len(rotation_causal)
            if best_rotation_mean is None or rotation_mean < best_rotation_mean:
                best_rotation_mean = rotation_mean
                best_rotation_states = (
                    rotation_q_state,
                    rotation_k_state,
                )
        if best_rotation_states is not None:
            q_state, k_state = best_rotation_states

    # A3 V importance 候选：Q/K state 已定稿（A1 终验门 + A2 旋转），
    # 仅更换 V 的 head 级 importance（当前 E[A^2] vs 一阶矩 E[A] vs
    # E[A^2]+E[A]^2），用完整 hif4_dynamic_quantize_v 部署路径在
    # calibration 前缀上重算真实 attention 输出误差；候选须通过同一
    # 门控（causal 主轨 + non-causal 安全轨），否则保持当前 importance。
    if v_importance_candidates and a1_context is not None:
        if a1_v_hats is None:
            a1_v_hats = [
                _dequantize_hif4(
                    hif4_dynamic_quantize_v(
                        v_quant, v_scale, kv_num_heads, head_dim, v_state
                    )
                ).to(torch.float32)
                for v_quant, v_scale in a1_v_pairs
            ]
        if base_causal is None:
            base_causal, base_safety = _attention_deployed_mse(
                a1_q_pairs,
                a1_k_pairs,
                a1_v_hats,
                a1_context["refs"],
                q_state,
                k_state,
                q_num_heads,
                kv_num_heads,
                head_dim,
            )
        best_v_state = None
        best_v_mean = None
        for candidate_importance in v_importance_candidates.values():
            candidate_state = _build_v_state(candidate_importance)
            candidate_v_hats = [
                _dequantize_hif4(
                    hif4_dynamic_quantize_v(
                        v_quant, v_scale, kv_num_heads, head_dim, candidate_state
                    )
                ).to(torch.float32)
                for v_quant, v_scale in a1_v_pairs
            ]
            candidate_causal, candidate_safety = _attention_deployed_mse(
                a1_q_pairs,
                a1_k_pairs,
                candidate_v_hats,
                a1_context["refs"],
                q_state,
                k_state,
                q_num_heads,
                kv_num_heads,
                head_dim,
            )
            if not _a1_gate_passes(
                candidate_causal, candidate_safety, base_causal, base_safety
            ):
                continue
            candidate_mean = sum(candidate_causal) / len(candidate_causal)
            if best_v_mean is None or candidate_mean < best_v_mean:
                best_v_mean = candidate_mean
                best_v_state = candidate_state
        if best_v_state is not None:
            v_state = best_v_state
    return {"q_state": q_state, "k_state": k_state, "v_state": v_state}


def _check_attention_state(
    state: Any,
    num_heads: int,
    head_dim: int,
    name: str,
) -> dict[str, Any]:
    if not isinstance(state, dict):
        raise TypeError(f"{name}_state must be a dict")
    if int(state.get("num_heads", -1)) != int(num_heads):
        raise ValueError(f"{name} head count does not match calibration state")
    if int(state.get("head_dim", -1)) != int(head_dim):
        raise ValueError(f"{name} head_dim does not match calibration state")
    return state


@torch.no_grad()
def hif4_dynamic_quantize_q(
    q_quant: torch.Tensor,
    q_scale: torch.Tensor,
    q_num_heads: int,
    head_dim: int,
    q_state: Any,
) -> dict[str, torch.Tensor]:
    state = _check_attention_state(q_state, q_num_heads, head_dim, "q")
    if int(q_quant.shape[-1]) != q_num_heads * head_dim:
        raise ValueError("Q width does not match q_num_heads * head_dim")
    return _nvfp4_to_hif4(
        q_quant,
        q_scale,
        multiplier=state["multiplier"],
        permutation=state["permutation"],
        attention_rotation=state.get("rotation"),
        rotation_num_heads=int(q_num_heads),
        importance=state["importance"],
        search_offsets=state["offsets"],
        error_threshold=float(state["error_threshold"]),
        accept_margin=float(state["accept_margin"]),
        max_refine_ratio=float(state["max_refine_ratio"]),
        max_refine_blocks=int(state["max_refine_blocks"]),
    )


@torch.no_grad()
def hif4_dynamic_quantize_k(
    k_quant: torch.Tensor,
    k_scale: torch.Tensor,
    kv_num_heads: int,
    head_dim: int,
    k_state: Any,
) -> dict[str, torch.Tensor]:
    state = _check_attention_state(k_state, kv_num_heads, head_dim, "k")
    if int(k_quant.shape[-1]) != kv_num_heads * head_dim:
        raise ValueError("K width does not match kv_num_heads * head_dim")
    return _nvfp4_to_hif4(
        k_quant,
        k_scale,
        multiplier=state["multiplier"],
        permutation=state["permutation"],
        attention_rotation=state.get("rotation"),
        rotation_num_heads=int(kv_num_heads),
        center_mode=int(state["center_mode"]),
        center_num_heads=kv_num_heads,
        center_head_dim=head_dim,
        importance=state["importance"],
        search_offsets=state["offsets"],
        error_threshold=float(state["error_threshold"]),
        accept_margin=float(state["accept_margin"]),
        max_refine_ratio=float(state["max_refine_ratio"]),
        max_refine_blocks=int(state["max_refine_blocks"]),
    )


@torch.no_grad()
def hif4_dynamic_quantize_v(
    v_quant: torch.Tensor,
    v_scale: torch.Tensor,
    kv_num_heads: int,
    head_dim: int,
    v_state: Any,
) -> dict[str, torch.Tensor]:
    state = _check_attention_state(v_state, kv_num_heads, head_dim, "v")
    if int(v_quant.shape[-1]) != kv_num_heads * head_dim:
        raise ValueError("V width does not match kv_num_heads * head_dim")
    return _nvfp4_to_hif4(
        v_quant,
        v_scale,
        importance=state["importance"],
        search_offsets=state["offsets"],
        error_threshold=float(state["error_threshold"]),
        accept_margin=float(state["accept_margin"]),
        max_refine_ratio=float(state["max_refine_ratio"]),
        max_refine_blocks=int(state["max_refine_blocks"]),
    )
