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
# C43: HiF4-aligned analytic CAT-64.  The transform is block diagonal and
# exactly equivalent in floating point: activation rows use M^T while the
# static weight carrier uses M^-T.  Only the selected block matrices are
# retained in activation_state; all covariance/weight products are released
# after calibration.  Candidate selection is operand-local and never builds
# A@W, which keeps the C43 mechanism separate from the later C45 selector.
_CAT64 = True
_CAT64_BLOCK_SIZE = 64
_CAT64_SHRINK = 0.05
_CAT64_RELATIVE_FLOOR = 1.0e-4
_CAT64_BETAS = (0.25,)
_CAT64_MIN_IMPROVEMENT = 1.0e-3
_CAT64_WORST_TOLERANCE = 0.03
_CAT64_MAX_CALIB_ROWS = 256
# C47: operand-only CAT-aware channel grouping.  A bounded hierarchical
# permutation is proposed before CAT so strongly coupled channels share the
# same 64-channel transform block.  It is kept independent from C30's
# residual-based prototype and uses a soft aggregate gate.
_CAT64_GROUPING = True
_CAT64_GROUPING_MAX_FEATURES = 2048
_CAT64_GROUPING_MIN_IMPROVEMENT = 5.0e-3
_CAT64_GROUPING_WORST_TOLERANCE = 0.03
# C45: calibration-product selector for the *static* weight quantizer.
#
# A@W is used here only as an offline objective for Q(W).  The online
# activation state is completely materialized before this selector runs, and
# no product/residual tensor is retained in that state.  The update is a
# bounded, block-conditional HiF4 projection: it proposes a few legal code
# updates for high-leverage 64-channel blocks, then accepts the whole static
# weight candidate using a soft mean/worst-fold product score.  There is no
# hard per-fold veto; the validation score is deliberately the main guard so
# a useful cross-window signal is not discarded by an over-strict threshold.
_WEIGHT_PRODUCT_SELECTOR = True
_WEIGHT_PRODUCT_SELECTOR_MAX_RATIO = 0.25
_WEIGHT_PRODUCT_SELECTOR_ALPHAS = (0.10, 0.25, 0.50)
_WEIGHT_PRODUCT_SELECTOR_DAMPING = 0.30
_WEIGHT_PRODUCT_SELECTOR_ROBUST_MIX = 0.25
_WEIGHT_PRODUCT_SELECTOR_MIN_GAIN = 1.0e-4
_WEIGHT_PRODUCT_SELECTOR_MIN_CHANNELS = 512
_WEIGHT_PRODUCT_SELECTOR_MAX_CALIB_ROWS = 128
# A 3072 dimension cap keeps the A@W product estimate well-conditioned with
# the two-window calibration budget used by the evaluator.  Wider matrices
# (GPT-2 medium's 4096-wide FFN and Qwen's 4864-wide FFN) retain the parent
# CAT/full-H solution instead of spending a high-variance update on two rows.
_WEIGHT_PRODUCT_SELECTOR_MAX_DIM = 4096
# C22: 64-dim incoherence transform (signed Hadamard, butterfly FWHT).
# Seed selection is two-stage: a cheap operand-local rank over 32 seeds,
# then a deployed two-fold validation of the top seeds against the parent
# transform.  Dynamic state stores only the two integers above.
# C22 was REJECTED on 2026-08-27 (v026): the R64 mixing regressed both
# operands on real data (ratio_A up to 1.37) and cost 1.52x calibration
# time, so the production default keeps the C21-C behavior.
_LINEAR_R64 = False
_LINEAR_R64_BLOCK = 64
_LINEAR_R64_STAGE1_SEEDS = tuple(range(32))
_LINEAR_R64_STAGE2_KEEP = 4
_LINEAR_R64_MIN_IMPROVEMENT = 0.005
_LINEAR_R64_WORST_TOLERANCE = 0.002
_LINEAR_R64_STAGE1_ROWS = 64
_LINEAR_R64_STAGE1_WEIGHT_ROWS = 128
# C23: full-64 weight Schur/GPTQ refinement.  Each transformed 64-channel
# block is re-solved against the full 64x64 damped activation Hessian with
# a 4-way scale beam; per-block fallback keeps the parent parameters
# wherever the solver does not strictly reduce the full-H loss.
# Budget constants (pre-registered 2026-08-27 before any C23 evaluation):
# _WEIGHT_FULL64_MAX_RATIO selects, per row chunk, the top block columns by
# parent full-H loss; unselected columns keep parent parameters, which is
# identical to the per-block fallback semantics.  _WEIGHT_FULL64_CHUNK_ROWS
# bounds the [chunk, blocks, 64] working tensors (memory-bounded chunking;
# the plan's 128 default is dominated by dispatch overhead on small-B
# layers, so production uses 1024 rows per ~3MB chunk).
# C39-FW: re-test the C23 full-64 weight solver on FFN-width Linear layers
# only.  This is intentionally a single-mechanism candidate derived from
# C21-C: q/k/v/o keep the exact C21-C path, while fc/proj are the diagnostic
# arm for separating the C38 FULL64 effect from its unrelated activation
# changes.  The official result is needed to calibrate this local proxy.
_WEIGHT_FULL64 = True
_WEIGHT_FULL64_BEAM_OFFSETS = (-2, -1, 0, 1, 2, 3)
_WEIGHT_FULL64_BEAM_KEEP = 4
_WEIGHT_FULL64_CHUNK_ROWS = 1024
_WEIGHT_FULL64_MAX_RATIO = 0.25
# C35 (2026-08-28): per-width full-64 coverage.  Narrow layers (q/k/v/o,
# <=1024 channels) have few 64-blocks and rows, so their refinement is cheap;
# wide FFN projectors (fc/proj, 2048/3072) dominate the calibration time.
# Giving narrow layers a fuller coverage and keeping wide layers conservative
# extracts weight-side precision (cap-oracle: 21pp weight-side gap) without
# breaching the official CPU-time envelope.
_WEIGHT_FULL64_NARROW_CHANNELS = 1024
_WEIGHT_FULL64_MAX_RATIO_NARROW = 1.0
_WEIGHT_FULL64_MAX_RATIO_WIDE = 0.25
_WEIGHT_FULL64_WIDE_ONLY = True
_WEIGHT_FULL64_DAMPINGS = (0.01, 0.03, 0.1)
_WEIGHT_FULL64_SIGNED_CODES = tuple(
    round(code * 0.25, 2) for code in range(-7, 8)
)
# C36 (2026-08-28): skip the second full-H coordinate descent after the
# lv2/lv3 toggle refinement.  The toggle refine already re-solves the
# hierarchy; the trailing sweep mostly re-confirms it.  Turning it off
# cuts the per-block CPU cost by about a third, buying FULL64 coverage
# inside the same official time envelope (A/B measured).
_WEIGHT_FULL64_SECOND_COORDINATE = False
# C44: replace the fixed wide-layer 25% cap with data-driven coverage.  The
# selected set is the smallest set of 64-channel blocks whose *parent* full-H
# loss reaches this fraction; a caller-supplied ratio remains a safety ceiling
# for memory/time, but the production C44 candidate uses 1.0.  The per-block
# diagonal-Hessian order below is the static act-order used by MR-GPTQ.
_WEIGHT_FULL64_DATA_DRIVEN_COVERAGE = False
_WEIGHT_FULL64_TARGET_COVERAGE = 0.97
_WEIGHT_FULL64_DATA_DRIVEN_MAX_RATIO = 1.0
# C45f: static adaptive-headroom candidate.  It reruns the existing
# CAT-coordinate FULL64 solve with a wider E6M2 neighbourhood, but only after
# activation_state has been frozen.  A@W then chooses parent vs headroom as a
# Q(W)-only candidate; no activation coverage or online state is changed.
_WEIGHT_HEADROOM = True
_WEIGHT_HEADROOM_BEAM_OFFSETS = (-4, -3, -2, -1, 0, 1, 2, 3, 4)
_WEIGHT_HEADROOM_MAX_RATIO = 1.0
_WEIGHT_HEADROOM_MIN_GAIN = 1.0e-4
# C39: cross-block conditional refinement.  C38's FULL64 solver optimizes
# each 64-channel diagonal Hessian block independently.  This stage couples
# adjacent 64-channel blocks through the off-diagonal Hessian terms and is
# deliberately separate from the existing solver so it can be ablated and
# audited in isolation.  The final state remains the ordinary HiF4 five-field
# weight representation; no activation/output product is constructed.
# Real-GPT2 screening (2026-08-28) showed that the single pooled calibration
# Hessian improves its own EHE^T proxy but regresses the deployed Linear
# score.  Keep the implementation available for the upcoming multi-fold
# robust variant, but do not ship the unrobust candidate by default.
_WEIGHT_CROSS64 = False
_WEIGHT_CROSS64_SUPERBLOCK = 128
_WEIGHT_CROSS64_SWEEPS = 1
_WEIGHT_CROSS64_MAX_RATIO_NARROW = 1.0
_WEIGHT_CROSS64_MAX_RATIO_WIDE = 0.25
_WEIGHT_CROSS64_NARROW_CHANNELS = 1024
_WEIGHT_CROSS64_DAMPING = 0.03
_WEIGHT_CROSS64_ACCEPT_EPS = 1.0e-9
_WEIGHT_CROSS64_STATS_TOKENS = 1024
# Soft robustness against calibration-window shift.  A value of 0 uses the
# mean fold loss; 0.5 blends the mean and worst calibration fold without
# imposing a hard per-fold rejection rule.
_WEIGHT_CROSS64_ROBUST_MAX_MIX = 0.5
# The activation-side importance in the existing pipeline is derived from
# the pre-C39 weight reconstruction.  Keep that proxy frozen while testing
# the new weight-only stage; otherwise an improved weight Hessian solution
# changes the activation state in the same pass and hides the isolated
# cross-block effect behind a second mechanism.
_WEIGHT_CROSS64_PRESERVE_ACTIVATION_IMPORTANCE = True
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
# C41: quantization-aware K center (mode 4), solved by fixed-point iteration.
# Flagging this off restores the parent behaviour exactly.
_ATTN_SCALE_AWARE_CENTER = True
# C41b: under GQA the KV heads are few (e.g. 2 for qwen2.5-0.5b), so the
# centered second moment is a high-variance estimate and the scale-aware
# center degraded the only GQA model.  Keep the parent behaviour there.
_ATTN_SCALE_AWARE_CENTER_GQA = False
_ATTN_CENTER_ALTERNATIONS = 3
if _ATTN_SCALE_AWARE_CENTER:
    _ATTN_CENTER_MODES = (0, 2, 4)
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

# C34: activation-side 16-channel quadratic refinement (2026-08-28).
# The weight side got full-64 refinement but the dynamic activation path
# only had 8-channel groups; cap-oracle showed the activation encoding is
# one of the two dominant error sources.  A refiner over 16-channel groups
# (adjacent 8-groups) closes part of that gap.  Restricted to narrow layers
# (channels <= 1024) to bound the stored gram16 state on wide FFNs, and to
# a low top-loss ratio to bound the per-sample dynamic cost.
_ACTIVATION_QUADRATIC16 = False  # REJECTED 2026-08-28: -3.43pp real-data
_ACTIVATION_QUADRATIC16_MAX_FEATURES = 1024
_ACTIVATION_QUADRATIC16_MAX_RATIO = 0.10
_ACTIVATION_QUADRATIC16_MAX_GROUPS = 4096
_ACTIVATION_QUADRATIC16_SWEEPS = 1
_ACTIVATION_QUADRATIC16_ACCEPT_MARGIN = 1.0e-5

# C37 (2026-08-28): sample-adaptive activation importance.  The static
# calibration importance (weight-column energy) biases the offset/refine
# selection toward the calibration centroid; for a test sample that differs
# from that centroid the ranking is stale.  Using the current sample's own
# per-channel RMS energy (an activation-only statistic, rule-zero legal)
# adapts the refinement to the sample.  Only used inside the per-sample
# dynamic path, so it cannot leak weight/token supervision.
_ACTIVATION_SAMPLE_IMPORTANCE = False  # REJECTED 2026-08-28: Linear bit-identical, Attention -2.7pp

# Permutation search bases.  The initial hierarchy-aware ordering combines the
# paired operands via max(log range); real-data diagnostics show the operand
# with the larger quantization burden (usually the weight/K side) often yields
# a better single-sided ordering.  Each basis is evaluated with the exact
# paired metric and accepted only when it clears the same safety gate as the
# smoothing candidates.
_PERMUTATION_BASES = True

# C30: Hessian-aware hierarchical permutation.  After the transform search
# selects its best candidate, one extra permutation candidate is built from
# the edge utility  edge(i, j) = |H_A[i, j]| * sqrt(r_i * r_j)  (activation
# Gram combined elementwise with the per-channel weight-quantization residual
# energy -- no cross-operand contraction) by greedy 4->8->16->32->64
# grouping.  REJECTED on real-data evaluation (2026-08-28): ungated direct
# replacement regresses Linear mean 0.5311 -> 0.5198 (-1.13pp, 6/6
# components negative, proj down to negative scores).  The earlier probe
# gains (+54.75%/+55.24%) were a frame artifact: the probes used
# state["smooth_inv"] (= 1/d) as d, inverting the weight-side scale.
_HIERARCHY_PERMUTATION = False

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

    dense8 = dense.reshape(rows, blocks, 8, 8).reshape(-1, 8)
    quantized8 = _dequantize_hif4(params).to(torch.float32).reshape(
        rows, blocks, 8, 8
    ).reshape(-1, 8)
    grams = group_gram8.unsqueeze(0).expand(rows, -1, -1, -1).reshape(
        -1, 8, 8
    )
    error = quantized8 - dense8
    losses = torch.einsum("ni,nij,nj->n", error, grams, error)
    finite = torch.isfinite(losses) & (losses > _EPS)
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
            losses.index_select(0, candidates), k=cap, largest=True
        ).indices
        candidates = candidates.index_select(0, order)

    x_selected = dense8.index_select(0, candidates)
    q_selected = quantized8.index_select(0, candidates).clone()
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
    denominator = (
        (
            scale[..., None]
            * lv2[..., None]
            * lv3.repeat_interleave(4, dim=-1)
        ).reshape(-1, 8)
    ).index_select(0, candidates)
    signed_codes = torch.arange(
        -7, 8, dtype=torch.float32, device=dense.device
    ) * 0.25

    for _ in range(int(sweeps)):
        for coordinate in range(8):
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


def _refine_activation_groups16(
    dense: torch.Tensor,
    params: dict[str, torch.Tensor],
    group_gram16: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """C34: coordinate-refine top-loss 16-channel activation groups.

    Mirrors the weight-side 16-group refiner, driven by the activation
    Hessian logits instead of the weight ones.  Activation refinement runs
    in the per-sample dynamic path, so the ratio is kept low to bound the
    dynamic cost; wide layers (channels > max_features) skip by design.
    """

    rows, channels = map(int, (dense.shape[0], dense.shape[1]))
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
                int(losses.numel()) * _ACTIVATION_QUADRATIC16_MAX_RATIO
            )
        ),
    )
    cap = min(
        cap, _ACTIVATION_QUADRATIC16_MAX_GROUPS, int(candidates.numel())
    )
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

    for _ in range(_ACTIVATION_QUADRATIC16_SWEEPS):
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
        1.0 - _ACTIVATION_QUADRATIC16_ACCEPT_MARGIN
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


def _full64_hessian_blocks(
    cov: torch.Tensor, channels: int
) -> torch.Tensor:
    """Extract the full contiguous 64x64 covariance blocks (C23 6.2).

    Returns ``[channels // 64, 64, 64]``; the full block is kept, never
    truncated to the 4/8/16 diagonal sub-blocks used by the parent
    quadratic refinements.
    """

    indices = torch.arange(channels, device=cov.device).reshape(
        -1, _HIF4_BLOCK_SIZE
    )
    return cov[indices[:, :, None], indices[:, None, :]]


def _cholesky_inverse_factor(
    h: torch.Tensor,
) -> Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Damped batched Cholesky-inverse factor (C23 6.2).

    Args:
        h: ``[blocks, 64, 64]`` block Hessians, already permuted into the
            GPTQ processing order by the caller.

    Returns:
        ``None`` when no block admits a factorization, otherwise a tuple
        ``(factor, h_damped, ok)``:

        - ``factor``: ``[blocks, 64, 64]`` upper-triangular ``U`` with
          ``h_damped^{-1} = U^T U``.  The GPTQ sequential update reads its
          rows directly (plan 6.5: ``error = (adj - q) / U[k, k]`` and
          ``adj[j] -= error * U[k, j]`` for later ``j``).  Blocks that
          fail every damping retry get an identity factor so downstream
          batched math stays finite; their results are rejected through
          the ``ok`` mask.
        - ``h_damped``: the damped Hessians actually factorized; the C23
          solver loss ``(q - w)^T H (q - w)`` must use the same matrix.
        - ``ok``: ``[blocks]`` bool mask of successfully factorized blocks.

    Damping retries ``0.01 / 0.03 / 0.1`` times ``mean(diag)`` as
    registered; blocks still failing after ``0.1`` fall back to the
    parent parameters in the caller.
    """

    num_blocks = int(h.shape[0])
    n = int(h.shape[-1])
    device = h.device
    matrix = h.detach().to(torch.float32)
    eye = torch.eye(n, dtype=torch.float32, device=device)
    diag_mean = matrix.diagonal(dim1=-2, dim2=-1).mean(dim=-1)
    # All finite blocks start as pending: per 6.2 the first attempt always
    # applies the 0.01 damping; 0.03/0.1 are retries on factorization
    # failure.
    ok = torch.isfinite(matrix).all(dim=(-2, -1)) & torch.isfinite(diag_mean)
    ok &= diag_mean > 0.0
    done = torch.zeros_like(ok)
    factor = torch.zeros(
        num_blocks, n, n, dtype=torch.float32, device=device
    )
    h_damped = matrix.clone()
    for damping in _WEIGHT_FULL64_DAMPINGS:
        pending = torch.nonzero(ok & ~done, as_tuple=False).reshape(-1)
        if int(pending.numel()) == 0:
            break
        lam = float(damping) * diag_mean.index_select(0, pending).clamp_min(
            0.0
        )
        trial = matrix.index_select(0, pending) + lam[:, None, None] * eye
        chol, info = torch.linalg.cholesky_ex(trial, check_errors=False)
        solved = info == 0
        if bool(solved.any()):
            solved_idx = pending[solved]
            inverse = torch.cholesky_inverse(chol[solved], upper=False)
            inv_chol, info2 = torch.linalg.cholesky_ex(
                inverse, upper=True, check_errors=False
            )
            accepted = info2 == 0
            if bool(accepted.any()):
                accepted_idx = solved_idx[accepted]
                factor.index_copy_(0, accepted_idx, inv_chol[accepted])
                h_damped.index_copy_(0, accepted_idx, trial[accepted])
                done.index_copy_(
                    0,
                    accepted_idx,
                    torch.ones(
                        int(accepted_idx.numel()),
                        dtype=torch.bool,
                        device=device,
                    ),
                )
    if not bool((ok & done).any()):
        return None
    ok = ok & done
    factor = torch.where(ok[:, None, None], factor, eye)
    return factor, h_damped, ok


def _gptq_initialize64(
    w: torch.Tensor,
    denom: torch.Tensor,
    order: torch.Tensor,
    factor: torch.Tensor,
) -> torch.Tensor:
    """Batched GPTQ sequential initialization (C23 6.4 step 4 / 6.5).

    Args:
        w: ``[pairs, blocks, 64]`` dense targets, original coordinates.
        denom: ``[pairs, blocks, 64]`` legal-grid denominators (scale x
            lv2 x lv3 per channel), original coordinates.
        order: ``[blocks, 64]`` int64 processing order (descending
            ``diag(H)``).
        factor: ``[blocks, 64, 64]`` upper Cholesky factor of the damped
            Hessian inverse, in processing coordinates.

    Returns:
        ``[pairs, blocks, 64]`` quantized values in original coordinates,
        each exactly on the legal ``denom * signed code`` grid.  The
        processing order only affects the solve; output coordinates are
        the input coordinates.
    """

    pairs = int(w.shape[0])
    n = int(w.shape[-1])
    index = order.unsqueeze(0).expand(pairs, -1, -1)
    w_perm = torch.gather(w, 2, index)
    denom_perm = torch.gather(denom, 2, index)
    adjusted = w_perm.clone()
    q_perm = torch.zeros_like(w_perm)
    diag = factor.diagonal(dim1=-2, dim2=-1)
    for k in range(n):
        u_row = factor[:, k, :]
        u_diag = diag[:, k].clamp_min(_EPS)
        adj = adjusted[:, :, k]
        den = denom_perm[:, :, k]
        code = torch.round(adj * (4.0 / den)).clamp_(-7.0, 7.0)
        q_k = code * 0.25 * den
        q_perm[:, :, k] = q_k
        error = (adj - q_k) / u_diag
        if k + 1 < n:
            adjusted[:, :, k + 1 :] -= error.unsqueeze(-1) * u_row[
                None, :, k + 1 :
            ]
    q = torch.zeros_like(w)
    q.scatter_(2, index, q_perm)
    return q


def _coordinate_descent64(
    q: torch.Tensor,
    w: torch.Tensor,
    h: torch.Tensor,
    denom: torch.Tensor,
) -> torch.Tensor:
    """One batched full-H coordinate-descent sweep (C23 6.6).

    Maintains ``g = H @ e`` incrementally.  For each of the 64 coordinates
    the 15 legal signed codes are enumerated, the exact quadratic loss
    change ``2*delta*g_i + delta^2*H_ii`` is evaluated, and the best
    strictly-improving move is applied.  The pairs/blocks dimensions stay
    fully batched; only the coordinate dimension loops.
    """

    device = q.device
    codes = torch.tensor(
        _WEIGHT_FULL64_SIGNED_CODES, dtype=torch.float32, device=device
    )
    diag = h.diagonal(dim1=-2, dim2=-1).clamp_min(_EPS)
    q = q.clone()
    g = torch.einsum("bij,pbj->pbi", h, q - w)
    for i in range(64):
        cur = q[:, :, i]
        cand = denom[:, :, i, None] * codes
        delta = cand - cur[:, :, None]
        change = delta * (
            g[:, :, i, None]
            + g[:, :, i, None]
            + delta * diag[None, :, i, None]
        )
        best_change, best = change.min(dim=-1)
        improve = best_change < -_EPS
        if not bool(improve.any()):
            continue
        step = delta.gather(-1, best.unsqueeze(-1)).squeeze(-1)
        step = step * improve.to(step.dtype)
        q[:, :, i] = cur + step
        g += step[:, :, None] * h[None, :, i, :]
    return q


def _hierarchy_toggle_refine64(
    q: torch.Tensor,
    w: torch.Tensor,
    h: torch.Tensor,
    denom: torch.Tensor,
    lv2: torch.Tensor,
    lv3: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched lv2/lv3 hierarchy bit-toggle refinement (C23 6.4 steps 6-7).

    Enumerates the 16 lv3 bits then the 8 lv2 bits.  A toggle rescales the
    decoded values of its channels by ``new/old`` (codes stay legal), the
    exact full-H loss change is evaluated from the maintained gradient,
    and accepted toggles update ``q``/``denom``/``lv2``/``lv3`` through
    boolean-mask batch updates.  No per-block branching.
    """

    q = q.clone()
    denom = denom.clone()
    lv2 = lv2.clone()
    lv3 = lv3.clone()
    g = torch.einsum("bij,pbj->pbi", h, q - w)

    def _apply_toggle(c0: int, span: int, ratio: torch.Tensor) -> torch.Tensor:
        delta = q[:, :, c0 : c0 + span] * (ratio - 1.0)[:, :, None]
        sub = h[:, c0 : c0 + span, c0 : c0 + span]
        change = 2.0 * (delta * g[:, :, c0 : c0 + span]).sum(dim=-1)
        change = change + torch.einsum("pbc,bcd,pbd->pb", delta, sub, delta)
        accept = torch.isfinite(change) & (change < -_EPS)
        delta = delta * accept.to(delta.dtype)[:, :, None]
        q[:, :, c0 : c0 + span] += delta
        g.add_(torch.einsum(
            "bcd,pbc->pbd", h[:, c0 : c0 + span, :], delta
        ))
        ones = torch.ones_like(ratio).unsqueeze(-1)
        denom[:, :, c0 : c0 + span] *= torch.where(
            accept[:, :, None], ratio.unsqueeze(-1), ones
        )
        return accept

    for group in range(8):
        for half in range(2):
            old = lv3[:, :, group, half]
            ratio = (3.0 - old) / old
            accept = _apply_toggle(group * 8 + half * 4, 4, ratio)
            lv3[:, :, group, half] = torch.where(accept, 3.0 - old, old)
    for group in range(8):
        old = lv2[:, :, group]
        ratio = (3.0 - old) / old
        accept = _apply_toggle(group * 8, 8, ratio)
        lv2[:, :, group] = torch.where(accept, 3.0 - old, old)
    return q, lv2, lv3, denom


def _refine_weight_blocks64(
    dense: torch.Tensor,
    params: dict[str, torch.Tensor],
    cov: torch.Tensor,
    max_ratio: float = _WEIGHT_FULL64_MAX_RATIO,
    beam_offsets: Optional[Sequence[int]] = None,
) -> dict[str, torch.Tensor]:
    """C23 full-64 weight refinement (plan section 6).

    Per row chunk, the top ``max_ratio`` block columns by
    parent full-H loss enter a ``_WEIGHT_FULL64_BEAM_KEEP``-way scale beam
    (codes ``standard_code + _WEIGHT_FULL64_BEAM_OFFSETS`` ranked by the
    exact-hierarchy initialization loss under the full damped Hessian).
    The four beam slots are folded into the batched pair dimension, so
    GPTQ initialization, the two full-H coordinate descents and the
    lv2/lv3 toggle refinement each run as one batched solve.  A beam
    replaces the parent parameters of a block only when its final
    full-H loss is finite and strictly lower than the parent's (and the
    block's Hessian factorized); unselected columns, non-improving
    blocks and failed factorizations keep the parent parameters, which
    is exactly the registered per-block fallback semantics.
    """

    if dense.ndim != 2:
        return params
    rows, channels = map(int, dense.shape)
    if rows <= 0 or channels % _HIF4_BLOCK_SIZE != 0:
        return params
    if tuple(cov.shape) != (channels, channels):
        return params
    blocks = channels // _HIF4_BLOCK_SIZE
    device = dense.device

    h_blocks = _full64_hessian_blocks(cov, channels)
    max_ratio = float(max_ratio)
    w_all = dense.detach().to(torch.float32).reshape(rows, blocks, 64)
    q_parent = _dequantize_hif4(params).to(torch.float32).reshape(
        rows, blocks, 64
    )
    parent_scale = (
        params["scale_factor"].detach().to(torch.float32).reshape(rows, blocks)
    )
    parent_lv2 = (
        params["scale_lv2"].detach().to(torch.float32).reshape(rows, blocks, 8)
    )
    parent_lv3 = (
        params["scale_lv3"]
        .detach()
        .to(torch.float32)
        .reshape(rows, blocks, 8, 2)
    )
    out_scale = parent_scale.clone()
    out_lv2 = parent_lv2.clone()
    out_lv3 = parent_lv3.clone()
    out_sign = (
        params["sign"].detach().to(torch.float32).reshape(rows, blocks, 64)
    ).clone()
    out_mant = (
        params["mant"].detach().to(torch.float32).reshape(rows, blocks, 64)
    ).clone()

    chunk = max(1, int(_WEIGHT_FULL64_CHUNK_ROWS))
    cap_cols = min(
        blocks,
        max(1, int(math.ceil(blocks * max_ratio))),
    )
    selected_offsets = (
        _WEIGHT_FULL64_BEAM_OFFSETS
        if beam_offsets is None
        else tuple(int(value) for value in beam_offsets)
    )
    if not selected_offsets:
        return params
    offsets = torch.tensor(
        selected_offsets, dtype=torch.int64, device=device
    )
    num_offsets = int(offsets.numel())
    beam_keep = min(int(_WEIGHT_FULL64_BEAM_KEEP), num_offsets)

    # Global column selection: the top ``cap_cols`` block columns by parent
    # full-H loss aggregated over all rows. Selection is chunk independent,
    # so the row chunking below is pure batching and exactly reproducible
    # for any chunk size (plan 6.8 chunking exactness test).
    e_all = q_parent - w_all
    col_loss = torch.einsum("rbi,bij,rbj->rb", e_all, h_blocks, e_all)
    col_agg = col_loss.sum(dim=0)
    col_agg = torch.where(
        torch.isfinite(col_agg),
        col_agg,
        torch.full_like(col_agg, -torch.inf),
    )
    if not bool((col_agg > _EPS).any()):
        return params
    if _WEIGHT_FULL64_DATA_DRIVEN_COVERAGE:
        # Select the smallest high-loss prefix reaching the registered
        # coverage target.  ``max_ratio`` remains a caller-visible ceiling,
        # so a future runtime budget can lower it without changing the
        # ranking rule or the act-order/GPTQ solve itself.
        finite_loss = torch.where(
            torch.isfinite(col_agg), col_agg.clamp_min(0.0), torch.zeros_like(col_agg)
        )
        total_loss = finite_loss.sum()
        if bool(torch.isfinite(total_loss)) and float(total_loss) > _EPS:
            ranked = torch.argsort(finite_loss, descending=True, stable=True)
            cumulative = finite_loss.index_select(0, ranked).cumsum(dim=0)
            target = cumulative[-1] * max(
                0.0, min(float(_WEIGHT_FULL64_TARGET_COVERAGE), 1.0)
            )
            needed = int(torch.searchsorted(cumulative, target).item()) + 1
            cap_cols = min(cap_cols, max(1, needed))
    if blocks > cap_cols:
        sel_cols = torch.sort(
            torch.topk(col_agg, k=cap_cols, largest=True).indices
        ).values
    else:
        sel_cols = torch.arange(blocks, device=device)
    num_sel = int(sel_cols.numel())

    h_sel = h_blocks.index_select(0, sel_cols)
    order = torch.argsort(
        -h_sel.diagonal(dim1=-2, dim2=-1), dim=1, stable=True
    )
    block_index = torch.arange(num_sel, device=device).view(num_sel, 1, 1)
    h_perm = h_sel[block_index, order[:, :, None], order[:, None, :]]
    chol_result = _cholesky_inverse_factor(h_perm)
    if chol_result is None:
        return params
    factor, h_damped_perm, ok_cols = chol_result
    if not bool(ok_cols.any()):
        return params
    inverse_order = torch.argsort(order, dim=1)
    h_damped = h_damped_perm[
        block_index,
        inverse_order[:, :, None],
        inverse_order[:, None, :],
    ]
    h_damped = torch.where(ok_cols[:, None, None], h_damped, h_sel)

    for row0 in range(0, rows, chunk):
        row1 = min(row0 + chunk, rows)
        w_sel = w_all[row0:row1].index_select(1, sel_cols)
        qp_sel = q_parent[row0:row1].index_select(1, sel_cols)
        scale_sel = parent_scale[row0:row1].index_select(1, sel_cols)
        lv2_sel = parent_lv2[row0:row1].index_select(1, sel_cols)
        lv3_sel = parent_lv3[row0:row1].index_select(1, sel_cols)
        num_rows = int(w_sel.shape[0])

        e_sel = qp_sel - w_sel
        parent_loss = torch.einsum(
            "rbi,bij,rbj->rb", e_sel, h_damped, e_sel
        )
        parent_loss = torch.where(
            torch.isfinite(parent_loss),
            parent_loss,
            torch.full_like(parent_loss, torch.inf),
        )

        amax = w_sel.abs().amax(dim=-1)
        std_code, _ = _standard_e6m2_scale(amax)
        cand_codes = (
            std_code.to(torch.int64).unsqueeze(0) + offsets.view(-1, 1, 1)
        ).clamp(min=0, max=254)
        cand_scales = _e6m2_decode(cand_codes)

        x_abs = w_sel.abs().reshape(num_rows, num_sel, 8, 2, 4)
        x_exp = x_abs.unsqueeze(0).expand(
            num_offsets, num_rows, num_sel, 8, 2, 4
        )
        _init_loss, init_lv2, init_lv3, init_mant = _solve_exact_hierarchy(
            x_exp, cand_scales, None, None, None
        )
        init_denom = (
            cand_scales[..., None, None, None]
            * init_lv2[..., None, None]
            * init_lv3[..., None]
        )
        sign_sel = torch.sign(w_sel).reshape(num_rows, num_sel, 8, 2, 4)
        q_init = (
            sign_sel.unsqueeze(0) * init_mant * init_denom
        ).flatten(start_dim=-3)
        e_init = q_init - w_sel.unsqueeze(0)
        rank_loss = torch.einsum(
            "krbi,bij,krbj->krb", e_init, h_damped, e_init
        )
        rank_loss = torch.where(
            torch.isfinite(rank_loss),
            rank_loss,
            torch.full_like(rank_loss, torch.inf),
        )
        beam_idx = torch.topk(
            rank_loss, k=beam_keep, dim=0, largest=False
        ).indices

        codes_b = torch.gather(cand_codes, 0, beam_idx)
        scale_b = _e6m2_decode(codes_b)
        lv2_b = torch.gather(
            init_lv2, 0, beam_idx.unsqueeze(-1).expand(-1, -1, -1, 8)
        )
        lv3_b = torch.gather(
            init_lv3,
            0,
            beam_idx.unsqueeze(-1)
            .unsqueeze(-1)
            .expand(-1, -1, -1, 8, 2),
        )

        # Fold the beam slots into the pair dimension so every solve stage
        # is a single batched pass (memory stays bounded by the chunk).
        pairs = beam_keep * num_rows
        w_pairs = (
            w_sel.unsqueeze(0)
            .expand(beam_keep, num_rows, num_sel, 64)
            .reshape(pairs, num_sel, 64)
        )
        scale_p = scale_b.reshape(pairs, num_sel)
        lv2_p = lv2_b.reshape(pairs, num_sel, 8)
        lv3_p = lv3_b.reshape(pairs, num_sel, 8, 2)
        denom_p = (
            scale_p[:, :, None, None, None]
            * lv2_p[:, :, :, None, None]
            * lv3_p[:, :, :, :, None]
        ).repeat_interleave(4, dim=-1)
        denom_p = denom_p.reshape(pairs, num_sel, 64)

        q_pairs = _gptq_initialize64(w_pairs, denom_p, order, factor)
        q_pairs = _coordinate_descent64(q_pairs, w_pairs, h_damped, denom_p)
        q_pairs, lv2_p, lv3_p, denom_p = _hierarchy_toggle_refine64(
            q_pairs, w_pairs, h_damped, denom_p, lv2_p, lv3_p
        )
        if _WEIGHT_FULL64_SECOND_COORDINATE:
            q_pairs = _coordinate_descent64(
                q_pairs, w_pairs, h_damped, denom_p
            )

        final_codes = torch.round(
            q_pairs * (4.0 / denom_p.clamp_min(_EPS))
        ).clamp_(-7.0, 7.0)
        e_final = q_pairs - w_pairs
        final_loss = torch.einsum(
            "pbi,bij,pbj->pb", e_final, h_damped, e_final
        ).reshape(beam_keep, num_rows, num_sel)
        final_loss = torch.where(
            torch.isfinite(final_loss),
            final_loss,
            torch.full_like(final_loss, torch.inf),
        )
        # Per (row, block): keep the best beam slot only when it strictly
        # beats the parent loss and the block's Hessian factorized.
        improve = (final_loss < parent_loss[None]) & ok_cols[None, None, :]
        masked = torch.where(
            improve, final_loss, torch.full_like(final_loss, torch.inf)
        )
        best_beam_loss, best_beam_idx = masked.min(dim=0)
        any_improve = best_beam_loss < parent_loss
        if not bool(any_improve.any()):
            continue

        codes_brc = final_codes.reshape(beam_keep, num_rows, num_sel, 64)
        idx_code = (
            best_beam_idx.unsqueeze(0)
            .unsqueeze(-1)
            .expand(1, num_rows, num_sel, 64)
        )
        best_codes_sel = torch.gather(codes_brc, 0, idx_code).squeeze(0)
        idx_lv2 = (
            best_beam_idx.unsqueeze(0)
            .unsqueeze(-1)
            .expand(1, num_rows, num_sel, 8)
        )
        best_lv2_sel = torch.gather(
            lv2_p.reshape(beam_keep, num_rows, num_sel, 8), 0, idx_lv2
        ).squeeze(0)
        idx_lv3 = (
            best_beam_idx.unsqueeze(0)
            .unsqueeze(-1)
            .unsqueeze(-1)
            .expand(1, num_rows, num_sel, 8, 2)
        )
        best_lv3_sel = torch.gather(
            lv3_p.reshape(beam_keep, num_rows, num_sel, 8, 2), 0, idx_lv3
        ).squeeze(0)
        idx_scale = (
            best_beam_idx.unsqueeze(0).expand(1, num_rows, num_sel)
        )
        best_scale_sel = torch.gather(scale_b, 0, idx_scale).squeeze(0)

        parent_sign_sel = out_sign[row0:row1].index_select(1, sel_cols)
        parent_mant_sel = out_mant[row0:row1].index_select(1, sel_cols)
        full_scale = out_scale[row0:row1].clone()
        full_lv2 = out_lv2[row0:row1].clone()
        full_lv3 = out_lv3[row0:row1].clone()
        full_sign = out_sign[row0:row1].clone()
        full_mant = out_mant[row0:row1].clone()
        full_scale[:, sel_cols] = torch.where(
            any_improve, best_scale_sel, scale_sel
        )
        full_lv2[:, sel_cols] = torch.where(
            any_improve[..., None], best_lv2_sel, lv2_sel
        )
        full_lv3[:, sel_cols] = torch.where(
            any_improve[..., None, None], best_lv3_sel, lv3_sel
        )
        full_sign[:, sel_cols] = torch.where(
            any_improve[..., None],
            torch.sign(best_codes_sel),
            parent_sign_sel,
        )
        full_mant[:, sel_cols] = torch.where(
            any_improve[..., None],
            best_codes_sel.abs() * 0.25,
            parent_mant_sel,
        )
        out_scale[row0:row1] = full_scale
        out_lv2[row0:row1] = full_lv2
        out_lv3[row0:row1] = full_lv3
        out_sign[row0:row1] = full_sign
        out_mant[row0:row1] = full_mant

    refined = dict(params)
    refined["scale_factor"] = out_scale.reshape(rows, blocks, 1, 1, 1)
    refined["scale_lv2"] = out_lv2.reshape(rows, blocks, 8, 1, 1)
    refined["scale_lv3"] = out_lv3.reshape(rows, blocks, 8, 2, 1)
    refined["sign"] = out_sign.reshape(rows, blocks, 8, 2, 4)
    refined["mant"] = out_mant.reshape(rows, blocks, 8, 2, 4)
    return refined


def _slice_hif4_weight_params(
    params: dict[str, torch.Tensor], block_start: int, block_count: int
) -> dict[str, torch.Tensor]:
    """Slice a contiguous group of 64-channel weight blocks."""

    start = int(block_start)
    end = start + int(block_count)
    sliced: dict[str, torch.Tensor] = {}
    for key, value in params.items():
        if torch.is_tensor(value) and value.ndim >= 2:
            sliced[key] = value[:, start:end].clone()
        else:
            sliced[key] = value
    return sliced


def _replace_hif4_weight_params(
    params: dict[str, torch.Tensor],
    replacement: dict[str, torch.Tensor],
    block_start: int,
    block_count: int,
) -> dict[str, torch.Tensor]:
    """Return ``params`` with a contiguous block slice replaced."""

    start = int(block_start)
    end = start + int(block_count)
    merged: dict[str, torch.Tensor] = {}
    for key, value in params.items():
        if torch.is_tensor(value):
            merged[key] = value.clone()
            if key in replacement and value.ndim >= 2:
                merged[key][:, start:end] = replacement[key].to(
                    device=value.device, dtype=value.dtype
                )
        else:
            merged[key] = value
    return merged


def _damped_hessian_inverse(
    hessian: torch.Tensor,
) -> Optional[torch.Tensor]:
    """Build a numerically safe inverse for one 64x64 Hessian block."""

    if hessian.ndim != 2 or tuple(hessian.shape) != (64, 64):
        return None
    h = hessian.detach().to(torch.float32)
    if not bool(torch.isfinite(h).all()):
        return None
    h = 0.5 * (h + h.t())
    diagonal = torch.diagonal(h)
    mean_diagonal = diagonal.mean().clamp_min(_EPS)
    eye = torch.eye(64, dtype=h.dtype, device=h.device)
    for damping in (
        float(_WEIGHT_CROSS64_DAMPING),
        0.1,
        0.3,
    ):
        damped = h + (damping * mean_diagonal) * eye
        chol, info = torch.linalg.cholesky_ex(damped)
        if int(info.item()) == 0 and bool(torch.isfinite(chol).all()):
            inverse = torch.cholesky_inverse(chol)
            if bool(torch.isfinite(inverse).all()):
                return inverse
    return None


def _hif4_weight_block_denominator(
    params: dict[str, torch.Tensor], block_index: int
) -> Optional[torch.Tensor]:
    """Return the current legal 64-channel grid for one weight block."""

    required = ("scale_factor", "scale_lv2", "scale_lv3")
    if any(key not in params or not torch.is_tensor(params[key]) for key in required):
        return None
    scale = params["scale_factor"][:, int(block_index)].reshape(-1, 1)
    lv2 = params["scale_lv2"][:, int(block_index)].reshape(-1, 8)
    lv3 = params["scale_lv3"][:, int(block_index)].reshape(-1, 8, 2)
    denominator = (
        scale[:, :, None, None]
        * lv2[:, :, None, None]
        * lv3[:, :, :, None]
    ).repeat_interleave(4, dim=-1)
    return denominator.reshape(-1, 1, 64).to(torch.float32)


def _cross64_quadratic_loss(
    quantized: torch.Tensor,
    dense: torch.Tensor,
    hessian: torch.Tensor,
) -> torch.Tensor:
    """Return the complete Hessian loss for a block or superblock."""

    error = quantized.to(torch.float32) - dense.to(torch.float32)
    h = hessian.to(device=error.device, dtype=torch.float32)
    return torch.einsum("ri,ij,rj->", error, h, error)


def _validate_cross64_fold_covariances(
    fold_pair_covariances: Optional[torch.Tensor],
    blocks: int,
    channels: int,
) -> Optional[torch.Tensor]:
    """Validate optional per-fold 128-channel covariance blocks."""

    if fold_pair_covariances is None:
        return None
    if not torch.is_tensor(fold_pair_covariances):
        return None
    expected_pairs = int(channels) // (2 * _HIF4_BLOCK_SIZE)
    if fold_pair_covariances.ndim != 4:
        return None
    if tuple(fold_pair_covariances.shape[1:]) != (
        expected_pairs,
        2 * _HIF4_BLOCK_SIZE,
        2 * _HIF4_BLOCK_SIZE,
    ):
        return None
    if int(blocks) != int(channels) // _HIF4_BLOCK_SIZE:
        return None
    if int(fold_pair_covariances.shape[0]) <= 0:
        return None
    if not bool(torch.isfinite(fold_pair_covariances).all()):
        return None
    return fold_pair_covariances.to(dtype=torch.float32)


def _cross64_fold_losses(
    quantized: torch.Tensor,
    dense: torch.Tensor,
    fold_h: Optional[torch.Tensor],
    pooled_h: torch.Tensor,
) -> torch.Tensor:
    """Evaluate a superblock under pooled or per-fold Hessians."""

    error = quantized.to(torch.float32) - dense.to(torch.float32)
    if fold_h is None:
        return torch.einsum("ri,ij,rj->", error, pooled_h, error).reshape(1)
    return torch.einsum(
        "ri,fij,rj->f", error, fold_h.to(error.device), error
    )


def _cross64_robust_loss(fold_losses: torch.Tensor) -> torch.Tensor:
    """Use a soft mean/worst-fold objective, never a hard fold gate."""

    losses = fold_losses.to(torch.float32).reshape(-1)
    if int(losses.numel()) == 0:
        return torch.tensor(float("inf"), dtype=torch.float32)
    mean_loss = losses.mean()
    worst_loss = losses.max()
    mix = max(0.0, min(float(_WEIGHT_CROSS64_ROBUST_MAX_MIX), 1.0))
    return mean_loss + mix * (worst_loss - mean_loss)


@torch.no_grad()
def _refine_weight_blocks_cross64(
    dense: torch.Tensor,
    params: dict[str, torch.Tensor],
    cov: torch.Tensor,
    fold_pair_covariances: Optional[torch.Tensor] = None,
) -> dict[str, torch.Tensor]:
    """C39 cross-block conditional refinement for static Linear weights.

    The existing FULL64 stage solves each 64-channel block independently.
    This stage performs block-coordinate minimization of the complete
    quadratic objective over adjacent 64-channel blocks.  For a current block
    ``b`` and fixed errors in the other blocks, the cross term is completed
    into a shifted local target:

        target_b = W_b - (sum_j E_j H_jb) H_bb^{-1}

    A fixed-hierarchy legal-grid coordinate solver then quantizes that target.
    A candidate is committed only when the original complete superblock loss
    decreases.  Keeping the hierarchy fixed in this first pass avoids
    repeating the expensive beam/GPTQ search for every conditional update;
    DHSS can later provide a separate scale candidate stage.  The routine
    uses only the activation covariance for the static weight objective and
    never constructs ``A @ W`` or any output residual.  It deliberately keeps
    the final representation unchanged.
    """

    if not _WEIGHT_CROSS64:
        return params
    if dense.ndim != 2 or cov.ndim != 2:
        return params
    rows, channels = map(int, dense.shape)
    if rows <= 0 or channels < 128 or channels % _HIF4_BLOCK_SIZE != 0:
        return params
    if tuple(cov.shape) != (channels, channels):
        return params
    if not bool(torch.isfinite(dense).all()) or not bool(torch.isfinite(cov).all()):
        return params

    blocks = channels // _HIF4_BLOCK_SIZE
    superblock = int(_WEIGHT_CROSS64_SUPERBLOCK)
    if superblock < 2 * _HIF4_BLOCK_SIZE or superblock % _HIF4_BLOCK_SIZE != 0:
        return params
    blocks_per_super = superblock // _HIF4_BLOCK_SIZE
    if blocks_per_super != 2:
        # The first implementation intentionally has a small, exactly
        # auditable conditional problem.  Larger groups can be added later
        # with the same complete-loss acceptance rule.
        return params
    fold_pair_covariances = _validate_cross64_fold_covariances(
        fold_pair_covariances,
        blocks,
        channels,
    )

    current_params = {
        key: value.clone() if torch.is_tensor(value) else value
        for key, value in params.items()
    }
    current_quantized = _dequantize_hif4(current_params).to(torch.float32)

    pair_starts = list(range(0, blocks - 1, 2))
    if not pair_starts:
        return params

    # Prioritize pairs whose currently discarded cross term is largest.  The
    # ratio controls compute on wide layers, while all narrow-layer pairs are
    # considered.  This is a deterministic budget allocation, not a score
    # gate; every selected pair still has a mathematical full-loss fallback.
    pair_scores: list[tuple[float, int]] = []
    for block_start in pair_starts:
        channel_start = block_start * _HIF4_BLOCK_SIZE
        channel_end = channel_start + 2 * _HIF4_BLOCK_SIZE
        h_pair = cov[channel_start:channel_end, channel_start:channel_end]
        e_pair = current_quantized[:, channel_start:channel_end] - dense[
            :, channel_start:channel_end
        ]
        e0 = e_pair[:, :_HIF4_BLOCK_SIZE]
        e1 = e_pair[:, _HIF4_BLOCK_SIZE:]
        h01 = h_pair[:_HIF4_BLOCK_SIZE, _HIF4_BLOCK_SIZE:]
        cross = 2.0 * torch.einsum("ri,ij,rj->", e0, h01, e1)
        score = abs(float(torch.nan_to_num(cross, nan=0.0).item()))
        pair_scores.append((score, block_start))
    max_ratio = (
        _WEIGHT_CROSS64_MAX_RATIO_NARROW
        if channels <= _WEIGHT_CROSS64_NARROW_CHANNELS
        else _WEIGHT_CROSS64_MAX_RATIO_WIDE
    )
    max_pairs = min(
        len(pair_starts),
        max(1, int(math.ceil(len(pair_starts) * float(max_ratio)))),
    )
    pair_scores.sort(key=lambda item: (-item[0], item[1]))
    selected_starts = {
        start for _score, start in pair_scores[:max_pairs]
    }

    for block_start in sorted(selected_starts):
        channel_start = block_start * _HIF4_BLOCK_SIZE
        channel_end = channel_start + 2 * _HIF4_BLOCK_SIZE
        pair_dense = dense[:, channel_start:channel_end]
        pair_h = 0.5 * cov[channel_start:channel_end, channel_start:channel_end] + (
            0.5 * cov[channel_start:channel_end, channel_start:channel_end].t()
        )
        pair_index = block_start // 2
        fold_h = None
        if fold_pair_covariances is not None:
            fold_h = fold_pair_covariances[:, pair_index]
            pair_h = fold_h.mean(dim=0)
            pair_h = 0.5 * (pair_h + pair_h.t())
        pair_quantized = current_quantized[:, channel_start:channel_end].clone()
        pair_fold_losses = _cross64_fold_losses(
            pair_quantized, pair_dense, fold_h, pair_h
        )
        pair_loss = _cross64_robust_loss(pair_fold_losses)
        if not bool(torch.isfinite(pair_loss)):
            continue

        local_inverses: list[Optional[torch.Tensor]] = []
        for local_block in range(2):
            lo = local_block * _HIF4_BLOCK_SIZE
            hi = lo + _HIF4_BLOCK_SIZE
            local_inverses.append(
                _damped_hessian_inverse(pair_h[lo:hi, lo:hi])
            )
        if any(inverse is None for inverse in local_inverses):
            continue

        for _ in range(max(1, int(_WEIGHT_CROSS64_SWEEPS))):
            changed = False
            for local_block in range(2):
                lo = local_block * _HIF4_BLOCK_SIZE
                hi = lo + _HIF4_BLOCK_SIZE
                other = 1 - local_block
                olo = other * _HIF4_BLOCK_SIZE
                ohi = olo + _HIF4_BLOCK_SIZE
                hbb = pair_h[lo:hi, lo:hi]
                h_other_b = pair_h[olo:ohi, lo:hi]
                other_error = pair_quantized[:, olo:ohi] - pair_dense[:, olo:ohi]
                cross = other_error @ h_other_b
                shift = cross @ local_inverses[local_block]
                target = pair_dense[:, lo:hi] - shift

                global_block = block_start + local_block
                denominator = _hif4_weight_block_denominator(
                    current_params, global_block
                )
                if denominator is None or not bool(
                    torch.isfinite(denominator).all()
                ):
                    continue
                target_3d = target.unsqueeze(1)
                candidate_init = (
                    torch.round(
                        target_3d * (4.0 / denominator.clamp_min(_EPS))
                    ).clamp_(-7.0, 7.0)
                    * 0.25
                    * denominator
                )
                candidate_3d = _coordinate_descent64(
                    candidate_init,
                    target_3d,
                    hbb.unsqueeze(0),
                    denominator,
                )
                candidate_quantized = candidate_3d[:, 0, :].to(torch.float32)
                trial_quantized = pair_quantized.clone()
                trial_quantized[:, lo:hi] = candidate_quantized
                trial_loss = _cross64_quadratic_loss(
                    trial_quantized, pair_dense, pair_h
                )
                trial_fold_losses = _cross64_fold_losses(
                    trial_quantized, pair_dense, fold_h, pair_h
                )
                trial_robust_loss = _cross64_robust_loss(trial_fold_losses)
                if not bool(torch.isfinite(trial_loss)) or not bool(
                    torch.isfinite(trial_robust_loss)
                ):
                    continue
                if float(trial_robust_loss.item()) >= float(pair_loss.item()) - float(_WEIGHT_CROSS64_ACCEPT_EPS):
                    continue

                candidate_block = _slice_hif4_weight_params(
                    current_params, global_block, 1
                )
                candidate_codes = torch.round(
                    candidate_quantized * (4.0 / denominator[:, 0, :].clamp_min(_EPS))
                ).clamp_(-7.0, 7.0)
                candidate_block["sign"] = torch.sign(candidate_codes).reshape(
                    candidate_block["sign"].shape
                )
                candidate_block["mant"] = (
                    candidate_codes.abs() * 0.25
                ).reshape(candidate_block["mant"].shape)
                current_params = _replace_hif4_weight_params(
                    current_params,
                    candidate_block,
                    global_block,
                    1,
                )
                pair_quantized = trial_quantized
                pair_fold_losses = trial_fold_losses
                pair_loss = trial_robust_loss
                changed = True

            if not changed:
                break
        current_quantized[:, channel_start:channel_end] = pair_quantized

    return current_params


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


class _ChannelGroups:
    """Union-find over channel indices with explicit member lists."""

    def __init__(self, count: int) -> None:
        self.parent = list(range(count))
        self.members: list[list[int]] = [[i] for i in range(count)]

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


def _hierarchy_edge_groups(utility: torch.Tensor, cap: int) -> list[list[int]]:
    """Seed channel groups by greedily joining the highest-utility edges."""

    channels = int(utility.shape[0])
    tri = torch.triu_indices(channels, channels, offset=1)
    values = utility[tri[0], tri[1]]
    order = torch.argsort(values, descending=True, stable=True)
    limit = min(int(order.numel()), 16 * channels)
    top = order[:limit].cpu().tolist()
    rows = tri[0].cpu().tolist()
    cols = tri[1].cpu().tolist()

    uf = _ChannelGroups(channels)
    for idx in top:
        i, j = rows[idx], cols[idx]
        ri, rj = uf.find(i), uf.find(j)
        if ri == rj:
            continue
        if len(uf.members[ri]) + len(uf.members[rj]) > cap:
            continue
        uf.union(ri, rj)

    groups = [m for m in uf.members if m]
    # Complete undersized groups: repeatedly merge the smallest group with
    # the smallest partner that fits the cap; groups that fit no partner
    # are left to merge at the next level up.
    while len(groups) > 1:
        groups.sort(key=lambda g: (len(g), min(g)))
        head = groups[0]
        if len(head) >= cap:
            break
        partner = None
        for idx in range(1, len(groups)):
            if len(head) + len(groups[idx]) <= cap:
                partner = idx
                break
        if partner is None:
            break
        tail = groups.pop(partner)
        groups[0] = sorted(head + tail)
    groups.sort(key=lambda g: min(g))
    return [sorted(g) for g in groups]


def _hierarchy_group_aggregate(
    utility: torch.Tensor, groups: list[list[int]]
) -> torch.Tensor:
    """Inter-group utility: entry [a, b] sums utility between groups a/b."""

    channels = int(utility.shape[0])
    indicator = torch.zeros(
        channels, len(groups), dtype=utility.dtype, device=utility.device
    )
    for gi, group in enumerate(groups):
        for ch in group:
            indicator[ch, gi] = 1.0
    return indicator.t() @ (utility @ indicator)


def _hierarchy_merge_groups(
    utility: torch.Tensor, groups: list[list[int]], cap: int
) -> list[list[int]]:
    """Merge groups pairwise by aggregated inter-group utility."""

    if len(groups) <= 1:
        return [list(g) for g in groups]
    agg = _hierarchy_group_aggregate(utility, groups)
    gsize = len(groups)
    tri = torch.triu_indices(gsize, gsize, offset=1)
    values = agg[tri[0], tri[1]]
    order = torch.argsort(values, descending=True, stable=True).cpu().tolist()
    rows = tri[0].cpu().tolist()
    cols = tri[1].cpu().tolist()

    uf = _ChannelGroups(gsize)
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


def _hierarchy_edge_permutation(utility: torch.Tensor) -> torch.Tensor:
    """Deterministic hierarchical 4->8->16->32->64 grouping permutation."""

    channels = int(utility.shape[0])
    groups = _hierarchy_edge_groups(utility, cap=4)
    for cap in (8, 16, 32, 64):
        if len(groups) <= 1:
            break
        groups = _hierarchy_merge_groups(utility, groups, cap)
    order: list[int] = []
    for group in groups:
        order.extend(group)
    if len(order) != channels or sorted(order) != list(range(channels)):
        return _identity_permutation(channels, utility.device)
    permutation = torch.tensor(
        order, dtype=torch.int64, device=utility.device
    )
    # Re-attach the utility's provenance to the returned permutation: the
    # grouping itself runs on Python-level index lists, which would drop
    # the operand taint chain and make the runtime compliance guard pass
    # the dual-side permutation silently instead of routing it to review.
    # The carrier is exactly zero, so the values never change.
    carrier = torch.nan_to_num(
        utility.sum(), nan=0.0, posinf=0.0, neginf=0.0
    ) * 0.0
    return permutation + carrier.to(torch.int64)


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


@torch.no_grad()
def _solve_k_center_scale_aware(
    dense: torch.Tensor,
    num_heads: int,
    head_dim: int,
    rounds: int,
) -> torch.Tensor:
    """C41: solve a quantization-aware K center by fixed-point iteration.

    ``K' = K - 1 c^T`` is an exact softmax invariance for every choice of
    ``c``, so the center can be optimized purely against HiF4 reconstruction
    error.  With the quantized codes held fixed, the MSE-optimal center is

        c = mean_tokens(K - dequant(Q(K - c)))

    which yields a simple fixed-point iteration.  Starting from ``c = 0``
    keeps the identity candidate admissible, so this can never be worse than
    the uncentered path when the gate also keeps the incumbent.
    """

    heads = int(num_heads)
    width = int(head_dim)
    if dense.ndim != 2 or int(dense.shape[0]) <= 0:
        raise ValueError("scale-aware centering expects a non-empty 2D tensor")
    if int(dense.shape[1]) != heads * width:
        raise ValueError("Invalid dimensions for scale-aware centering")
    grouped = dense.reshape(-1, heads, width).to(torch.float32)
    center = torch.zeros(
        (heads, width), dtype=torch.float32, device=grouped.device
    )
    for _ in range(max(1, int(rounds))):
        shifted = (grouped - center).reshape(-1, heads * width)
        params = _dense_to_hif4(shifted, search_offsets=())
        rebuilt = _dequantize_hif4(params).reshape(-1, heads, width)
        updated = (grouped - rebuilt).mean(dim=0)
        if not torch.isfinite(updated).all():
            return torch.zeros(
                (heads, width), dtype=torch.float32, device="cpu"
            )
        delta = float((updated - center).abs().max())
        center = updated
        if delta <= 1.0e-6:
            break
    return center.detach().to(device="cpu").contiguous()


def _center_attention_k(
    dense: torch.Tensor,
    num_heads: int,
    head_dim: int,
    center_mode: int,
    center_value: Optional[torch.Tensor] = None,
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
    elif mode == 4:
        if center_value is None:
            raise ValueError("scale-aware centering requires a center vector")
        center = center_value.detach().to(
            device=grouped.device, dtype=grouped.dtype
        )
        if tuple(int(size) for size in center.shape) != (
            int(num_heads),
            int(head_dim),
        ):
            raise ValueError("Invalid scale-aware center shape")
        center = center.reshape(1, int(num_heads), int(head_dim))
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
    cat_transform: Optional[torch.Tensor] = None,
    center_mode: int = 0,
    center_num_heads: Optional[int] = None,
    center_head_dim: Optional[int] = None,
    center_value: Optional[torch.Tensor] = None,
    importance: Optional[torch.Tensor] = None,
    group_gram: Optional[torch.Tensor] = None,
    group_gram8: Optional[torch.Tensor] = None,
    group_gram16: Optional[torch.Tensor] = None,
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
            center_value,
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
    if cat_transform is not None:
        dense = _apply_cat64_rows(dense, cat_transform, inverse=False)
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
    refine_importance = importance
    if _ACTIVATION_SAMPLE_IMPORTANCE and dense.ndim == 2:
        refine_importance = torch.sqrt(
            dense.square().mean(dim=0).clamp_min(_EPS)
        )
    params = _dense_to_hif4(
        dense,
        importance=refine_importance,
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
            max_ratio=_ACTIVATION_QUADRATIC8_MAX_RATIO,
            max_groups=_ACTIVATION_QUADRATIC8_MAX_GROUPS,
            sweeps=_ACTIVATION_QUADRATIC8_SWEEPS,
            accept_margin=_ACTIVATION_QUADRATIC8_ACCEPT_MARGIN,
        )
    if (
        _ACTIVATION_QUADRATIC16
        and group_gram16 is not None
        and channels <= _ACTIVATION_QUADRATIC16_MAX_FEATURES
    ):
        gram16 = group_gram16.detach().to(
            device=dense.device, dtype=torch.float32
        )
        params = _refine_activation_groups16(dense, params, gram16)
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


def _fwht_last_dim(x: torch.Tensor) -> torch.Tensor:
    """Butterfly fast Walsh-Hadamard transform along the last dimension.

    Equivalent to ``x @ H_n`` for the normalized Sylvester Hadamard matrix
    ``H_n`` (which is symmetric), but never materializes the dense matrix.
    The input is never modified in place; float32/bfloat16 and CPU/CUDA all
    run the same deterministic op sequence.
    """

    n = int(x.shape[-1])
    if n < 1 or (n & (n - 1)) != 0:
        raise ValueError(f"FWHT width must be a power of two, got {n}")
    lead = tuple(x.shape[:-1])
    y = x.reshape(-1, n).clone()
    width = 1
    while width < n:
        y = y.reshape(-1, n // (2 * width), 2, width)
        a = y[:, :, 0, :]
        b = y[:, :, 1, :]
        y = torch.stack((a + b, a - b), dim=2).reshape(-1, n)
        width *= 2
    y = y * (1.0 / math.sqrt(float(n)))
    return y.reshape(*lead, n)


def _linear_r64_signs(
    channels: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Deterministic per-channel sign vector shared by both Linear sides."""

    indices = torch.arange(int(channels), dtype=torch.int64, device=device)
    bits = (
        indices * 1_103_515_245 + int(seed) * 214_013 + 12_345
    ).bitwise_and(1 << 30)
    return torch.where(bits == 0, 1.0, -1.0).to(dtype=dtype)


def _apply_linear_r64(x: torch.Tensor, seed: int) -> torch.Tensor:
    """Apply the signed orthogonal R64 incoherence transform.

    ``R64 = diag(signs) · H64`` applied on the last dimension via the
    butterfly FWHT (no dense [64, 64] matrix is ever built).  The transform
    is exactly orthogonal, so the inverse applies the FWHT and multiplies
    the same signs back.
    """

    channels = int(x.shape[-1])
    if channels % _LINEAR_R64_BLOCK != 0:
        raise ValueError(
            f"Feature width {channels} is not divisible by "
            f"{_LINEAR_R64_BLOCK}"
        )
    signs = _linear_r64_signs(channels, seed, x.device, x.dtype)
    grouped = x.reshape(
        *x.shape[:-1], channels // _LINEAR_R64_BLOCK, _LINEAR_R64_BLOCK
    )
    grouped = grouped * signs.reshape(
        channels // _LINEAR_R64_BLOCK, _LINEAR_R64_BLOCK
    )
    return _fwht_last_dim(grouped).reshape(*x.shape)


def _block_hadamard_transform(
    dense: torch.Tensor,
    block_size: int,
    seed: int = 0,
) -> torch.Tensor:
    """Apply a deterministic signed orthogonal transform to feature blocks.

    The signs avoid concentrating positively correlated channels in the DC
    Hadamard coefficient.  They are derived from the absolute feature index,
    so calibration and dynamic quantization only share ``block_size`` and a
    small integer ``seed``.  Size 64 routes through the butterfly FWHT; the
    smaller sizes keep the dense matrix product.
    """

    size = int(block_size)
    if size == 0:
        return dense
    if size == _LINEAR_R64_BLOCK:
        return _apply_linear_r64(dense, int(seed))
    channels = int(dense.shape[-1])
    if channels % size != 0:
        raise ValueError(
            f"Feature width {channels} is not divisible by block size {size}"
        )
    signs = _linear_r64_signs(channels, seed, dense.device, dense.dtype)
    grouped = dense.reshape(*dense.shape[:-1], channels // size, size)
    grouped = grouped * signs.reshape(channels // size, size)
    h = _hadamard_matrix(size, dense.device, dense.dtype)
    return torch.matmul(grouped, h).reshape_as(dense)


def _spd_matrix_power(
    matrix: torch.Tensor,
    power: float,
    relative_floor: float = _CAT64_RELATIVE_FLOOR,
) -> torch.Tensor:
    """Evaluate a symmetric positive-definite matrix power in float32.

    The CAT statistics are empirical Gram matrices and can be rank deficient
    when a calibration window is short.  A relative eigenvalue floor keeps
    the inverse path finite without imposing a condition-number rejection.
    The helper accepts either one matrix or a batch of matrices.
    """

    value = matrix.detach().to(dtype=torch.float32)
    if value.ndim < 2 or int(value.shape[-1]) != int(value.shape[-2]):
        raise ValueError("SPD matrix power expects square matrices")
    value = 0.5 * (value + value.transpose(-1, -2))
    n = int(value.shape[-1])
    trace = value.diagonal(dim1=-2, dim2=-1).sum(dim=-1) / float(n)
    floor = trace.abs().clamp_min(_EPS) * max(float(relative_floor), 0.0)
    floor = floor.unsqueeze(-1)
    eigenvalues, eigenvectors = torch.linalg.eigh(value)
    eigenvalues = torch.where(
        torch.isfinite(eigenvalues), eigenvalues, floor
    ).clamp_min(floor)
    powered = eigenvalues.pow(float(power))
    result = (eigenvectors * powered.unsqueeze(-2)).matmul(
        eigenvectors.transpose(-1, -2)
    )
    return 0.5 * (result + result.transpose(-1, -2))


def _spd_geometric_mean(
    left: torch.Tensor,
    right: torch.Tensor,
    relative_floor: float = _CAT64_RELATIVE_FLOOR,
) -> torch.Tensor:
    """Affine-invariant geometric mean ``left # right`` for SPD batches."""

    left_half = _spd_matrix_power(left, 0.5, relative_floor)
    left_inv_half = _spd_matrix_power(left, -0.5, relative_floor)
    middle = left_inv_half.matmul(right).matmul(left_inv_half)
    middle_half = _spd_matrix_power(middle, 0.5, relative_floor)
    result = left_half.matmul(middle_half).matmul(left_half)
    return 0.5 * (result + result.transpose(-1, -2))


def _cat64_blocks(
    activation_covariance: torch.Tensor,
    weight_gram: torch.Tensor,
    strength: float,
    *,
    shrink: float = _CAT64_SHRINK,
    relative_floor: float = _CAT64_RELATIVE_FLOOR,
) -> torch.Tensor:
    """Construct normalized CAT-64 transforms for independent channel blocks.

    ``activation_covariance`` and ``weight_gram`` have shape
    ``[num_blocks, 64, 64]``.  The determinant-normalized square root of
    ``Sigma_w # inv(Sigma_x)`` is raised to ``strength`` in log coordinates,
    so strength zero is the exact identity and strength one is the full
    analytic CAT solution.
    """

    x_cov = activation_covariance.detach().to(dtype=torch.float32)
    w_gram = weight_gram.detach().to(dtype=torch.float32)
    if x_cov.ndim != 3 or w_gram.ndim != 3:
        raise ValueError("CAT-64 statistics must be rank-3 batches")
    if tuple(x_cov.shape) != tuple(w_gram.shape):
        raise ValueError("CAT-64 statistics must have matching shapes")
    blocks, rows, cols = map(int, x_cov.shape)
    if rows != _CAT64_BLOCK_SIZE or cols != _CAT64_BLOCK_SIZE:
        raise ValueError("CAT-64 statistics must use 64x64 blocks")
    identity = torch.eye(
        _CAT64_BLOCK_SIZE, dtype=torch.float32, device=x_cov.device
    ).expand(blocks, -1, -1).clone()
    beta = float(strength)
    if not math.isfinite(beta) or beta <= 0.0:
        return identity
    beta = min(beta, 1.0)

    def _shrink(value: torch.Tensor) -> torch.Tensor:
        value = 0.5 * (value + value.transpose(-1, -2))
        mean = (
            value.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
            / float(_CAT64_BLOCK_SIZE)
        ).clamp_min(_EPS)
        amount = max(0.0, min(float(shrink), 1.0))
        value = (1.0 - amount) * value + amount * mean[..., None, None] * torch.eye(
            _CAT64_BLOCK_SIZE, dtype=value.dtype, device=value.device
        )
        return 0.5 * (value + value.transpose(-1, -2))

    try:
        x_cov = _shrink(x_cov)
        w_gram = _shrink(w_gram)
        x_inv = _spd_matrix_power(x_cov, -1.0, relative_floor)
        geometric = _spd_geometric_mean(w_gram, x_inv, relative_floor)
        base = _spd_matrix_power(geometric, 0.5, relative_floor)
        eigenvalues, eigenvectors = torch.linalg.eigh(base)
        eigenvalues = eigenvalues.clamp_min(_EPS)
        log_values = eigenvalues.log()
        log_values = log_values - log_values.mean(dim=-1, keepdim=True)
        values = (beta * log_values).exp()
        transforms = (eigenvectors * values.unsqueeze(-2)).matmul(
            eigenvectors.transpose(-1, -2)
        )
        transforms = 0.5 * (transforms + transforms.transpose(-1, -2))
        finite = torch.isfinite(transforms).all(dim=-1).all(dim=-1)
        transforms = torch.where(
            finite[:, None, None], transforms, identity
        )
        return transforms.contiguous()
    except (RuntimeError, ValueError):
        # A numerical failure on a candidate is a local identity candidate,
        # not a reason to reject the whole layer or impose a hard gate.
        return identity


def _apply_cat64_rows(
    dense: torch.Tensor,
    transforms: Optional[torch.Tensor],
    *,
    inverse: bool = False,
) -> torch.Tensor:
    """Apply a block-diagonal CAT transform to row-major data."""

    if transforms is None:
        return dense
    channels = int(dense.shape[-1])
    block = int(_CAT64_BLOCK_SIZE)
    if channels % block != 0:
        raise ValueError("CAT-64 transform width does not divide channels")
    matrix = transforms.detach().to(device=dense.device, dtype=torch.float32)
    expected = (channels // block, block, block)
    if tuple(matrix.shape) != expected:
        raise ValueError(
            f"CAT-64 transform shape {tuple(matrix.shape)} != {expected}"
        )
    if inverse:
        matrix = torch.linalg.inv(matrix)
    grouped = dense.to(dtype=torch.float32).reshape(
        *dense.shape[:-1], channels // block, block
    )
    transformed = torch.einsum("...bi,bji->...bj", grouped, matrix)
    return transformed.reshape_as(grouped).reshape_as(dense)


def _apply_cat64_covariance(
    covariance: torch.Tensor,
    transforms: Optional[torch.Tensor],
) -> torch.Tensor:
    """Apply ``blockdiag(transforms) C blockdiag(transforms)^T``."""

    if transforms is None:
        return covariance
    channels = int(covariance.shape[-1])
    block = int(_CAT64_BLOCK_SIZE)
    if tuple(covariance.shape) != (channels, channels) or channels % block != 0:
        raise ValueError("CAT covariance must be a square 2D divisible matrix")
    matrix = transforms.detach().to(
        device=covariance.device, dtype=covariance.dtype
    )
    blocks = channels // block
    if tuple(matrix.shape) != (blocks, block, block):
        raise ValueError("CAT covariance transform shape mismatch")
    grouped = covariance.reshape(blocks, block, blocks, block)
    left = torch.einsum("bpi,bicj->bpcj", matrix, grouped)
    transformed = torch.einsum("bpcj,cqj->bpcq", left, matrix)
    return transformed.reshape(channels, channels)


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
    cat_transform: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply one side of the exactly equivalent Linear transform."""

    scale = d if weight_side else d.reciprocal()
    transformed = (dense * scale.unsqueeze(0)).index_select(-1, permutation)
    transformed = _block_hadamard_transform(
        transformed, block_smooth_size, block_smooth_seed
    )
    return _apply_cat64_rows(
        transformed, cat_transform, inverse=bool(weight_side)
    )


def _transformed_second_moment(
    second_moment: torch.Tensor,
    d: torch.Tensor,
    permutation: torch.Tensor,
    block_smooth_size: int,
    block_smooth_seed: int = 0,
    cat_transform: Optional[torch.Tensor] = None,
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
    if cat_transform is not None:
        matrix = cat_transform.detach().to(
            device=moment.device, dtype=torch.float32
        )
        blocks = int(moment.numel()) // int(_CAT64_BLOCK_SIZE)
        if tuple(matrix.shape) != (
            blocks,
            _CAT64_BLOCK_SIZE,
            _CAT64_BLOCK_SIZE,
        ):
            raise ValueError("CAT second-moment transform shape mismatch")
        grouped = moment.reshape(blocks, _CAT64_BLOCK_SIZE)
        moment = torch.einsum(
            "bij,bj->bi", matrix.square(), grouped
        ).reshape(-1)
    return moment


def _transformed_covariance(
    covariance: torch.Tensor,
    d: torch.Tensor,
    permutation: torch.Tensor,
    block_smooth_size: int,
    block_smooth_seed: int = 0,
    cat_transform: Optional[torch.Tensor] = None,
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
    if cat_transform is not None:
        cov = _apply_cat64_covariance(cov, cat_transform)
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


def _cat64_operand_metrics(
    weight_rows: torch.Tensor,
    activation_second_moment: torch.Tensor,
    activation_samples: Sequence[torch.Tensor],
    d: torch.Tensor,
    permutation: torch.Tensor,
    block_smooth_size: int,
    block_smooth_seed: int,
    cat_transform: Optional[torch.Tensor],
) -> tuple[float, tuple[float, ...], float]:
    """Return CAT operand losses and a diagonal alignment score.

    This is deliberately separated from the evaluator's product objective:
    CAT selection can only see the activation and weight operands.  The
    returned losses use the same standard HiF4 codec as the parent search,
    while ``alignment`` estimates ``trace(W H W^T)`` from the diagonal
    activation second moment.
    """

    channels = int(weight_rows.shape[1])
    weight_smooth = _linear_pair_transform(
        weight_rows,
        d,
        permutation,
        block_smooth_size,
        block_smooth_seed,
        weight_side=True,
        cat_transform=cat_transform,
    )
    h_x = _transformed_second_moment(
        activation_second_moment,
        d,
        permutation,
        block_smooth_size,
        block_smooth_seed,
        cat_transform,
    )
    weight_params = _dense_to_hif4(weight_smooth)
    weight_hat = _dequantize_hif4(weight_params)
    weight_energy = (weight_smooth.square() * h_x.unsqueeze(0)).sum()
    weight_loss = torch.nan_to_num(
        ((weight_smooth - weight_hat).square() * h_x.unsqueeze(0)).sum()
        / (weight_energy + _EPS),
        nan=1.0e30,
        posinf=1.0e30,
        neginf=1.0e30,
    )
    h_w = _normalize_importance(weight_hat.square().sum(dim=0), channels)
    if h_w is None:
        h_w = torch.ones(channels, dtype=torch.float32, device=weight_rows.device)

    activation_losses: list[float] = []
    for sample in activation_samples:
        transformed = _linear_pair_transform(
            sample,
            d,
            permutation,
            block_smooth_size,
            block_smooth_seed,
            weight_side=False,
            cat_transform=cat_transform,
        )
        reconstructed = _dequantize_hif4(_dense_to_hif4(transformed))
        numerator = ((transformed - reconstructed).square() * h_w.unsqueeze(0)).sum()
        denominator = (transformed.square() * h_w.unsqueeze(0)).sum()
        activation_losses.append(
            float(
                torch.nan_to_num(
                    numerator / (denominator + _EPS),
                    nan=1.0e30,
                    posinf=1.0e30,
                    neginf=1.0e30,
                )
            )
        )

    alignment_denominator = (
        weight_smooth.square().sum() * h_x.sum()
    ).clamp_min(_EPS)
    alignment = torch.nan_to_num(
        weight_energy / alignment_denominator,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    return float(weight_loss), tuple(activation_losses), float(alignment)


def _cat64_robust_objective(
    baseline: tuple[float, tuple[float, ...], float],
    candidate: tuple[float, tuple[float, ...], float],
) -> tuple[float, tuple[float, ...]]:
    """Compare one CAT candidate with the base transform using soft folds."""

    base_weight, base_activation, base_alignment = baseline
    cand_weight, cand_activation, cand_alignment = candidate
    if not math.isfinite(base_weight) or not math.isfinite(cand_weight):
        return 1.0e30, (1.0e30,)
    weight_ratio = cand_weight / max(base_weight, _EPS)
    alignment_ratio = base_alignment / max(cand_alignment, _EPS)
    if not math.isfinite(weight_ratio) or not math.isfinite(alignment_ratio):
        return 1.0e30, (1.0e30,)
    if not base_activation:
        folds = (0.40 * weight_ratio + 0.20 * alignment_ratio,)
    else:
        folds = tuple(
            0.40 * weight_ratio
            + 0.40 * (
                candidate_loss / max(base_loss, _EPS)
            )
            + 0.20 * alignment_ratio
            for base_loss, candidate_loss in zip(
                base_activation, cand_activation
            )
        )
    mean = sum(folds) / float(len(folds))
    robust = mean + 0.15 * (max(folds) - mean)
    return float(robust), tuple(float(v) for v in folds)


def _select_cat64_transform(
    weight_rows: torch.Tensor,
    activation_second_moment: torch.Tensor,
    activation_samples: Sequence[torch.Tensor],
    d: torch.Tensor,
    permutation: torch.Tensor,
    block_smooth_size: int,
    block_smooth_seed: int,
) -> tuple[Optional[torch.Tensor], float, tuple[float, ...]]:
    """Select a determinant-normalized CAT-64 transform from local losses."""

    channels = int(weight_rows.shape[1])
    blocks = channels // int(_CAT64_BLOCK_SIZE)
    if (
        not _CAT64
        or channels < int(_CAT64_BLOCK_SIZE)
        or channels % int(_CAT64_BLOCK_SIZE) != 0
        or not activation_samples
    ):
        return None, 1.0, (1.0,)

    base_activation_rows = torch.cat(
        [
            _linear_pair_transform(
                sample,
                d,
                permutation,
                block_smooth_size,
                block_smooth_seed,
                weight_side=False,
            )
            for sample in activation_samples
        ],
        dim=0,
    )
    base_activation_rows = _sample_rows(
        base_activation_rows, _CAT64_MAX_CALIB_ROWS
    )
    base_weight_rows = _linear_pair_transform(
        weight_rows,
        d,
        permutation,
        block_smooth_size,
        block_smooth_seed,
        weight_side=True,
    )
    x_grouped = base_activation_rows.reshape(-1, blocks, _CAT64_BLOCK_SIZE)
    w_grouped = base_weight_rows.reshape(-1, blocks, _CAT64_BLOCK_SIZE)
    count_x = float(max(int(x_grouped.shape[0]), 1))
    count_w = float(max(int(w_grouped.shape[0]), 1))
    x_covariance = torch.einsum(
        "tbi,tbj->bij", x_grouped, x_grouped
    ) / count_x
    weight_gram = torch.einsum(
        "obi,obj->bij", w_grouped, w_grouped
    ) / count_w

    baseline = _cat64_operand_metrics(
        weight_rows,
        activation_second_moment,
        activation_samples,
        d,
        permutation,
        block_smooth_size,
        block_smooth_seed,
        None,
    )
    best_objective, best_folds = _cat64_robust_objective(baseline, baseline)
    best_transform: Optional[torch.Tensor] = None
    for beta in _CAT64_BETAS:
        transforms = _cat64_blocks(
            x_covariance,
            weight_gram,
            float(beta),
        )
        candidate = _cat64_operand_metrics(
            weight_rows,
            activation_second_moment,
            activation_samples,
            d,
            permutation,
            block_smooth_size,
            block_smooth_seed,
            transforms,
        )
        objective, folds = _cat64_robust_objective(baseline, candidate)
        if (
            objective < best_objective
            and objective <= 1.0 - float(_CAT64_MIN_IMPROVEMENT)
        ):
            best_objective = objective
            best_folds = folds
            best_transform = transforms
    return best_transform, float(best_objective), best_folds


def _r64_operand_losses(
    weight_rows: torch.Tensor,
    second_moment: torch.Tensor,
    activation_samples: Sequence[torch.Tensor],
    d: torch.Tensor,
    permutation: torch.Tensor,
    size: int,
    seed: int,
) -> tuple[float, tuple[float, ...]]:
    """Operand-separated standard-HiF4 losses of one block transform.

    The weight side reports the ``H_A``-weighted relative reconstruction
    loss; the activation side reports the per-sample importance-weighted
    reconstruction losses.  No Linear output is ever constructed.
    """

    channels = int(weight_rows.shape[1])
    weight_smooth = _linear_pair_transform(
        weight_rows,
        d,
        permutation,
        int(size),
        int(seed),
        weight_side=True,
    )
    h_x = _transformed_second_moment(second_moment, d, permutation, size)
    weight_hat = _dequantize_hif4(_dense_to_hif4(weight_smooth))
    weight_loss = float(
        torch.nan_to_num(
            ((weight_smooth - weight_hat).square() * h_x.unsqueeze(0)).sum()
            / (
                (weight_smooth.square() * h_x.unsqueeze(0)).sum() + _EPS
            ),
            nan=1.0e30,
            posinf=1.0e30,
            neginf=1.0e30,
        )
    )
    h_w = _normalize_importance(
        weight_hat.square().sum(dim=0), channels
    )
    if h_w is None:
        h_w = torch.ones(channels, dtype=torch.float32, device=weight_rows.device)

    act_losses: list[float] = []
    for sample in activation_samples:
        smooth = _linear_pair_transform(
            sample,
            d,
            permutation,
            int(size),
            int(seed),
            weight_side=False,
        )
        recon = _dequantize_hif4(_dense_to_hif4(smooth))
        error = ((smooth - recon).square() * h_w.unsqueeze(0)).sum()
        energy = (smooth.square() * h_w.unsqueeze(0)).sum()
        act_losses.append(
            float(
                torch.nan_to_num(
                    error / (energy + _EPS),
                    nan=1.0e30,
                    posinf=1.0e30,
                    neginf=1.0e30,
                )
            )
        )
    return weight_loss, tuple(act_losses)


def _hierarchy_edge_utility(
    rows: torch.Tensor, residual_energy: torch.Tensor
) -> torch.Tensor:
    """C30 edge utility: |Gram| scaled elementwise by residual energies.

    ``rows`` are calibration activation rows; ``residual_energy`` is the
    per-channel weight-quantization residual energy.  The two operand-local
    statistics are combined elementwise (an outer product of the channel
    statistic with itself times the activation Gram magnitude) -- no
    cross-operand contraction is ever formed.
    """

    gram = rows.t().to(torch.float32) @ rows.to(torch.float32)
    gram = gram.abs() / float(max(int(rows.shape[0]), 1))
    scale = torch.sqrt(
        torch.outer(residual_energy, residual_energy).clamp_min(0.0)
    )
    return gram * scale


def _cat64_grouping_permutation(
    weight: torch.Tensor,
    activation_samples: Sequence[torch.Tensor],
    d: torch.Tensor,
    permutation: torch.Tensor,
) -> Optional[torch.Tensor]:
    """Propose a CAT-aware permutation from operand covariance utility.

    The returned order is composed with the already selected SmoothQuant
    permutation.  Utility is computed in that parent coordinate system from
    ``A^T A`` and ``W^T W`` only; no output product or evaluator state is
    constructed.  The feature cap bounds the O(C^2) covariance work on wide
    FFN layers while retaining the inexpensive 768/1024-channel attention
    and up-projection cases.
    """

    channels = int(weight.shape[1])
    if (
        not _CAT64_GROUPING
        or channels < _CAT64_BLOCK_SIZE
        or channels % _CAT64_BLOCK_SIZE != 0
        or channels > _CAT64_GROUPING_MAX_FEATURES
        or not activation_samples
    ):
        return None
    device = weight.device
    try:
        rows = torch.cat(
            [sample.to(device=device, dtype=torch.float32)
             for sample in activation_samples],
            dim=0,
        )
        rows = _sample_rows(rows, _CAT64_MAX_CALIB_ROWS)
        weight_rows = _sample_rows(
            weight.to(dtype=torch.float32), _LINEAR_WEIGHT_EVAL_ROWS
        )
        # Work after the selected diagonal scale/permutation and before any
        # optional small Hadamard.  This makes the composed permutation an
        # exact input to the existing transform order.
        rows = rows * d.reciprocal().to(device=device).reshape(1, -1)
        rows = rows.index_select(-1, permutation)
        weight_rows = weight_rows * d.to(device=device).reshape(1, -1)
        weight_rows = weight_rows.index_select(-1, permutation)
        count_x = float(max(int(rows.shape[0]), 1))
        count_w = float(max(int(weight_rows.shape[0]), 1))
        x_cov = rows.t().mm(rows) / count_x
        w_cov = weight_rows.t().mm(weight_rows) / count_w
        x_diag = x_cov.diagonal().abs().clamp_min(_EPS)
        w_diag = w_cov.diagonal().abs().clamp_min(_EPS)
        utility = x_cov.abs() * torch.sqrt(
            torch.outer(w_diag, w_diag)
        )
        utility = utility + w_cov.abs() * torch.sqrt(
            torch.outer(x_diag, x_diag)
        )
        utility = torch.nan_to_num(
            utility,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        utility.fill_diagonal_(0.0)
        local = _hierarchy_edge_permutation(utility)
        identity = _identity_permutation(channels, device)
        if torch.equal(local, identity):
            return None
        return permutation.index_select(
            0, local.to(device=permutation.device, dtype=torch.int64)
        )
    except (RuntimeError, ValueError):
        return None


def _hierarchy_permutation_candidate(
    weight: torch.Tensor,
    weight_sample: torch.Tensor,
    activation_samples: Sequence[torch.Tensor],
    second_moment: torch.Tensor,
    d: torch.Tensor,
    permutation: torch.Tensor,
    size: int,
    seed: int,
) -> Optional[torch.Tensor]:
    """Build the C30 hierarchical edge permutation (ungated).

    Returns the replacement permutation, or ``None`` to keep the parent.
    """

    channels = int(weight.shape[1])
    if channels < 16 or not activation_samples:
        return None
    device = weight.device

    # Parent weight-quantization residual energy per channel (standard
    # HiF4 quantization of the transform-selected weight).
    weight_smooth = _linear_pair_transform(
        weight, d, permutation, int(size), int(seed), weight_side=True
    )
    weight_hat = _dequantize_hif4(_dense_to_hif4(weight_smooth))
    residual = torch.zeros_like(weight_smooth)
    residual.index_copy_(1, permutation, weight_smooth - weight_hat)
    residual_energy = residual.square().sum(dim=0).sqrt()

    all_rows = torch.cat(
        [s.to(device=device, dtype=torch.float32) for s in activation_samples],
        dim=0,
    )
    candidate_perm = _hierarchy_edge_permutation(
        _hierarchy_edge_utility(all_rows, residual_energy)
    ).to(device)
    if torch.equal(candidate_perm, permutation):
        return None
    return candidate_perm


def _rank_r64_seeds(
    weight_rows: torch.Tensor,
    activation_rows: torch.Tensor,
    second_moment: torch.Tensor,
    d: torch.Tensor,
    permutation: torch.Tensor,
) -> list[int]:
    """Stage A: cheap operand-local ranking of the R64 seeds.

    Uses at most 64 activation rows and 128 weight rows with plain
    standard HiF4 (no refinement), ranking each seed by the sum of the
    activation hard reconstruction loss and the ``H_A``-weighted weight
    loss.  Returns all seeds ordered best-first.
    """

    scores: list[tuple[float, int]] = []
    for seed in _LINEAR_R64_STAGE1_SEEDS:
        weight_loss, act_losses = _r64_operand_losses(
            weight_rows,
            second_moment,
            (activation_rows,),
            d,
            permutation,
            _LINEAR_R64_BLOCK,
            int(seed),
        )
        act_loss = act_losses[0] if act_losses else 0.0
        scores.append((weight_loss + act_loss, int(seed)))
    scores.sort(key=lambda item: (item[0], item[1]))
    return [seed for _score, seed in scores]


def _r64_two_fold_check(
    weight_rows: torch.Tensor,
    activation_samples: Sequence[torch.Tensor],
    d: torch.Tensor,
    permutation: torch.Tensor,
    parent_size: int,
    parent_seed: int,
    seed: int,
) -> bool:
    """Stage B two-fold validation of one R64 seed against the parent.

    Fold ``i`` computes the fold statistics from batch ``i`` and scores on
    every other batch.  The seed must keep the activation-only metric no
    worse than the parent transform on both folds, and its operand-separated
    robust metric ``max(ratio_A, ratio_W) + 0.10 * max(0, tail - 1)`` must
    beat the parent (metric < 1).
    """

    fold_count = min(2, len(activation_samples))
    for fold_index in range(fold_count):
        stats_batch = activation_samples[fold_index]
        fold_second = stats_batch.square().mean(dim=0)
        eval_batches = [
            batch
            for index, batch in enumerate(activation_samples)
            if index != fold_index
        ]
        if not eval_batches:
            eval_batches = [stats_batch]
        w_seed, act_seed = _r64_operand_losses(
            weight_rows,
            fold_second,
            eval_batches,
            d,
            permutation,
            _LINEAR_R64_BLOCK,
            int(seed),
        )
        w_parent, act_parent = _r64_operand_losses(
            weight_rows,
            fold_second,
            eval_batches,
            d,
            permutation,
            int(parent_size),
            int(parent_seed),
        )
        act_seed_mean = sum(act_seed) / float(len(act_seed))
        act_parent_mean = sum(act_parent) / float(len(act_parent))
        if act_seed_mean > act_parent_mean:
            return False
        ratio_a = act_seed_mean / max(act_parent_mean, 1.0e-12)
        ratio_w = w_seed / max(w_parent, 1.0e-12)
        tail_ratio = max(act_seed) / max(max(act_parent), 1.0e-12)
        metric = max(ratio_a, ratio_w) + 0.10 * max(0.0, tail_ratio - 1.0)
        if not math.isfinite(metric) or metric >= 1.0:
            return False
    return True


def _select_r64_candidate(
    weight_sample: torch.Tensor,
    activation_samples: Sequence[torch.Tensor],
    activation_second_moment: torch.Tensor,
    d: torch.Tensor,
    permutation: torch.Tensor,
    parent_size: int,
    parent_seed: int,
    baseline_metrics: tuple[float, tuple[float, ...]],
) -> tuple[int, tuple[float, tuple[float, ...]]]:
    """Two-stage R64 seed selection on operand-local metrics only.

    Stage A ranks all seeds on cheap sampled operands; Stage B validates the
    top seeds with the two-fold check and the deployed full-data candidate
    metric guarded by ``_candidate_is_safe``.  Returns ``(-1, baseline)``
    when every seed regresses, keeping the parent transform.
    """

    if not _LINEAR_R64:
        return -1, baseline_metrics
    activation_rows = _sample_rows(
        torch.cat(list(activation_samples), dim=0), _LINEAR_R64_STAGE1_ROWS
    )
    weight_rows = _sample_rows(
        weight_sample, _LINEAR_R64_STAGE1_WEIGHT_ROWS
    )
    ranked = _rank_r64_seeds(
        weight_rows,
        activation_rows,
        activation_second_moment,
        d,
        permutation,
    )
    best_seed = -1
    best_metrics = baseline_metrics
    for seed in ranked[: int(_LINEAR_R64_STAGE2_KEEP)]:
        if not _r64_two_fold_check(
            weight_sample,
            activation_samples,
            d,
            permutation,
            int(parent_size),
            int(parent_seed),
            seed,
        ):
            continue
        metrics = _linear_candidate_metrics(
            weight_sample,
            activation_second_moment,
            activation_samples,
            d,
            permutation,
            _LINEAR_R64_BLOCK,
            seed,
        )
        if (
            metrics[0] < best_metrics[0]
            and _candidate_is_safe(
                metrics,
                baseline_metrics,
                min_mean_improvement=_LINEAR_R64_MIN_IMPROVEMENT,
                worst_tolerance=_LINEAR_R64_WORST_TOLERANCE,
            )
        ):
            best_metrics = metrics
            best_seed = int(seed)
    return best_seed, best_metrics


def _activation8_refinement_is_safe(
    activation_samples: Sequence[torch.Tensor],
    d: torch.Tensor,
    permutation: torch.Tensor,
    block_smooth_size: int,
    block_smooth_seed: int,
    importance: Optional[torch.Tensor],
    group_gram4: Optional[torch.Tensor],
    group_gram8: torch.Tensor,
    activation_ratio: float,
    cat_transform: Optional[torch.Tensor] = None,
) -> bool:
    """Activation-only calibration gate for the 8x8 dynamic refinement.

    Compares the activation-local reconstruction loss of the base encoder
    output against the refined output on the transformed calibration
    samples themselves.  The gate never reads the weight, weight state,
    or any Linear output.
    """

    base_losses: list[float] = []
    refined_losses: list[float] = []
    for sample in activation_samples:
        transformed = _linear_pair_transform(
            sample,
            d,
            permutation,
            block_smooth_size,
            block_smooth_seed,
            weight_side=False,
            cat_transform=cat_transform,
        )
        channels = int(transformed.shape[-1])
        blocks = channels // _HIF4_BLOCK_SIZE
        gram4 = None
        if group_gram4 is not None:
            gram4 = (
                group_gram4.detach()
                .to(device=transformed.device, dtype=torch.float32)
                .reshape(blocks, 8, 2, 4, 4)
                .unsqueeze(0)
                .expand(int(transformed.shape[0]), blocks, 8, 2, 4, 4)
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
        refined_params = _refine_weight_groups8(
            transformed,
            base_params,
            group_gram8,
            max_ratio=_ACTIVATION_QUADRATIC8_MAX_RATIO,
            max_groups=_ACTIVATION_QUADRATIC8_MAX_GROUPS,
            sweeps=_ACTIVATION_QUADRATIC8_SWEEPS,
            accept_margin=_ACTIVATION_QUADRATIC8_ACCEPT_MARGIN,
        )
        base_output = _dequantize_hif4(base_params)
        refined_output = _dequantize_hif4(refined_params)
        denominator = transformed.square().sum() + _EPS
        base_losses.append(
            float((transformed - base_output).square().sum() / denominator)
        )
        refined_losses.append(
            float((transformed - refined_output).square().sum() / denominator)
        )

    if not base_losses or len(base_losses) != len(refined_losses):
        return False
    if not all(map(math.isfinite, base_losses + refined_losses)):
        return False
    base_mean = sum(base_losses) / float(len(base_losses))
    refined_mean = sum(refined_losses) / float(len(refined_losses))
    mean_safe = refined_mean <= base_mean * (
        1.0 - _ACTIVATION_QUADRATIC8_GATE_MIN_IMPROVEMENT
    )
    worst_safe = all(
        refined <= base * (1.0 + _ACTIVATION_QUADRATIC8_GATE_WORST_TOLERANCE)
        for base, refined in zip(base_losses, refined_losses)
    )
    return bool(mean_safe and worst_safe)


def _cpu_state_tensor(x: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(
        x.detach().to(device="cpu", dtype=torch.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).contiguous()


def _weight_product_score(
    dense_activations: Sequence[torch.Tensor],
    dense_weight: torch.Tensor,
    quantized_weight: torch.Tensor,
    product_activations: Optional[Sequence[torch.Tensor]] = None,
) -> tuple[float, tuple[float, ...]]:
    """Score a static Q(W) candidate with offline calibration products.

    The products are intentionally confined to this offline weight selector.
    Only Python scalar fold scores leave the function; no product tensor is
    used to build or update ``activation_state``.  A soft mean/worst-fold
    blend keeps the selector useful when two calibration windows are small
    without turning one noisy fold into a hard rejection gate.
    """

    if not dense_activations:
        return 1.0e30, (1.0e30,)
    dense_weight = dense_weight.detach().to(dtype=torch.float32)
    quantized_weight = quantized_weight.detach().to(
        device=dense_weight.device, dtype=torch.float32
    )
    if dense_weight.ndim != 2 or tuple(quantized_weight.shape) != tuple(
        dense_weight.shape
    ):
        return 1.0e30, (1.0e30,)

    scores: list[float] = []
    for index, sample in enumerate(dense_activations):
        activation = sample.detach().to(
            device=dense_weight.device, dtype=torch.float32
        )
        if activation.ndim != 2 or int(activation.shape[1]) != int(
            dense_weight.shape[1]
        ):
            continue
        activation = _sample_rows(
            activation, _WEIGHT_PRODUCT_SELECTOR_MAX_CALIB_ROWS
        )
        player_activation = activation
        if product_activations is not None and index < len(product_activations):
            candidate_activation = product_activations[index].detach().to(
                device=dense_weight.device, dtype=torch.float32
            )
            if candidate_activation.ndim == 2 and tuple(
                candidate_activation.shape
            ) == tuple(sample.shape):
                player_activation = _sample_rows(
                    candidate_activation, _WEIGHT_PRODUCT_SELECTOR_MAX_CALIB_ROWS
                )
        reference_product = activation.mm(dense_weight.t())
        quantized_product = player_activation.mm(quantized_weight.t())
        error = (reference_product - quantized_product).square().mean()
        energy = reference_product.square().mean().clamp_min(_EPS)
        scores.append(
            float(
                torch.nan_to_num(
                    error / energy,
                    nan=1.0e30,
                    posinf=1.0e30,
                    neginf=1.0e30,
                )
            )
        )
    if not scores:
        return 1.0e30, (1.0e30,)
    mean_score = sum(scores) / float(len(scores))
    worst_score = max(scores)
    mix = max(
        0.0, min(float(_WEIGHT_PRODUCT_SELECTOR_ROBUST_MIX), 1.0)
    )
    return mean_score + mix * (worst_score - mean_score), tuple(scores)


@torch.no_grad()
def _select_static_weight_product_candidate(
    dense_weight: torch.Tensor,
    parent_params: dict[str, torch.Tensor],
    dense_activations: Sequence[torch.Tensor],
    product_activations: Optional[Sequence[torch.Tensor]] = None,
) -> dict[str, torch.Tensor]:
    """Use offline A@W to improve only the already-frozen static Q(W).

    A small conditional update is solved in the legal 64-channel HiF4 grid.
    The fit side uses the first calibration window(s), while the acceptance
    score evaluates every available window with a soft robust aggregate.  In
    particular, the product score is never consulted by the online activation
    quantizer and no product-derived tensor is returned through state.
    """

    if not _WEIGHT_PRODUCT_SELECTOR or not dense_activations:
        return parent_params
    if dense_weight.ndim != 2:
        return parent_params
    rows, channels = map(int, dense_weight.shape)
    if channels < int(_WEIGHT_PRODUCT_SELECTOR_MIN_CHANNELS):
        return parent_params
    if channels % _HIF4_BLOCK_SIZE != 0:
        return parent_params
    parent_weight = _dequantize_hif4(parent_params).to(
        device=dense_weight.device, dtype=torch.float32
    )
    dense_weight = dense_weight.detach().to(dtype=torch.float32)
    if tuple(parent_weight.shape) != (rows, channels):
        return parent_params

    samples: list[torch.Tensor] = []
    player_samples: list[torch.Tensor] = []
    for index, sample in enumerate(dense_activations):
        value = sample.detach().to(
            device=dense_weight.device, dtype=torch.float32
        )
        if value.ndim != 2 or int(value.shape[1]) != channels:
            continue
        samples.append(_sample_rows(value, _WEIGHT_PRODUCT_SELECTOR_MAX_CALIB_ROWS))
        player_value = value
        if product_activations is not None and index < len(product_activations):
            candidate_value = product_activations[index].detach().to(
                device=dense_weight.device, dtype=torch.float32
            )
            if candidate_value.ndim == 2 and tuple(candidate_value.shape) == tuple(
                value.shape
            ):
                player_value = candidate_value
        player_samples.append(
            _sample_rows(player_value, _WEIGHT_PRODUCT_SELECTOR_MAX_CALIB_ROWS)
        )
    if not samples:
        return parent_params

    # Use one or two folds for the conditional fit.  The final scalar score
    # below still sees all folds, so an overfit update has an easy path back to
    # the unchanged parent candidate.
    fit_count = max(1, len(samples))
    fit_samples = samples[:fit_count]
    fit_player_samples = player_samples[:fit_count]
    fit_rows = torch.cat(fit_player_samples, dim=0)
    if int(fit_rows.shape[0]) == 0:
        return parent_params
    fit_covariance = fit_rows.t().mm(fit_rows) / float(fit_rows.shape[0])
    h_blocks = _full64_hessian_blocks(fit_covariance, channels)
    diagonal_mean = h_blocks.diagonal(dim1=-2, dim2=-1).mean(dim=-1)
    if not bool(torch.isfinite(h_blocks).all()):
        return parent_params

    blocks = channels // _HIF4_BLOCK_SIZE
    denominator = (
        parent_params["scale_factor"].to(torch.float32)
        * parent_params["scale_lv2"].to(torch.float32)
        * parent_params["scale_lv3"].to(torch.float32)
    ).repeat_interleave(4, dim=-1).reshape(rows, blocks, _HIF4_BLOCK_SIZE)
    current_blocks = parent_weight.reshape(rows, blocks, _HIF4_BLOCK_SIZE).clone()

    # Form the complete calibration products once for the fit windows.  The
    # residual is an offline Q(W) objective; only its scalar ranking survives
    # the selector.
    fold_differences = [
        dense_sample.mm(dense_weight.t())
        - player_sample.mm(parent_weight.t())
        for dense_sample, player_sample in zip(fit_samples, fit_player_samples)
    ]
    initial_gradients = torch.stack(
        [
            difference.t().mm(player_sample)
            / float(max(int(player_sample.shape[0]), 1))
            for difference, player_sample in zip(
                fold_differences, fit_player_samples
            )
        ],
        dim=0,
    )
    initial_gradient = initial_gradients.mean(dim=0).reshape(
        rows, blocks, _HIF4_BLOCK_SIZE
    )
    block_scores = initial_gradient.square().sum(dim=(0, 2)) / (
        diagonal_mean.clamp_min(_EPS)
    )
    selected_count = min(
        blocks,
        max(
            1,
            int(
                math.ceil(
                    blocks * float(_WEIGHT_PRODUCT_SELECTOR_MAX_RATIO)
                )
            ),
        ),
    )
    finite_scores = torch.where(
        torch.isfinite(block_scores), block_scores, torch.zeros_like(block_scores)
    )
    selected_blocks = torch.topk(
        finite_scores, k=selected_count, largest=True
    ).indices.tolist()
    if not selected_blocks or not bool((finite_scores > _EPS).any()):
        return parent_params

    eye = torch.eye(
        _HIF4_BLOCK_SIZE, dtype=torch.float32, device=dense_weight.device
    )
    damped_h = h_blocks + (
        float(_WEIGHT_PRODUCT_SELECTOR_DAMPING)
        * diagonal_mean.clamp_min(_EPS)
    )[:, None, None] * eye

    any_change = False
    for block_index in selected_blocks:
        lo = int(block_index) * _HIF4_BLOCK_SIZE
        hi = lo + _HIF4_BLOCK_SIZE
        local_h = damped_h[block_index]
        inverse, info = torch.linalg.inv_ex(local_h)
        if int(info.reshape(-1)[0]) != 0 or not bool(
            torch.isfinite(inverse).all()
        ):
            continue
        gradients = torch.stack(
            [
                difference.t().mm(player_sample[:, lo:hi])
                / float(max(int(player_sample.shape[0]), 1))
                for difference, player_sample in zip(
                    fold_differences, fit_player_samples
                )
            ],
            dim=0,
        )
        gradient = gradients.mean(dim=0)
        conditional_step = gradient.mm(inverse)
        current = current_blocks[:, block_index : block_index + 1]
        local_denominator = denominator[:, block_index : block_index + 1]
        best_change = float("inf")
        best_local: Optional[torch.Tensor] = None
        local_h_batch = local_h.unsqueeze(0)
        for alpha in _WEIGHT_PRODUCT_SELECTOR_ALPHAS:
            target = current + float(alpha) * conditional_step.unsqueeze(1)
            trial = _coordinate_descent64(
                current,
                target,
                local_h_batch,
                local_denominator,
            )
            delta = trial[:, 0] - current[:, 0]
            change = (
                torch.einsum("ri,ij,rj->", delta, h_blocks[block_index], delta)
                - 2.0 * torch.einsum("ri,ri->", gradient, delta)
            )
            change_value = float(
                torch.nan_to_num(
                    change, nan=float("inf"), posinf=float("inf"), neginf=float("inf")
                )
            )
            if change_value < best_change:
                best_change = change_value
                best_local = trial
        if best_local is None or not math.isfinite(best_change) or best_change >= 0.0:
            continue
        delta = best_local[:, 0] - current[:, 0]
        current_blocks[:, block_index] = best_local[:, 0]
        fold_differences = [
            difference - player_sample[:, lo:hi].mm(delta.t())
            for difference, player_sample in zip(
                fold_differences, fit_player_samples
            )
        ]
        any_change = True

    if not any_change:
        return parent_params

    denominator = denominator.clamp_min(_EPS)
    codes = torch.round(current_blocks * (4.0 / denominator)).clamp_(-7.0, 7.0)
    candidate_params = dict(parent_params)
    candidate_params["sign"] = torch.sign(codes).reshape_as(parent_params["sign"])
    candidate_params["mant"] = (codes.abs() * 0.25).reshape_as(
        parent_params["mant"]
    )
    candidate_weight = _dequantize_hif4(candidate_params).to(torch.float32)
    parent_score, _ = _weight_product_score(
        samples, dense_weight, parent_weight, player_samples
    )
    candidate_score, _ = _weight_product_score(
        samples, dense_weight, candidate_weight, player_samples
    )
    if candidate_score <= parent_score * (
        1.0 - float(_WEIGHT_PRODUCT_SELECTOR_MIN_GAIN)
    ):
        return candidate_params
    return parent_params


@torch.no_grad()
def _select_static_weight_product_pool(
    dense_weight: torch.Tensor,
    parent_params: dict[str, torch.Tensor],
    candidate_params: dict[str, torch.Tensor],
    dense_activations: Sequence[torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Choose a static Q(W) candidate from a small offline product pool."""

    if not dense_activations:
        return parent_params
    parent_weight = _dequantize_hif4(parent_params).to(torch.float32)
    candidate_weight = _dequantize_hif4(candidate_params).to(torch.float32)
    parent_score, _ = _weight_product_score(
        dense_activations, dense_weight, parent_weight
    )
    candidate_score, _ = _weight_product_score(
        dense_activations, dense_weight, candidate_weight
    )
    if candidate_score <= parent_score * (
        1.0 - float(_WEIGHT_HEADROOM_MIN_GAIN)
    ):
        return candidate_params
    return parent_params


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
    block_baseline_metrics = _linear_candidate_metrics(
        weight_sample,
        activation_second_moment,
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
                block_metrics = _linear_candidate_metrics(
                    weight_sample,
                    activation_second_moment,
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
    elif (
        _LINEAR_R64
        and in_features % _LINEAR_R64_BLOCK == 0
        and best_block_smooth_size != _LINEAR_R64_BLOCK
    ):
        # C22: two-stage R64 incoherence-transform seed selection on
        # operand-local metrics only; -1 keeps the parent transform.
        r64_seed, r64_metrics = _select_r64_candidate(
            weight_sample,
            activation_samples,
            activation_second_moment,
            best_d,
            best_perm,
            best_block_smooth_size,
            best_block_smooth_seed,
            block_best_metrics,
        )
        if r64_seed >= 0:
            best_block_smooth_size = _LINEAR_R64_BLOCK
            best_block_smooth_seed = int(r64_seed)
            block_best_metrics = r64_metrics

    # C30: one additional permutation candidate from the hierarchical edge
    # utility, gated operand-separately with a two-fold stability check.
    # Only applied when the block-smooth transform stayed off: the block
    # size/seed search above was tuned for the parent permutation.
    if (
        _HIERARCHY_PERMUTATION
        and best_block_smooth_size == 0
        and len(activation_samples) >= 2
        and in_features >= 16
    ):
        c30_perm = _hierarchy_permutation_candidate(
            weight,
            weight_sample,
            activation_samples,
            activation_second_moment,
            best_d,
            best_perm,
            best_block_smooth_size,
            best_block_smooth_seed,
        )
        if c30_perm is not None:
            best_perm = c30_perm

    # C47: form one CAT-aware grouping candidate from operand covariance
    # utility.  The candidate is evaluated with the already selected parent
    # transform and a soft fold gate; no output product is involved.
    if _CAT64_GROUPING:
        grouping_perm = _cat64_grouping_permutation(
            weight,
            activation_samples,
            best_d,
            best_perm,
        )
        if grouping_perm is not None:
            grouping_baseline = _linear_candidate_metrics(
                weight_sample,
                activation_second_moment,
                activation_samples,
                best_d,
                best_perm,
                best_block_smooth_size,
                best_block_smooth_seed,
            )
            grouping_metrics = _linear_candidate_metrics(
                weight_sample,
                activation_second_moment,
                activation_samples,
                best_d,
                grouping_perm,
                best_block_smooth_size,
                best_block_smooth_seed,
            )
            if (
                grouping_metrics[0] < grouping_baseline[0]
                and _candidate_is_safe(
                    grouping_metrics,
                    grouping_baseline,
                    min_mean_improvement=_CAT64_GROUPING_MIN_IMPROVEMENT,
                    worst_tolerance=_CAT64_GROUPING_WORST_TOLERANCE,
                )
            ):
                best_perm = grouping_perm

    # C43: build the analytic CAT-64 transform after the parent diagonal,
    # permutation and small Hadamard choices are frozen.  Candidate ranking
    # uses operand-local reconstruction losses only; the later C45 A@W
    # selector is intentionally not part of this candidate.
    best_cat_transform: Optional[torch.Tensor] = None
    if _CAT64 and in_features % _CAT64_BLOCK_SIZE == 0:
        best_cat_transform, _cat_objective, _cat_folds = _select_cat64_transform(
            weight_sample,
            activation_second_moment,
            activation_samples,
            best_d,
            best_perm,
            best_block_smooth_size,
            best_block_smooth_seed,
        )

    weight_smooth = _linear_pair_transform(
        weight,
        best_d,
        best_perm,
        best_block_smooth_size,
        best_block_smooth_seed,
        weight_side=True,
        cat_transform=best_cat_transform,
    )
    h_x_smooth = _transformed_second_moment(
        activation_second_moment,
        best_d,
        best_perm,
        best_block_smooth_size,
        best_block_smooth_seed,
        best_cat_transform,
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
            best_cat_transform,
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
    # C23: full-64 weight refinement against the complete transformed
    # activation covariance (the same cov used for the group grams above).
    # Gate on use_quadratic so the Hessian is always available.
    # C35: per-width coverage -- narrow layers (<=1024 channels, cheap
    # 64-blocks) get a higher ratio; wide FFN projectors keep the global
    # budget so the official CPU-time envelope is respected.
    is_wide_layer = (
        in_features >= _WIDE_LAYER_MIN_DIM
        or out_features >= _WIDE_LAYER_MIN_DIM
    )
    full64_scope_enabled = (
        _WEIGHT_FULL64
        and use_quadratic
        and (
            not _WEIGHT_FULL64_WIDE_ONLY
            or is_wide_layer
        )
    )
    if full64_scope_enabled:
        if _WEIGHT_FULL64_DATA_DRIVEN_COVERAGE:
            full64_ratio = _WEIGHT_FULL64_DATA_DRIVEN_MAX_RATIO
        else:
            full64_ratio = (
                _WEIGHT_FULL64_MAX_RATIO_WIDE
                if is_wide_layer
                else _WEIGHT_FULL64_MAX_RATIO_NARROW
            )
        weight_params = _refine_weight_blocks64(
            weight_smooth, weight_params, gram, max_ratio=full64_ratio
        )
    # The cross-block prototype is retained as a dormant research hook, but
    # it must not add a second dequantization or alter the released candidate
    # while its flag is off.
    weight_hat_for_activation = None
    if _WEIGHT_CROSS64 and use_quadratic:
        weight_hat_for_activation = _dequantize_hif4(weight_params)
        weight_params = _refine_weight_blocks_cross64(
            weight_smooth, weight_params, gram
        )

    weight_hat = _dequantize_hif4(weight_params)
    if (
        _WEIGHT_CROSS64
        and _WEIGHT_CROSS64_PRESERVE_ACTIVATION_IMPORTANCE
        and weight_hat_for_activation is not None
    ):
        activation_weight_hat = weight_hat_for_activation
    else:
        activation_weight_hat = weight_hat
    activation_importance = _normalize_importance(
        activation_weight_hat.square().sum(dim=0), in_features
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
            transformed = _apply_cat64_rows(
                transformed, best_cat_transform, inverse=False
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
            use_group8 = True
            if (
                _ACTIVATION_QUADRATIC8_CALIBRATION_GATE
                and in_features <= _ACTIVATION_QUADRATIC8_GATE_MAX_FEATURES
            ):
                use_group8 = _activation8_refinement_is_safe(
                    activation_samples,
                    best_d,
                    best_perm,
                    best_block_smooth_size,
                    best_block_smooth_seed,
                    activation_importance,
                    activation_gram_state.to(weight.device),
                    group_gram8,
                    activation_ratio,
                    best_cat_transform,
                )
            if use_group8:
                activation_gram8_state = _cpu_state_tensor(group_gram8)

    activation_gram16_state = None
    if (
        _ACTIVATION_QUADRATIC16
        and in_features <= _ACTIVATION_QUADRATIC16_MAX_FEATURES
        and len(activation_samples) >= 1
    ):
        transformed_rows = torch.cat(
            [
                _linear_pair_transform(
                    s.to(device=weight.device, dtype=torch.float32),
                    best_d,
                    best_perm,
                    int(best_block_smooth_size),
                    int(best_block_smooth_seed),
                    weight_side=False,
                    cat_transform=best_cat_transform,
                )
                for s in activation_samples
            ],
            dim=0,
        )
        h_a = (
            transformed_rows.t() @ transformed_rows
        ) / float(transformed_rows.shape[0])
        activation_gram16_state = _cpu_state_tensor(
            _flat_group_gram16(h_a, in_features)
        )

    activation_state = {
        "smooth_inv": smooth_inv_state,
        "permutation": permutation_state,
        "block_smooth_size": int(best_block_smooth_size),
        "block_smooth_seed": int(best_block_smooth_seed),
        "importance": _cpu_state_tensor(activation_importance),
        "gram": activation_gram_state,
        "gram8": activation_gram8_state,
        "gram16": activation_gram16_state,
        "offsets": torch.tensor(_DYNAMIC_OFFSETS, dtype=torch.int8, device="cpu"),
        "error_threshold": _ACTIVATION_REFINE_ERROR_THRESHOLD,
        "accept_margin": _ACTIVATION_REFINE_ACCEPT_MARGIN,
        "max_refine_ratio": float(activation_ratio),
        "max_refine_blocks": _ACTIVATION_REFINE_MAX_BLOCKS,
        "in_features": int(in_features),
        "version": 4,
    }
    if best_cat_transform is not None:
        activation_state["cat_transform"] = _cpu_state_tensor(
            best_cat_transform
        )
    # C45: the online activation state is now frozen.  The remaining stages
    # can only choose a legal replacement for static Q(W); product-derived
    # tensors never enter the state above or the dynamic activation path.
    if (
        (_WEIGHT_PRODUCT_SELECTOR or _WEIGHT_HEADROOM)
        and (
            in_features >= _WIDE_LAYER_MIN_DIM
            or out_features >= _WIDE_LAYER_MIN_DIM
        )
        and in_features >= _WEIGHT_PRODUCT_SELECTOR_MIN_CHANNELS
    ):
        transformed_calibration = [
            _linear_pair_transform(
                sample.to(device=weight.device, dtype=torch.float32),
                best_d,
                best_perm,
                int(best_block_smooth_size),
                int(best_block_smooth_seed),
                weight_side=False,
                cat_transform=best_cat_transform,
            )
            for sample in activation_samples
        ]
        if _WEIGHT_HEADROOM and use_quadratic:
            headroom_params = _refine_weight_blocks64(
                weight_smooth,
                weight_params,
                gram,
                max_ratio=_WEIGHT_HEADROOM_MAX_RATIO,
                beam_offsets=_WEIGHT_HEADROOM_BEAM_OFFSETS,
            )
            weight_params = _select_static_weight_product_pool(
                weight_smooth,
                weight_params,
                headroom_params,
                transformed_calibration,
            )
        # The product selector consumes only these raw, transformed
        # calibration activations.  It does not call or mutate the online
        # activation quantizer, and it returns only a static Q(W) candidate.
        if (
            _WEIGHT_PRODUCT_SELECTOR
            and out_features <= _WEIGHT_PRODUCT_SELECTOR_MAX_DIM
        ):
            weight_params = _select_static_weight_product_candidate(
                weight_smooth,
                weight_params,
                transformed_calibration,
            )
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
        cat_transform=activation_state.get("cat_transform"),
        importance=activation_state["importance"],
        group_gram=activation_state.get("gram"),
        group_gram8=activation_state.get("gram8"),
        group_gram16=activation_state.get("gram16"),
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
    center_value: Optional[torch.Tensor] = None,
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
                k_full, kv_num_heads, head_dim, center_mode, center_value
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
            k_sample, kv_num_heads, head_dim, center_mode, center_value
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
    k_sac_sum_square = torch.zeros_like(k_sum_square)
    k_sac_peak_square = torch.zeros_like(k_sum_square)
    # C41: solve the quantization-aware K center once, before the statistics
    # loop, so that mode 4 has its own second moment / peak estimates.
    sac_center = None
    if _ATTN_SCALE_AWARE_CENTER and (
        _ATTN_SCALE_AWARE_CENTER_GQA or q_num_heads == kv_num_heads
    ):
        sac_pieces = []
        for sample in calib_qkv_list:
            k_dense = _dequantize_nvfp4_float32(sample["k"][0], sample["k"][1])
            sac_pieces.append(_sample_rows(k_dense, _ATTN_STATS_TOKENS))
        if sac_pieces:
            sac_center = _solve_k_center_scale_aware(
                torch.cat(sac_pieces, dim=0),
                kv_num_heads,
                head_dim,
                _ATTN_CENTER_ALTERNATIONS,
            )
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
            k_sac_sum_square = k_sac_sum_square.to(q.device)
            k_sac_peak_square = k_sac_peak_square.to(q.device)
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
        if sac_center is not None:
            k_sac_stats = _center_attention_k(
                k_stats.reshape(-1, kv_channels),
                kv_num_heads,
                head_dim,
                4,
                sac_center,
            ).reshape(-1, kv_num_heads, head_dim)
            k_sac_sum_square += k_sac_stats.square().sum(dim=0)
            k_sac_peak_square += k_sac_stats.abs().amax(dim=0).square()
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
    k_sac_second_moment = k_sac_sum_square / float(max(k_token_count, 1))
    k_sac_peak = torch.sqrt(k_sac_peak_square / float(max(sample_count, 1)))

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
            if center_mode == 4:
                if not _ATTN_SCALE_AWARE_CENTER:
                    continue
                if (
                    not _ATTN_SCALE_AWARE_CENTER_GQA
                    and q_num_heads != kv_num_heads
                ):
                    continue
            if center_mode in (2, 3):
                effective_second = k_mid_second_moment
                effective_peak = k_mid_peak
            elif center_mode == 4:
                effective_second = k_sac_second_moment
                effective_peak = k_sac_peak
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
                    sac_center if center_mode == 4 else None,
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

        if best_center_mode == 4:
            selected_k_peak = k_sac_peak
            selected_k_second = k_sac_second_moment
        elif best_center_mode in (2, 3):
            selected_k_peak = k_mid_peak
            selected_k_second = k_mid_second_moment
        else:
            selected_k_peak = k_peak
            selected_k_second = k_second_moment
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
                sac_center if best_center_mode == 4 else None,
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
                    sac_center if best_center_mode == 4 else None,
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
        if int(center_mode) == 4:
            eff_k_second = k_sac_second_moment
        elif int(center_mode) in (2, 3):
            eff_k_second = k_mid_second_moment
        else:
            eff_k_second = k_second_moment
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
                    sample,
                    kv_num_heads,
                    head_dim,
                    int(center_mode),
                    sac_center if int(center_mode) == 4 else None,
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
        # C41: only carry the center vector when mode 4 is actually selected,
        # so the state key set stays identical to the parent otherwise.
        if int(center_mode) == 4 and sac_center is not None:
            k_state["center_value"] = _cpu_state_tensor(sac_center)
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
        center_value=state.get("center_value"),
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
