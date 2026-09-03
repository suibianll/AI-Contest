# v162 侧向计划完成后的低复杂度算法扩展实施计划

> 状态：**ACTIVE**
>
> 创建：2026-09-03
>
> 激活：2026-09-03（用户指示切换；§13 执行顺序改为 Attention 优先）。移交自
> [`侧向隔离计划`](../../archive/plans/2026-09-03-v162-official-side-isolation-optimization-plan-superseded.md)：
> v165 timeout、v167 本地 REJECTED、v166 已官方提交待回传。激活时父版本：
> `P_L = v163（4587/202s；v166 回传 S_L > 4587 则更新）`、`P_A = v164（13945/204s）`。

## 1. 激活条件与边界

激活时必须在同一提交完成以下动作：

1. 当前活动计划中的 Linear rank-1、低复杂度 Gram Attention 及可能的组合候选均已有最终状态，
   或已明确取消；
2. 当前活动计划移入 `docs/superpowers/archive/plans/`；
3. 本文件移到 `docs/superpowers/plans/` 根目录，状态改为 `ACTIVE`；
4. 更新 `docs/superpowers/plans/README.md`、`docs/current-solution-status.md` 和根 README，确保
   根目录仍只有一份活动计划；
5. 根据届时的官方结果重新确定两侧父版本。若当前计划没有产生更优侧版本，则 Linear 父侧仍为
   v163（v160 Linear + standard Attention，`4587 / 202s`），Attention 父侧仍为 v164
   （standard Linear + v160 Attention，`13945 / 204s`）。

本计划不恢复已经关闭的 full64 多轮 sweep、Householder、Hadamard seed 搜索、A@W 邻域调参或
Cross-Gram64 per-call 动态精化。

## 2. v165 给出的设计约束

v165（standard Linear + v161 Attention，SHA
`033E85D5DAF1A820BACDB14F9E35183C485E8DD489D118899A1AE3CB491D8C1D`）官方
`timeout（>300s，无分数）`。v164 同侧对照是 `13945 / 204s`，所以 Cross-Gram64 per-call
动态精化的官方增量成本下界约为 `>96s`。timeout 没有提供精度结论，但明确给出以下实现约束：

- Attention 动态 API 不再执行完整 `64×64` Gram contraction；
- 不运行随 token 数增长的候选循环、多轮 coordinate sweep 或逐 block Python 调度；
- 新 Attention 方法只能把复杂计算放在 calibration，并把结果编译成逐元素、逐 head 或一次
  固定 hierarchy encode；
- 每个候选必须继续输出标准合法 HiF4 五字段，不能增加 decoder、side channel 或 attention
  内部 hook。

## 3. 统一父版本、代码入口与归因方式

### 3.1 候选构造

每次激活一个工作包时，先确定当前官方最好的单侧父版本：

```text
P_L = 当前官方最好 Linear + standard Attention
P_A = standard Linear + 当前官方最好 Attention
```

新 Linear 候选从 `P_L` 复制，只改以下两个 API 及其直接 helper：

```text
hif4_calibration_and_quantize_weight
hif4_dynamic_quantize_activation
```

四个 Attention API 必须与 v162 standard Attention 逐位一致。

新 Attention 候选从 `P_A` 复制，只改以下四个 API 及其直接 helper：

```text
hif4_calibration_attention
hif4_dynamic_quantize_q
hif4_dynamic_quantize_k
hif4_dynamic_quantize_v
```

两个 Linear API 必须与 v162 standard Linear 逐位一致。正式提交仍是单文件、自包含的
`solution.py`，不得 import 归档源码。

### 3.2 允许复用的现有内部入口

实现优先复用当前代码中的稳定函数，而不是另写第二套 codec：

```text
_dequantize_nvfp4_float32   NVFP4 -> FP32
_dense_to_hif4             标准 HiF4 编码入口
_solve_exact_hierarchy     固定 E6M2 scale 下精确选择 lv2/lv3/mantissa
_pack_hif4_params          生成合法五字段
_dequantize_hif4           五字段回解码
_nvfp4_to_hif4             动态 Q/K/V/Activation 公共入口
_linear_pair_transform     Linear 等价变换公共入口
_attention_forward         本地真实 attention 输出
```

新 helper 放在所依赖函数之后、六 API 之前；不为单次操作创建类或通用框架。

### 3.3 官方差分

设候选官方结果为 `S_new / T_new`，当前同侧父版本为 `S_parent / T_parent`：

```text
step_gain       = S_new - S_parent
side_contrib    = S_new - 1001
Linear ratio    = (S_new - 4587) / 3586       # 仅报告 Linear 候选
Attention ratio = (S_new - 13945) / 12944     # 仅报告 Attention 候选
```

如果当前计划已经更新了父侧锚点，同时报告相对新父侧的 `step_gain`，但保留以上 v160 固定口径以
便跨版本比较。timeout 时不计算任何精度比例。

## 4. Attention 工作包 A1：解析 logits 增益校正

### 4.1 假设

量化后的 Q/K logits 可能产生稳定的乘性收缩或膨胀。此前 reciprocal scaling 保持连续 QK
不变，只重新分配两侧动态范围；A1 则允许一个低自由度的乘性校正，使量化 logits 更接近浮点
logits。它不需要 Gram state 或动态候选搜索。

### 4.2 固定数学规则

对每个 KV head 及其共享的 GQA query-head group，在两个 calibration folds 分别计算：

```text
L_f  = row_center(Q_f K_f^T / sqrt(d))
Lq_f = row_center(Qhat_f Khat_f^T / sqrt(d))

raw_gamma_f = sum(Lq_f * L_f) / (sum(Lq_f^2) + eps)
raw_gamma_f = clamp(raw_gamma_f, 0.5, 2.0)

gamma = exp(0.5 * median_f(log(raw_gamma_f)))
g_q = sqrt(gamma)
g_k = sqrt(gamma)
```

`row_center` 去掉 softmax 不可辨识的逐 query 常数。`0.5` 是预注册的 log-domain 向 1 收缩系数，
不做搜索。GQA 中一个 KV head 的 `gamma` 重复到其所属的全部 query heads。

### 4.3 具体代码修改

新增常量：

```python
_ATTN_LOGIT_GAIN = True
_ATTN_LOGIT_GAIN_MIN = 0.5
_ATTN_LOGIT_GAIN_MAX = 2.0
_ATTN_LOGIT_GAIN_LOG_SHRINK = 0.5
_ATTN_LOGIT_GAIN_TOKENS = 128
```

新增 helper：

```python
def _fit_attention_logit_gain(
    q_pairs, k_pairs, q_state, k_state,
    q_num_heads, kv_num_heads, head_dim,
) -> tuple[torch.Tensor, dict[str, int]]:
    ...
```

算法流程：

1. 用已经选定的父 `q_state/k_state` 调 `_nvfp4_to_hif4`，再用 `_dequantize_hif4` 得到
   `Qhat/Khat`；
2. 按 calibration list 偶数/奇数索引形成两个 folds；
3. reshape 为 `[tokens, heads, head_dim]`，GQA 下把 K repeat 到对应 Q heads；
4. 每个样本固定只取前 128 tokens；计算 causal 有效区域的 `L/Lq`，每个 query row 只在其
   合法 prefix keys 上去均值，mask 区域不进入统计；
5. 对属于同一 KV head 的 Q heads 合并 numerator/denominator；
6. 按 §4.2 得到 `gamma`；若 numerator、denominator 非有限，则该 head 固定 `gamma=1`；
7. 返回 CPU `float32[kv_num_heads]` 和有效 head 计数。

在 `hif4_calibration_attention` 已完成父 `q_state/k_state` 选择、即将 return 前：

```python
gain, stats = _fit_attention_logit_gain(...)
q_gain = gain.repeat_interleave(q_num_heads // kv_num_heads)
q_gain = q_gain[:, None].expand(-1, head_dim).reshape(-1).sqrt()
k_gain = gain[:, None].expand(-1, head_dim).reshape(-1).sqrt()

q_base = ones if q_state["multiplier"] is None else q_state["multiplier"]
k_base = ones if k_state["multiplier"] is None else k_state["multiplier"]
q_state["multiplier"] = cpu(q_base * q_gain)
k_state["multiplier"] = cpu(k_base * k_gain)
q_state["logit_gain"] = cpu(gain)
k_state["logit_gain"] = cpu(gain)
```

动态 Q/K API 不增加新矩阵运算：校正已折叠进现有 `multiplier` 路径。`logit_gain` 只用于审计，
不在动态 API 重复乘。

### 4.4 复杂度和记录

```text
calibration: O(F * H_q * T^2 * d)   # 复用短 calibration scorer
dynamic:     O(TD)                   # 已有 multiplier 广播乘法
state:       O(H_k)
```

必须记录：有效 head 数、`gamma` min/median/max、Q/K state 是否变化、浮点/父量化/校正量化 logits
slope，以及 Q/K/QK/QKV/probability/output delta。

## 5. Attention 工作包 A2：V 输出偏差质心补偿

### 5.1 假设

对一个 attention row，权重和为 1。因此给同一 KV head 的所有 V token 加固定向量 `b_h`，会给
该 head 的 attention output 加近似相同的 `b_h`。如果父 V 编码存在跨 token 稳定的输出偏差，
可以在 V 量化前用一个小向量抵消；动态成本只是一遍广播加法。

### 5.2 固定数学规则

保持父 Q/K 不变。每个 calibration fold 计算：

```text
O_ref    = Attention(Q, K, V)
O_parent = Attention(Qhat, Khat, Vhat_parent)

b_f,h = mean_query_and_group_heads(O_ref - O_parent)
b_h   = 0.5 * coordinatewise_median_f(b_f,h)
```

GQA 中把属于同一 KV head 的 query heads 合并。每个 calibration 样本固定只取前 128 tokens；
固定收缩系数为 `0.5`，不搜索。

### 5.3 具体代码修改

新增常量和 helper：

```python
_ATTN_V_BIAS = True
_ATTN_V_BIAS_SHRINK = 0.5
_ATTN_V_BIAS_TOKENS = 128

def _fit_attention_v_bias(
    q_pairs, k_pairs, v_pairs,
    q_state, k_state, v_state,
    q_num_heads, kv_num_heads, head_dim,
) -> tuple[torch.Tensor, dict[str, float]]:
    ...
```

给 `_nvfp4_to_hif4` 增加一个默认关闭参数：

```python
additive_bias: Optional[torch.Tensor] = None
```

在 NVFP4 解码后、任何 multiplier/permutation/rotation 之前执行：

```python
if additive_bias is not None:
    bias = additive_bias.to(dense.device, torch.float32).reshape(-1)
    dense.add_(bias.reshape(*([1] * (dense.ndim - 1)), channels))
```

`hif4_calibration_attention` 在父 `v_state` 完成后调用 `_fit_attention_v_bias`，保存：

```python
v_state["additive_bias"] = cpu(b.reshape(-1))
v_state["bias_version"] = 1
```

`hif4_dynamic_quantize_v` 只增加：

```python
additive_bias=state.get("additive_bias")
```

Q/K API 不改。候选不对量化结果做事后高精度加法，最终输出仍是普通 HiF4 参数。

### 5.4 复杂度和记录

```text
calibration: O(F * H_q * T^2 * d)
dynamic:     O(TD)
state:       O(H_k * d)
```

必须记录：`b` norm/max、父 V 与 bias-V 的普通 MSE、attention output bias、V-only 与 QKV delta。

## 6. Attention 工作包 A3：动态 scale 搜索的静态策略编译

### 6.1 假设

当前动态编码对每个样本尝试多个 E6M2 offset，并执行 refine。v165 表明官方硬件不适合 per-call
候选计算。A3 在 calibration 阶段为 Q/K/V 各选一个 layer-global offset，动态阶段只进行一次
scale/hierarchy 求解。这既是独立编码机制，也是明确的降复杂度方案。

### 6.2 固定候选和选择顺序

固定候选：

```text
offsets = (-1, 0, 1, 2, 3)
```

不做笛卡尔积。按 `Q -> K -> V` 顺序选择，每一步冻结前一步 winner、其余 operand 保持父编码；
每个 offset 用两个 folds、每个样本前 128 tokens 的真实 attention output MSE 计算：

```text
score(offset) = mean(fold_loss) + 0.25 * max(fold_loss)
```

相同分数选择绝对值更小、再选择数值更小的 offset。`0.25` 固定，不调整。

### 6.3 具体代码修改

给 `_dense_to_hif4` 和 `_nvfp4_to_hif4` 增加：

```python
fixed_scale_offset: Optional[int] = None
```

在 `_dense_to_hif4` 得到 `standard_code` 后增加独立分支：

```python
if fixed_scale_offset is not None:
    code = clamp_e6m2_code(standard_code + fixed_scale_offset)
    scale = _e6m2_decode(code)
    loss, lv2, lv3, mant = _solve_exact_hierarchy(
        x_abs, scale, importance, sign, group_gram
    )
    return _pack_hif4_params(prefix, blocks, scale, lv2, lv3, sign, mant)
```

该分支不进入 `search_offsets`、edge extension 或后续 coordinate refine。新增校准 helper：

```python
def _compile_attention_fixed_offsets(...):
    # 返回 q_offset, k_offset, v_offset 以及两个 fold 的损失表
```

保存到三个 state：

```python
state["fixed_scale_offset"] = int(winner)
state["offsets"] = empty int8 CPU tensor
state["max_refine_ratio"] = 0.0
```

三个动态 API 把 `fixed_scale_offset` 传给 `_nvfp4_to_hif4`。不得同时保留原多 offset refine，避免
算法和时间归因混杂。

### 6.4 复杂度和记录

```text
calibration: O(15 * F * attention_short_panel)
dynamic:     O(TD), one hierarchy solve
state:       O(1) per operand
```

必须记录：Q/K/V winner、每个候选的两折 loss、动态 `_dense_to_hif4` 调用次数、相对父版本减少的
候选数和 API 时间。官方是否提升由真实分数决定，不因本地精度轻微下降自动取消首次提交。

## 7. Attention 工作包 A4：矩匹配 mantissa 阈值

### 7.1 假设

标准 nearest rounding 的阈值固定为 `0.5`，但低比特长尾分布可能产生稳定的幅值偏差。A4
不搜索运行时候选，而是在 calibration 编译一个 Q、K、V 各自的标量阈值，使归一化绝对值编码
残差均值接近零；使用绝对值后，二分目标随 threshold 单调不增。

### 7.2 固定求解

定义：

```text
code_tau(z) = clamp(floor(z + 1 - tau), 0, 7)
tau in [0.25, 0.75]
```

对每个 operand/layer、每个 calibration fold，用固定 8 次二分求解：

```text
mean(abs(Q_tau(x)) - abs(x)) = 0
```

最终阈值：

```text
tau = 0.5 + 0.5 * (median_f(tau_f) - 0.5)
```

不按 head、长度、layer role 再细分，不调整二分次数或区间。

### 7.3 具体代码修改

新增 `_round_mantissa_threshold(x_abs, local_scale, threshold)`，并给
`_solve_exact_hierarchy`、`_dense_to_hif4`、`_nvfp4_to_hif4` 增加默认 `None` 的
`rounding_threshold` 参数。只有 `group_gram is None` 时使用 threshold；若某父路径传入
`group_gram`，该 operand 保持父 `_adaround_mantissa`，不能悄悄改变两种算法。

`hif4_calibration_attention` 保存：

```python
q_state["rounding_threshold"] = float(q_tau)
k_state["rounding_threshold"] = float(k_tau)
v_state["rounding_threshold"] = float(v_tau)
```

三个动态 API 原样透传。在线只用一次 `floor` 替代 `round`，没有候选循环。

### 7.4 复杂度和记录

```text
calibration: O(8FTD) per operand
dynamic:     O(TD), same asymptotic as parent
state:       3 scalars
```

必须记录三个 fold threshold、部署 threshold、残差均值、clipping rate 和真实 attention delta。

## 8. Linear 工作包 L1：WUSH 与现有 CAT-64 的公式审计及移植

### 8.1 目的

WUSH 是数据感知非正交 block transform，但当前代码已有基于 activation covariance 与 weight
Gram 的 CAT-64。先判断二者是否只是不同记号；数学或数值等价时不建立版本、不提交重复算法。

### 8.2 审计流程

1. 从 WUSH 官方实现只抄录 transform 数学，不复制训练、模型或 kernel 框架；
2. 新增临时纯函数 `_wush64_blocks(x_second_moment, w_second_moment)`，固定 block `64`、固定
   canonical Hadamard、无 seed；
3. 对 v160 calibration 的每个 64-block，分别生成 determinant-normalized `R_wush` 与现有
   `_cat64_blocks(..., strength=0.25)` 的 `R_cat`；
4. 消除整体标量和可能的转置约定后计算：

```text
matrix_delta = ||R_wush - R_cat||F / ||R_cat||F
output_delta = ||X R_wush (W R_wush^-T)^T - XW^T|| / ||XW^T||
codec_delta  = deployed_loss(R_wush) - deployed_loss(R_cat)
```

5. 若所有审计 block 的 `matrix_delta <= 1e-6` 或最终 HiF4 五字段逐位相同，记录
   `DUPLICATE / NO VERSION`，立即进入 L2；
6. 只有变换和编码输出均不同，才把临时函数转为候选实现。

### 8.3 非重复时的代码修改

新增正式 helper：

```python
_wush64_blocks(...)
_apply_wush64_rows(dense, transforms, inverse=False)
```

在 `hif4_calibration_and_quantize_weight` 中，最终 Smooth/permutation 后计算一次 WUSH transform；
用它替换 CAT transform，不与 CAT 叠加。静态 Weight 使用 `inverse=True`，Activation state 保存
CPU `wush_transform`，动态 Activation 使用 `inverse=False`。连续域必须验证：

```text
A' = A R^T
W' = W R^-1
A' W'^T = A W^T
```

`_linear_pair_transform` 增加一个互斥的 `wush_transform` 参数；CAT 与 WUSH 同时非空直接报实现
错误。强度、block、Hadamard seed 均固定，不做候选网格。

### 8.4 复杂度和记录

```text
calibration: O((D/64) * 64^3 + ND)
weight transform: O(R_o * D * 64)
dynamic: O(T * D * 64)
state: O((D/64) * 64^2)
```

必须记录等价审计表、condition number、连续输出误差、变换接受 block 数，以及相对 CAT 的
Weight-only/Activation-only/Both interaction。

## 9. Linear 工作包 L2：HiF4 层级约束 Babai 解码

### 9.1 目标函数

在最终部署变换坐标系计算：

```text
H = X_cal^T X_cal / N
min_q (w - q)^T H (w - q)
```

对一个 64-block，先沿用父编码的合法 `scale_factor/lv2/lv3`。每个坐标的量化步长为：

```text
step_j = 0.25 * scale_factor * lv2_j * lv3_j
q_j = step_j * z_j, z_j in {-7, ..., 7}
```

### 9.2 固定 Babai 流程

1. 固定 `B=64`；从最终变换后的 calibration activation 构造 `H_b`；
2. damping 固定为 `0.01 * mean(diag(H_b))`；
3. 固定处理顺序为 damped Hessian 对角线降序；
4. 对 `Bmat = chol(H_b)^T @ diag(step)` 做 batched QR；
5. 对每个 Weight row 从最后一维向前执行 nearest-plane rounding，code clip 到 `[-7,7]`；
6. 若发生 clipping，只允许一次 no-clipping rescale：把顶层 E6M2 scale 提到能容纳未裁剪 code
   的最小合法 code，用 `_solve_exact_hierarchy` 重算 lv2/lv3，再执行一次 Babai；
7. 计算 parent 与 Babai 的同一个 block-Hessian loss，逐 block 保留较小者；
8. 原子写回该 block 的 `scale_factor/lv2/lv3/sign/mant`，不得只写 mantissa。

不运行第二 sweep，不搜索 damping/order/scale offset。

### 9.3 具体代码修改

新增：

```python
_WEIGHT_HIF4_BABAI = True
_WEIGHT_HIF4_BABAI_BLOCK = 64
_WEIGHT_HIF4_BABAI_DAMPING = 0.01

def _babai_weight_blocks64(
    dense_weight, parent_params, activation_hessian
) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    ...
```

在 `hif4_calibration_and_quantize_weight` 中，所有等价变换确定、父 `weight_params` 已生成之后调用
一次 `_babai_weight_blocks64`。它只替换 `weight_params`；`activation_state` 和
`hif4_dynamic_quantize_activation` 与父版本逐位一致。

### 9.4 复杂度和记录

```text
Hessian blocks: O(N * D * 64)
factorization:  O((D/64) * 64^3)
row decoding:   O(R_o * D * 64)
dynamic:        zero added work
state:          zero added state
```

必须记录 attempted/accepted/rescaled/clipped block 数、parent/Babai Hessian loss、五字段变化率、
W-only/A-only/Both/interaction 和各 role 分布。

## 10. Linear 工作包 L3：固定宽度 HiF4 Trellis/VQ

### 10.1 与 L2 的区别

Babai 每一步只保留一个 partial solution；L3 保留固定 8 条路径，允许早期 rounding 决策被后续
相关坐标纠正。它仍只输出标准 HiF4 code，不引入 QTIP/GPTVQ 的自定义 decoder。

### 10.2 固定搜索

每个 64-block 使用父版本已经确定的合法 scale/lv2/lv3：

```text
stage count = 16                 # 每 stage 连续 4 坐标
beam width  = 8
branch      = {nearest-1, nearest, nearest+1}
```

流程：

1. 复用 L2 的 damped `H_b` 和固定对角降序，但把顺序按连续四坐标打包；
2. 初始化一个空路径；
3. 每个 stage 对当前 8 条路径生成最多 `8 * 3^4` 个合法 child；
4. 用已确定坐标的精确 quadratic partial loss 加未确定坐标的对角下界排序；
5. 固定保留最小 8 条；相同 loss 按 code 的字典序决定，保证确定性；
6. 第 16 stage 后与 parent 完整 Hessian loss 比较，逐 block 选择较小者；
7. 只写回 sign/mant；scale/lv2/lv3 保持父值，使 L3 与 L2 的层级 rescale 机制隔离。

### 10.3 具体代码修改

新增 `_trellis_weight_blocks64(...)`，调用位置与 L2 相同，但 L2 flag 必须关闭。使用 tensorized
`topk`，不按 row/block 建 Python 候选循环；Python 只允许固定 16 个 stage 循环。

### 10.4 复杂度和记录

```text
calibration: O(R_o * (D/64) * 16 * 8 * 81)
dynamic:     zero added work
state:       zero added state
```

必须记录每 stage survivor loss、最终 accepted block 数、相对 Babai 可达但不是同一候选的比例。
L2 官方负向不会取消 L3，因为 L3 是不同解码算法；L3 内部不得再调整 beam 或 branch。

## 11. Linear 工作包 L4：Kronecker 压缩的解析 CAT

### 11.1 目标

完整 64×64 非正交变换表达力高，但动态乘法和 state 较重。L4 把解析 CAT target 投影成两个
8×8 因子，形成固定：

```text
R = R_left tensor-product R_right
```

它不是 rank 扫描，也不是 Householder；每个 64-block 始终是 `8×8` reshape。

### 11.2 固定闭式投影

1. 用 `_cat64_blocks(..., strength=0.25)` 得到 SPD target `M`；
2. 求 `G = logm(M)`；
3. reshape `G` 为 `[8,8,8,8]`，用 partial trace 得到 Kronecker-sum 因子：

```text
A = partial_trace_right(G) / 8
B = partial_trace_left(G) / 8
A <- A - trace(A)/8 * I
B <- B - trace(B)/8 * I
R_left  = expm(A)
R_right = expm(B)
```

4. 用 determinant normalization 消除整体尺度；
5. 不拟合 strength，不搜索因子形状。

### 11.3 具体代码修改

新增：

```python
_kron64_factors(cat_target) -> left, right
_apply_kron64_rows(dense, left, right, inverse=False)
```

应用时把每个连续 64 向量 reshape 为 `8×8`，依次左乘/右乘两个因子；Weight 使用精确逆因子，
Activation 使用正因子。`_linear_pair_transform` 增加互斥的 `kron_left/kron_right` 参数。

`activation_state` 新增两个 CPU tensor：

```python
"kron_left":  [num_blocks, 8, 8]
"kron_right": [num_blocks, 8, 8]
```

动态 Activation 在 permutation/Hadamard 后、HiF4 encode 前调用一次 `_apply_kron64_rows`。

### 11.4 复杂度和记录

```text
calibration eig/log/exp: O((D/64) * 64^3)
weight transform:        O(R_o * D * 16)
dynamic transform:       O(T * D * 16)
state:                   O((D/64) * 128)
```

与 dense CAT 的动态 `O(TD64)` 相比，Kronecker 常数约降为两次 8×8 乘。必须记录 CAT target
投影误差、连续输出误差、condition number、动态时间和两侧 interaction。

## 12. 每个工作包的固定执行流程

每个 A/L 工作包严格走以下步骤，不把多个工作包一次合并：

1. **父版本锁定**：复制当前官方最好单侧源码，记录父 SHA 和 standard control SHA；
2. **实现**：只增加本工作包常量/helper/state/API 接线；
3. **静态检查**：脱离仓库导入六 API，运行 `reference_hif4.validate_state` 和
   `validate_hif4_params`；
4. **机制 reachability**：记录 attempted/accepted、state key、参数范围和至少一个真实输出变化；
5. **control**：非目标侧与 v162 standard case-by-case 逐位一致；
6. **compact 配对**：使用同 cache、同 device 和 `--baseline-json`；保存 JSON/report；
7. **default 单侧审计**：记录 mean/median/q25/worst/负 case、分 role/head/length、API/wall；
8. **跨模型封存检查**：固定 GPT-2，再用 Pythia-160m 或 OPT-125m；不得根据结果反调参数；
9. **官方一次提交**：只要接口/状态/有限性/control/reachability 正常，且跨模型没有整体结构性
   反向，即提交一次；不设置 `+300` 或本地均值正向门槛；
10. **登记**：更新 result、独立官方日志、活动计划、状态文档和 `solutions/README.md`；
11. **父侧更新**：`S_new > S_parent` 且未 timeout 时成为新单侧父版本，否则下一个不同工作包
    从旧父侧继续；
12. **提交仓库**：`git diff --check`、commit、push、核验工作区。

推荐命令模板：

```powershell
# Attention
.venv\Scripts\python.exe evaluator/official_eval.py --solution <candidate.py> --attention-only --compact-panel --cache-mode read --nvfp4-cache-mode auto --capture-device cuda --algorithm-device cuda --baseline-json <parent.json>

# Linear
.venv\Scripts\python.exe evaluator/official_eval.py --solution <candidate.py> --linear-only --compact-panel --cache-mode read --nvfp4-cache-mode auto --capture-device cuda --algorithm-device cuda --baseline-json <parent.json>
```

## 13. 固定执行顺序

激活后顺序如下：

```text
A1 logits gain
-> A2 V bias
-> A3 fixed-offset compiler
-> A4 rounding threshold
-> L1 WUSH/CAT audit（等价则不分配版本）
-> L2 Babai
-> L3 trellis
-> L4 Kronecker CAT
```

排序依据（2026-09-03 激活时按用户指示改为 Attention 优先）：官方两侧贡献
`12944:3586 ≈ 3.61:1`，Attention 候选的单次官方提交期望回报更高；且 Linear 侧 v166
rank-1 已在官方通道中待回传，Linear 名额暂有在途测量。侧内相对顺序不变：A3 优先于
A4，因为它同时降低动态候选复杂度；L1 审计先行确认 WUSH 与既有 CAT-64 的关系；
L4 最后执行，因为与既有 CAT 家族重叠最大。

两侧出现新的官方最好版本后，才构造一个组合版本，实测：

```text
interaction = S_LA - S_L - S_A + 1001
closure     = (S_LA - 17532) / 4233
gap         = 21765 - S_LA
```

官方时间不得相加预测。

## 14. 终止与失败处理

- `ERROR`：导入失败、state 非法、NaN/Inf、control 改变或机制死分支；修实现但不改数学参数；
- `TIMEOUT`：不计算精度；同一机制只允许一次保持输出目标的复杂度重构；
- `REJECTED`：官方有分数且不高于同侧父版本，或跨模型出现整体结构性反向；不做邻域调参；
- `RETAINED`：官方高于父版本且 `<300s`；成为新单侧父版本；
- `DUPLICATE`：公式或输出与已实现机制等价；不分配版本、不提交官方。

所有 A1--A4 和 L1--L4 均获得上述状态后，本计划完成。

## 15. 文献依据

- WUSH：<https://arxiv.org/abs/2512.00956>
- GPTQ/Babai 几何：<https://arxiv.org/abs/2507.18553>
- QTIP trellis：<https://arxiv.org/abs/2406.11235>
- GPTVQ：<https://github.com/Qualcomm-AI-research/gptvq>
- FlatQuant：<https://arxiv.org/abs/2410.09426>
- AffineQuant：<https://arxiv.org/abs/2403.12544>
- H-Scale：<https://arxiv.org/abs/2608.28113>
- KIVI：<https://arxiv.org/abs/2402.02750>
- Quantized Keys Steal Attention：<https://arxiv.org/abs/2605.26266>
- KVLinC：<https://arxiv.org/abs/2510.05373>

这些论文只提供数学来源；论文分数、kernel 加速或模型结论不用于预测本竞赛官方结果。

## 16. 执行记录

- **2026-09-03 激活**（用户指示，§13 改为 Attention 优先）：侧向隔离计划归档移交；
  v165 官方 timeout、v167 低秩 Gram 码本本地 REJECTED（真实 QK 交叉 Gram 高秩，
  rank-2 耦合破坏深层哨兵，λ=0 消融与父版本逐位一致证明实现正确）、v166 rank-1
  Linear 已官方提交待回传。父版本：`P_L = v163`（若 v166 回传 `S_L > 4587` 则更新
  并在此登记）、`P_A = v164`。下一动作：A1 解析 logits 增益校正（从 P_A = v164 构造）。
