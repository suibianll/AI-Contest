# HiF4 Linear 22000+ 分完整优化计划

日期：2026-08-29  
状态：Active  
目标：在新版官方面板上保持 `22000+`，从本地归档冠军 `22557 / 217.2s` 继续逼近外部参考 `24153 / 239s`
主攻方向：Linear；Attention 只保留已验证的 C41b 增益，不在本计划中继续扩张  
官方硬约束：六个 API 不变、HiF4 五字段合法、总时间严格 `<420s`（7 分钟）

本地评测重构：默认以 Qwen2.5-0.5B（GQA/RoPE/SwiGLU）为主模型，将冻结语料
的 Linear/Attention 平均 case gain 投影到 **250/200 固定面板**；GPT-2、OPT、
Pythia 只作软 guardrail。原始 `official_flow_total` 仍保留作回溯，不再因模型
层数和本地窗口数直接累加主排序。

> **官方口径更新（2026-08-29）**：评测集扩大为 250 个 Linear case 与 200 个
> Attention case，分数和时间均会高于旧口径。用户确认 v031/C39-FW 为
> `21864 / 161.3s`、v034/C41b 为 `21864 / 159.4s`、v051/C47b 为
> `22451 / 234s`、v066/C66 为 `22557 / 217.2s`；[`youxilee/hif4`](https://github.com/youxilee/hif4) 的外部
> 参考结果为 `24153 / 239s`。旧锚点只保留作历史对照。

## 0. 执行摘要

当前算法已经把标准 HiF4、局部 scale 搜索、4/8/16 二阶精修和部分 FULL64 GPTQ
推到较高完成度。继续扩大 coverage、增加 offset 或增加坐标下降 sweep，只能得到边际
收益，无法承担旧口径 `14613 -> 22000+` 所需的结构级提升；新版面板下本地
`22557` 已越过 22000，较 v051 再提升 `106` 分并减少 `16.8s`；后续重点是逼近
外部参考 `24153`，当前差距为 `1596` 分和 `21.8s`。

本计划选择唯一高潜力主线：

```text
C41b 合规父版本
  -> C43 HiF4-aligned CAT-64 等价变换
  -> C44 CAT 坐标系下的 HiF4 MR-GPTQ
  -> C45 A@W 驱动的静态 Weight headroom / clipping 选择
  -> C46 解析 CAT 初始化的小步学习式 refinement
  -> 官方提交校准 22000+ 是否达到
```

核心判断：

1. **最高潜力来自 alignment，而不是继续压局部 reconstruction error。** 当前代码已经
   处理 concentration、SmoothQuant 和局部 Hessian，却没有显式优化 Weight 与 Activation
   主变化方向的 alignment。
2. **CAT-64 是第一优先级。** CAT 从 `A^T A` 与 `W^T W` 解析构造可逆变换，论文在
   `gate_proj/down_proj` 等困难层观察到最高约 `10 dB` 的 alignment/SQNR 提升，量级接近
   同时增加 Weight 与 Activation 约 2 bit。
3. **MR-GPTQ 必须在 CAT 之后。** FP4/microscaling 的共享 scale 会改变传统 rotation、
   act-order 和 GPTQ 的收益关系；应先按 HiF4 64-group 固定 scale/hierarchy，再临时按
   activation importance 排序做 GPTQ，最后还原坐标。
4. **`A@W` 应当大胆但低自由度地使用。** 它可以合法选择静态 `Q(W)`；不应用于选择
   CAT、Activation scale、rotation、coverage 或任何 `activation_state`。用 `A@W` 在每块
   少量离散 Weight 候选中选择，泛化风险远低于拟合完整输出残差。
5. **不采用过严门限。** calibration 内使用软均值/尾部混合目标，不要求每个 fold、每个
   模型、每个层都严格正向。只有合规、合法性、非有限数值和官方 `<420s` 是硬门。

### 0.1 新版结果与外部差距（2026-08-29）

用户随后确认 v066/C66 为 **`22557 / 217.2s`**，已超过 v051/C47b 的
`22451 / 234s`，但仍比外部 [`youxilee/hif4`](https://github.com/youxilee/hif4)
的用户提供结果 `24153 / 239s` 低 `1596` 分。外部最新公开实现的最大结构差距是
v2.6 的 64-channel X/W 联合残差补偿（校准期只更新静态 `Q(W)`，Gauss--Seidel
最多 3 轮）；C70 的直接移植在 GPT-2 small 上显著提升，但在 OPT/Qwen 代理回退，
说明它必须和父版本、跨 fold 软选择一起验证，不能直接全量打开。
官方新增的两个用例呈现千问 30B 特征，故后续压力测试必须覆盖宽 FFN down-proj
与 GQA，而不能只用 Qwen2.5-0.5B 代理。完整审计及 C70/C71 的失败消融见
[`2026-08-29-external-hif4-gap-analysis.md`](../../../logs/candidates/2026-08-29-external-hif4-gap-analysis.md)。

## 1. 当前基线、证据与问题定义

### 1.1 当前有效父版本

官方已确认的本地父锚点是 C66；当前根是其后仅做本地代理微调的 C69。后续
外部差距实验应从 C69 分支，并保留 C66 的官方 SHA/分数记录：

```text
solutions/20260829_v066_c66-activation-ratio100_scoreNA_timeNA/solution.py
solutions/20260829_v069_c69-activation-gram8-ratio12_scoreNA_timeNA/solution.py
```

C66 的新版官方结果为 `22557 / 217.2s`；C69 没有新的官方分数，五模型本地
official-flow proxy 为 `1044.706838`。直接移植外部联合补偿的 C70、以及 proj
最终量化器排名的 C71 均已完成隔离并拒绝，不能把它们当作当前父版本。

历史 C41b 结果：

| 指标 | C39 | C41b | 增量 |
|---|---:|---:|---:|
| 五模型 official-flow total | 996.745557 | 997.221971 | +0.476414 |
| Linear | 逐位相同 | 逐位相同 | 0 |
| Attention | 仅 MHA K-center 改善 | 改善 | +0.476414 |
| 官方锚点（旧面板） | 14613 / 159.2s | — | 历史 |
| v031 / C39-FW（250/200 新面板） | **21864 / 161.3s** | — | 用户确认 |
| v034 / C41b（250/200 新面板） | **21864 / 159.4s** | — | 用户确认 |
| v051 / C47b（250/200 新面板） | 22451 / 234s | — | 前一本地冠军 |
| v066 / C66（250/200 新面板） | **22557 / 217.2s** | — | 官方确认的本地冠军 |
| 外部 `youxilee/hif4` | **24153 / 239s** | — | 参考实现 |

历史产品补偿实验在 GPT-2 small 正向，却在 GPT-2 medium、OPT、Qwen 上反向，不能
作为父版本：

| 实验 | GPT-2 small Linear Δ | GPT-2 medium Linear Δ | Qwen Linear Δ | 结论 |
|---|---:|---:|---:|---|
| C42c 产品补偿 | +1.2684 | -3.6798 | -6.0287 | 高自由度 calibration 过拟合 |
| C42d 共识补偿 | +1.1451 | -3.1044 | -6.3825 | 文档内共识仍不足 |

因此 C70/C71 之后的 Phase 0 已完成：根已恢复 C69，产品补偿和不稳定外部候选均
移出 active path；不得把失败候选叠加到下一轮。

### 1.2 为什么 Linear 是主战场

五模型本地 official-flow 中，Linear 约占总分的 80% 以上；Attention 的 C41b 总增量只有
约 `0.048%`。旧面板下 `14613 -> 22000` 需要结构级 Linear 增量；新版面板的
v066/C66 已达到 `22557`，后续应以逼近外部 `24153` 为目标，不依赖 Attention
微调或 offset 网格扩张。

旧本地口径只作为工程量级参考：22000 大约要求 Linear mean 从约 `0.535` 推到 `0.68`
附近。由于 C38/C40 出现过本地正向、官方反向，该映射不得用于伪造官方分数，只用作
“是否达到结构级改善”的检查点。

### 1.3 已验证失败路径

以下路径不得作为本计划主线重复：

- 全层或高自由度 `A@W` 残差补偿；
- Cross64/Block-LDLQ 跨块条件重求解；
- 全局 R64 随机 Hadamard seed 扫描；
- 仅扩大 FULL64 coverage；
- 仅增加 dynamic/weight offset；
- 直接全量移植外部 v2.6 joint refine（C70）或 proj 最终量化器排名（C71），
  未经过当前父版本的跨 fold 稳定性审计；
- 要求所有 calibration fold、模型和层逐一不退化的过严 gate。

### 1.4 外部差距的修订优先级

当前 CAT、FULL64、静态 `A@W` Q(W) 和 Gram-8 已在 C69 中存在，下一阶段按以下
顺序推进：

1. **跨 fold 软候选池**：以 C69 的 `Q(W)` 为 parent，生成外部 joint residual
   候选、现有 FULL64 候选和 parent 三者，按 calibration fold 的均值/尾部混合
   目标选择静态权重；不要求每层或每 fold 都严格正向。
2. **proj 专属 H32/H64**：只作为候选池成员而非无条件替换，沿用最终量化器的
   目标但保留 parent 回退，重点观察 Qwen-30B 类宽 down-proj。
3. **性能压缩**：对确认有收益的候选做分块、缓存和向量化，最终只检查官方
   `<420s`，不再使用旧的 `<300s` 内部门槛。

官方新增的两个 Qwen-30B 特征用例优先用于第 1、2 步的独立验证；在拿到完整用例
前，Qwen2.5-0.5B 与 synthetic shape 只提供方向信号，不能用于官方绝对分数拟合。

它们可以保留为诊断工具，但不能占用正式候选 ID。

## 2. 论文与工程依据

### 2.1 CAT：直接优化缺失的 alignment

[Dissecting Quantization Error: A Concentration-Alignment Perspective](https://arxiv.org/abs/2603.04359)
把 Linear SQNR 分解为 bit-width、concentration 与 alignment。正交 rotation/Hadamard 可以
改善 concentration，却不能改变 alignment。论文给出的解析最优 alignment 变换由
Activation 与 Weight autocorrelation 的矩阵几何均值决定，并用 block-diagonal CAT 做
低成本近似。

论文的重要工程结论：

- `down_proj/o_proj/v_proj` 等层的 alignment 往往最差；
- 部分层存在超过 `10 dB` 的 alignment 改进空间；
- CAT block 在 W4A4 上稳定达到或超过 QuaRot、SpinQuant；
- CAT block 在多个模型上的结果接近 FlatQuant，并且不需要训练；
- 全矩阵 CAT 太贵，block-diagonal 是可落地近似。

### 2.2 MR-GPTQ：FP4 必须格式专用

[Bridging the Gap Between Promise and Performance for Microscaling FP4 Quantization](https://arxiv.org/abs/2509.23202)
指出传统 PTQ 方法直接移植到 NVFP4/MXFP4 会失效：共享 scale 与 FP4 网格改变了 rotation、
GPTQ 和 activation ordering 的最优行为。论文的 MR-GPTQ 使用 block-wise rotation、
格式专用 scale search 和静态 act-order。

与本赛题最相关的做法：

1. 先按原始 group 固定 scale/grid；
2. GPTQ 求解期间临时按 activation importance 重排坐标；
3. 求解结束后还原坐标；
4. 不破坏 microscaling/HiF4 group 结构；
5. block rotation 与 online quantization 融合，避免全局 rotation 的 FP4 不兼容。

[Block Rotation is All You Need for MXFP4 Quantization](https://arxiv.org/abs/2511.04214)
进一步说明全局 rotation 可能与 microscaling 的 block scale 冲突，局部 block rotation 更
适合 FP4。该结论支持本计划选择 CAT-64，而不是恢复全局 R64。

### 2.3 Adaptive headroom

[Four Over Six: More Accurate NVFP4 Quantization with Adaptive Block Scaling](https://arxiv.org/abs/2512.02010)
发现 FP4 的主要误差常出现在 near-maximal values，`amax/max_code` 并不总是最优；每块
只增加少量 headroom 候选即可明显改善 NVFP4。HiF4 虽然码本不同，但同样具有共享 scale
与稀疏高值域，适合把 `inner_target in {4,5,6,7}` 作为顶层 E6M2 scale 候选来源。

### 2.4 学习式等价变换

- [OmniQuant](https://arxiv.org/abs/2308.13137)：Learnable Weight Clipping 与 Learnable
  Equivalent Transformation；
- [AffineQuant](https://arxiv.org/abs/2403.12544)：直接学习可逆仿射变换；
- [FlatQuant](https://arxiv.org/abs/2410.09426)：Kronecker/factorized affine transform；
- [SpinQuant](https://arxiv.org/abs/2405.16406)：学习 rotation 优于固定随机 rotation；
- [AMD MXFP4 rotation + SmoothQuant 实践](https://rocm.blogs.amd.com/software-tools-optimization/mxfp4-online-rotation/README.html)：
  联合训练 rotation 与 scale 优于单独优化，但固定 Hadamard 并非所有层都受益。

这些结果支持 C46，但不支持一开始就从随机矩阵做长时间训练。C46 必须从解析 CAT 初始化，
只学习小扰动。

## 3. 合规数据流设计

### 3.1 严格边界

离线 `hif4_calibration_and_quantize_weight` 中允许：

```text
A -> A^T A
W -> W^T W
(A^T A, W^T W) -> CAT / SmoothQuant 等等价变换
A @ W^T -> 参考 Linear 输出
A @ Q_k(W)^T -> 静态 Weight 候选输出
output loss -> 只选择 weight_params
```

禁止：

```text
A @ W 或其损失 -> CAT / rotation / activation scale / offset
A @ W 或其损失 -> activation refinement coverage
A @ W 派生 Tensor -> activation_state
Q(A) @ Q(W) 的联合残差 -> 反推或选择 Q(A)
Weight residual operator -> activation_state
```

### 3.2 状态冻结顺序

实现必须按以下顺序组织：

```text
1. 从 A、W 的 operand statistics 选择 CAT / transform
2. 完成 activation_state 构造并冻结
3. 构造静态 Weight 候选
4. 可选使用 A@W 对少量 Q(W) 候选评分
5. 只替换 weight_params
6. 返回已经冻结的 activation_state
```

不得依赖 Python 控制流或 taint 工具漏洞让输出派生选择间接影响 state。

### 3.3 状态节点限制的正确解释

赛事限制是状态总节点数 `<4096`，不是 Tensor 元素数 `<4096`。因此 CAT transform 可以
存为一个 CPU Tensor：

```text
cat_transform: [num_blocks, 64, 64]
```

它只占一个状态节点。无需因过度防御退化为单一共享 `64x64` transform。真正需要控制的是：

- 动态矩阵变换时间；
- CPU -> algorithm device 的传输；
- float32 精度与 contiguous；
- 逆矩阵只用于离线 Weight，不写入 state。

## 4. Phase 0：恢复父版本与预注册

### 4.1 恢复步骤

1. 从 C41b 归档复制到临时候选并验证 SHA；
2. 不使用 destructive git 命令覆盖用户文件；
3. 保存当前 C42 实验 diff 与评测报告供诊断；
4. 恢复根 `solution.py` 到 C41b 行为；
5. 运行 feature-off、合规、release、五字段合法性测试；
6. 以 C41b 重新生成 GPT-2 small 快筛基线，确认与归档一致。

### 4.2 C43 预注册

在任何实现前写入 execution log：

```text
Candidate: C43
Parent: C41b / archived SHA256
Unique mechanism: HiF4-aligned analytic CAT-64
Attention behavior: bit-identical to C41b
Output-supervision use: none for transform selection
Expected API time: <220s
Primary metric: Qwen `primary_panel_score_total` on the fixed 250/200 panel;
guardrail model means are diagnostic only
```

## 5. C43：HiF4-aligned CAT-64

### 5.1 数学定义

对一个 64-channel block，使用列向量约定：

```text
Sigma_x = E[x x^T]
Sigma_w = W^T W / out_features
```

加入软 shrinkage：

```text
shrink(S, rho) = (1-rho) S + rho * trace(S)/64 * I
```

首版固定 `rho=0.05`，Cholesky/eigh 失败时依次尝试 `0.10/0.20`，不采用严格 condition
number gate。

矩阵几何均值：

```text
A # B = A^(1/2) (A^(-1/2) B A^(-1/2))^(1/2) A^(1/2)
```

alignment transform：

```text
M = (Sigma_w # inv(Sigma_x))^(1/2)
```

为避免尺度整体漂移，对 `log eigenvalue(M)` 去均值，使 `det(M)` 的几何均值为 1。

候选强度：

```text
M_beta = exp(beta * log(M))
beta in {0.00, 0.25, 0.50, 0.75, 1.00}
```

`beta=0` 是 identity 自然候选，不需要额外硬 fallback。

### 5.2 Row-major 实现约定

当前矩阵为：

```text
activation X: [tokens, in_features]
weight W:     [out_features, in_features]
```

定义：

```text
X_t = X @ M^T
W_t = W @ M^-1
```

因此：

```text
X_t @ W_t^T = X @ M^T @ M^-T @ W^T = X @ W^T
```

动态 Activation 只保存并应用 `M`；Weight calibration 计算 `M^-1` 后立即释放。

### 5.3 候选选择目标

CAT 会影响 `Q(A)`，因此选择指标必须 operand-separated：

```text
L_A = mean_folds(
    ||A_t - Q(A_t)||^2 / (||A_t||^2 + eps)
)

L_W = trace((W_t-Q(W_t)) H_t (W_t-Q(W_t))^T)
      / trace(W_t H_t W_t^T)

alignment = trace(W_t H_t W_t^T)
            / (||W_t||_F^2 * trace(H_t))
```

归一化 soft objective：

```text
J_beta = 0.40 * L_A / L_A_identity
       + 0.40 * L_W / L_W_identity
       + 0.20 * alignment_identity / alignment_beta
```

fold robustness：

```text
J_robust = mean(J_fold) + 0.15 * (max(J_fold) - mean(J_fold))
```

采用宽松选择：

- 只要 `J_robust` 相对 identity 改善 `0.1%` 即采用；
- 不要求每个 fold 改善；
- 单 fold 最多允许约 `3%` 退化；
- 若所有非 identity 候选无正收益，保留 identity；
- 不允许用 evaluator test Linear 输出回退单层 CAT。

### 5.4 第一版不做的内容

C43 不得同时加入：

- Hadamard composition；
- channel regrouping/permutation；
- Weight solver coverage 变化；
- adaptive headroom；
- learned transform；
- `A@W` transform gate。

这些必须分配独立 candidate ID，避免无法归因。

### 5.5 代码结构

建议新增：

```python
def _spd_matrix_power(matrix, power, relative_floor): ...
def _spd_geometric_mean(a, b): ...
def _cat64_blocks(activation_cov, weight_gram, strength): ...
def _apply_cat64_rows(x, transforms): ...
def _apply_cat64_weight(weight, inverse_transforms): ...
def _cat_operand_metrics(...): ...
def _select_cat64_strength(...): ...
```

状态新增：

```python
"cat_transform": CPU float32 [blocks, 64, 64] or None
"cat_block_size": 64
```

动态顺序：

```text
NVFP4 dequant
-> existing diagonal smooth/permutation（首版保留父版本次序）
-> CAT-64
-> existing HiF4 codec/refinement
```

Weight 使用严格对应的逆次序和逆矩阵。

### 5.6 C43 单元测试

```text
test_cat_spd_power_round_trip
test_cat_geometric_mean_is_spd
test_cat64_identity_strength_is_exact
test_cat64_linear_equivalence_float32
test_cat64_inverse_matches_transform
test_cat64_state_is_cpu_contiguous_finite
test_cat64_state_node_count_is_legal
test_cat64_dynamic_params_are_legal
test_cat64_feature_off_matches_c41b
test_cat64_does_not_use_linear_product_selector
test_cat64_attention_is_bit_identical
```

数学等价测试允许计算测试夹具中的 Linear output；提交 API 内不得为了选择 `Q(A)` 计算
该输出。

## 6. C44：CAT 坐标系下的 HiF4 MR-GPTQ

### 6.1 唯一机制

固定 C43 的 transform 与 Activation 路径，只替换静态 Weight FULL64 solver。

### 6.2 Static act-order

每个 64 block：

1. 先基于原坐标生成 HiF4 顶层 E6M2 scale 候选；
2. 固定 scale、lv2/lv3 group 边界；
3. 按 `diag(H)` 从大到小生成 processing order；
4. GPTQ 求解时临时重排 `w/H/denominator`；
5. 完成 error feedback、coordinate descent 与 hierarchy toggle；
6. 将 mantissa/sign 还原原始坐标；
7. 输出仍是标准 HiF4 五字段。

act-order 不进入 activation state，也不改变在线布局。

### 6.3 Scale beam

C44 只使用父版本 scale 邻域：

```text
base_code + {-2, -1, 0, 1, 2, 3}
```

先用 exact-hierarchy full-H loss 排序，保留 4 个 beam。Adaptive `4/5/6/7` headroom 留到
C45，避免混合机制。

### 6.4 求解流程

对每个 beam：

```text
exact hierarchy initialization
-> damped Cholesky inverse factor
-> static act-order GPTQ initialization
-> one full64 coordinate sweep
-> 16 lv3 toggles
-> 8 lv2 toggles
-> optional second coordinate sweep（仅困难块）
-> exact full-H loss
```

困难块由父版本 full-H loss 数据驱动选择：选择覆盖总 Weight full-H loss `97%` 的最小集合。
不再固定 `25%` 或盲目 `100%`。

### 6.5 宽松接受

每个 block 只要求：

```text
candidate_loss < parent_loss * (1 - 1e-5)
```

这是数值单调性，不是泛化硬门。层级不设置 fold unanimity；用 pooled Hessian 加轻量软稳健项：

```text
L = mean(L_fold) + 0.10 * (max(L_fold)-mean(L_fold))
```

### 6.6 C44 测试

```text
test_mr_gptq_act_order_restores_coordinates
test_mr_gptq_grid_is_fixed_before_ordering
test_mr_gptq_full_h_loss_is_monotonic
test_mr_gptq_hierarchy_fields_are_legal
test_mr_gptq_chunking_is_exact
test_mr_gptq_data_driven_coverage
test_mr_gptq_nonfinite_fallback
test_mr_gptq_feature_off_matches_c43
```

## 7. C45：HiF4 adaptive headroom + LWC

### 7.1 唯一机制

固定 C44 的 CAT、Activation state 与 MR-GPTQ，只扩展 Weight 顶层 scale/clipping 候选。

### 7.2 候选生成

对每个 64 Weight block 生成：

```text
amax / 7
amax / 6
amax / 5
amax / 4
weighted least-squares scale
p99.0 clipped scale
p99.5 clipped scale
parent winning scale
```

全部映射到合法 E6M2 code；去重后通常只剩 3--7 个候选。每个候选必须重新求解
lv2/lv3/mantissa，不能只替换 `scale_factor`。

### 7.3 A@W 选择器

CAT 与 Activation state 在进入本阶段前已经冻结。允许为静态 Weight 候选计算：

```text
Y_ref = A @ W^T
Y_k   = A @ Q_k(W)^T
L_k   = mean((Y_ref-Y_k)^2) / (mean(Y_ref^2)+eps)
```

为控制成本：

- 先用 full-H proxy 从每块候选中保留 top-2；
- 将 top-2 合成为 layer candidate；
- 每层只做 2--4 次真实 `A@W` candidate scoring；
- 使用 128 calibration rows；
- 不保存 output/residual；
- selector 返回后立即释放临时 Tensor；
- output-derived 值不得影响 `activation_state`。

soft selector：

```text
L_soft = mean(L_fold) + 0.15 * (max(L_fold)-mean(L_fold))
```

只要求相对父 Weight 候选改善 `0.05%`；不要求所有 fold 改善。允许最差 fold 约 `3%`
轻微退化，但若 layer aggregate 变差则保持父候选。

### 7.4 为什么不会重复 C42 失败

C42 为数十万 mantissa 构造连续高自由度修正，校准 token 远少于参数数目。C45 只在
每块少量 scale/clipping 离散候选中选择，且候选先由 full-H proxy 收缩。`A@W` 只是裁判，
不是生成器，因此容量和过拟合风险显著更低。

### 7.5 C45 测试

```text
test_headroom_candidates_are_legal_e6m2
test_headroom_reoptimizes_hierarchy
test_lwc_candidate_count_is_bounded
test_product_selector_only_changes_weight_params
test_product_selector_does_not_taint_activation_state
test_product_selector_parent_fallback
test_product_selector_uses_soft_fold_objective
test_c45_feature_off_matches_c44
```

## 8. C46：解析 CAT 初始化的学习式 refinement

### 8.1 启动条件

以下信号用于安排 C46 优先级，不作为硬门：

- C45 的 Qwen shaped-panel Linear mean 出现稳定提升；
- guardrail 模型未出现重复的结构性回退；
- CAT alignment 提升明显，但 quantized operand loss 仍有大空间。

若 CAT 本身没有结构级信号，不得靠长时间优化掩盖问题。

### 8.2 参数化

推荐两种方式，先做 A，失败后才做 B：

```text
A. M = M_CAT @ exp(S), S 为 block 8x8 Kronecker/低秩对称扰动
B. M = M_CAT @ H(v1) @ ... @ H(vr), r in {2,4}
```

禁止从随机 dense `64x64` 矩阵初始化。

### 8.3 优化目标

由于 transform 影响 `Q(A)`，优化目标仍必须 operand-separated：

```text
L = 0.45 * activation_local_hard_proxy
  + 0.45 * weight_hessian_hard_proxy
  + 0.05 * ||S||^2
  + 0.05 * fold_variance
```

配置：

```text
steps = 6
lr = 0.02
activation_rows = 64 per fold
weight_rows = 128 sampled only for STE ranking
gradient_clip = 1.0
```

优化完成后必须用真实离散 C45 路径重算，不得用 STE loss 直接接受。

### 8.4 宽松 hard validation

- 用 soft mean/worst 目标比较解析 CAT 与 learned CAT；
- 最小改善 `0.1%`；
- 不要求两个 fold 都改善；
- learned transform 的 singular value 采用连续 clamp `[0.125, 8]`，不设更严 condition gate；
- nonfinite 或 inverse 失败才回退解析 CAT。

## 9. 可选后续候选，不与主线混合

### C47：CAT-aware channel grouping

CAT-64 只处理当前连续 64 channels。若诊断显示 block 之间 alignment 差异很大，可先用
permutation 把通道重新分组，再求 CAT：

```text
utility(i,j) = |Sigma_x[i,j]| * sqrt(Sigma_w[i,i] Sigma_w[j,j])
             + |Sigma_w[i,j]| * sqrt(Sigma_x[i,i] Sigma_x[j,j])
```

用层次合并形成 4 -> 8 -> 16 -> 32 -> 64 group。该候选只改变 grouping，不同时修改
CAT 公式和 solver。

### C48：CAT + micro-Hadamard

仅在 CAT 已改善 alignment、但 concentration 仍是主瓶颈时测试：

```text
T = H16/32/64 @ M_CAT
```

Hadamard size 必须作为独立候选；不默认 H64。FP4 论文和 AMD 实践均说明固定 rotation
可能伤害特定层。

## 10. 统一评测矩阵

### 10.1 快筛

每个 candidate 先运行：

```powershell
.\.venv\Scripts\python.exe -m pytest -q `
  tests\test_linear_compliance_guard.py `
  tests\test_release_candidate.py `
  tests\test_weight_full64.py

.\.venv\Scripts\python.exe evaluator\real_model_suite.py `
  --models gpt2-small `
  --candidates c39 `
  --solution solution.py `
  --candidate-name <candidate> `
  --panel-profile qwen-official --primary-model gpt2-small `
  --device cpu --algorithm-device cuda --cache-mode read `
  --output <json> --report <report>
```

快筛只用于发现明显 bug。GPT-2 small 正向不能单独晋级。

### 10.2 高风险筛查

建议立即运行：

```text
gpt2-medium
qwen2.5-0.5b
```

这是 C42 暴露过拟合的两个敏感模型。它们只用于发现结构性回退，允许轻微负向，
不采用“两个都必须正”的硬门，也不覆盖 Qwen 主排序。

### 10.3 五模型正式开发矩阵

```text
gpt2-small
gpt2-medium
opt-125m
pythia-160m
qwen2.5-0.5b
```

主指标：

```text
Qwen primary_panel_score_total
  = 250 * mean(Linear case scores) + 200 * mean(Attention case scores)
```

次指标：

```text
guardrail model panel mean / per-model delta
proj/o/fc family aggregate
global Linear MSE gain
API time by model
```

不把 component macro、global-MSE、native case sum 或 guardrail 回退覆盖主排序。

### 10.4 不过度防御的晋级规则

除硬合法性外，采用以下软规则：

- Qwen 主面板增量是第一条件；
- guardrail 模型增量只作为稳定性证据，不设正向或百分比硬门；
- 允许 guardrail 轻微负向，只有重复且明显的结构性回退才提升人工复核优先级；
- 不因一个 layer、一个 role 或一个 fold 小幅退化否决总候选；
- 候选总增量很小但理论正确时可作为可叠加机制保留；
- 官方提交前 API 时间以 `<420s`（7 分钟）为唯一硬门，不使用更严的一票否决；
- 等于或超过 `420s`、非法 state、非法 HiF4、nonfinite 才是硬失败。

## 11. 运行时间与实现预算

| Candidate | Calibration 新增 | Dynamic 新增 | 最慢 API 时间（仅 420s 硬门，不作更严晋级门） |
|---|---|---|---:|
| C43 CAT-64 | batched eigh/inverse | block `64x64` transform | <420s |
| C44 MR-GPTQ | Weight-only full64 | 0 | <420s |
| C45 headroom/LWC | 2--4 layer product score | 0 | <420s |
| C46 learned CAT | 6-step calibration | 与 C43 相同 | <420s |

优化要求：

- CAT blocks 用 batched `torch.linalg.eigh`；
- dynamic CAT 用 `[tokens, blocks, 64] x [blocks,64,64]` batched einsum/bmm；
- CAT state 每次 API 调用只搬运一次；
- Weight inverse transform 分 row chunk；
- MR-GPTQ rows/blocks 维必须向量化；
- 只允许在 64 coordinate、beam、少量 candidate 维度循环；
- `A@W` scorer 复用 `Y_ref`，每层只计算一次；
- 产品候选只保留 top-2/4，不做无界搜索。

## 12. 正确性与合规测试清单

每个 candidate 必须通过：

```text
python -m py_compile solution.py
git diff --check
pytest -q
```

必须确认：

1. 六个官方 API 签名不变；
2. Weight/Activation/Q/K/V 五字段 shape、dtype、值域合法；
3. state 仅含允许类型、CPU Tensor、depth <=8、node <4096；
4. state Tensor finite、contiguous、无 grad；
5. feature flag 关闭时与父版本字段级相同；
6. CAT FP32 等价误差在合理浮点容差内；
7. `A@W` 派生值不进入 activation state；
8. Linear-only candidate 的 Attention 与 C41b 逐 case 相同；
9. 不读取文件、网络、模型名称或 evaluator 结果；
10. 不用官方分数或 test split 逐层调 threshold。

## 13. 候选归档与日志

每个阶段必须独立归档：

```text
solutions/YYYYMMDD_vNNN_c43-cat64_scoreNA_timeNA/
solutions/YYYYMMDD_vNNN_c44-cat64-mrgptq_scoreNA_timeNA/
solutions/YYYYMMDD_vNNN_c45-hif4-headroom-lwc_scoreNA_timeNA/
solutions/YYYYMMDD_vNNN_c46-learned-cat_scoreNA_timeNA/
```

`result.md` 至少记录：

```text
Parent ID / SHA256
Unique mechanism
Feature flags
CAT strength selection rate
CAT singular value distribution
alignment before/after
activation local error before/after
weight full-H error before/after
headroom candidate distribution
A@W selector acceptance rate
per-model Linear and Attention
per-role q/k/v/o/fc/proj
state tensor shapes and node count
calibration/dynamic/API time
official status
decision and next checkpoint
```

未知官方结果使用 `scoreNA_timeNA`，不得把本地分数写成官方分数。

## 14. 决策检查点

### Checkpoint A：C43 CAT-64

- 若 alignment 显著改善且 Qwen 主面板为正：进入 C44；guardrail 仅作复核信号；
- 若 alignment 改善但 quantized loss 不变：先测试 C47 grouping，不直接增加训练；
- 若 alignment 和 concentration 都无改善：CAT block 公式/方向实现可能错误，停止后续；
- 若只有 GPT-2 small 正向、medium/Qwen 大负向：判定 calibration 选择过拟合，不晋级。

### Checkpoint B：C44 MR-GPTQ

- 若 Weight full-H error 下降且 Qwen 主面板为正：进入 C45；
- 若局部 Weight loss 下降但真实 Linear 反向：停止扩大 coverage，检查 CAT 坐标下的 act-order；
- 若 API 时间接近 420s：先减少 loss coverage，不缩小 CAT 表达能力。

### Checkpoint C：C45 headroom/LWC

- 若 selector acceptance 接近 0：候选与父 scale 等价，需要改候选生成；
- 若 acceptance 接近 100% 且跨模型反向：`A@W` candidate set 容量过大，减少到 top-2；
- 若旧口径 Linear 约 `0.63--0.67`：进入 C46；
- 若达到 `0.68+`：先官方提交建立新锚点，再决定是否训练 refinement。

### Checkpoint D：C46 learned CAT

- 若 learned 只比解析 CAT 提升很小：保留解析 CAT，减少复杂度；
- 若达到工程目标且 `<420s`：官方提交；
- 若官方仍明显低于 22000：用官方结果只判断机制方向，不拟合逐层参数。

## 15. 分数目标分解

以下只是工程目标，不是官方分数承诺：

| 阶段 | 期望 Linear 旧口径增量 | 累计目标 | 主要来源 |
|---|---:|---:|---|
| C43 CAT-64 | +8--15pp | 0.61--0.68 | alignment + concentration |
| C44 MR-GPTQ | +2--5pp | 0.63--0.71 | FP4 format-specific Weight solve |
| C45 headroom/LWC | +1--3pp | 0.64--0.74 | near-max value representation |
| C46 learned CAT | +1--4pp | 0.66--0.78 | analytic transform refinement |

CAT 是唯一可能单阶段提供 `8pp+` 的候选。如果 C43 实际只有 `<1pp`，则 22000 目标不能靠
本计划剩余微调自动达成，必须重新诊断 CAT 与 HiF4 hierarchy 的交互，而不是盲目执行
C44--C46。

## 16. 明确禁止的捷径

1. 不把 C42 高自由度产品补偿改个名字重新启用；
2. 不用 `Q(A)@Q(W)` 联合残差选择 `Q(A)`；
3. 不让 `A@W` 决定 CAT、rotation、Activation offset 或 coverage；
4. 不用 test/validation 输出逐层回退 CAT；
5. 不把全局 R64 当作 CAT；
6. 不只扩大 FULL64 coverage 并声称实现 MR-GPTQ；
7. 不只增加 offset 并声称实现 adaptive headroom；
8. 不用严格 fold unanimity 把绝大多数候选挡回 identity；
9. 不因追求稳健把允许的 CAT dense state 错误限制到 4096 个元素；
10. 不根据官方分数反向拟合 beta、damping、coverage 或模型 shape gate。

## 17. 最终执行顺序

```text
Phase 0  恢复 C41b、移出 C42 实验路径、预注册 C43
  |
C43      解析 HiF4-aligned CAT-64
  |
Checkpoint A
  |
C44      CAT 坐标系下的 MR-GPTQ/static act-order
  |
Checkpoint B
  |
C45      adaptive headroom + LWC + 合规 A@W Weight selector
  |
Checkpoint C / 可选首次官方提交
  |
C46      解析 CAT 初始化的学习式小扰动
  |
Checkpoint D / 官方提交验证 22000+
  |
可选 C47 channel grouping 或 C48 micro-Hadamard
```

每个箭头都代表：预注册、单机制实现、单元测试、合规测试、GPT-2 small 快筛、
medium/Qwen 敏感筛查、Qwen 主面板评测、其余模型软 guardrail、时间评测、归档、
决策。不得跨候选一次性合并实现。
