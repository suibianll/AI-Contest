# 官方评测规律辨识与 Linear 结构实验计划

> 状态：**ACTIVE**
>
> 创建：2026-09-03
>
> 官方父版本：v160，`17532 / 232s`，归档源码 SHA256
> `33B1D061CE6BFCD92659C597BE4830BB9B910E646FF518433DA67B925AE8680D`
>
> 根 `solution.py` SHA 与 v160 归档不同；所有实验必须从 v160 归档源码分支，不能把根文件
> 当作逐位相同的源码父本。

## 1. 目标与边界

本阶段不尝试拟合官方隐藏 case 权重，只回答三个可证伪问题：

1. v158 Attention Matrix-Smooth 在 v159 Linear 上是否仍贡献约同量级收益，还是存在明显
   Linear×Attention 交互；
2. 本地等价批处理的时间收益是否能迁移到官方硬件；
3. 统一的低自由度坐标重分布能否产生跨 shape 的大范围收益。

Attention 新门控、seed/alpha/offset 扫描、ROAB、PAWV、length/layer/role 路由继续冻结。
官方反馈只用于接受或拒绝上述预注册假设；每个假设最多提交一个算法候选，不根据官方分数
追加邻近参数版本。

## 2. 已知事实与必须先纠正的错误

### 2.1 官方锚点

| 单元 | Linear | Attention | 官方分数/时间 |
| --- | --- | --- | ---: |
| `S00` v86 | v86 | v86 | `16744 / 222.7s` |
| `S01` v158 | v86 | v158 Matrix-Smooth | `16861 / 223s` |
| `S11` v159/v160 | v159 GPTQ | v158 Matrix-Smooth | `17532 / 232s`（v160） |
| `S10` | v159 GPTQ | v86 | **未知，本计划 E1 补齐** |

v155/v156 的 `10^-4` 级局部变化没有迁移；v158 的较广 Attention 变化得到 `+117`；v159
的大范围联合 Linear 变化得到 `+671`。因此本计划把“影响覆盖面”和“父版本交互”作为观察量，
不再把本地 aggregate mean 当作官方分数代理。

## 3. 通用实验纪律

每个实验必须先写清 parent SHA、唯一变化、预期调用路径、focus/control 和失败条件。统一执行：

1. 单文件导入与六 API 合法性 smoke；
2. 静态 reachability 检查：目标分支必须有实际调用计数，不能仅凭输出变化判断；
3. Qwen compact 父子配对；只有满足该实验自己的门禁才运行 Qwen default；
4. default 通过后运行 GPT-2；最终算法候选再运行 OPT-125m；
5. 保存 source SHA、JSON/report、调用计数、API 分解、case identity 和决定；
6. 只有计划明确标记“可提交官方”的候选才交给用户提交，本地流程不得自行提交。

官方 score/time 与本地 proxy/time 分栏记录。一次官方结果只能验证当前假设，不能用来估计
未观测的 layer/role 权重，也不能触发 threshold/seed/alpha 的邻域搜索。

## 4. E0：官方确定性与时间噪声控制（可选，两次提交之一）

### 假设

官方分数对相同 SHA 是确定的；同 SHA 重跑可估计时间噪声。没有这个控制时，数秒级时间差不能
归因于代码优化。

### 实验体

- 直接重新提交 v160 归档 SHA `33B1D061…`，不重新生成源码；
- 记录原 v160 `17532/232s` 与重复运行的 score/time；
- 若分数不同，立即停止所有“官方规律拟合”，把官方评分视为非确定或评测集发生变化；
- 时间噪声定义为 `noise_time = abs(T_repeat - 232)`，E2 只有在时间差明显大于该噪声时才解释。

### 产物

`logs/execution/2026-09-03-e0-v160-official-repeat.md`，只记录官方回传，不创建新版本目录。

## 5. E1：补齐 Linear×Attention 官方 2×2 因子实验（最高优先级）

### 假设

v158 Matrix-Smooth 的官方收益可能依赖 v86 Linear。补齐 `S10` 后才能判断 Linear 与 Attention
能否独立优化，不能继续用 `v86→v158 +117` 直接外推到 v159。

### 候选构造

建立 `workbench/e1_v159linear_v86attention.py`：

- Linear 两个 API 必须来自 v160/v159 Linear；
- Attention 四个 API 及其闭包必须恢复 exact-v86 行为；优先从 v160 归档关闭
  `_ATTN_PAIR_MATRIX_SMOOTH` 和 A2 GQA center，若字段/输出不能逐位复现 v86，则从 v86 归档
  精确移植 Attention 闭包；
- 不改变 codec、Linear state、调用图和其他 Attention 候选。

### 本地身份门禁

1. 对同一 default cache，候选 Linear 168/168 输出与 v160 逐位一致；
2. 候选 Attention state 的键、shape、dtype 以及 Q/K/V 120/120 输出与 v86 逐位一致；
3. API 调用数仍为 `168/168/24/120/120/120`；任一不满足则禁止官方提交。

这不是本地效果筛选；只要身份门禁成立，就提交一次官方评测，因为未知量正是官方交互。

### 官方结果计算

令新结果为 `S10`：

```text
Linear effect under v86 Attention     = S10 - 16744
Attention effect under v159 Linear    = 17532 - S10
interaction                           = 17532 - S10 - 16861 + 16744
                                      = 17415 - S10
```

- `S10 = 17415`：在当前四单元上呈加性；
- `S10 < 17415`：组合存在正交互，拆开后损失超过单项相加；
- `S10 > 17415`：存在负交互，v158 Attention 在 v159 Linear 上部分抵消；
- 只记录原始 interaction，不围绕 17415 调任何参数。

### 产物

- `artifacts/official_eval/e1-v159linear-v86attention-default.json`
- `logs/official_eval/e1-v159linear-v86attention-default.md`
- 官方回传后建立一个带准确 score/time/SHA 的归档版本；未回传前保持 `unregistered/NA`。

## 6. E2：逐位等价的官方时间 A/B（可选，需 E0）

### 假设

L1 在本地把 Linear default API 从约 `269.4s` 降到 `231.4s`，但官方硬件是否受同一热点支配
未知。用逐位等价实现才能把官方时间差归因于调度/批处理，而不是算法分支。

### 候选构造

建立 `workbench/e2_l1_unbatched.py`，从 v160 归档只撤销 L1 combo 批编码：把
`_linear_output_candidate_metrics_combos` 的批量编码恢复为相同顺序的逐候选
`_linear_output_candidate_metrics` 调用。A1、A2、候选集合、比较顺序和所有常量保持不变。

### 门禁与解释

- Qwen 完整 default 的 288 个 case、最终 state 和六 API 输出必须逐位等于 v160；
- 本地 Linear calibration 应回到批处理前量级，证明慢路径确实执行；
- 与 E0 同一时间窗口提交一次，官方 score 必须仍为 17532；
- 只有 `abs(T_E2 - T_E0_repeat) > max(5s, 2*noise_time)` 才判定官方硬件存在可测时间效应；
  否则结论为“低于噪声/计时粒度”，不是“官方不执行校准热点”。

E2 只研究时间，不作为新父版本，也不继续做 A1 的 1 秒级官方 A/B。

## 7. E3：统一 64-block Householder 坐标重分布

E3 测试一个独立假设：v159 的剩余瓶颈可能在进入 HiF4 前的通道几何，固定的低自由度正交
重分布能否改善最终 `Q(A)Q(W)^T`。

### 固定算法

在 v159 已选定的 smooth/permutation/Hadamard 坐标之后，对每个连续 64-channel block：

```text
C_A = X_b^T X_b / trace(X_b^T X_b)
C_W = W_b^T W_b / trace(W_b^T W_b)
C   = 0.5 * (C_A + C_W)
```

从固定向量 `ones/sqrt(64)` 开始做恰好 4 次 power iteration 得主方向 `u`；目标均衡向量
`t = sign(u)/sqrt(64)`，零符号按 +1；令 `v=(u-t)/||u-t||`，构造
`H=I-2vv^T`。若分母小于固定数值稳定 epsilon，则该 block 用 identity。

- Weight：`W_b' = W_b H`；
- Activation：`X_b' = X_b H`；
- 因 H 正交且对称，`X_b'(W_b')^T = X_b W_b^T`；
- Weight Hessian 在最终坐标用 `H^T G_b H`，Activation Gram 继续从最终量化 Weight 计算；
- state 每 block 只保存一个 64 向量 `v`，dynamic 用一次向量化
  `x - 2(x·v)v`，禁止 Python token/block 循环。

没有 seed、alpha、rank 或 block-size 候选；所有 layer/role 使用同一规则，不设模型或 shape
路由。正式候选无 parent/proposal 搜索，避免双倍 GPTQ 校准和稀疏 gate。

### 验收

1. CPU/CUDA、三类 Linear shape 上连续域相对误差 `<1e-5`；
2. 静态复杂度为每个 tensor 一次 block projection + axpy，在线 O(nC)，不改变 API 次数；
3. Qwen compact 要求 mean delta > 0、median delta >= 0、正 case 多于负 case、各 shape family
   mean 不为负、worst-quartile mean delta >= -1e-3，且非零变化覆盖至少 25% case；Qwen
   default 要求正 case 多于负 case、validation/test 成对同号率 >= 75%，未修改 Attention
   120/120 逐位一致；
4. GPT-2 与 OPT 的 shape-family 主效应、W/A interaction 与 Qwen 同构；
5. 本地完整 API total 不超过 v160 同机父版本 +30s。

任一失败即关闭该固定 Householder 机制，不尝试 2/8 次迭代、其他 target、rank-2、不同 block
或 role gate。全部通过才提交一次官方。

## 8. 实验结果如何回答官方规律

| 结果组合 | 可支持的结论 | 后续动作 |
| --- | --- | --- |
| E1 interaction 接近 0 | 当前 Linear/Attention 近似可分离 | 后续继续单侧开发 |
| E1 interaction 明显非零 | 官方结果强依赖父版本组合 | 所有机制必须做 2×2 或 exact-parent A/B |
| E2 官方明显变慢 | 官方时间受 Weight candidate encoding 支配 | 保留批处理，时间预算按该路径控制 |
| E2 落入噪声 | 无法从官方计时识别该热点 | 不宣称无效，只停止该时间探针 |
| E3 本地与官方均正向 | 官方可能奖励低自由度坐标几何改善 | 下一计划研究同一固定低秩族，不回调本实验参数 |
| E3 本地广泛正向、官方零或负向 | 本地坐标收益不能迁移 | 关闭该机制，不围绕参数调优 |
| E3 本地门禁失败 | 当前固定 Householder 机制缺少泛化收益 | 不提交官方，等待新机制或外部材料 |

以上结论都只限已观测父版本与机制，不写成官方模型、权重或隐藏 case 的事实。

## 9. 执行顺序与停止条件

1. **E1**：先补齐唯一缺失的 2×2 官方单元，这是最高信息量的一次提交；
2. **E0/E2（可选）**：只有用户希望专门研究官方时间时执行，必须成对；
3. **E3**：实现并测试固定 Householder；只有通过三模型和复杂度门禁才提交；
4. 每个官方候选只提交一次。官方失败后不做邻域调参，不把官方分数回填成本地 loss 权重。

若 E1、E3 均结束，本计划归档；下一计划必须依据这些正交实验的结论重新定义算法族，不能
恢复已关闭的 seed/alpha/offset/ROAB/PAWV/Attention 小门控。
