# v162 独立 Attention 优化执行日志（A 代理）

> 任务书：[workpackages/v162-attention.md](../../docs/superpowers/plans/workpackages/v162-attention.md)
> 总计划：[2026-09-06-v162-independent-linear-attention-plan.md](../../docs/superpowers/plans/2026-09-06-v162-independent-linear-attention-plan.md)
> 零点：v162 standard-baseline-both，SHA `56101559D267D962084CD67A9F9AF8EB924501B17AB408EAF676081876CC000A`，官方 1001/146s。
> 本侧冻结：两个 Linear API、dynamic V、standard codec。工作目录 `workbench/v162_attention/`。

## A0 基线与去重（2026-09-06）

- baseline 复制到 `workbench/v162_attention/baseline/solution.py`，SHA256 复核一致
  （`56101559…C000A`，与总计划唯一零点相同）。
- 环境：`.venv` CUDA 可用（RTX 3060 Ti）。GPU 排队按总计划 §2 使用
  `artifacts/proxy_v3/v162-independent/gpu.lock` 原子创建；锁存在则等待，不删除对方锁。
- 面板事实（来自 cache 与 evaluator 源码）：
  - Qwen2.5-0.5B：hidden 896、q_heads 14、kv_heads 2、head_dim 64（G=2 组、每组 7 个 Q head）；
  - calibration 5 窗口，长度 (10, 128, 512, 1024, 1024)；gate=输入顺序最后一窗（1024），其余 4 窗训练；
  - proxy-v3 每 shard 4 层 × 2 in-dist 窗口 = 8 attention cases，六 shard 共 48；
  - evaluator `_attention` 为 non-causal GQA（repeat_interleave），gain=(MSE_STD−MSE_PLAYER)/MSE_STD。

### novelty 表（任务书 A0.3）

| 维度 | 08-26 learned butterfly（Phase D，仅计划未执行） | v187/v188 importance | 09-06 Jacobian pair-matrix | 本轮机制 |
|---|---|---|---|---|
| 参数空间 | 6 蝶形 stage × 8 共享角度，`[kv_heads,1,6,8]`，无 D×D 密集矩阵 | 对角 importance（无旋转、无新参数） | 固定 2×2 SPD pair transform（闭式拟合，非梯度） | 每 KV group 全密度正交 `R_g∈R^{64×64}`，Cayley(Θ)+Hadamard 初始化 |
| 损失 | MSE ratio + 尾部 relu 惩罚（`_attention_ratio_objective`） | operand/坐标敏感度 | logit→输出 Jacobian Frobenius 加权协方差 | 完整 Attention 输出 MSE / MSE_STD（标准量化 V 进 player 路径） |
| 硬编码器 | STE `_ste_hif4` | 无（解析权重） | 真实部署 gate | v162 标准 HiF4 硬码，STE 仅 backward |
| V 是否量化 | 未明确（沿用旧例程） | n/a | 用 v-state 实际部署输出 | 是：Vh=标准 HiF4，进 O_hat 与 O_std |
| GQA | 未按 group 共享（逐 kv_heads 角度） | n/a | 2×2 pair（非 group 级） | 同 group Q head 与 K head 共享同一 R_g，连续 QK 不变量精确保持 |
| 训练/holdout | 偶数样本训练/奇数验证 | 校准内 | 奇偶窗口隔离 | 4 训练窗 + 第 5 窗 gate + evaluator validation/test 独立 split |
| 可达计数 | 未执行 | 已执行（v187 官方 9167） | 已执行（TIME_REJECTED） | 记录 attempted/accepted 与 gate 决策 |
| 父/耗时 | C21-C（未达实现） | v185/v186 | v189 | v162 零点；32 步/层训练全在 calibration API 内计时 |

结论：三个历史机制均与本轮「group 级全密度正交矩阵 + 完整输出损失梯度学习」不数学等价；
learned butterfly 从未执行，不存在等价的有效失败实验 → 非 DUPLICATE_CLOSED，继续 A1。
另注意：08-26 C22 的 Linear 侧 signed Hadamard R64 被拒（operand-local 指标），与本轮
Attention 输出目标不同面板不同目标，不构成本机制的反证。

### 本地最高精度锚点（提交参照）

- v162 零点本侧 gain=0（构造性）。
- 强对照 v168（`5988AE47…01AC79`）与最新完整父 v189 Attention 侧需在同协议下成对读取，
  不得混用旧 panel 均值；v189 default Attention 侧历史读数 `0.752173407020` 仅作背景，不作直接比较。
- 用户目标：本地 Attention gain 达 0.9；每超过本工作流已测最高即 commit & push 本侧文件。

## A1 最小合法实现（进行中）

- 候选骨架：复制 v162，新增 attention 专用 helper，不改 `_standard_params`/Linear/V 路径。
- state 约定：`q_state`/`k_state` 各含独立 `R`（float32 CPU，`[G,D,D]`）与形状元数据；`v_state={}`。
- 动态路径：NVFP4 解码 → head reshape → group R matmul → v162 标准 HiF4 编码。无候选循环、无跨调用缓存。
- 研究臂开关：`_ROT_ARM ∈ {identity,h,learned,gate}`（研究用副本），部署候选固定 `gate`。

## A1 最小合法实现（2026-09-06 完成）

- 候选：`workbench/v162_attention/candidate/solution.py`（SHA `19159AB9904FB3F1…`，
  完整 SHA 见 git 提交）。v162 标准 codec 原样保留；新增 attention 专用 helper；
  `_standard_params`/Linear/V 路径未动。
- 机制：每 KV group 一个 `R_g∈R^{64×64}`，Cayley(Θ)+规范化 Sylvester Hadamard 初始化；
  同组 7 个 Q head 与 1 个 K head 共享 R_g；动态路径 NVFP4 解码 → group matmul → v162 标准 HiF4 编码；
  state 仅存 CPU float32 `r`（`[G,D,D]`）与审计字段；无候选循环、无跨调用缓存。
- 训练（校准内，arm=learned/gate）：4 训练窗（等间隔确定性采样 ≤128 KV/≤32 Q token，
  Q 行取自 KV 子集）+ 第 5 窗 gate（完整长度、Q 分块 512、K/V 全量）；
  Adam lr=0.01×32 步、clip 1.0、reg `1e-3·mean((C−I)²)`；STE 仅 backward；非有限即 ERROR。
- 部署选择（arm=gate）：先 learned vs H（gate 上严格更好才选 learned），再胜者 vs identity
  （严格更好才部署，平局归 identity）。
- 必要测试 `workbench/v162_attention/tests_a1.py` 全部通过（CPU 模式，不占 GPU）：
  - T1 R=I 六 API 与 v162 逐位对齐（含 state 审计字典合法）✓
  - T2 合成正交 R 的 GQA 连续 QK 误差 2.67e-05 ≤ 4.26e-04 ✓
  - T3 token 数/调用次序互换一致、无跨调用状态污染 ✓
  - T4 五字段过 reference 合法性，decoded == 研究硬前向（逐位）✓
  - T5 STE 前向==decode(encode(x))、梯度有限非零；合成数据训练冒烟 loss 0.9139、正交误差 3.58e-07 ✓
  - T6 Linear 与 V 状态/编码/输出逐位不变 ✓
- 实现期修复：einsum 组维（`gkd`）、gate 评估 K/V 全量（non-causal 不随 Q 分块）、
  state 审计浮点 NaN→-1.0 哨兵、`_cayley_orthogonal` 命名统一。

## A2 固定配置训练实验（2026-09-06 进行中）

- config.json 已在读取训练结果前保存（任务书 A2 表，无邻域扫描）。
- GPU 排队按计划协议执行（`gpu.lock` 原子创建）；期间 Linear 代理多次持锁（l1~l4 run），
  本侧全部等待，未删除对方锁。
- gate 诊断（真实 proxy-v2 cache 校准窗，归一化完整输出损失，identity=1.0，越低越好）：

  | 层 | H | learned | gate 部署 |
  |---|---|---|---|
  | 0 | 6.248150 | 5.970765 | identity |
  | 8 | 1.095325 | 1.125787 | identity |
  | 15 | 0.683861 | **0.619129** | learned |
  | 23 | **0.792165** | 0.795395 | h |

  校准耗时 0.9–3.1 s/层。层 0/8 的 Hadamard/learned 明显劣化，gate 正确回退 identity；
  层 15/23 有材料正信号。数据：`research/a2_gate_diagnostics.json`。
- 四臂 + 强对照 evaluator 运行（shards 0,2,3,5 attention-only，baseline=v162 零点）进行中：
  a2-gate / a2-h / a2-learned / a2-strong-v168 / a2-strong-v189；臂变体 SHA：
  h=`F156D719D561A3C5…`、learned=`8C7C88B7C658501E…`。

（后续条目按 run 追加）
