# 根组件时间-收益审计（v202 根，SHA 56DC805D…EFCB2BD）

日期：2026-09-09。工作目录 `workbench/full_solution/root-time-audit/`。
不改 `solution.py`、不改 `evaluator/`、未运行 `eval.py`（无校准缓存写入）。

## 0. 方法与水文说明

- 根 SHA256 已核验：`56dc805d6e5a3aef896db8021045740292735725d688b48e3d4393e55efcb2bd`。
- 生效调用链（同名覆盖确认）：Linear = 11645 包装 → `_COMBINED_LINEAR_BASE_CALIBRATION`
  = `_STATIC_ACTORDER_BASE_CALIBRATION` = **7802 主校准**；Attention = 11529 包装（R3）
  → `_V189_CALIBRATION_ATTENTION` = **9250 v189 栈校准**。11052 的标准 v162 版本被 11645 再次覆盖，不生效。
- 插桩：importlib 加载根为模块；monkeypatch 包装 28 个组件函数（CUDA synchronize +
  perf_counter）；另用 `sys.settrace` 对校准主体做逐行计时（inline 段如 rank-1/rank-2 幂迭代、
  h_inv cholesky 循环无法 monkeypatch，只能靠行追踪）。`@torch.no_grad()` 装饰的函数共享
  `decorate_context` code object，第一轮追踪因此错配，已通过 `__wrapped__` 解包修正。
- 真实输入：`artifacts/official_eval/cache/qwen3.5-4b-proxy-v2.pt`，取法复刻
  `evaluator/proxy_v3_eval.py`（Linear：每 (layer,role) 2 个校准窗口 NVFP4 pair；Attention：
  全部 5 个校准窗口 q/k/v pair）。实测角色与形状（24 层 × 7 role，q_heads=16 kv_heads=4 head_dim=256）：

  | role | 形状 (out×in) | quadratic/GPTQ 资格（in≤3072） |
  |---|---|---|
  | fc_gate / fc_up | 9216×2560 | 是 |
  | proj | 2560×9216 | 否 |
  | o | 2560×4096 | 否 |
  | q | 4096×2560 | 是 |
  | k / v | 1024×2560 | 是 |

  （任务书中"d_in=11008"在 4B 代理上不存在；最大 d_in 为 proj/o 侧 9216/4096，已覆盖。
  fc_gate/fc_up 后补测，见 §2。）

- **测量水文**：任务期间 GPU 实际被多个进程共享（nvidia-smi 显示 27 个 compute app、空闲时
  ~39% util），各轮 wall 噪声可达 2–4×。处理：以各 case 跨轮**最小 wall**（最安静窗口）为准
  （Linear proj 2.96/o 1.39/q 5.33/k 2.29/v 2.54s；Attention 5.5–5.9s/层），并与同机六 shard
  全量评测锚点交叉验证（§1）。逐行追踪轮因每行 cuda.synchronize 会**高估 Python 循环段**
  （rank 幂迭代、搜索循环）份额、相对稀释 GPU 段份额；下文份额以此方向性偏差标注。
- 同机全量锚点（v195 根六 shard eval-v3，calibration cache miss）：Linear 校准 API **772.4s /
  168 次 ≈ 4.6s/次**；Attention 校准 API **29.0s / 6 层 ≈ 4.8s/层**。本地逐 case 测量与此一致，
  说明最小 wall 轮可信。

## 1. 官方分数贡献引用（历史官方回传，直接引用不换算）

| 机制 | 官方贡献 |
|---|---|
| v159 Linear 栈整体（SmoothQuant/perm/block-Hadamard 搜索、quadratic Gram、Weight GPTQ、E6M2 offset/refinement 等内件） | +671（内件未单独定价） |
| R3 Attention（`_a2_train_rotation` 32 步学习旋转 + 融合 gate） | +396 |
| v195（K-center 多窗口梯度聚合修复） | +21 |
| compiled sample-energy 块序（Linear 激活 GPTQ 排序） | +20 |
| v189 静态块序 | +17 |
| v166 rank-1 残差（Linear） | +3（侧隔离） |
| L-R2 rank-2 残差（Linear） | +1 |
| v186 E6M2 scale +4 窗口 | +1 |
| v202（sample-energy 校准期预编译） | +0 / −8s |

## 2. Linear 校准组件 × 时间（layer 0 各 role）

份额来自逐行追踪轮（Python 循环段高估、GPU 段被稀释）；秒数为最小 wall 轮的包装实测
（`*` 为估计值：fc_gate/fc_up 的真 wall 由全量锚点 4.6s×7 − 其他 role 反推 ≈ 8.9s/个；
residual/h_inv 等未包装段 = wall − 已包装组件之差）。

| 组件 | proj 2560×9216 (2.96s) | o 2560×4096 (1.39s) | q 4096×2560 (5.33s) | k 1024×2560 (2.29s) | v 1024×2560 (2.54s) | fc_gate 9216×2560 (~8.9s*) |
|---|---|---|---|---|---|---|
| **Weight GPTQ（`_gptq_quantize_weight`，quadratic 路径）** | 不适用 | 不适用 | **4.09s ≈ 77%** | **1.37s ≈ 60%** | **1.43s ≈ 56%** | **28.1/33.8s = 83%（追踪轮），真值估 ~7.4s* ≥83%** |
| **v166 rank-1 + L-R2 rank-2 残差拟合**（幂迭代，inline） | ~1.3–1.5s* ≈ 45–50%（追踪 71.8% 为高估） | ~0.7s* ≈ 50% | ~0.6s* ≈ 11% | ~0.5s* ≈ 22% | ~0.6s* ≈ 24% | ~0.8s*（追踪 11.2%） |
| SmoothQuant/perm/block-Hadamard 搜索族（hybrid metrics + combos + perm bases） | ~0.6s ≈ 20% | ~0.4s ≈ 29% | ~0.5s ≈ 9% | ~0.5s ≈ 22% | ~0.5s ≈ 20% | ~1.2s ≈ 13% |
| 激活 h_inv（cholesky + inverse，激活 GPTQ 编译） | ~0.4s* ≈ 12–14% | ~0.05s* | ~0.05s | ~0.07s | ~0.06s | ~0.06s |
| v202 sample-energy 块序预编译 | 0.12s ≈ 4% | 0.002s | 0.05s | 0.005s | 0.001s | 0.009s |
| NVFP4 解码 + 校准统计 + 采样 | 0.06s ≈ 2% | ~0.01s | ~0.01s | ~0.01s | ~0.01s | ~0.1s |
| quadratic Gram 变换 / ratio capture / state build / 其它 | ~0.1s | ~0.1s | ~0.1s | ~0.05s | ~0.05s | ~0.15s |

追踪轮细节：rank-2 在残差块内普遍是 rank-1 的 1.5–2.7×（`_rank2_top2` 每折 128 步 ×2 方向 +
`_proj` 投影）；逐行分桶数据见 `tables.md` 与 `results-fc_gate.json`。

当前配置下**零成本**的组件（不需 ablation）：`_ADAPTIVE_OFFSETS=False`、
`_ADAPTIVE_ACT_GPTQ_REG=False`、`_WEIGHT_E2E_REFINE=False`、`_BLOCK_SWAP_ROUNDS=0`、
joint 搜索（形状门控未触发）、E2E verify 未触发、block-swap 关闭。

按每层 7 role 汇总估计：Linear 校准时间 **Weight GPTQ ≈ 65–70%**（fc_gate+fc_up 两个 role
就贡献 ~15s/32s），**残差拟合块 ≈ 15–20%**，**搜索族 ≈ 12–15%**，其余 <5%。

## 3. Attention 校准组件 × 时间（层 0 / 5 / 15，wall 5.91 / 5.73 / 5.55s）

| 组件 | 秒（3 层范围） | 占校准 | 官方贡献 |
|---|---|---|---|
| **R3 `_a2_train_rotation`（32 步解析梯度训练，Cayley 正交）** | 2.13–2.18s | **~37%** | R3 +396 的核心（不可单独砍） |
| **C76.4 variable H16/H32 旋转搜索**（v189 栈内，多 seed × 多块尺寸 × deployed MSE 门控） | 1.54–1.74s | **~28–30%** | 未单独定价（v189 栈内件） |
| 双轨 Q/K 候选搜索（`_run_selection` ×2，smooth d × perm × center × block，41 次 `_attention_candidate_metrics`） | 0.99–1.04s | ~17–18% | 未单独定价 |
| A1 终验门（`_attention_deployed_mse` ×2） | 0.24–0.27s | ~4.5% | 未单独定价 |
| logit gain 拟合（`_fit_attention_logit_gain`） | 0.19–0.24s | ~3.5–4% | 未单独定价 |
| R3 gate 损失 ×2（`_a2_true_path_gate_loss`） | 0.17–0.19s | ~3% | R3 一部分 |
| v158 pair-matrix smooth | 0.10–0.15s | ~2% | 未单独定价 |
| `_build_qk_states` + Q/K 变换/重要性构建 | 0.30–0.39s（嵌套帧合计） | ~6–7% | 部署必需 |
| 统计循环 + A1 恒等参考 + SAC K-center（C41） | ~0.10s | ~2% | 部署必需 |
| Fisher importance（C76.2）、A2 固定 H64、A3 V importance 候选 | 0.00s | 0% | 当前配置未触发/关闭 |

`_attention_deployed_mse` 全程 16 次调用共 ~1.6s，是 C76.4（其内部门控）+ A1 终验门 +
pair-matrix/logit-gain 门控的公共成本；上表已按调用点分摊到各桶。

## 4. 无法干净分离的组件（如实标注）

- **v186 E6M2 scale +4 窗口**：对应 `_DYNAMIC_OFFSETS = (-1,1,2,3,4)`（权重侧 `_WEIGHT_OFFSETS=(-1,1,2,3)`）
  的 codec 内 offset 搜索，融合在 `_dense_to_hif4` / `_activation_gptq_quantize` 内部的逐块搜索里，
  校准粒度无法拆分其份额（未分离）。官方 +1。
- **E6M2 offset / refinement（v159 栈内件）**：同为 codec 内 search_offsets + max_refine 机制，
  时间计入 Weight GPTQ / `_dense_to_hif4` 桶（未分离）。
- **quadratic AdaRound**：当前根没有独立激活的 AdaRound 例程；quadratic 路径实际形态是
  `cov_sum → _transformed_covariance → gram_full → _gptq_quantize_weight`，其时间已在
  "Weight GPTQ" 与 "quadratic Gram 变换" 两桶。

## 5. Ablation 短名单（按"本地份额高 × 官方贡献小/未测"排序）

> 注意：本地时间对官方时间**无预测力**（v194 教训：本地 calibration −22.5% 官方反而 +5s），
> 下表"省时"均为本地相对份额，最终时间与分数只能靠官方提交实测；且本次测量在共享 GPU 上完成，
> 绝对秒数只能按最小 wall 轮解读。

1. **L-R2 rank-2 残差拟合（Linear）** — 官方仅 **+1**（已定价、最小）；本地是残差块的主成本
   （rank-2 ≈ rank-1 的 1.5–2.7×，残差块合计占 Linear 校准 ~15–20%，非 GPTQ 层上为最大单块，
   proj 层追踪份额 52%）。机制开关现成：`_WEIGHT_RESIDUAL_RANK 2→1` 即回退 v166 单 rank。
   预期：砍掉换 Linear 校准 ~10% 级本地时间，已知分数代价 +1。**性价比最高**。
2. **C76.4 variable H16/H32 旋转搜索（Attention）** — 官方贡献**未单独定价**（v189 栈内件）；
   本地占 Attention 校准 **~28–30%**（6 层合计 ~9–10s 本地），是 Attention 侧最大的可裁块。
   它是纯搜索（多 seed × H16/H32/H64 × deployed-MSE 门控），砍掉不改变 R3/A1 等其他机制的
   语义，只需接受其 winner 可能消失的风险。建议以标准 Linear + 本候选的官方侧隔离提交定价：
   若侧分 ≤ 基线，砍掉即省 Attention 校准近三分之一。
3. **Weight GPTQ / quadratic 路径（Linear，fc_gate/fc_up/q/k/v 五 role）** — 本地最大时间块
   （五 role 内 56–83%，Linear 校准总体 ~65–70%），官方贡献**未单独定价**（v159 +671 的内件）。
   这是主量化路径，直接砍掉（回退 `_dense_to_hif4`）风险最高，不应作为"省时"手段盲目裁；
   建议作为**定价诊断**：一个"GPTQ→plain encode"的单机制官方候选，若官方掉分远小于预期，
   则回退 plain encode 可释放 Linear 校准一大半时间预算给新机制。

候补（份额较小或已定价，暂不推荐）：双轨 Q/K 搜索 ~17%（未定价，但承载 v159 以来全部 Q/K
变换选择）；logit gain ~4%、pair-matrix smooth ~2%（未定价）；激活 h_inv cholesky（仅 proj 类
大 d_in 层显著，服务 compiled activation GPTQ 路径，关联 +20/+17 已定价机制）；v202 预编译
~0.1–0.2s（保留）。R3 训练 ~37% 是 +396 的核心，v195 +21 已定价，均不在 ablation 范围。

## 6. 产物清单

- `audit.py` / `analyze.py`：插桩与分桶脚本（进程内 monkeypatch + settrace，不落盘改源码）。
- `results-fc_gate.json`（fc_gate 逐行追踪补测）、
  `results-wrap1.json` / `results-wrap2.json`（无追踪包装轮 ×2，含水文噪声）、
  `results-nosync.json`（无同步逐行轮，受共享 GPU 污染，仅存档）。
- `tables.md`：逐组件全量分桶表（Linear 5 role 同步追踪轮 + Attention 第一轮；Linear 5 role 的
  原始 JSON 在 fc_gate 补测时被同名覆盖，分桶结果完整保留在本文件）。
- 锚点来源：`artifacts/proxy_v3/full_solution/linear-sample-energy-fusion-six-shards/candidate/baseline-linear-shard*.json`
  与 `attn-legal-hierarchy-selection-six-shards-fixed/candidate/baseline-attention-shard*.json`
  （v195 根 SHA 839adb1e…，校准结构与 v202 仅差 order 预编译）。
