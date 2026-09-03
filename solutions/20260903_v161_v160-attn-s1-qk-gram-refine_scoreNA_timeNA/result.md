# v161 候选：S1 交叉算子 Gram64 per-call 精化

> 状态：**CANDIDATE — 本地全漏斗通过，等待一次官方提交**
>
> 父版本：v160 归档，SHA `33B1D061CE6BFCD92659C597BE4830BB9B910E646FF518433DA67B925AE8680D`
> （官方 `17532 / 232s`）
>
> 候选 SHA256：`27EEE4710B0170384A17E2F3E9AB87B3437E7B224883150D70BEBF8A5FB11848`
>
> 官方结果：`unregistered / NA`

## 1. 唯一算法变化

Attention Q/K 的 per-call 交叉算子 Gram64 精化（v128 家族机制的 v160 坐标系移植）：

- **校准期**（追加，不动任何既有 gate/state）：在全部 v160 既有拟合（A1 终验门、A2、
  Matrix-Smooth、V importance）完成后，用最终部署坐标变换链重放校准 Q/K 样本，计算
  交叉块 Gram——Q 每 head 每 64-block 用同组 K 的 `X^T X`，K 用对应 Q 组的——存入
  `q_state["gram64"]` / `k_state["gram64"]`（CPU 有限张量，合法 state）；
- **动态期**（追加）：`_nvfp4_to_hif4` 产出 params 后，对当前序列码字做 3-sweep 有界
  坐标下降（sweeps=3 为 v128 固定值，不扫描），最小化块内 `error^T · G64 · error`；
  Q/K 的 G64 即 QK logits 误差的精确二阶 Hessian；
- **V 路径 bit-exact 不变**；Linear 全部冻结；无搜索、无在线矩阵求逆、无候选循环。

连续域变换不变（精化只动离散码字）；Gram 在最终部署坐标计算（AGENTS.md §2 约束）。

## 2. 验证漏斗结果（全部通过）

| 阶段 | 结果 | 证据 |
| --- | --- | --- |
| A 单元 | 6 API 脱离仓库导入；交叉 Gram 对称半正定、形状正确；refine 单调（gram loss `404.2→313.2`，重建 MSE `0.000621→0.000765`，符合"用重建换 logits"的机制设计）；params 布局不变 | `workbench/s1_unit_check.py` |
| B Qwen compact 4 哨兵 | `0.797462→0.819033`，paired `+0.021571`（2+/2−），median `+0.000561`，API `10.4s`（parent `10.8s`） | `s1-gram-refine-attn-compact.json` |
| **C Qwen default 120** | `0.742354→0.794856`，**paired `+0.052502`（106+/14−/0，touch 88.3%）**，median `+0.040285`，MSE ratio `0.817`；attention API `85.995s` vs parent `57.97s`（**+28.0s ≤ +40s 门禁**；dyn Q/K `0.092s/call`，calib `2.62s/call`） | `s1-gram-refine-attn-default.json` |
| C GPT-2 compact 同号 | **`+0.067751`（3+/1−）**，median `+0.056450`，与 Qwen 同向同量级 | `s1-gram-refine-gpt2-attn-compact.json` |
| D | SHA、单文件隔离导入（单元检查第一段）、本 result.md | — |

**D1 预注册判别器（OPA-1 账本 §5）：touch 88.3% ≥ 50% ✓、improved 106 > regressed 14 ✓、
median +0.040 ≥ 0 ✓ → 预测官方正向**（现有 D1 证据 3/3；本次提交若正向则 4/4）。

## 3. 时间预算

- 本地 default：attention 侧 +28.0s（calib +7.3s 含 gram 计算、dyn Q/K +19.7s）；
- 官方外推：v160 官方 232s 中 attention 侧约 46s（按本地比例），S1 增量按同机比
  ~25s → **官方估算 ~257s < 300s，余量 ~43s**；
- 本地秒数不换算官方分数，只作复杂度门禁。

## 4. 官方提交与解释（预注册）

- 一次官方提交，用户执行；预测：官方 Δ > 0（D1）；
- 官方正向 → per-call 自适应精化可迁移，下一机制候选 S2（校准搜索解析化）；
- 官方零/负 → D1 对 per-call 族失效，降级记录，不邻域调参（sweeps/gram/块不动）；
- 官方 timeout → 时间门禁失效，REJECTED；
- 官方回传同时记录 P9 检验（17816 的 +284 缺口归因），不作调参输入。

## 5. 复现

```powershell
# Qwen attention default（parent 与候选）
.venv\Scripts\python.exe -u evaluator\official_eval.py --solution <v160归档或v161> --attention-only --cache-mode read --nvfp4-cache-mode auto --capture-device cuda --algorithm-device cuda --baseline-json artifacts\official_eval\s1-parent-v160-attn-default.json ...

# GPT-2 compact
.venv\Scripts\python.exe -u evaluator\cross_model_eval.py --model gpt2 --solution <候选> --attention-only --compact-panel --cache-mode read --capture-device cuda --algorithm-device cuda --baseline-json artifacts\official_eval\s1-parent-v160-gpt2-attn-compact.json ...
```

设计依据：[`活动计划`](../../docs/superpowers/plans/2026-09-03-attention-per-call-refinement-plan.md)、
[`Step 0 消融`](../logs/execution/2026-09-03-s0-v128-family-ablation.md)。
