# v161 候选：S1 交叉算子 Gram64 per-call 精化

> 状态：**TIMEOUT / REJECTED（官方，2026-09-03 用户回传：`>300s`，无分数）**
>
> 父版本：v160 归档，SHA `33B1D061CE6BFCD92659C597BE4830BB9B910E646FF518433DA67B925AE8680D`
> （官方 `17532 / 232s`）
>
> 候选 SHA256：`27EEE4710B0170384A17E2F3E9AB87B3437E7B224883150D70BEBF8A5FB11848`
>
> 官方结果：**timeout（`>300s`，无分数回传）**

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

## 2. 验证漏斗结果（本地全部通过；官方 timeout 推翻时间外推）

| 阶段 | 结果 | 证据 |
| --- | --- | --- |
| A 单元 | 6 API 脱离仓库导入；交叉 Gram 对称半正定、形状正确；refine 单调（gram loss `404.2→313.2`，重建 MSE `0.000621→0.000765`，符合"用重建换 logits"的机制设计）；params 布局不变 | `workbench/s1_unit_check.py` |
| B Qwen compact 4 哨兵 | `0.797462→0.819033`，paired `+0.021571`（2+/2−），median `+0.000561`，API `10.4s`（parent `10.8s`） | `s1-gram-refine-attn-compact.json` |
| C Qwen default 120 | `0.742354→0.794856`，**paired `+0.052502`（106+/14−/0，touch 88.3%）**，median `+0.040285`，MSE ratio `0.817`；attention API `85.995s` vs parent `57.97s`（+28.0s，通过本地 +40s 门禁；dyn Q/K `0.092s/call` CUDA，calib `2.62s/call`） | `s1-gram-refine-attn-default.json` |
| C GPT-2 compact 同号 | `+0.067751`（3+/1−），median `+0.056450`，与 Qwen 同向同量级 | `s1-gram-refine-gpt2-attn-compact.json` |
| D | SHA、单文件隔离导入、本 result.md | — |
| **E 官方** | **timeout（`>300s`，无分数）** | 用户回传 2026-09-03 |

D1 预注册判别器本地满足（touch 88.3% ≥ 50%、106 > 14、median +0.040 ≥ 0），但官方
timeout 使精度方向未被检验——D1 证据维持 3/3，v161 不计入（无官方分数）。P9 检验
（17816 的 +284 缺口归因）同样无法记录。

## 3. 官方超时归因（时间外推门禁失效）

- v161 相对 v160 的唯一时间增量：校准期 gram 计算（本地 CUDA `+7.3s`）+ 动态 Q/K
  3-sweep 精化（本地 CUDA `+19.7s`，`0.092s/call`）；官方外推 `~257s < 300s`；
- 实际官方 timeout 说明：官方机（鲲鹏，本地不可复制）上 per-call 小张量算子
  （einsum/topk/gather/逐坐标闭式更新）的有效成本是本地 CUDA 外推的数倍以上，
  v160 官方 232s 的 68s 余量被完全耗尽；
- 家族对照：v138（删除 dyn refine 与校准搜索）官方 `208s` 通过；v128/v129/v130/v131
  （含 dyn refine）官方全部 timeout；v161（只保留 dyn refine + gram，无校准搜索）仍
  timeout。**结论：v128 家族超时元凶不只是校准期候选搜索，动态 per-call 精化本身在
  官方机上也超预算**——此前"dyn refine 仅 0.08s/call"的本地 CUDA 口径核算不适用于
  官方硬件；
- 精度余量（本地 default +0.0525、GPT-2 同号）在官方 300s 预算内无法回收，per-call
  动态自适应族结构性关闭。

## 4. 官方提交与解释（预注册，已回传）

- 一次官方提交，用户执行；预测：官方 Δ > 0（D1）；
- 官方 timeout → 时间门禁失效，REJECTED（本行生效）；
- 按预注册不缩 sweeps 重试（sweeps/gram/块/topk 均不动）；S2（校准搜索解析化）前置
  条件"S1 官方正向"不满足，不启动；
- 本计划归档；本地已知机制族全部闭环（Linear 结构 full64/Householder、Attention
  解析静态族、Attention per-call 动态族）。

## 5. 复现

```powershell
# Qwen attention default（parent 与候选）
.venv\Scripts\python.exe -u evaluator\official_eval.py --solution <v160归档或v161> --attention-only --cache-mode read --nvfp4-cache-mode auto --capture-device cuda --algorithm-device cuda --baseline-json artifacts\official_eval\s1-parent-v160-attn-default.json ...

# GPT-2 compact
.venv\Scripts\python.exe -u evaluator\cross_model_eval.py --model gpt2 --solution <候选> --attention-only --compact-panel --cache-mode read --capture-device cuda --algorithm-device cuda --baseline-json artifacts\official_eval\s1-parent-v160-gpt2-attn-compact.json ...
```

设计依据：[`归档计划`](../../docs/superpowers/archive/plans/2026-09-03-attention-per-call-refinement-plan-superseded.md)、
[`Step 0 消融`](../logs/execution/2026-09-03-s0-v128-family-ablation.md)、
[`官方超时日志`](../logs/execution/2026-09-03-v161-official-timeout.md)。
