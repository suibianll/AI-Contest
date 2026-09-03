# v177 预研：C2 per-(KV-head, 8 通道组) logits 增益（REJECTED）

> 状态：**REJECTED（本地预研明确负优化，不提交官方、不占配额）**
>
> 性质：C2 计划 §2b 门禁要求 C1 官方回传后才启动"官方候选"；本版本是**本地预研
> （v168 基底）**——提前验证 C2 核心数学信号的目的是避免浪费官方提交配额，不
> 违反门禁（门禁约束的是官方提交，不是本地诊断）。
>
> 构造：基于 P_A = v168（standard Linear + A1 logits gain）+ C2 组级增益；
> A1 保持开启（C2 是 A1 细粒度化，乘性叠加在既有 multiplier 之上）。
>
> 候选 SHA256：`C5A573256FE430C7D45350E077665C3E9C2004F37FA23D36C96A63F2B83C340E`
>
> 官方结果：`not submitted（本地 REJECTED）`

## 1. 机制（计划 2026-09-04 C2，§2b 预注册数学）

- C2 = A1 的细粒度化：A1 是 per-KV-head 单一 logits 增益（B=1 特例）；
  C2 把 head_dim 划分为 B=8 连续通道组，per-(KV-head, 组) 独立乘性增益。
- 每 KV head 独立解闭式 8 参数线性最小二乘
  `min || center − Σ_b g_{h,b}·center_q^{(b)} ||²`（B×B Normal equation，
  calibration-only）；偶数/奇数 fold 分别拟合，log-median + A1 同款
  log-shrink(0.5)/clamp(sqrt(0.5)..sqrt(2.0))。
- 折叠：Q 与 K 同一通道组内同乘 sqrt(g_{h,b})——保持组内 QᵀK 乘性缩放
  结构，即量化后 `Σ_b g_{h,b}·partial_b` 逼近 float logits。
- 实现细节：per-Q-head 循环累积 (B×B) Gram 与 cross 到 fold 累加器
  （float64 CPU），避免大中间张量；B=8、seg=head_dim/8。
- 动态零新增：只改 state 中 Q/K multiplier。

## 2. 本地实测（描述性；官方不参与——已 REJECTED）

| 项目 | 结果 |
| --- | --- |
| 隔离导入 + 六 API | OK |
| 机制 reachability | group_logit_gain 写入 Q/K state，896/128 通道非恒等（随机数据 min 0.997 / max 1.001） |
| attention compact 4（配对 v168） | **mean Δgain −0.006858**、median −0.006017、1+/3−/0= |
| attention default 120（配对 v168） | **mean Δgain −0.006643**、median −0.004289、**41+/79−/0=（win 0.342）** |
| 组件分解（default） | QK-only −0.0065、**QK interaction −0.0935**（负交互）；k-only +0.0335、q-only +0.0536 |
| 时间 | attention default 校准 60.1s（v168 基线 68.4s）、动态 Q/K/V 3.4s，无风险（但与拒绝无关） |

**判读**：120 cases 上 win 0.342、QK interaction 显著为负——组级独立增益的
QK 联合反噬超过单侧表面积改善（q-only/k-only 为正但被 interaction 反转），
A1 的单一 per-KV-head 增益已捕获大部分可校正结构，B=8 细粒度化是负优化。
本预研对应计划 §2b 的「C1 官方负 → C2 从 P_A=v168」分支，该分支明确 REJECTED。

## 3. 对 C2 家族的影响

- 按计划 §2b：C2 官方提交仍须等 C1 官方回传裁决；若 C1 官方负，C2 分支
  直接用本预研结论 **REJECTED，不提交**，省配额。
- 若 C1 官方正，C2 应从 C1（v176）基底继续（C1 语义并入基础 multiplier）——
  该分支与 v168 基底不同，需 C1 回传后再单独本地预研，不能引用本预研作为
  该分支的裁决。但 v168 基底的强负交互提示组级增益机制整体风险高。

## 4. 复现

```powershell
.venv\Scripts\python.exe -u evaluator\official_eval.py --solution solutions\20260904_v177_c2-group-logit-gain_rejected\solution.py --attention-only --compact-panel --cache-mode read --nvfp4-cache-mode auto --capture-device cuda --algorithm-device cuda --baseline-json artifacts\official_eval\v168-compact-attn.json --output artifacts\official_eval\v177-compact-attn.json --report logs\official_eval\v177-compact-attn.md

.venv\Scripts\python.exe -u evaluator\official_eval.py --solution solutions\20260904_v177_c2-group-logit-gain_rejected\solution.py --attention-only --cache-mode read --nvfp4-cache-mode auto --capture-device cuda --algorithm-device cuda --baseline-json artifacts\official_eval\v168-attn-default.json
```