# v178 预研：C2 per-(KV-head, 8 通道组) logits 增益 on C1（REJECTED）

> 状态：**REJECTED（本地预研明确负优化，不提交官方、不占配额）**
>
> 性质：C2 计划 §2b 的「C1 官方正 → C2 从 C1 候选继续」分支预演。C1 官方未回传，
> 但提前验证该分支本地信号，避免 C1 官方正后盲目提交浪费配额；门禁约束的是官方
> 提交，不约束本地诊断。
>
> 构造：基于 v176（= v168 + C1 K-side outlier-channel 等化）+ C2 组级增益；
> A1 与 C1 保持开启（C2 乘性叠加在既有 multiplier 之上）。
>
> 官方结果：`not submitted（本地 REJECTED）`

## 1. 机制与判读口径

- C2 数学同 v177（计划 §2b：per-(KV-head, B=8 通道组) 闭式 8 参数最小二乘
  logits 增益，fold median + A1 同款 shrink/clamp，同组同乘 sqrt(g)）。
- 与 v177 的唯一差异：父基底为 v176（含 C1 的 k_eq/q_eq），C2 的 g 拟合与
  折叠都发生在「A1 × C1」之后的 multiplier 之上——即 C2 捕获的是 C1 处理后
  的残差偏置。这正对应计划 §2b 的 C1 正分支组成。

## 2. 本地实测（描述性；官方不参与——已 REJECTED）

| 项目 | 结果 |
| --- | --- |
| 隔离导入 + 六 API | OK（py_compile + 评测器加载通过） |
| attention compact 4（配对 v176） | **mean Δgain −0.009194**、median −0.002469、1+/3−/0= |
| attention default 120（配对 v168，同 v177 口径） | **mean Δgain −0.008610**、median −0.002788、**47+/73−/0=（win 0.392）** |
| 时间 | attention default 校准 60.2s、动态 Q/K/V 3.5s，无风险（与拒绝无关） |

**对比 v177（C2 on v168）**：default mean −0.006643（41+/79−）→ v178（C2 on
C1）−0.008610（47+/73−）。C2 在 C1 基底上同样明确负，且整体更负——8 组独立
增益的负交互在 C1 处理后仍存在，A1 单一增益与 C1 通道等化已捕获可校正结构，
组级细粒度化无剩余余量。

## 3. C2 家族裁决

- v177（C2 on v168）= REJECTED；v178（C2 on C1）= REJECTED。
- **C2 的两个预注册分支本地均明确负优化，C2 家族本地关闭**：无论 C1 官方正
  负，C2 都不应提交官方（省 1 配额）。C2 从候选清单移除，后续按计划切换
  C3 条件对照（仅 C1/C2 官方均负时——C2 本地已否，等价于仅 C1 官方负时
  考虑一次 C3 对照）。

## 4. 复现

```powershell
.venv\Scripts\python.exe -u evaluator\official_eval.py --solution solutions\20260904_v178_c2-on-c1-pre-research_rejected\solution.py --attention-only --compact-panel --cache-mode read --nvfp4-cache-mode auto --capture-device cuda --algorithm-device cuda --baseline-json artifacts\official_eval\v176-compact-attn.json --output artifacts\official_eval\v178-compact-attn.json --report logs\official_eval\v178-compact-attn.md

.venv\Scripts\python.exe -u evaluator\official_eval.py --solution solutions\20260904_v178_c2-on-c1-pre-research_rejected\solution.py --attention-only --cache-mode read --nvfp4-cache-mode auto --capture-device cuda --algorithm-device cuda --baseline-json artifacts\official_eval\v168-attn-default.json --output artifacts\official_eval\v178-attn-default.json --report logs\official_eval\v178-attn-default.md
```