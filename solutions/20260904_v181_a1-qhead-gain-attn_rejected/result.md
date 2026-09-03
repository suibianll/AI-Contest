# v181 预研：D2 per-Q-head logits gain（A1 head 维分解对照）（REJECTED）

> 状态：**REJECTED（本地预研明确负优化，不提交官方、不占配额）**
>
> 性质：2026-09-04-post-official-a1-freedom-plan §2 D2 对照——把 A1 的
> per-KV-head logits gain 分解到 per-Q-head（14 个独立 gain），K 保持
> per-KV-head 共享，检验 GQA 组内一致性结构是否承重。
>
> 构造：基于 v175（P_A = v168 A1 logits gain）的 Q/K multiplier 路径叠加
> per-Q-head 乘性 gain（校准期闭式回归，D1 折叠关闭、A1 对称父上的干净对照）。
>
> 官方结果：`not submitted（本地 REJECTED）`

## 1. 机制（计划 2026-09-04 D2）

- A1 是 per-KV-head（GQA 组内 7 个 Q head 共享一个 KV-head gain），D2 给每个
  Q head 独立 gain，仅注入 Q multiplier（gain 的 sqrt），K 保持 per-KV-head 共享。
- 这打破连续域 QK^T 内积不变（每个 Q-head logits 行独立缩放），与 C2（通道维
  细粒度）正交但同属「打破 A1 组内一致结构」的风险族。
- 动态零新增：只改校准期 state 中 Q multiplier 值；v165 约束满足。
- 单配置预注册，不搜索 head 粒度/增益阈值邻域。

## 2. 本地实测（描述性；官方不参与——已 REJECTED）

| 项目 | 结果 |
| --- | --- |
| 隔离导入 + 六 API | OK（校准 + 动态 Q/K smoke 通过） |
| 纯 D2 default 120（配对 v168） | **mean Δgain −0.002746**、median −0.000086、54+/66−/0=、median MSE ratio 1.000333、QK interaction +50.96 |
| D1+D2 叠加 default 120（配对 v168） | mean Δgain −0.002019、median +0.000026、60+/60−/0= |
| 纯 D2 compact 4（配对 v175） | mean +0.000595、4+/0−（哨兵噪声，default 才是决策面板） |

**判读**：default 120 上纯 D2 中位数与 win rate 均负向（median −0.000086、
54/66），与 v180（D1）default +0.000356、69+/51− 相反；per-Q-head 分解打破
A1 已验证的 GQA 组内一致结构，重复 C2 式负交互（QK interaction 巨大正但
Q-only/K-only 深度负，说明收益只在 A1 组内共享的折叠结构下成立）。对照结论：
**A1 的 group-consistent 结构是承重组件，D2 家族关闭**。

## 3. 对候选清单的影响

- D2 对照完成，确认 A1 group-consistent 结构承重；per-Q-head 粒度不注册候选。
- D1（v180）官方已 `17597/242s` RETAINED（+3 vs v175），D3 组合由 v180 本身完成；
  本轮计划 D1/D2/D3 全部裁决完毕，不新增配额消耗。
- 禁止围绕 alpha 或 per-head 粒度邻域继续扫描。

## 4. 复现

```powershell
# 纯 D2（D1 折叠关闭）default 120，配对 v168（= v175 attention 父侧）
.venv\Scripts\python.exe -u evaluator\official_eval.py --solution solutions\20260904_v181_a1-qhead-gain-attn_rejected\solution.py --attention-only --cache-mode read --nvfp4-cache-mode auto --capture-device cuda --algorithm-device cuda --baseline-json artifacts\official_eval\v168-attn-default.json --output artifacts\official_eval\v181d2-only-attn-default.json --report logs\official_eval\v181d2-only-attn-default.md
```

归档 SHA：`dc0fcdb0d22fd6bd4436cc5d9eda4c2fe84b7f48f9943e0d4bb70171d61164f3`
（D1 折叠 OFF 的纯 D2 版；D1+D2 叠加中间版 SHA `86b0762c99ef2ce6...` 仅作探索记录）。
