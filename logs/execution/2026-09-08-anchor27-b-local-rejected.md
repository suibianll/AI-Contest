# A27-B 执行日志：LOCAL_REJECTED（2026-09-08）

循环框架：[误差账本驱动的持续研究循环](../../docs/superpowers/plans/workpackages/2026-09-08-continuous-research-loop.md) R-2（出卡→实现→验证→gate→面板→登记）。
完整证据链与逐层分解见 [anchor27-b/report.md](../../workbench/continuous_attention/anchor27-b/report.md)。本日志只记执行事实与裁决依据。

## 事实

- 机制：低维谱系数 S（每 KV 组 8 维 c_g，S_g = U·diag(band_expand(c_g))·Uᵀ，U = 归一化 Kronecker 序 Hadamard 256×256，8 个连续 32 宽谱带）；训练 = 真实五字段读出 MSE 中心差分符号梯度（ε=0.05、η=0.05、8 步坐标轮换）；部署栈单主干 B = v189 基座校准（无旧训练），gate 末折严格小于接受、平局保 B，拒绝层回退 base_stack。
- 候选 SHA：`8306444fd8a5ecdd39e1a48e8fb63672d473be0b5bf6baa87def63ec87e17147`。
- 验证链：verify_math 6/6（V4 切片读出与全动态 API 逐位 0.0）；smoke PASS；gate_check 4/6 接受（L0 −0.08%、L1 −3.8%、L5 −7.3%、L22 −0.26%；L8 +5.0%、L15 +1.1% 拒绝）。
- 4B 面板（六 shard × 12 case，shard-balanced 层映射 s0→L0、s1→L1、s2→L8、s3→L15、s4→L22、s5→L5）：
  - vs R3（`a5c679d7...`，官方 14405/238s）：加权 Δmean = **−0.006442**（10正/38负/24零）。
  - vs v189 B（`26120224...`，官方 17616/275s）：加权 Δmean = −0.001640（20正/28负/24零）。
  - R3 旧训练 4B 净贡献（R3−B）：+0.004677；逐层恒等式 vs_r3 = vs_b − (R3−B) 全部吻合。

## 预注册判据裁决

mechanism.md 第 6 条："6 层 gate 接受数 < 3/6，或 4B 72 例目标侧 attention Δmean(vs R3) 未改善（≤0）→ 关闭本卡。"
实测：gate 4/6（第一条通过）；Δmean(vs R3) = −0.006442 ≤ 0（**第二条触发**）→ LOCAL_REJECTED，不提交官方。父 A23 与官方状态不变（SIDE_PARENT_STAYS_A23）。

## shard2/3 零差异反常裁决

面板初查 shard2/3 的 Δmean 与 L1 恰为 0.000000（两基线下皆然），触发三种假设：case 缓存复用 / 层映射错误 / 标准路径回退。逐 case 裁决（对比 candidate/baseline/R3-B 三组 case_scores）：

1. vs_b shard2/3 全零 → 候选在 L8/L15 逐位 = v189 基座：这两层 gate 拒绝，`fallback_states="base_stack"`，与 gate_check 一致。
2. R3−B shard2/3 全零 → R3 旧训练在 L8/L15 本就零部署（R3 gate 型旧训练未覆盖全部 FA 层；A22-1 历史验证"强制拒绝候选≡R3 逐位"为旁证）。
3. ∴ 候选 ≡ R3 ≡ B 在 L8/L15 逐位成立，零差异为数学必然。层映射由 case 级 layer 字段证实（shard2 case0 layer=8）。

## 根因（三层，相互独立）

1. **机制净效果负**：vs_b = −0.00164，最有利对照（纯基座）下仍负；L0/L22 微正、L1/L5 负（L5 −0.0106 最重）。
2. **gate 校准折过拟合不 transfer**：gate 末折 4/6 改善（−0.08%~−7.3%）但同层 4B 独立窗 3 负 1 正。32 低维参数/层未阻止过拟合——根源是目标分布（校准折 token）而非参数量。
3. **单主干结构负担**：丢 R3 旧训练损失 +0.004677，与 A25 官方 −380 教训同构；但扣除后机制本身仍负，两层根因独立成立。

## 关闭边界

- 关闭：Kronecker-Hadamard 8 谱带 + 中心差分符号梯度实现（含冻结 ε/η/步数/token_cap）。
- 不关闭：F2 格（0.192 仍 OPEN）；其他正交基/解析结构的码分配机制；逐层 gate 回退结构（安全性再次逐位确认）。
- F2 格"折内训练类"两路线关闭：A26-A（解析一阶代理，与真实编码器系统性反向）、A27-B（真实读出+数值优化，分布不 transfer）。

## 工程教训（固化）

- token-major 切片 (T, heads, dim) 的 `[:, g]` 挤压维度；组内 attention 必须先 transpose 到 (heads, tokens, dim) 再算 token-token logits——否则 logits 退化为 (T, pg, 1)，softmax 后广播不报错但数值全错（V5 曾静默 garbage，靠 debug_shapes.py 实测形状定位）。
- 保维切片用 `g:g+1`；`torch.diag_embed` 作用对象是带扩展后的 256 维对角线而非 8 维系数。
- 修复后统一由 gen_solution.py 重新生成（生成器为源），SHA 从 4c10076a... 变为 8306444f...。

## 工件

- 归档：`workbench/continuous_attention/anchor27-b/`（mechanism.md、config.json、gen_solution.py、solution.py、verify_math、smoke、gate_check、probe_cost、run.py、report.md）。
- 面板：`artifacts/proxy_v3/continuous/attention/anchor27-b/{vs_r3,vs_b}/`（六 shard analysis JSON+md + manifest）。
- 登记：state.json（status/next_action/recent_steps）、queue.md §8-§10。
