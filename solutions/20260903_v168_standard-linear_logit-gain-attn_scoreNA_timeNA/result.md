# v168 候选：A1 解析 logits 增益校正 + standard Linear（低复杂度扩展计划首个工作包）

> 状态：**RETAINED — 官方 `14005 / 210s`（2026-09-03 用户回传，同日修正：初报 17248/237s 有误）；A1 晋级为新 Attention 父侧**
>
> 官方共同基线：v162 `1001 / 146s`；Attention 官方父侧：v164（standard Linear + v160 Attention）
> `13945 / 204s`
>
> 父版本：v164 归档（Attention = v160 逐位；Linear = standard tail，未改动）
>
> 候选 SHA256：`5988AE47EAC2E7DDE7488E06B8F91939F5660A585034280A6D68A8FB6701AC79`
>
> 官方结果：**14005 / 210s**（step_gain `+60`，Attention ratio `0.46%`）

## 1. 唯一算法机制（预注册，低复杂度扩展计划 §4）

校准期在父 q_state/k_state 最终确定后、return 前拟合每 KV head 的乘性 logits 增益：

```text
L_f  = row_center(Q_f K_f^T / sqrt(d))      # causal 合法前缀内逐行去均值
Lq_f = row_center(Qhat_f Khat_f^T / sqrt(d)) # 父部署路径（含全部变换）
raw_gamma_f = <Lq_f, L_f> / (<Lq_f, Lq_f> + eps)   # clamp [0.5, 2.0]
gamma = exp(0.5 * median_f(log(raw_gamma_f)))       # 预注册 log 域 0.5 收缩
g_q = g_k = sqrt(gamma)                             # 折入两侧 multiplier
```

- 折 = 校准窗口偶/奇索引；每样本取前 128 tokens；GQA 组内共享 gamma；
- **动态 API 零新增运算**：校正折叠进现有 `multiplier` 逐通道路径（K center 之后、
  permutation 之前；head-major 布局下 64-block 与 head 对齐，逐 head 常数增益近似
  齐次通过 amax 编码）；state 新增 `logit_gain`（仅审计）；
- 完全满足 v165 约束（无 Gram contraction、无候选循环、复杂计算只在 calibration）。

## 2. 机制诊断（§4.4 记录要求）

| 项目 | 数值 |
| --- | --- |
| 真实 gamma（default 24 层 × 2 KV head） | min `0.9876` / median `1.0001` / max `1.0063`；48 head 中 8 个偏离 >0.005 |
| 折间一致性（compact 4 层） | 折间离散 0.002–0.006 << 偏置量：**偏置稳定存在但幅度 ≤1.5%** |
| logits slope（compact 4 层） | parent 0.983–1.007（无一致偏置方向）→ corrected 0.986–0.991 |
| layer 16 | gamma `[0.9935, 0.9876]`（全网络最大校正量）→ default 回归集中层（见下） |
| Q/K state | multiplier 全部 24 层变化；V state 未动 |

## 3. 本地验证（描述性；官方裁决）

| 检查 | 结果 |
| --- | --- |
| 隔离导入 + 六 API | OK |
| attention compact 4（配对 v164） | mean **0.797753** vs 父 0.797462（+0.000291，四哨兵混合微小变动） |
| attention default 120（配对 v164） | mean **0.741474** vs 父 0.742354（**−0.00088**）；median +0.000175、`66+/54−/0`、worst `−0.160`（layer16 len10）、best +0.034；按长度 len10 −0.0034 / len128 +0.0000 / len512 −0.0028 / len1024 +0.0009 |
| 回归结构 | layer 16 全 5 长度负（−0.16/−0.061/−0.040/−0.021/...）——该校准 gamma 偏差最大（0.9876），校正量最大且校准→测试不迁移；其余 23 层噪声级 |
| GPT-2 compact 4（配对 v160 attention = v164 同侧） | mean **0.443404** vs 父 0.440956（**+0.002447**，`3+/1−`，worst −0.007）；logit_mse 组件 3/4 层改善；无整体结构性反向 |
| API 时间（GPT-2 A/B） | calibration 5.10→5.35s（1.05×，每 state +~60ms）；动态 Q/K/V 与父基本持平 |
| 非目标侧 | V 逐位一致（合成探针验证）；Linear = standard tail 未触碰 |

证据：`v168-compact-attn.json`、`v168-attn-default.json`、`v168-gpt2-attn-compact.json`
（`artifacts/official_eval/`，对应 report；gamma/slope/折诊断在会话记录与本文件 §2）。

## 4. 风险记录（不阻止提交，预注册流程 §12 step 9）

- **期望官方效应量级小**：机制校正的偏置本身 ≤1.5%，且 default 出现 layer 16 型
  过校正（最大校正量恰是最大回归层）；本地 default mean 微负 −0.0009、GPT-2 微正
  +0.0024——两侧本地信号均在小量级混合；
- 该机制的官方测量价值：干净裁决 A1 假设（"稳定乘性 logits 偏置可校正"在官方
  panel 上是否有净收益）；`S_A ≤ 13945` 即关闭 A1 转 A2；
- 官方时间无风险：本地校准增量 ~1.4s/24 states，动态零增量（v164 官方 204s 基础上）。

## 5. 官方提交与判读（预注册 §3.3）

```text
step_gain   = S_new - 13945
side_contrib = S_new - 1001
Attention ratio = (S_new - 13945) / 12944
```

官方回传 `S_new = 14005 / 210s`（2026-09-03 同日修正：初报 `17248 / 237s` 有误）：

```text
step_gain    = 14005 − 13945 = +60
side_contrib = 14005 − 1001  = 13004
Attention ratio = 60 / 12944 ≈ 0.0046（0.46%）
```

**A1 RETAINED**：`14005 > 13945`，`210s < 300s`（相对 v164 仅 `+6s`）——v168 成为
新 Attention 父侧 `P_A`。小幅正向增量（与 v166 Linear 的 `+3` 同量级）；本地 proxy
同样无正向信号（Qwen default mean `−0.00088`、GPT-2 `+0.0024`）——本地-官方不可
排序纪律仍成立，但初报的"最大单机制增量/完全反转"结论随修正撤销；官方 logits
偏置的可校正空间为小量级。注：A2（v169）、A3（v170）均在等待回传期间按独立父侧
v164/v168 构造并完成裁决，归因不受本修正影响。

## 6. 复现

```powershell
.venv\Scripts\python.exe -u evaluator\official_eval.py --solution solutions\20260903_v168_standard-linear_logit-gain-attn_scoreNA_timeNA\solution.py --attention-only --cache-mode read --nvfp4-cache-mode auto --capture-device cuda --algorithm-device cuda --baseline-json artifacts\official_eval\sidecal-v164-attn-default.json --output artifacts\official_eval\v168-attn-default.json --report logs\official_eval\v168-attn-default.md

.venv\Scripts\python.exe -u evaluator\cross_model_eval.py --model gpt2 --solution solutions\20260903_v168_standard-linear_logit-gain-attn_scoreNA_timeNA\solution.py --attention-only --compact-panel --cache-mode read --capture-device cuda --algorithm-device cuda --baseline-json artifacts\official_eval\s1-parent-v160-gpt2-attn-compact.json --output artifacts\official_eval\v168-gpt2-attn-compact.json --report logs\official_eval\v168-gpt2-attn-compact.md
```
