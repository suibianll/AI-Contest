# v171 候选：A4 矩匹配 mantissa 阈值 + standard Linear（低复杂度扩展计划第四包）

> 状态：**CANDIDATE — 本地机制完整、控制干净；按用户指示纳入官方提交队列（本地 proxy
> 不裁决，官方判定）**
>
> Attention 官方父侧：v168（A1 晋级，`14005 / 210s`，同日修正：初报 17248/237s 有误）
>
> 候选 SHA256：`4469B85B53F5ADEFC6CFE4FBF136BDD4D7FF9FFC48A815592C95864A7287A844`
>
> 官方结果：`unregistered / NA`

## 1. 唯一算法机制（预注册，低复杂度扩展计划 §7）

每个 operand/layer 校准一个标量 mantissa 舍入阈值 tau（Q/K/V 各一）：每折 8 次二分
`mean(|Q_tau(x)| − |x|) = 0`，`code = floor(z + 1 − tau)`（z = 归一化绝对值），
`tau = 0.5 + 0.5·(median_f(tau_f) − 0.5)`，区间 [0.25, 0.75]。仅在 `group_gram is None`
的 mantissa 路径生效（有 gram 的保持父 adaround，不混用两种算法）；动态只用一次
floor 替代 round，无候选循环。新增 `_params_denominator`/`_round_mantissa_threshold`，
`_solve_exact_hierarchy`/`_dense_to_hif4`/`_nvfp4_to_hif4`/三动态 API 透传
`rounding_threshold`。A1 增益、A2/A3 机制均未含入（本包从 v168 干净构造）。

## 2. 本地验证（描述性；官方裁决）

| 检查 | 结果 |
| --- | --- |
| 隔离导入 + 六 API | OK |
| 机制可达 | 是（24 层 q/k/v state 均写入 rounding_threshold，mantissa 变化） |
| attention compact 4（配对 v168） | 0.797457 vs 父 0.797753（−0.0003，近中性） |
| attention default 120（配对 v168） | 0.740808 vs 父 0.741474（**−0.0007**、median −0.0003、`56+/64−`、worst −0.027）——与 v168 本地信号同量级 |
| GPT-2 compact 4（配对 v168） | **+0.009573**（`2+/2−`，worst −0.005）——不构成结构性反向 |
| opt-125m attn 60（配对 v160 父） | **mean Δgain −0.004172**、median −0.004290、26+/34−/0=（win 0.433）——轻微负向，同 Qwen 方向一致 |
| API 时间 | default 120：64.0s vs v168 72.1s（本地 −8s） |

证据：`v171-compact-attn.json`、`v171-attn-default.json`、`v171-gpt2-attn-compact.json`
（`artifacts/official_eval/`，对应 `logs/official_eval/` report）。

## 3. 判读（按计划 §3.3）

```text
step_gain    = S(v171) − 13945
Attention ratio = step_gain / 12944
```

`S(v171) > 13945` → A4 RETAINED 成为新 Attention 父侧；`≤` → REJECTED 转 L1-L4
（Linear 侧）。按用户指示所有计划内优化均实现并提交官方裁决，本地 proxy 不否决。

## 4. 复现

```powershell
.venv\Scripts\python.exe -u evaluator\official_eval.py --solution solutions\20260903_v171_standard-linear_moment-threshold-attn_scoreNA_timeNA\solution.py --attention-only --cache-mode read --nvfp4-cache-mode auto --capture-device cuda --algorithm-device cuda --baseline-json artifacts\official_eval\v168-attn-default.json --output artifacts\official_eval\v171-attn-default.json --report logs\official_eval\v171-attn-default.md

.venv\Scripts\python.exe -u evaluator\cross_model_eval.py --model gpt2 --solution solutions\20260903_v171_standard-linear_moment-threshold-attn_scoreNA_timeNA\solution.py --attention-only --compact-panel --cache-mode read --capture-device cuda --algorithm-device cuda --baseline-json artifacts\official_eval\v168-gpt2-attn-compact.json --output artifacts\official_eval\v171-gpt2-attn-compact.json --report logs\official_eval\v171-gpt2-attn-compact.md
```
