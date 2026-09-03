# v169 候选：A2 V 输出偏差质心补偿 + standard Linear（低复杂度扩展计划第二包）

> 状态：**REJECTED（本地否决，未提交官方；2026-09-03）**
>
> 否决依据：计划 §12 step 9 "跨模型没有整体结构性反向" 条款被违反（GPT-2 `0/4` 全负），
> 且机制级诊断显示前提缺席；按 §14 该状态为 REJECTED，不消耗官方提交。
>
> 官方共同基线：v162 `1001 / 146s`；Attention 官方父侧：v164（standard Linear + v160
> Attention）`13945 / 204s`（A2 从 P_A = v164 构造；A1 的 v168 官方结果未回传，不叠加）
>
> 候选 SHA256：`3E9307BC45EDE56E240380E09905C9FEF8577C9A461FC752ACDD8105EF67DAE8`
>
> 官方结果：`unregistered / NA`（未提交）

## 1. 唯一算法机制（预注册，低复杂度扩展计划 §5）

softmax 行和为 1，给同一 KV head 的所有 V token 加常向量 `b_h` 使该 head 输出平移
~`b_h`。校准期偶/奇折分别计算：

```text
b_f,h = mean_query_and_group_heads(O_ref − O_parent)   # causal，前 128 tokens
b_h   = 0.5 * coordinatewise_median_f(b_f,h)           # 预注册收缩，不搜索
```

`_nvfp4_to_hif4` 新增默认关闭参数 `additive_bias`（NVFP4 解码后、center/multiplier/
permutation/rotation 之前广播相加）；`v_state` 新增 `additive_bias/bias_version`；
V 动态 API 仅加一行传参；**Q/K API 与 state 逐位未动**；动态成本一遍广播加法。

## 2. 控制与合法性（全部通过）

| 检查 | 结果 |
| --- | --- |
| 隔离导入 + 六 API | OK |
| Q/K state vs v164 | **逐位一致**（四层 multiplier/importance/permutation 等全字段） |
| V state vs v164 | 基础字段逐位一致，仅新增 `additive_bias/bias_version` |
| state 合法性/有限输出 | OK |
| 机制可达 | 是（b 非零，V 输出实际变化） |

## 3. 否决证据（三层独立）

**机制级（前提缺席）**：compact 四层真实数据上，输出偏置修正幅度为
`−1.7% / −0.0% / −0.1% / +0.6%`（L23 反向）。父输出偏置由 Q/K 量化误差主导，
V 稳定分量可忽略——A2 的补偿目标在真实数据上基本不存在。b 本身：
max_abs `0.002–0.023`、head norm median `0.002–0.055`；V operand MSE 如预期变差
（如 L8 `128.29→128.95`）。

**Qwen panel**：compact mean `0.793085` vs 父 `0.797462`（−0.0044，四哨兵全负）；
default 120 mean `0.733044` vs 父 `0.742354`（**−0.0093**、median −0.0027、
`21+/99−`、worst −0.30）；V-only 组件 mean delta **−0.0154**——偏置的 operand
MSE 代价系统性超过其输出收益。

**GPT-2 跨模型**：mean **−0.0172，`0/4` 全负**（worst −0.051）；v_only −0.0152。
整体结构性反向成立，触发计划 §12 step 9 的阻止条款。

## 4. 纪律与后续

- 按预注册公式与 0.5 收缩实现并仅运行一次；不调 shrink/折定义/token 数/聚合方式
  （邻域禁令）；
- 不提交官方（§12 step 9 条款 + §14 REJECTED：跨模型整体结构性反向）；
- **A2 关闭，转 A3（动态 scale 搜索的静态策略编译）**；v168（A1）官方回传不受影响。

## 5. 证据

`v169-compact-attn.json`、`v169-attn-default.json`、`v169-gpt2-attn-compact.json`
（`artifacts/official_eval/`，对应 `logs/official_eval/` report）；机制级诊断
（b 统计/输出偏置前后/V MSE）在会话记录与本文件 §3。

## 6. 复现

```powershell
.venv\Scripts\python.exe -u evaluator\official_eval.py --solution solutions\20260903_v169_standard-linear_v-bias-attn_rejected\solution.py --attention-only --cache-mode read --nvfp4-cache-mode auto --capture-device cuda --algorithm-device cuda --baseline-json artifacts\official_eval\sidecal-v164-attn-default.json --output artifacts\official_eval\v169-attn-default.json --report logs\official_eval\v169-attn-default.md

.venv\Scripts\python.exe -u evaluator\cross_model_eval.py --model gpt2 --solution solutions\20260903_v169_standard-linear_v-bias-attn_rejected\solution.py --attention-only --compact-panel --cache-mode read --capture-device cuda --algorithm-device cuda --baseline-json artifacts\official_eval\s1-parent-v160-gpt2-attn-compact.json --output artifacts\official_eval\v169-gpt2-attn-compact.json --report logs\official_eval\v169-gpt2-attn-compact.md
```
