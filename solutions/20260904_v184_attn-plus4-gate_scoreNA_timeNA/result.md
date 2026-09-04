# v184 候选：attention 双窗全流程 +4 scale-code 门控（Attention）

> 状态：**本地三模型验证完成，待提交决策（时间风险见 §5）；配额 5/10 未消耗**
>
> 来源诊断：[`2026-09-04-big-gain-space-diagnostics.md`](../../logs/execution/2026-09-04-big-gain-space-diagnostics.md)
> ——oracle 分解证明 +4 单码捕获 99.6% 穷举增益（+0.0103），且增益主体来自
> 「gate 决策在 5 码窗口下翻转」，非纯在线窗口或 gamma 重拟合（两者实测各仅
> +0.001）。
>
> 构造：v182 官方父（`17598/273s`）+ `_calibrate_attention_core` 窗口参数化 +
> `hif4_calibration_attention` 双窗 wrapper：每层完整校准跑两遍（标准 4 码 +
> 5 码 +4），用 `_attention_deployed_mse` + `_a1_gate_passes`
> （safety_tolerance=0.0，与 A1/rotation 门同判据）选优。拒绝层位级 = v182。

## 1. 机制

- 在线 Q/K/V scale 搜索窗口 `(-1,1,2,3)` vs `(-1,1,2,3,4)`，每层独立。
- 增益机制（oracle 分解链）：5 码全流程校准使 rotation/pair-matrix/A1 等
  gate 决策翻转（L11：校准 MSE 0.002428→0.001274 减半），纯窗口 swap 或
  gamma 重拟合单独实测均仅 +0.001。
- 每层门控：双窗分支部署 MSE 比较 + `_a1_gate_passes`
  （≥0.5% 改进 + 逐窗 2% 容差 + 安全轨严格），拒绝层保持 v182 位级。
- 单预注册配置（仅加 +4 一码，不扫其他码/窗口）。

## 2. 硬检查

| 检查项 | 结果 |
| --- | --- |
| 单文件六 API、合法 state、有限输出 | OK |
| 拒绝层位级 = v182（GPT-2 0/0/60 no_effect 佐证） | OK |
| Linear 侧未动（attention-only 隔离验证） | OK |
| 在线无新增算子（仅接受层 offsets 多 1 码） | OK |

## 3. 本地三模型验证（配对 v180/v182 attention baseline）

| 模型 | panel | 门决策 | by-layer |
| --- | --- | --- | --- |
| Qwen default 120 | **+0.006580**（11/8/101） | 4/24 接受 | **L11 +0.15844（5/5，与探针一致）**、L17 +0.0002；L16 被门拒（5 码分支校准 0.001993 劣于纯窗口 0.001818） |
| GPT-2 default 60 | 0.000 no_effect | 12/12 拒绝 | 探针 L8 +0.245 被校准判据否决 |
| OPT-125m default 60 | −0.000200（7/3/50） | 2/12 接受 | 探针 L8 −0.215 大负基本防住 |

对比未门控探针（Qwen +0.0103 / GPT-2 +0.0231 / OPT −0.0183）：门保留最大
真实增益（Qwen L11），否决校准不可靠的 GPT-2 L8，防住 OPT L8 大负。
GPT-2/OPT 整体非正 → 按规则标记 **model-specific-risk**（D1 先例：GPT-2 负
→ 官方 +3）。

## 4. 量级参照

Qwen +0.0066 = D1 本地信号（+0.000356→官方+3）的 **18.5 倍**；为 A1 以来
门控后最大本地信号（v159 +0.149→官方 +671 的 1/23）。

## 5. 时间风险（官方决策关键）

- Qwen attention default wall：126.6s vs v180 74.5s（**校准 ×2，+52s 本地**）；
  5 码单窗校准本身与 4 码同速（探针 70.5s）。
- 官方换算不可靠（AGENTS：本地 CUDA 时间门禁对官方预测失效）；粗估官方
  attention calibration 增量 +20~40s。
- **v182 父官方余量 27s：有超限风险**；v180 父余量 58s 较安全但基线少 1 分。
- GPT-2 12/12 拒绝提示：若官方模型行为似 GPT-2，双窗成本全付零收益。

## 6. 官方裁决规则（预注册）

`S(v184) > 17598` 且 `<300s` → RETAINED；`≤17598` 且 `<300s` → REJECTED
（+4 窗口族关闭，不扫邻域）；`>300s` → TIMEOUT（不缩窗重试；可评估从
v180 时间父重构同机制，属父版本选择非机制邻域）。

## 7. 复现

```powershell
.venv\Scripts\python.exe -u evaluator\official_eval.py --solution solutions\20260904_v184_attn-plus4-gate_scoreNA_timeNA\solution.py --attention-only --cache-mode read --nvfp4-cache-mode auto --capture-device cuda --algorithm-device cuda --baseline-json artifacts\official_eval\v180-attn-default.json --output artifacts\official_eval\v184-attn-default.json --report logs\official_eval\v184-attn-default.md
```

源码 SHA256：`a77f77dab13f207c41cbc4636abf603b28891e191ce038780dfd93868df104e1`。
根 `solution.py` 保持 v182
官方父；v184 官方 RETAINED 前不切换。
