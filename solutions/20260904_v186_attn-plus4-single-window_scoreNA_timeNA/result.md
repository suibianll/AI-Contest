# v186 候选：attention 在线 scale 窗口 +4 单码（单窗版）

> 状态：**候选待官方评测（配额 6/10）；全部门禁通过**
>
> 来源诊断：[`2026-09-04-big-gain-space-diagnostics.md`](../../logs/execution/2026-09-04-big-gain-space-diagnostics.md)
> + v184 TIMEOUT 归因（[`v184_timeout_attribution.py`](../../logs/execution/v184_timeout_attribution.py)）。
>
> 构造：v182 完整官方父（`17598/273s`）+ **仅 1 行常量**：
> `_DYNAMIC_OFFSETS (-1,1,2,3) → (-1,1,2,3,4)`。Linear 与 v182 逐位一致。

## 1. 机制（oracle 分解的最终最小产物）

- 在线 Q/K/V 动态编码的 E6M2 scale 搜索窗口增加 +4 单码。
- oracle 分解链：±24 穷举 +0.010386 → ±8/±4 位级相同 → 负侧补码仅
  +0.000102 → 显式 0 无效果 → **仅 +4 一码捕获 99.6%（+0.010344）**。
- 根因：`_REFINE_EDGE_EXTENSION` 爬山式扩展在 E6M2 跨 binade 处失效
  （最优在 +4 的块，+3 常比标准差，不触发扩展）。
- v184 双窗 TIMEOUT 证明：增益只需窗口本身（含校准决策在 5 码下的自然
  重排），不需要额外门控校准（单窗 5 码校准与 4 码同速：67.6s vs 66.1s）。

## 2. 门禁（全过）

| 门禁 | 结果 |
| --- | --- |
| 官方时间模型 | 预测 **274.0s**（273 + 0.694×1.5s）< 280 PASS |
| Δmean > 0 | **+0.010344**（A1 后最大本地信号，D1 的 29 倍） |
| L1 < 0.02（AGENTS 新规则） | **0.0155 PASS**（120 case 逐对计算） |
| 单文件孤立导入六 API | OK |
| diff | 仅 1 行常量（git diff 核验） |
| v186 归档复核 | 与探针位级一致（+0.010344, 65/53/2），校准时间中性 |

## 3. 本地证据（配对 v180 attention baseline = v182 attention 逐位）

| 场景 | 结果 |
| --- | --- |
| Qwen default 120 | **+0.010344**（65/53/2，median +0.000085，MSE ratio 0.9995） |
| Qwen by-layer | L11 +0.158、L16 +0.093、L23 +0.006；L13/L14 −0.003~−0.005 |
| GPT-2 default 60（记录项） | +0.023123（36/24）；OPT default 60（记录项）−0.018267（30/30）——按 2026-09-04 修订清单，跨模型探针 ρ=−0.07/−0.20 不作为晋级/否决依据 |
| 校准时间 | wall 75.4s vs v180 74.5s（中性）；API A_calib 67.6s vs 66.1s |

## 4. 官方裁决规则（预注册）

`S(v186) > 17598` 且 `<300s` → RETAINED；`≤17598` 且 `<300s` → REJECTED
（+4 窗口族关闭，不扫其他码）；`>300s` → TIMEOUT（不缩窗重试）。
量级参照：L1=0.0155 在历史 17 官方配对中属安全带（该门禁曾拦下三次最大损失）。

## 5. 复现

```powershell
.venv\Scripts\python.exe -u evaluator\official_eval.py --solution solutions\20260904_v186_attn-plus4-single-window_scoreNA_timeNA\solution.py --attention-only --cache-mode read --nvfp4-cache-mode auto --capture-device cuda --algorithm-device cuda --baseline-json artifacts\official_eval\v180-attn-default.json --output artifacts\official_eval\v186-attn-default.json --report logs\official_eval\v186-attn-default.md
```

源码 SHA256：`f8495dca20334acbdad16fc18ee41a4970f31e1837fdeedcee9c70aee54e7eb8`。
根 `solution.py` 保持 v182 官方父；v186 官方 RETAINED 前不切换。
