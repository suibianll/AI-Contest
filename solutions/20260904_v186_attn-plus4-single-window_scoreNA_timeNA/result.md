# v186 候选：attention 在线 scale 窗口 +4 单码（单窗版）

> 状态：**RETAINED（官方 2026-09-04）— v186** **`17599/272s`，成为新完整官方父版本**
>
> 来源诊断：[`2026-09-04-big-gain-space-diagnostics.md`](../../logs/execution/2026-09-04-big-gain-space-diagnostics.md)
>
> - v184 TIMEOUT 归因（[`v184_timeout_attribution.py`](../../logs/execution/v184_timeout_attribution.py)）。
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

| 门禁                    | 结果                                        |
| --------------------- | ----------------------------------------- |
| 官方时间模型                | 预测 **274.0s**（273 + 0.694×1.5s）< 280 PASS |
| Δmean > 0             | **+0.010344**（A1 后最大本地信号，D1 的 29 倍）       |
| L1 < 0.02（AGENTS 新规则） | **0.0155 PASS**（120 case 逐对计算）            |
| 单文件孤立导入六 API          | OK                                        |
| diff                  | 仅 1 行常量（git diff 核验）                      |
| v186 归档复核             | 与探针位级一致（+0.010344, 65/53/2），校准时间中性        |

## 3. 本地证据（配对 v180 attention baseline = v182 attention 逐位）

| 场景                    | 结果                                                                                                     |
| --------------------- | ------------------------------------------------------------------------------------------------------ |
| Qwen default 120      | **+0.010344**（65/53/2，median +0.000085，MSE ratio 0.9995）                                               |
| Qwen by-layer         | L11 +0.158、L16 +0.093、L23 +0.006；L13/L14 −0.003\~−0.005                                                |
| GPT-2 default 60（记录项） | +0.023123（36/24）；OPT default 60（记录项）−0.018267（30/30）——按 2026-09-04 修订清单，跨模型探针 ρ=−0.07/−0.20 不作为晋级/否决依据 |
| 校准时间                  | wall 75.4s vs v180 74.5s（中性）；API A\_calib 67.6s vs 66.1s                                               |

## 4. 官方裁决

**官方结果（2026-09-04 用户回传）：`17599 / 272s`，RETAINED，成为新完整官方父。**

- `S(v186) = 17599 > S(v182) = 17598`，且 `272s < 300s` → RETAINED。

- step\_gain **+1**；时间 **−1s**（校准中性预测验证成立：时间模型预测 274.0s，
  实测 272s，误差 2s，在模型 MAE 10.1s 带内）。

- 本地 Δmean +0.010344（A1 后最大信号）→ 官方 +1：再次确认本地均值不换算官方
  分数（LOO MAE≈1108 vs 增益 1\~123），但符号门禁（Δmean>0、L1<0.02）零误。

- 距榜首 21765 差 **4166**；时间余量 28s。

- 预注册规则：+4 窗口族已获官方正裁决；不扫其他码（+5/-2 等）邻域。

- v186 是完整组合版本；隐含 Attention 单侧增量不登记为独立测量。

## 5. 复现

```powershell
.venv\Scripts\python.exe -u evaluator\official_eval.py --solution solutions\20260904_v186_attn-plus4-single-window_scoreNA_timeNA\solution.py --attention-only --cache-mode read --nvfp4-cache-mode auto --capture-device cuda --algorithm-device cuda --baseline-json artifacts\official_eval\v180-attn-default.json --output artifacts\official_eval\v186-attn-default.json --report logs\official_eval\v186-attn-default.md
```

源码 SHA256：`f8495dca20334acbdad16fc18ee41a4970f31e1837fdeedcee9c70aee54e7eb8`。
根 `solution.py` 保持 v182 官方父；v186 官方 RETAINED 前不切换。
