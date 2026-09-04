# S1 交叉算子 Gram64 per-call 精化 — 本地全漏斗通过，产出 v161 官方候选

日期：2026-09-03
状态：**DONE — v161 CANDIDATE，等待一次官方提交**

## 1. 执行记录

按[`活动计划`](../../docs/superpowers/plans/2026-09-03-attention-per-call-refinement-plan.md)
完成 Step 0 消融与 S1 候选全漏斗：

### Step 0（零实现消融，见独立日志）

v128/v129/v138 归档 attention 同协议运行：v128 `+0.0636`（3+/1−）确认余量属 per-call
自适应族（v138 静态比 v160 还差 −0.014）；v129 与 v128 逐位相同（搜索砍半无损）；
dyn refine 实测 `0.095s/call`。

### S1 实现（`workbench/s1_qk_gram_refine.py`，从 v160 归档分支）

- 常量：`_ATTN_GRAM_REFINE=True`、`_ATTN_GRAM_SWEEPS=3`（v128 固定值）；
- 移植：`_SIGNED_LEVELS`、`_params_denominator`、`_write_codes`、
  `_refine_activation_gram`（v128 的 topk+批处理坐标下降，块常量改为根文件命名
  `_HIF4_BLOCK_SIZE`）、`_qk_cross_gram64`；
- 校准尾追加：最终部署坐标的交叉 Gram 存入 state；
- 动态 Q/K 追加 refine；V 不动。

### 漏斗结果

| 阶段 | 结果 |
| --- | --- |
| A 单元 | 全过（gram loss 单调 404→313；PSD；6 API；布局不变） |
| B Qwen compact | `+0.021571`（2+/2−），API 持平 |
| C Qwen default 120 | **`+0.052502`（106+/14−，touch 88.3%，median +0.040）**，API +28.0s ≤ +40s 门禁 |
| C GPT-2 compact | **`+0.067751`（3+/1−）同号** |
> **[2026-09-04 修订]** 「GPT-2 同号」**不计入漏斗通过项**（探针与官方排序 ρ = −0.071）。
> 本表 "C GPT-2" 行与下方 "D1 全过 → 预测官方正向" 的判断依据中，GPT-2 一项失效；
> 最终 v161 官方 timeout 也印证了本地正向（含跨模型同号）不迁移官方。
> 见 [修订清单 §1](../../docs/stale-information-inventory-2026-09-04.md)。
| D | SHA `27EEE471...1848`；隔离导入 ✓；归档 `solutions/20260903_v161_v160-attn-s1-qk-gram-refine_scoreNA_timeNA/` |

**D1 预注册判别器全过 → 预测官方正向**（3/3 → 若正向 4/4）。

## 2. 时间与预算

本地 attention API `57.97→85.99s`（+28.0s：calib +7.3 / dyn Q/K +19.7）；官方外推
~257s < 300s（余量 ~43s）。

## 3. 决定

- v161 候选已归档，交用户做一次官方提交；
- 官方正向 → S2（校准搜索解析化，仍单机制）；零/负 → D1 降级记录，不邻域调参；
  timeout → REJECTED；
- root `solution.py` 在官方回传前不切换；
- 官方回传同时记录 P9 检验。
