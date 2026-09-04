# 2026-09-04 oracle 第四轮（E1 裁决）+ v188 Jacobian 移植提交记录

## A. oracle 方法论第四轮：窗口族与 E1 关闭

- `diag_oracle_round4_e1.py`：oracle（±24）vs 部署 +4 窗口 default 120 残差复核。
  全局残差 `+0.000150`（5+/4−，87/96 cell 逐位一致）；+4 已捕获 ±24 穷举增益的
  98.7%。窗口族关闭，无进一步码邻域价值。
- 校准 state 对比（`diag_l21_state_oracle/v186.jsonl`）：其余 23 层 state 逐位一致
  （仅 offsets 字段差异）；唯一实质差异是 **L21 离散分支翻转**（rotation_block
  32→64、pair_transform 启用、importance 尖刺化 0.05 底 + 大尖峰）。L21 分长度
  残差 `+0.091@10 / −0.081@128 / −0.020@512 / +0.024@1024` 符号混合，层均值
  `+0.0036`，全局净值 `+0.00015` → 红鲱鱼确认。**E1（校准宽窗评估/在线窄窗）
  死亡**：增益不来自平滑的 state 质量，而是一次分支选择翻转且本身不可靠。
- 剩余误差集中度（plus4 = 当前 attention 父）：top 24/120 case 占 48.4%；按长度
  1024 占 43.5%、512 占 22.7%；最差层 L21（13.9%）、L23（12.8%）、L9（9.1%）。

## B. 下一机制假设：v188 = v186 + v187 Jacobian 敏感度移植

假设逻辑：剩余误差集中长序列（softmax 跨 key 耦合最强、v186 现役二阶矩
importance 启发式最失效处）；v187 完整一阶 Jacobian 官方 +721。发现
`_ATTN_FISHER_IMPORTANCE=False`（v186 现役 importance 为二阶矩启发式或
pair-smooth 统计量）。

实现（root solution.py，SHA `1D14B465...7D2`）：常量块（L412-425，v187 预注册值
原样）、`_transform_attention_sample` + `_attention_jacobian_sensitivity_kv`
（L2871-3007，v187 公式适配 v186 最终变换坐标系）、校准最终步（L10819-10948，
logit gain 之后 LOO 门控替换 Q/K importance）。在线路径零新增计算。

## C. 本地验证（全部 vs v186 同 cache 同 panel）

| 步骤 | 结果 |
|---|---|
| compact 4 哨兵 | no_effect（0/0/4），校准 API +1.18s |
| gate 诊断（`diag_v188_gate.jsonl`） | 2/24 层通过（L12 med +0.0398/min +0.0007；L22 med +0.0952/min +0.0001）；L0/L8 中位数强正（+4.9%/+6.1%）但单 fold 回归超 −0.01 被拒；L9/L16 候选明确更差 |
| default 120 | Δmean `+0.000426`、L1 `0.001114`、6+/4−/110=；变化明细 128/512/1024 净全正（+0.092 合计）、len-10 净负（−0.037） |
| OOD（父+候选各一次） | Δgap `+0.000426`（≤0.01 门内通过；OOD profile 校准下门控 0/24 全拒，OOD 输出与父逐位一致） |
| gpt2（记录项） | Δmean `−0.000950`（4+/6−/50=）；校准 API +1.885s/12 层 |
| 时间 | Qwen 校准 API +3.14s 本地 → 时间模型预测官方 ≈274s（<280 门内） |

关键归因：L12/L22 恰为 v186 中无 pair_transform 的层；约 18 层 importance 槽位被
pair-smooth 输出拟合 importance 占据，Jacobian 无法胜出 → **静态 importance
机制族在 v186 上实测饱和**。最差剩余误差层 L21/L23/L9 全部拒绝。

## D. 裁决

用户选择提交（配额 9/10，剩余 1）。预注册：score>17599 且 <300s RETAINED；
≤17599 且 <300s REJECTED；>300s TIMEOUT；非晋级即根回滚 v186。
归档：`solutions/20260904_v188_attn-jacobian-port_scoreNA_timeNA/`。
