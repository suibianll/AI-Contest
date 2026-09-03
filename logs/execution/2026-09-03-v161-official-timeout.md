# v161 官方超时记录

日期：2026-09-03
候选：`20260903_v161_v160-attn-s1-qk-gram-refine_scoreNA_timeout`
来源：用户回传官方评测结果。

## 官方裁决

| 候选 | 官方分数 | 官方时间 | 裁决 |
|---|---:|---:|---|
| v161 S1 qk-gram-refine | 未返回 | `>300s` | **timeout（官方，用户确认）** |

官方评分没有返回可登记分数，因此 Official score 保持 `NA`；官方时间只登记为
超出 300 秒，不用本地代理数字替代。

## 归因（时间外推门禁失效）

- v161 = v160（官方 `17532 / 232s`）+ 校准期交叉 Gram 计算 + 动态 Q/K 3-sweep 精化，
  其余逐位冻结；本地 CUDA 口径 attention API `57.97→85.99s`（+28.0s，其中 calib
  +7.3s、dyn Q/K +19.7s 即 `0.092s/call`），官方外推 `~257s < 300s`；
- 官方实际 timeout：官方机（鲲鹏）上 per-call 小张量算子成本远超本地 CUDA 外推比，
  v160 的 68s 官方余量被动态精化耗尽；
- 家族证据链：v138（删除 dyn refine + 校准搜索）官方 `208s` 通过；v128/v129/v130/
  v131（含 dyn refine）官方全部 timeout；v161（无校准搜索、只保留 dyn refine + gram）
  仍 timeout。**修正此前核算结论：v128 家族超时元凶不只是校准期候选搜索
  （199.8s/24 calls），动态 per-call 精化本身（本地 `0.08–0.09s/call` CUDA）在官方
  硬件上即超预算**；
- 精度方向未被官方否证（timeout ≠ WA），但本地精度余量（Qwen default 120 paired
  `+0.0525`、106+/14−；GPT-2 `+0.0678` 同号）在官方 300s 内无法回收。

## 后续处理（按预注册）

- S1 正式 `TIMEOUT / REJECTED`；不缩 sweeps 重试，不调 gram/块/topk；
- S2（校准搜索解析化）前置条件"S1 官方正向"不满足，不启动；
- per-call 动态自适应族关闭；本地时间门禁（CUDA 口径 parent+40s）对官方时间的预测
  能力记为失效，后续任何含在线逐 call 计算的候选默认按官方不可行处理，除非增量
  算子是纯解析标量/查表级；
- D1 判别器证据维持 3/3（v161 无官方分数，不计入）；P9 检验无法记录；
- 活动计划归档：`docs/superpowers/archive/plans/2026-09-03-attention-per-call-refinement-plan-superseded.md`；
- 本地已知机制族全部闭环（Linear 结构 full64/Householder、Attention 解析静态族、
  Attention per-call 动态族）；下一步为外部材料搜索或用户指定新机制，不再从已关闭
  族内微调。

## 本地对应记录

同候选本地 default attention proxy：attention_mean `0.742354→0.794856`（paired
`+0.052502`，`106+/14−/0`，touch 88.3%），attention API `85.995s`（parent `57.97s`）。
这组数值只解释机制与本地成本，不能推导官方分数或官方秒数。

## 归档处理

源码 SHA256 为 `27EEE4710B0170384A17E2F3E9AB87B3437E7B224883150D70BEBF8A5FB11848`。
目录已更名为 `20260903_v161_v160-attn-s1-qk-gram-refine_scoreNA_timeout`；root
`solution.py` 不切换（保持 v160 系）。
