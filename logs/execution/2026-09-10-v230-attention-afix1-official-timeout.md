# v230（Attention A-FIX1）官方超时回传（2026-09-10）

用户原话：“v230-attention也超时了”。官方结果登记为 TIMEOUT (>300s)，REJECTED；未提供精确秒数或分数，两者记 null/NA，不填本地耗时或推算值。

- 类型：完整六 API 候选，旧完整父 v202 Linear + v195 Attention + A-FIX1（A2 训练前向从裸
  `_dense_to_hif4` 对齐为完整部署编码，STE 反向不变）。与同日回传 RETAINED 的 v230 Linear
  （L-EM2）是同编号不同候选，本条只绑定 Attention A-FIX1。
- 原归档：`solutions/20260910_v230_attention-afix1-train-deploy-align_officialNA_timeNA/`。
- 当前归档：`solutions/20260910_v230_attention-afix1-train-deploy-align_rejected_scoreNA_timeNA/`。
- 官方计分 SHA / 归档 SHA：`c2ff4ea0d6a3823e29351b616c330fa9358b588934e73019c382183130dfcd6f`。
  绑定依据：用户明确识别 "v230-attention"，按唯一登记的 v230 Attention 候选绑定；归档源码
  哈希现场匹配。未取得官方上传文件的独立哈希。
- 根不变：v230 Linear（L-EM2）+ v195 Attention，`18428/292s`（余量 8s）。
- 本地六 shard `−0.004884`（29/31/12）保留为诊断，未获官方精度定价。

## 判读

1. 只关闭此实现，不从超时推出"训练/部署前向对齐无精度价值"；本地本就净负，官方精度未知。
2. 时间归因：该候选在旧父（281s）上校准约 1.4×（shard0 校准 API 11.017s vs 根约 8s），
   超时与"v229 校准期 gate 前向"同成本类——**校准期新增前向在官方机上不可行**；在当前根
   292s（余量 8s）下更无空间。训练/部署对齐路线重试前必须先消除对齐前向的校准期成本，
   不缩步数/窗口重试本实现。
3. 2026-09-10 注意力四卡至此全部有官方或本地裁决：v227（本地负）、v228（本地负）、
   v229（TIMEOUT + 侧隔离 −2）、A-QC1（NO_EFFECT）、v230 A-FIX1（TIMEOUT）。
   Attention 侧无存活卡片；后续只由新证据（21071 锚点源码绑定、Linear 释时后按规则
   降成本重开事件搜索族）驱动。

同步官方结果登记、归档 result.md、A-FIX1 执行日志与计划卡、版本索引、当前状态、
计划入口、已关闭机制证据、瓶颈审计与轮次总结。未重跑评测，未修改候选源码。
