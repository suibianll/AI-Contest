# v229 官方超时回传（2026-09-10）

用户原话：“v229超时了”。官方结果登记为 TIMEOUT (>300s)，REJECTED；未提供精确秒数或分数，两者记 null/NA，不填本地耗时或推算值。

- 类型：完整六 API 候选，v202 Linear + v195 Attention 完整父 + A-MC1 K per-call 均值再定心。
- 原归档：`solutions/20260910_v229_attention-amc1-k-mean-recenter_officialNA_timeNA/`。
- 当前归档：`solutions/20260910_v229_attention-amc1-k-mean-recenter_rejected_scoreNA_timeNA/`。
- 官方计分 SHA / 归档 SHA：`d1c23fa11198e56f15ac8f64e033c00333dcd2d5660cec773598624c4b247f4d`。
  绑定依据：用户明确识别 v229，按唯一登记的 v229 完整候选绑定；现场核验归档源码哈希匹配。未取得官方上传文件的独立哈希。
- 当前根仍为 v202 Linear + v195 Attention，18053/281s；现场 SHA 为
  `56DC805D6E5A3AEF896DB8021045740292735725D688B48E3D4393E55EFCB2BD`。
- 本地六 shard +0.014923（26/10/36）保留为诊断，未获官方精度定价。

只关闭此完整实现，不从超时推出 K 定心无精度价值或整个 Attention 空间饱和；不把超时归因到具体 API，不从本地缓存命中耗时估算官方开销。此前分析中“优先获取 v229 官方结果”已由本次回传完成。

**补充（2026-09-10 侧隔离回传）**：标准 Linear + v229 Attention 侧隔离官方 `14424/245s`，相对 v195 侧基准 `14426/243s` 为 −2/+2s，A-MC1 精度价值已获官方定价（−2，无正向价值）；K 平移类正式关闭。见[侧隔离回传](2026-09-10-v229-side-isolation-official.md)。

同步官方结果 JSON、配置、候选 result、根 README、当前状态、版本索引、计划入口、A-MC1 计划与瓶颈分析。原始本地日志和 JSON/report 保留，未重跑评测，未修改提交源码，也未重开邻域或追加新实验。
