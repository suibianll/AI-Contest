# v121 C1b structured refresh2 accepted

- 状态：accepted precision parent；根常量 `_ACT_STRUCTURED_LRH_REFRESH_MODE = "sweep2"`。
- source `solution.py` SHA256（规范 LF）：`17f99a198c9e13c2cb2518d14b02973bc71336adfc90e6bf884e89d22717af7b`。
- 父版本：v119 C1a；v120 block-refresh screen 被拒绝。
- screen：Linear mean `0.5333964596`，较 v118 `+0.0000211411`；proj `0.4135807679`。
- full Qwen：Linear mean `0.5096135327`，Attention mean `0.8420394885`，panel `295.8112808759`，native total `423.2960848901`。
- 相对 v119：Linear `+0.0000122773`，panel `+0.0030693200`，proj `+0.0000859410`；其余 role 和 Attention 不变。
- API `2180.4501505s`，wall `2212.6619801s`，超过 420s；accuracy-first 阶段不因时间拒绝，C3 再压缩。
- 合成单调性与 38 项目标测试通过；静态/运行时 compliance 无 violation。
- `official_flow_valid=false` 仅因 API 超过 420s；本地精度字段完整可复现，不可直接当官方分数。

完整 JSON、screen、synthetic 和 full 报告均保存在本目录。

---

> **2026-08-31 归档修复批注**：本目录 `solution.py` 携带 B2 PAWV 变长 calibration bug
> （官方长度 `[10,128,512,1024,1024]` 触发 `[10,10] += [128,128]` RuntimeError），已在
> 原文件上按 v127 逻辑修复（按长度分组的 keyed diagonal），并通过官方长度形状复现。
> 修复后 v5 `sampled-means-v1` 复评：Linear `0.516685`、Attention `0.828395`、
> Local API `832.920s`（仍远超 300s，不能官方提交）。官方分类不变（timeout）。见
> [`pawv 归档修复与 v5 复评`](../../logs/execution/2026-08-31-pawv-archive-fix-and-v5-reeval.md) 与
> [`v121 修复复评`](../../logs/execution/2026-08-31-v121-pawv-fixed-sampled.md)。
