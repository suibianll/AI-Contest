# Attention 相邻 pair 4×4 执行记录

日期：2026-09-06

计划：`2026-09-06-attention-crosspair-4x4-plan`

状态：**CLOSED / C2_OOD_REJECTED**

父版本：v189，SHA256 `261202248a0146a2ee45f3df60bd1979bb8171b7c162921013b0024c848617af`

候选源码：`workbench/attention_crosspair_block_solution.py`

候选 SHA256：`3BA683A046083A5E9E043CC135532D01912A2CC437A5290ABB2D5D6E695E75F3`

## 执行

- C0 通过：六 API 可导入；4×4 连续 GQA `QK` 最大绝对误差 `9.54e-7`；旧 2×2 identity
  回退通过。
- C1 shard0：mean delta `+0.001844`，L1 `0.001844`，正/负/零 `2/0/6`。
- C1 shard1：mean delta `+0.009313`，L1 `0.010591`，正/负/零 `7/1/0`。
- C2 ID 六 shard：candidate Attention mean `0.758912691`，baseline `0.752772355`，
  delta `+0.006140336`。
- C2 OOD shard0：mean delta `-0.001240`；shard1：`-0.015858`，L1 `0.021338`，
  最坏 delta `-0.072815`（layer 19、validation、length 10），评测按 OOD 负向停止。

原始 eval-v3 结果：

- `artifacts/proxy_v3/attention-crosspair-4x4-20260906/c1/`
- `artifacts/proxy_v3/attention-crosspair-4x4-20260906/c2/`
- `artifacts/proxy_v3/attention-crosspair-4x4-20260906/c2-ood/`

结论：ID 局部正向没有通过 OOD 泛化门禁；该 4×4 机制关闭为 **REJECTED**，不运行
fresh default，不提交官方，不扫描 stride/ridge/收缩/矩阵维度邻域；根 `solution.py`
仍为 v186。
