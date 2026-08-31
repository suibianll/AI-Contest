# v125 C1c structured rank-8 / max-blocks-8（精度-only）

日期：2026-08-31  
父版本：v124 C1c rank-8 / max-blocks-4  
状态：**precision-only accepted；runtime invalid，不作为提交候选**

## 唯一变量

固定 C1c kernel rank `S=8`，把每行 proposal 的 selected block budget 从 `4` 提高到
`8`；C1b `sweep2`、完整部署 `G_q=W_q^T W_q` 行级 gate、HiF4 五字段和 Attention
路径均保持不变。候选由结构化近似生成，最终仍用实际部署 Gram 重新计算整行目标：

\[
J(e)=e^T G_qe,\qquad
\Delta J=2e^TG_q\Delta e+\Delta e^TG_q\Delta e.
\]

只有 `J(candidate) <= J(parent)` 的行才写回，因此本候选没有把结构化近似当作最终
裁判。

## Qwen 固定缓存结果

配置：Qwen2.5-0.5B、24 层、`seq=128`、`calib=2`、`test=4`、`amax6`、CPU、只读
cache、`qwen-official` panel（250 Linear + 200 Attention）。

| 版本 | screen Linear | full Linear mean | Attention mean | panel | native total | API time | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| v124 parent | 0.53343639 | 0.5096493233 | 0.8420394885 | 295.8202285 | 423.3201361 | 2323.911s | parent |
| v125 block-8 | **0.53358298** | **0.5097598050** | 0.8420394885 | **295.8478490** | **423.3943799** | 2653.580s | 精度正向，超时 |

增量：full panel `+0.0276204`，Linear mean `+0.0001105`；Attention 逐位不变。API
时间增加 `+329.669s`（约 `+14.2%`），超过官方 `420s` 限制约 `2233.580s`，因此
`official_flow_valid=false`、`panel_valid=false`。这次 full 结果仍保留作精度上界证据，
不能当成可提交版本。

## 复现产物

- screen：[`v125-c1c-block8-qwen-screen.json`](v125-c1c-block8-qwen-screen.json)
- full：[`v125-c1c-block8-qwen-full.json`](v125-c1c-block8-qwen-full.json)
- full report：[`2026-08-31-v125-c1c-block8-qwen-full.md`](2026-08-31-v125-c1c-block8-qwen-full.md)
- source：[`solution.py`](solution.py)

根源码（规范 LF）SHA256：`c9b419717e38bcec69d907d1cab6638409f1fa9a3072892dde9494ef9da3cc8e`。

下一步不再增加 block budget；转入 active plan 的 C2 低成本跨模型 guardrail 和 C3
state/runtime 压缩，提交版本优先回到 `<420s` 的时间 parent。
