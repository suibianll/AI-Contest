# A7 量化后权重 Gram 激活 Hessian

日期：2026-08-30  
状态：`archived-rejected`；layer-1 正向，24 层迁移失败。

## 算法

现有激活 HSDQ 用浮点变换后权重 (W) 的 block Gram：

\[
H_b=W_b^T W_b.
\]

但真实 Linear 输出使用已经量化的权重 (W_q=Q(W))。A7 改用：

\[
H_b^{(q)}=W_{q,b}^{T}W_{q,b},
\qquad
\Delta L=2e^T H_b^{(q)}\Delta+\Delta^T H_b^{(q)}\Delta,
\]

并将 (H_b^{(q)}) 作为静态 `gram64` 写入 `activation_state`。它不访问 evaluator
输出，也不把 `A@W` 或 residual 写入在线状态；理论上更贴近实际输出

\[
Q(A)Q(W)^T.
\]

## 结果

| 版本 | layer-1 panel | full panel | Linear mean | Attention mean | API time | 裁决 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| A7 layer-1 | `336.562922` | — | `0.605174` | `0.926347` | `24.89s` | 单层正向 |
| A7 full | — | `290.226694` | `0.487275` | `0.842039` | `470.581s` | **拒绝** |
| v100 baseline | `336.037091` | **`293.797301`** | `0.603071` | `0.842039` | `392.424s` | active |

相对 v100：layer-1 panel `+0.525831`，但 full panel `−3.570607`；Linear mean
从 `0.501558` 降到 `0.487275`，API 增加 `78.16s` 并超过 `420s`。这说明用更
贴近部署权重的 Hessian 仍然严重依赖层/校准分布，不能从单层结果外推全层收益。

## 结论与归档

- A7 的数学目标是合理的，但当前 cross-fold solver 与量化后 Gram 的组合造成跨层
  回退；不得保留在主线。
- 量化后 Gram 的层级差异可作为未来诊断，但若重启必须加入跨层/跨模型 gate，且
  需要先解决约 `78s` 的额外 API 成本。
- 失败源码和完整 layer/full JSON 归档在
  `solutions/20260830_v104_a7-quant-weight-gram-rejected_score290.226694_time471s/`。
