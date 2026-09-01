# v134 Linear output-supervised activation cross64

日期：2026-09-01  
父版本：v133  
源 SHA256：`5837e765e478b1a16a5e3170ace40fbadb670871e47c5ee2c8c748102a30478d`

## 算法变化

v134 把 `A@W` 放开的空间用于在线 Linear 激活量化。校准阶段对最终部署权重
`W_q` 与变换前教师权重 `W_t` 保存连续 64 通道块：

\[
H_b=W_{q,b}^{T}W_{q,b},\qquad D_b=W_{q,b}^{T}W_{t,b}.
\]

在线候选 `q_b` 不再只最小化 operand-local 的 `H_b(q_b-a_b)`，而是使用实际输出目标的
`H_bq_b-D_ba_b` 梯度；坐标候选仍通过合法 HiF4 层级编码和增量二次型接受。原先先构造完整
`W_q^TW_q` 再取 block 的实现改为 batched block matmul，避免无用的跨 block 矩阵。

## official-shape-v1 空闲复测

| run | Linear mean | Attention mean | API total | wall |
|---|---:|---:|---:|---:|
| first | 0.5073195049 | 0.8342564884 | 289.042407 s | 312.315192 s |
| rerun2 | 0.5073195049 | 0.8342564884 | 289.832117 s | 313.181455 s |

两次 450-case 结果逐位一致；API 代理均低于 300 秒。官方结果仍为未登记，不能由本地
代理推断官方通过。完整 JSON/报告：

- [`first JSON`](../../artifacts/official_eval/v134-linear-output-activation-cross64-official-shape-v1.json)
- [`rerun2 JSON`](../../artifacts/official_eval/v134-linear-output-activation-cross64-rerun2-official-shape-v1.json)

## 决策

v134 提升 Linear `+0.0237091`（相对 v133）且时间仍在本地 API 代理内，因此提升为当前根。
下一实验为 L3 低成本连续补偿中心，先验证无需新增 state 的 Jacobi 暖启动。
