# A5 frozen-Q(A) ridge/Qronos — rejected (runtime)

日期：2026-08-30  
状态：`archived-rejected`；主代码已恢复 stable parent。官方评测不可用，以下为
固定 Qwen2.5-0.5B cache 的本地 shaped panel。

## 实现

先冻结一个合法激活重建 \(Z=Q(A)\)，然后只在离线 weight calibration 中使用
teacher output \(Y=AW^T\)。对每个输入宽度不超过 1024 的 Linear role，按计划用
Woodbury 形式避免分配 \(d\times d\) 矩阵：

\[
\widetilde W^T=Z^T(ZZ^T+\lambda I)^{-1}Y,
\qquad
W_\eta=(1-\eta)W+\eta\widetilde W,
\]

其中本轮 \(\eta=1/8\)、\(\lambda/\operatorname{tr}(ZZ^T)=10^{-4}\)。\(W_\eta\)
再经过合法 HiF4 编码，并用独立 fold 的 frozen-output objective 选择；任何
output/residual 都不会写入 activation_state。为通过运行时合规 taint 检查，
候选评分后的权重在 objective 边界重新 materialize 为普通 CPU tensor。

## 结果

| 范围 | Linear mean | Attention mean | panel | API time |
| --- | ---: | ---: | ---: | ---: |
| layer-1 A5 frozen-Qronos | `0.603071` | `0.926339` | `336.035344` | `17.43s` |
| 24-layer A5 frozen-Qronos | `0.501558` | `0.841829` | `293.755106` | `455.73s` |
| 24-layer stable parent | `0.501558` | `0.841829` | `293.755106` | `382.15s` |

精度完全持平，没有达到 panel 的严格提升门禁；校准时间增加约 `75.4s`，使
API 超过 420s `35.73s`。因此候选被拒绝并回退。

## 裁决与证据

- 候选源码、JSON、报告保存在
  `solutions/20260830_v094_a5-frozen-qronos-rejected_score293.755106_time456s/`。
- 合规测试 `tests/test_linear_compliance_guard.py` 13/13 通过；候选只影响
  weight_params。
- 结论：在当前两折与固定 `eta=1/8` 下，冻结-Q(A) 连续 ridge 目标在离散 HiF4
  投影后没有可迁移增益；不继续扩大该参数网格，转向独立的 activation-side
  Global-LRH 上限实验。
