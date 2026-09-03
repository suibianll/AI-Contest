# candidate — proxy-v2

本报告是本地 proxy；隐藏官方数据和鲲鹏硬件不可本地复制。

- evaluation scope: `linear-only-default-panel` / `linear-only-proxy-ranking-within-identical-cache`
- proxy ranking comparable: `False`; official-score equivalent: `False`
- Linear calibration indices: `[0, 1]`; all captured lengths: `[10, 128, 512, 1024, 1024]`
- cases: `168 Linear + 0 Attention` (stratified real-W/A panel by default)
- calibration calls: `168 weight + 0 attention` (shared state)
- input codec: `e4m3-subnormal-ceil-v1` / mode `amax6`
- test splits: `['test', 'validation']`
- source SHA256: `896b4aca9f9f0c55d91c439e628b59d0b04d3bd77e23aa6f17144b0d665793d7`
- data pack: `{'train': 'e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7', 'validation': '204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c', 'test': '5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91'}`

| 指标 | 值 |
|---|---:|
| Linear mean | 0.000000000 |
| Attention mean | NA (not run) |
| Overall mean (all captured cases) | 0.000000000 |
| Linear role macro mean | 0.000000000 |
| Attention layer macro mean | 0.000000000 |
| Candidate wall | 24.161s |
| Candidate API total | 1.682s |

## Linear 泛化与尾部分析

均值只作位置统计；优先检查 median、worst-quartile、负 case、跨 split/长度和 W/A/interaction 分布。

| 分组 | cases | mean | median | q25 | worst-quartile mean | min | 正/负/零 | median player/std MSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 168 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0/0/168 | 1.000000 |
| family:fc | 48 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0/0/48 | 1.000000 |
| family:o | 24 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0/0/24 | 1.000000 |
| family:proj | 24 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0/0/24 | 1.000000 |
| family:qkv | 72 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0/0/72 | 1.000000 |
| role:fc_gate | 24 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0/0/24 | 1.000000 |
| role:fc_up | 24 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0/0/24 | 1.000000 |
| role:k | 24 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0/0/24 | 1.000000 |
| role:o | 24 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0/0/24 | 1.000000 |
| role:proj | 24 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0/0/24 | 1.000000 |
| role:q | 24 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0/0/24 | 1.000000 |
| role:v | 24 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0/0/24 | 1.000000 |

跨 layer/shape/split/test_length 的同结构统计，以及 W-only/A-only/Both/interaction 的完整分布位于 JSON `analysis.linear_generalization`。

## 误差源分解（evaluator-only）

控制臂只在评测器内重算，不增加六个候选 API 调用，也不改变主分数。gain 为相对标准输出误差的改善；interaction 为正表示超加性互补，负表示收益重叠或递减。

### Linear：W / A / 交互

| 分组 | cases | W-only gain | A-only gain | Both gain | interaction | W operand rel-MSE | A operand rel-MSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| overall | 168 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 6.835814e-03 | 8.626838e-03 |
| family:fc | 48 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 6.726051e-03 | 8.568782e-03 |
| family:o | 24 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 6.679961e-03 | 8.062684e-03 |
| family:proj | 24 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 6.863414e-03 | 7.657579e-03 |
| family:qkv | 72 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 6.951741e-03 | 9.176679e-03 |
| role:fc_gate | 24 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 6.629023e-03 | 8.522993e-03 |
| role:fc_up | 24 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 6.823080e-03 | 8.614570e-03 |
| role:k | 24 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 6.999664e-03 | 9.216427e-03 |
| role:o | 24 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 6.679961e-03 | 8.062684e-03 |
| role:proj | 24 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 6.863414e-03 | 7.657579e-03 |
| role:q | 24 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 6.885952e-03 | 9.113443e-03 |
| role:v | 24 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 6.969608e-03 | 9.200166e-03 |
| shape:hidden_to_hidden | 48 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 6.782956e-03 | 8.588063e-03 |
| shape:hidden_to_wide | 96 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 6.855344e-03 | 8.888539e-03 |
| shape:wide_to_hidden | 24 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 6.863414e-03 | 7.657579e-03 |

Linear overall interpretation: `mixed_or_neutral`.
Linear 的完整 layer/role/window 结果位于 JSON 的 `decomposition.linear` 和 `case_scores.linear`；优先查看 role 与 shape 行，再定位对应 layer/case。

Attention 分解：已通过 `--no-decomposition` 关闭。

官方成绩只保留为独立历史字段，不参与本地 proxy 评分或时间换算。
