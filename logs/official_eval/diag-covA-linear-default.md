# candidate — proxy-v2

本报告是本地 proxy；隐藏官方数据和鲲鹏硬件不可本地复制。

- evaluation scope: `linear-only-default-panel` / `linear-only-proxy-ranking-within-identical-cache`
- proxy ranking comparable: `False`; official-score equivalent: `False`
- Linear calibration indices: `[0, 1]`; all captured lengths: `[10, 128, 512, 1024, 1024]`
- cases: `168 Linear + 0 Attention` (stratified real-W/A panel by default)
- calibration calls: `168 weight + 0 attention` (shared state)
- input codec: `e4m3-subnormal-ceil-v1` / mode `amax6`
- test splits: `['test', 'validation']`
- source SHA256: `db55264da0a4e880c0aa30ebc59e15a5b2dfe38281cd93bfe78beac09068c100`
- data pack: `{'train': 'e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7', 'validation': '204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c', 'test': '5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91'}`

| 指标 | 值 |
|---|---:|
| Linear mean | 0.636609487 |
| Attention mean | NA (not run) |
| Overall mean (all captured cases) | 0.636609487 |
| Linear role macro mean | 0.636609487 |
| Attention layer macro mean | 0.000000000 |
| Candidate wall | 360.813s |
| Candidate API total | 337.908s |

## Linear 泛化与尾部分析

均值只作位置统计；优先检查 median、worst-quartile、负 case、跨 split/长度和 W/A/interaction 分布。

| 分组 | cases | mean | median | q25 | worst-quartile mean | min | 正/负/零 | median player/std MSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 168 | 0.636609 | 0.635532 | 0.536987 | 0.446654 | -0.043193 | 167/1/0 | 0.364468 |
| family:fc | 48 | 0.541531 | 0.546142 | 0.481115 | 0.429155 | 0.385555 | 48/0/0 | 0.453858 |
| family:o | 24 | 0.516566 | 0.508368 | 0.411804 | 0.302555 | -0.043193 | 23/1/0 | 0.491632 |
| family:proj | 24 | 0.559647 | 0.563847 | 0.531657 | 0.482293 | 0.336650 | 24/0/0 | 0.436153 |
| family:qkv | 72 | 0.765664 | 0.756428 | 0.732911 | 0.699084 | 0.640659 | 72/0/0 | 0.243572 |
| role:fc_gate | 24 | 0.569396 | 0.572599 | 0.527178 | 0.455283 | 0.385555 | 24/0/0 | 0.427401 |
| role:fc_up | 24 | 0.513666 | 0.513123 | 0.454454 | 0.417331 | 0.406128 | 24/0/0 | 0.486877 |
| role:k | 24 | 0.777317 | 0.774311 | 0.742128 | 0.722620 | 0.678765 | 24/0/0 | 0.225689 |
| role:o | 24 | 0.516566 | 0.508368 | 0.411804 | 0.302555 | -0.043193 | 23/1/0 | 0.491632 |
| role:proj | 24 | 0.559647 | 0.563847 | 0.531657 | 0.482293 | 0.336650 | 24/0/0 | 0.436153 |
| role:q | 24 | 0.751610 | 0.754748 | 0.693377 | 0.669376 | 0.640659 | 24/0/0 | 0.245252 |
| role:v | 24 | 0.768064 | 0.757649 | 0.733624 | 0.718502 | 0.701699 | 24/0/0 | 0.242351 |

跨 layer/shape/split/test_length 的同结构统计，以及 W-only/A-only/Both/interaction 的完整分布位于 JSON `analysis.linear_generalization`。

## 父版本配对效果

基线：`candidate`；候选：`candidate`。

| 范围 | cases | mean Δgain | median Δgain | 改善/回归/不变 | median MSE ratio | 结论 |
|---|---:|---:|---:|---:|---:|---|
| Linear overall | 168 | 0.000000 | 0.000000 | 0/0/168 | 1.000000 | no_effect |
| Attention overall | 0 | 0.000000 | 0.000000 | 0/0/0 | - | no_cases |

完整 role/family、W/A 来源、最坏 case、Attention 控制臂和 API 时间差分位于 JSON `paired_effect`。

## 误差源分解（evaluator-only）

控制臂只在评测器内重算，不增加六个候选 API 调用，也不改变主分数。gain 为相对标准输出误差的改善；interaction 为正表示超加性互补，负表示收益重叠或递减。

### Linear：W / A / 交互

| 分组 | cases | W-only gain | A-only gain | Both gain | interaction | W operand rel-MSE | A operand rel-MSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| overall | 168 | -389.205806 | -130.517101 | 0.636609 | 520.359517 | 2.193338e+00 | 1.521784e+00 |
| family:fc | 48 | -157.267550 | -137.272110 | 0.541531 | 295.081191 | 2.087120e+00 | 1.562662e+00 |
| family:o | 24 | -184.463014 | -108.547995 | 0.516566 | 293.527576 | 2.179579e+00 | 1.586049e+00 |
| family:proj | 24 | -1572.370928 | -155.793020 | 0.559647 | 1728.723594 | 1.860782e+00 | 1.640534e+00 |
| family:qkv | 72 | -217.690534 | -124.911490 | 0.765664 | 343.367688 | 2.379589e+00 | 1.433527e+00 |
| role:fc_gate | 24 | -180.786976 | -177.930921 | 0.569396 | 359.287292 | 2.162998e+00 | 1.589925e+00 |
| role:fc_up | 24 | -133.748123 | -96.613299 | 0.513666 | 230.875089 | 2.011243e+00 | 1.535398e+00 |
| role:k | 24 | -275.016729 | -166.026212 | 0.777317 | 441.820258 | 2.535365e+00 | 1.461509e+00 |
| role:o | 24 | -184.463014 | -108.547995 | 0.516566 | 293.527576 | 2.179579e+00 | 1.586049e+00 |
| role:proj | 24 | -1572.370928 | -155.793020 | 0.559647 | 1728.723594 | 1.860782e+00 | 1.640534e+00 |
| role:q | 24 | -223.686961 | -116.030688 | 0.751610 | 340.469259 | 2.339128e+00 | 1.422897e+00 |
| role:v | 24 | -154.367913 | -92.677570 | 0.768064 | 247.813547 | 2.264273e+00 | 1.416175e+00 |
| shape:hidden_to_hidden | 48 | -204.074987 | -112.289342 | 0.634088 | 316.998417 | 2.259354e+00 | 1.504473e+00 |
| shape:hidden_to_wide | 96 | -185.979935 | -133.312001 | 0.657111 | 319.949047 | 2.243470e+00 | 1.500752e+00 |
| shape:wide_to_hidden | 24 | -1572.370928 | -155.793020 | 0.559647 | 1728.723594 | 1.860782e+00 | 1.640534e+00 |

Linear overall interpretation: `paired_coordinate_coupling_likely`.
Linear 的完整 layer/role/window 结果位于 JSON 的 `decomposition.linear` 和 `case_scores.linear`；优先查看 role 与 shape 行，再定位对应 layer/case。

Attention 分解：已通过 `--no-decomposition` 关闭。

官方成绩只保留为独立历史字段，不参与本地 proxy 评分或时间换算。
