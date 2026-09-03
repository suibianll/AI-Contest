# v158-parent — proxy-v2

本报告是本地 proxy；隐藏官方数据和鲲鹏硬件不可本地复制。

- evaluation scope: `linear-only-default-panel` / `linear-only-proxy-ranking-within-identical-cache`
- proxy ranking comparable: `False`; official-score equivalent: `False`
- Linear calibration indices: `[0, 1]`; all captured lengths: `[10, 128, 512, 1024, 1024]`
- cases: `168 Linear + 0 Attention` (stratified real-W/A panel by default)
- calibration calls: `168 weight + 0 attention` (shared state)
- input codec: `e4m3-subnormal-ceil-v1` / mode `amax6`
- test splits: `['test', 'validation']`
- source SHA256: `18f9de037a29ad96ee06fb5c73095e9ad36d0d04da2953162181be3aea528277`
- data pack: `{'train': 'e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7', 'validation': '204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c', 'test': '5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91'}`

| 指标 | 值 |
|---|---:|
| Linear mean | 0.448179673 |
| Attention mean | NA (not run) |
| Overall mean (all captured cases) | 0.448179673 |
| Linear role macro mean | 0.448179673 |
| Attention layer macro mean | 0.000000000 |
| Candidate wall | 296.113s |
| Candidate API total | 259.707s |

## Linear 泛化与尾部分析

均值只作位置统计；优先检查 median、worst-quartile、负 case、跨 split/长度和 W/A/interaction 分布。

| 分组 | cases | mean | median | q25 | worst-quartile mean | min | 正/负/零 | median player/std MSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 168 | 0.448180 | 0.489773 | 0.342733 | 0.162481 | -0.484511 | 161/7/0 | 0.510227 |
| family:fc | 48 | 0.409512 | 0.420687 | 0.350097 | 0.300302 | 0.221486 | 48/0/0 | 0.579313 |
| family:o | 24 | 0.361368 | 0.294434 | 0.244078 | 0.125135 | -0.256624 | 23/1/0 | 0.705566 |
| family:proj | 24 | 0.155105 | 0.151237 | 0.024866 | -0.184663 | -0.484511 | 18/6/0 | 0.848763 |
| family:qkv | 72 | 0.600587 | 0.594603 | 0.546537 | 0.506828 | 0.446870 | 72/0/0 | 0.405397 |
| role:fc_gate | 24 | 0.409206 | 0.420687 | 0.345203 | 0.278868 | 0.221486 | 24/0/0 | 0.579313 |
| role:fc_up | 24 | 0.409818 | 0.416989 | 0.353223 | 0.322569 | 0.290658 | 24/0/0 | 0.583011 |
| role:k | 24 | 0.602044 | 0.594344 | 0.568697 | 0.529659 | 0.472447 | 24/0/0 | 0.405656 |
| role:o | 24 | 0.361368 | 0.294434 | 0.244078 | 0.125135 | -0.256624 | 23/1/0 | 0.705566 |
| role:proj | 24 | 0.155105 | 0.151237 | 0.024866 | -0.184663 | -0.484511 | 18/6/0 | 0.848763 |
| role:q | 24 | 0.583115 | 0.560571 | 0.510178 | 0.484160 | 0.446870 | 24/0/0 | 0.439429 |
| role:v | 24 | 0.616602 | 0.606603 | 0.575344 | 0.529633 | 0.492184 | 24/0/0 | 0.393397 |

跨 layer/shape/split/test_length 的同结构统计，以及 W-only/A-only/Both/interaction 的完整分布位于 JSON `analysis.linear_generalization`。

## 误差源分解（evaluator-only）

控制臂只在评测器内重算，不增加六个候选 API 调用，也不改变主分数。gain 为相对标准输出误差的改善；interaction 为正表示超加性互补，负表示收益重叠或递减。

### Linear：W / A / 交互

| 分组 | cases | W-only gain | A-only gain | Both gain | interaction | W operand rel-MSE | A operand rel-MSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| overall | 168 | -180.457229 | -34.723528 | 0.448180 | 215.628936 | 4.112577e-01 | 4.465998e-01 |
| family:fc | 48 | -116.306951 | -25.479625 | 0.409512 | 142.196088 | 2.311660e-01 | 3.034954e-01 |
| family:o | 24 | -219.571765 | -45.077931 | 0.361368 | 265.011065 | 8.764757e-01 | 6.882004e-01 |
| family:proj | 24 | -421.171705 | -94.738358 | 0.155105 | 516.065167 | 1.021495e+00 | 9.877018e-01 |
| family:qkv | 72 | -129.947743 | -17.429718 | 0.600587 | 147.978048 | 1.728337e-01 | 2.811019e-01 |
| role:fc_gate | 24 | -173.089537 | -28.491629 | 0.409206 | 201.990372 | 7.073658e-02 | 1.592674e-01 |
| role:fc_up | 24 | -59.524365 | -22.467621 | 0.409818 | 82.401804 | 3.915954e-01 | 4.477233e-01 |
| role:k | 24 | -138.311062 | -23.535176 | 0.602044 | 162.448282 | 1.767590e-01 | 2.213322e-01 |
| role:o | 24 | -219.571765 | -45.077931 | 0.361368 | 265.011065 | 8.764757e-01 | 6.882004e-01 |
| role:proj | 24 | -421.171705 | -94.738358 | 0.155105 | 516.065167 | 1.021495e+00 | 9.877018e-01 |
| role:q | 24 | -143.011209 | -18.021345 | 0.583115 | 161.615668 | 1.969030e-01 | 2.916486e-01 |
| role:v | 24 | -108.520958 | -10.732633 | 0.616602 | 119.870193 | 1.448392e-01 | 3.303248e-01 |
| shape:hidden_to_hidden | 48 | -181.291487 | -31.549638 | 0.472242 | 213.313367 | 5.366893e-01 | 4.899245e-01 |
| shape:hidden_to_wide | 96 | -119.861480 | -21.306765 | 0.509417 | 141.677663 | 1.959826e-01 | 2.896619e-01 |
| shape:wide_to_hidden | 24 | -421.171705 | -94.738358 | 0.155105 | 516.065167 | 1.021495e+00 | 9.877018e-01 |

Linear overall interpretation: `paired_coordinate_coupling_likely`.
Linear 的完整 layer/role/window 结果位于 JSON 的 `decomposition.linear` 和 `case_scores.linear`；优先查看 role 与 shape 行，再定位对应 layer/case。

Attention 分解：已通过 `--no-decomposition` 关闭。

官方成绩只保留为独立历史字段，不参与本地 proxy 评分或时间换算。
