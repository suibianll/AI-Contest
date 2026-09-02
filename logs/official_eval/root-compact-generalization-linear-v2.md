# root-compact-generalization-v2 — proxy-v2

本报告是本地 proxy；隐藏官方数据和鲲鹏硬件不可本地复制。

- evaluation scope: `linear-only-compact-generalization-panel` / `linear-only-low-cost-cross-holdout-mechanism-diagnosis`
- proxy ranking comparable: `False`; official-score equivalent: `False`
- Linear calibration indices: `[1, 2]`; all captured lengths: `[10, 128, 512, 1024, 1024]`
- cases: `56 Linear + 0 Attention` (stratified real-W/A panel by default)
- calibration calls: `28 weight + 0 attention` (shared state)
- input codec: `e4m3-subnormal-ceil-v1` / mode `amax6`
- test splits: `['test', 'validation']`
- source SHA256: `44e37709a02b962cdaedfc57e3ad999b2c9a2c0606b8b9db7e4e81dc4dc92672`
- data pack: `{'train': 'e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7', 'validation': '204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c', 'test': '5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91'}`

| 指标 | 值 |
|---|---:|
| Linear mean | 0.611041176 |
| Attention mean | NA (not run) |
| Overall mean (all captured cases) | 0.611041176 |
| Linear role macro mean | 0.611041176 |
| Attention layer macro mean | 0.000000000 |
| Candidate wall | 45.438s |
| Candidate API total | 40.408s |

## Linear 泛化与尾部分析

均值只作位置统计；优先检查 median、worst-quartile、负 case、跨 split/长度和 W/A/interaction 分布。

| 分组 | cases | mean | median | q25 | worst-quartile mean | min | 正/负/零 | median player/std MSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 56 | 0.611041 | 0.553221 | 0.469447 | 0.421898 | 0.383918 | 56/0/0 | 0.446779 |
| family:fc | 16 | 0.496750 | 0.495631 | 0.467105 | 0.441985 | 0.419541 | 16/0/0 | 0.504369 |
| family:o | 8 | 0.479489 | 0.445528 | 0.402360 | 0.387122 | 0.383918 | 8/0/0 | 0.554472 |
| family:proj | 8 | 0.440669 | 0.431588 | 0.409719 | 0.400173 | 0.396449 | 8/0/0 | 0.568412 |
| family:qkv | 24 | 0.787877 | 0.802203 | 0.750459 | 0.689189 | 0.627255 | 24/0/0 | 0.197797 |
| role:fc_gate | 8 | 0.516341 | 0.509327 | 0.490751 | 0.465480 | 0.462228 | 8/0/0 | 0.490673 |
| role:fc_up | 8 | 0.477159 | 0.483210 | 0.445938 | 0.428460 | 0.419541 | 8/0/0 | 0.516790 |
| role:k | 8 | 0.820010 | 0.809237 | 0.783623 | 0.761678 | 0.757729 | 8/0/0 | 0.190763 |
| role:o | 8 | 0.479489 | 0.445528 | 0.402360 | 0.387122 | 0.383918 | 8/0/0 | 0.554472 |
| role:proj | 8 | 0.440669 | 0.431588 | 0.409719 | 0.400173 | 0.396449 | 8/0/0 | 0.568412 |
| role:q | 8 | 0.753880 | 0.756781 | 0.687703 | 0.634032 | 0.627255 | 8/0/0 | 0.243219 |
| role:v | 8 | 0.789741 | 0.800468 | 0.767887 | 0.725299 | 0.721951 | 8/0/0 | 0.199532 |

跨 layer/shape/split/test_length 的同结构统计，以及 W-only/A-only/Both/interaction 的完整分布位于 JSON `analysis.linear_generalization`。

Cross-holdout：`28/28` 对 validation/test 同号；gain gap median `0.009432`、max `0.063375`；成对 minimum-gain median `0.546436`。最不稳定 pair 位于 JSON `worst_gap_pairs`。

## 误差源分解（evaluator-only）

控制臂只在评测器内重算，不增加六个候选 API 调用，也不改变主分数。gain 为相对标准输出误差的改善；interaction 为正表示超加性互补，负表示收益重叠或递减。

### Linear：W / A / 交互

| 分组 | cases | W-only gain | A-only gain | Both gain | interaction | W operand rel-MSE | A operand rel-MSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| overall | 56 | -197.646665 | -137.979477 | 0.611041 | 336.237183 | 1.892512e+00 | 1.394221e+00 |
| family:fc | 16 | -163.193530 | -141.335279 | 0.496750 | 305.025559 | 2.048362e+00 | 1.581806e+00 |
| family:o | 8 | -173.290740 | -53.631751 | 0.479489 | 227.401980 | 5.759579e-01 | 6.327347e-01 |
| family:proj | 8 | -195.803452 | -184.504214 | 0.440669 | 380.748335 | 2.020202e+00 | 1.937354e+00 |
| family:qkv | 24 | -229.348467 | -148.349940 | 0.787877 | 378.486283 | 2.184899e+00 | 1.341949e+00 |
| role:fc_gate | 8 | -191.850304 | -175.668191 | 0.516341 | 368.034836 | 2.076639e+00 | 1.581572e+00 |
| role:fc_up | 8 | -134.536756 | -107.002366 | 0.477159 | 242.016281 | 2.020085e+00 | 1.582039e+00 |
| role:k | 8 | -317.584151 | -183.460514 | 0.820010 | 501.864675 | 2.233014e+00 | 1.349814e+00 |
| role:o | 8 | -173.290740 | -53.631751 | 0.479489 | 227.401980 | 5.759579e-01 | 6.327347e-01 |
| role:proj | 8 | -195.803452 | -184.504214 | 0.440669 | 380.748335 | 2.020202e+00 | 1.937354e+00 |
| role:q | 8 | -214.370761 | -166.858918 | 0.753880 | 381.983558 | 2.183121e+00 | 1.341454e+00 |
| role:v | 8 | -156.090488 | -94.730388 | 0.789741 | 251.610617 | 2.138563e+00 | 1.334580e+00 |
| shape:hidden_to_hidden | 16 | -193.830751 | -110.245334 | 0.616684 | 304.692769 | 1.379539e+00 | 9.870942e-01 |
| shape:hidden_to_wide | 32 | -200.015425 | -140.215365 | 0.650813 | 340.881602 | 2.117075e+00 | 1.462001e+00 |
| shape:wide_to_hidden | 8 | -195.803452 | -184.504214 | 0.440669 | 380.748335 | 2.020202e+00 | 1.937354e+00 |

Linear overall interpretation: `paired_coordinate_coupling_likely`.
Linear 的完整 layer/role/window 结果位于 JSON 的 `decomposition.linear` 和 `case_scores.linear`；优先查看 role 与 shape 行，再定位对应 layer/case。

Attention 分解：已通过 `--no-decomposition` 关闭。

官方成绩只保留为独立历史字段，不参与本地 proxy 评分或时间换算。
