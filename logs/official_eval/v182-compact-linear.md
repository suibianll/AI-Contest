# candidate — proxy-v2

本报告是本地 proxy；隐藏官方数据和鲲鹏硬件不可本地复制。

- evaluation scope: `linear-only-compact-generalization-panel` / `linear-only-low-cost-cross-holdout-mechanism-diagnosis`
- proxy ranking comparable: `False`; official-score equivalent: `False`
- Linear calibration indices: `[1, 2]`; all captured lengths: `[10, 128, 512, 1024, 1024]`
- cases: `56 Linear + 0 Attention` (stratified real-W/A panel by default)
- calibration calls: `28 weight + 0 attention` (shared state)
- input codec: `e4m3-subnormal-ceil-v1` / mode `amax6`
- test splits: `['test', 'validation']`
- source SHA256: `f3e39e993a436e217cb4811525c81239f82a6ec58845a0646e183a824c33a438`
- data pack: `{'train': 'e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7', 'validation': '204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c', 'test': '5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91'}`

| 指标 | 值 |
|---|---:|
| Linear mean | 0.705534906 |
| Attention mean | NA (not run) |
| Overall mean (all captured cases) | 0.705534906 |
| Linear role macro mean | 0.705534906 |
| Attention layer macro mean | 0.000000000 |
| Candidate wall | 71.561s |
| Candidate API total | 66.776s |

## Linear 泛化与尾部分析

均值只作位置统计；优先检查 median、worst-quartile、负 case、跨 split/长度和 W/A/interaction 分布。

| 分组 | cases | mean | median | q25 | worst-quartile mean | min | 正/负/零 | median player/std MSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 56 | 0.705535 | 0.684189 | 0.591902 | 0.539742 | 0.420840 | 56/0/0 | 0.315811 |
| family:fc | 16 | 0.563422 | 0.558746 | 0.543689 | 0.495642 | 0.420840 | 16/0/0 | 0.441254 |
| family:o | 8 | 0.687273 | 0.641927 | 0.618530 | 0.607591 | 0.601658 | 8/0/0 | 0.358073 |
| family:proj | 8 | 0.650671 | 0.607867 | 0.578487 | 0.526400 | 0.524968 | 8/0/0 | 0.392133 |
| family:qkv | 24 | 0.824652 | 0.825622 | 0.766851 | 0.745611 | 0.706422 | 24/0/0 | 0.174378 |
| role:fc_gate | 8 | 0.595125 | 0.585223 | 0.558953 | 0.557237 | 0.556145 | 8/0/0 | 0.414777 |
| role:fc_up | 8 | 0.531718 | 0.539265 | 0.520034 | 0.463870 | 0.420840 | 8/0/0 | 0.460735 |
| role:k | 8 | 0.831923 | 0.829911 | 0.787643 | 0.755617 | 0.751622 | 8/0/0 | 0.170089 |
| role:o | 8 | 0.687273 | 0.641927 | 0.618530 | 0.607591 | 0.601658 | 8/0/0 | 0.358073 |
| role:proj | 8 | 0.650671 | 0.607867 | 0.578487 | 0.526400 | 0.524968 | 8/0/0 | 0.392133 |
| role:q | 8 | 0.821832 | 0.819797 | 0.762705 | 0.724723 | 0.706422 | 8/0/0 | 0.180203 |
| role:v | 8 | 0.820202 | 0.823318 | 0.788431 | 0.756495 | 0.754812 | 8/0/0 | 0.176682 |

跨 layer/shape/split/test_length 的同结构统计，以及 W-only/A-only/Both/interaction 的完整分布位于 JSON `analysis.linear_generalization`。

Cross-holdout：`28/28` 对 validation/test 同号；gain gap median `0.016487`、max `0.161985`；成对 minimum-gain median `0.682799`。最不稳定 pair 位于 JSON `worst_gap_pairs`。

## 父版本配对效果

基线：`candidate`；候选：`candidate`。

| 范围 | cases | mean Δgain | median Δgain | 改善/回归/不变 | median MSE ratio | 结论 |
|---|---:|---:|---:|---:|---:|---|
| Linear overall | 56 | -0.000093 | 0.000073 | 30/26/0 | 0.999585 | mixed |
| Attention overall | 0 | 0.000000 | 0.000000 | 0/0/0 | - | no_cases |

完整 role/family、W/A 来源、最坏 case、Attention 控制臂和 API 时间差分位于 JSON `paired_effect`。

## 误差源分解（evaluator-only）

控制臂只在评测器内重算，不增加六个候选 API 调用，也不改变主分数。gain 为相对标准输出误差的改善；interaction 为正表示超加性互补，负表示收益重叠或递减。

### Linear：W / A / 交互

| 分组 | cases | W-only gain | A-only gain | Both gain | interaction | W operand rel-MSE | A operand rel-MSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| overall | 56 | -226.272086 | -151.226487 | 0.705535 | 378.204107 | 2.239532e+00 | 1.538066e+00 |
| family:fc | 16 | -190.189387 | -153.243209 | 0.563422 | 343.996017 | 2.135971e+00 | 1.642043e+00 |
| family:o | 8 | -197.754942 | -155.847053 | 0.687273 | 354.289267 | 2.099231e+00 | 1.662944e+00 |
| family:proj | 8 | -226.108165 | -153.872545 | 0.650671 | 380.631381 | 2.136918e+00 | 1.622399e+00 |
| family:qkv | 24 | -259.887573 | -147.459798 | 0.824652 | 408.172023 | 2.389545e+00 | 1.399012e+00 |
| role:fc_gate | 8 | -235.310943 | -197.483618 | 0.595125 | 433.389686 | 2.173244e+00 | 1.657082e+00 |
| role:fc_up | 8 | -145.067831 | -109.002800 | 0.531718 | 254.602349 | 2.098698e+00 | 1.627003e+00 |
| role:k | 8 | -290.346482 | -188.395301 | 0.831923 | 479.573705 | 2.388026e+00 | 1.430680e+00 |
| role:o | 8 | -197.754942 | -155.847053 | 0.687273 | 354.289267 | 2.099231e+00 | 1.662944e+00 |
| role:proj | 8 | -226.108165 | -153.872545 | 0.650671 | 380.631381 | 2.136918e+00 | 1.622399e+00 |
| role:q | 8 | -302.379968 | -153.865620 | 0.821832 | 457.067420 | 2.395969e+00 | 1.391398e+00 |
| role:v | 8 | -186.936269 | -100.118472 | 0.820202 | 287.874943 | 2.384641e+00 | 1.374958e+00 |
| shape:hidden_to_hidden | 16 | -250.067455 | -154.856337 | 0.754552 | 405.678344 | 2.247600e+00 | 1.527171e+00 |
| shape:hidden_to_wide | 32 | -214.415381 | -148.750048 | 0.694742 | 363.860171 | 2.261152e+00 | 1.522431e+00 |
| shape:wide_to_hidden | 8 | -226.108165 | -153.872545 | 0.650671 | 380.631381 | 2.136918e+00 | 1.622399e+00 |

Linear overall interpretation: `paired_coordinate_coupling_likely`.
Linear 的完整 layer/role/window 结果位于 JSON 的 `decomposition.linear` 和 `case_scores.linear`；优先查看 role 与 shape 行，再定位对应 layer/case。

Attention 分解：已通过 `--no-decomposition` 关闭。

官方成绩只保留为独立历史字段，不参与本地 proxy 评分或时间换算。
