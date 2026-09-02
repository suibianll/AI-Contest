# ablate_normsmooth_off — proxy-v2

本报告是本地 proxy；隐藏官方数据和鲲鹏硬件不可本地复制。

- evaluation scope: `linear-only-compact-generalization-panel` / `linear-only-low-cost-cross-holdout-mechanism-diagnosis`
- proxy ranking comparable: `False`; official-score equivalent: `False`
- Linear calibration indices: `[1, 2]`; all captured lengths: `[10, 128, 512, 1024, 1024]`
- cases: `56 Linear + 0 Attention` (stratified real-W/A panel by default)
- calibration calls: `28 weight + 0 attention` (shared state)
- input codec: `e4m3-subnormal-ceil-v1` / mode `amax6`
- test splits: `['test', 'validation']`
- source SHA256: `ecb4d59b79aaa07c543108e992a3263b6aa0d7ee5e47f090c4554e5715f36933`
- data pack: `{'train': 'e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7', 'validation': '204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c', 'test': '5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91'}`

| 指标 | 值 |
|---|---:|
| Linear mean | 0.680523728 |
| Attention mean | NA (not run) |
| Overall mean (all captured cases) | 0.680523728 |
| Linear role macro mean | 0.680523728 |
| Attention layer macro mean | 0.000000000 |
| Candidate wall | 47.845s |
| Candidate API total | 42.727s |

## Linear 泛化与尾部分析

均值只作位置统计；优先检查 median、worst-quartile、负 case、跨 split/长度和 W/A/interaction 分布。

| 分组 | cases | mean | median | q25 | worst-quartile mean | min | 正/负/零 | median player/std MSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 56 | 0.680524 | 0.675217 | 0.546393 | 0.513642 | 0.408555 | 56/0/0 | 0.324783 |
| family:fc | 16 | 0.547820 | 0.547493 | 0.517377 | 0.481593 | 0.408555 | 16/0/0 | 0.452507 |
| family:o | 8 | 0.635171 | 0.620317 | 0.544312 | 0.533446 | 0.527205 | 8/0/0 | 0.379683 |
| family:proj | 8 | 0.558732 | 0.548479 | 0.525413 | 0.502981 | 0.501380 | 8/0/0 | 0.451521 |
| family:qkv | 24 | 0.824707 | 0.827681 | 0.769158 | 0.744312 | 0.704469 | 24/0/0 | 0.172319 |
| role:fc_gate | 8 | 0.582096 | 0.572265 | 0.557534 | 0.531312 | 0.514210 | 8/0/0 | 0.427735 |
| role:fc_up | 8 | 0.513545 | 0.518967 | 0.506621 | 0.450361 | 0.408555 | 8/0/0 | 0.481033 |
| role:k | 8 | 0.832083 | 0.830941 | 0.788751 | 0.759352 | 0.755499 | 8/0/0 | 0.169059 |
| role:o | 8 | 0.635171 | 0.620317 | 0.544312 | 0.533446 | 0.527205 | 8/0/0 | 0.379683 |
| role:proj | 8 | 0.558732 | 0.548479 | 0.525413 | 0.502981 | 0.501380 | 8/0/0 | 0.451521 |
| role:q | 8 | 0.823199 | 0.826312 | 0.762560 | 0.720641 | 0.704469 | 8/0/0 | 0.173688 |
| role:v | 8 | 0.818840 | 0.822147 | 0.790447 | 0.752943 | 0.742827 | 8/0/0 | 0.177853 |

跨 layer/shape/split/test_length 的同结构统计，以及 W-only/A-only/Both/interaction 的完整分布位于 JSON `analysis.linear_generalization`。

Cross-holdout：`28/28` 对 validation/test 同号；gain gap median `0.018419`、max `0.110946`；成对 minimum-gain median `0.668343`。最不稳定 pair 位于 JSON `worst_gap_pairs`。

## 父版本配对效果

基线：`candidate`；候选：`ablate_normsmooth_off`。

| 范围 | cases | mean Δgain | median Δgain | 改善/回归/不变 | median MSE ratio | 结论 |
|---|---:|---:|---:|---:|---:|---|
| Linear overall | 56 | -0.024984 | -0.006309 | 11/35/10 | 1.020465 | mixed |
| Attention overall | 0 | 0.000000 | 0.000000 | 0/0/0 | - | no_cases |

完整 role/family、W/A 来源、最坏 case、Attention 控制臂和 API 时间差分位于 JSON `paired_effect`。

## 误差源分解（evaluator-only）

控制臂只在评测器内重算，不增加六个候选 API 调用，也不改变主分数。gain 为相对标准输出误差的改善；interaction 为正表示超加性互补，负表示收益重叠或递减。

### Linear：W / A / 交互

| 分组 | cases | W-only gain | A-only gain | Both gain | interaction | W operand rel-MSE | A operand rel-MSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| overall | 56 | -213.050302 | -160.769282 | 0.680524 | 374.500107 | 2.186910e+00 | 1.659922e+00 |
| family:fc | 16 | -183.201522 | -162.390928 | 0.547820 | 346.140271 | 2.094277e+00 | 1.738445e+00 |
| family:o | 8 | -226.961444 | -171.403062 | 0.635171 | 398.999677 | 2.160246e+00 | 1.831744e+00 |
| family:proj | 8 | -182.534985 | -178.547657 | 0.558732 | 361.641374 | 2.025109e+00 | 1.979360e+00 |
| family:qkv | 24 | -238.484213 | -150.217466 | 0.824707 | 389.526385 | 2.311487e+00 | 1.443820e+00 |
| role:fc_gate | 8 | -228.911900 | -206.664303 | 0.582096 | 436.158299 | 2.137405e+00 | 1.708711e+00 |
| role:fc_up | 8 | -137.491145 | -118.117552 | 0.513545 | 256.122242 | 2.051149e+00 | 1.768178e+00 |
| role:k | 8 | -271.913828 | -191.975651 | 0.832083 | 464.721562 | 2.308138e+00 | 1.472195e+00 |
| role:o | 8 | -226.961444 | -171.403062 | 0.635171 | 398.999677 | 2.160246e+00 | 1.831744e+00 |
| role:proj | 8 | -182.534985 | -178.547657 | 0.558732 | 361.641374 | 2.025109e+00 | 1.979360e+00 |
| role:q | 8 | -265.847426 | -156.496622 | 0.823199 | 423.167247 | 2.268793e+00 | 1.451407e+00 |
| role:v | 8 | -177.691384 | -102.180123 | 0.818840 | 280.690348 | 2.357530e+00 | 1.407858e+00 |
| shape:hidden_to_hidden | 16 | -246.404435 | -163.949842 | 0.729185 | 411.083462 | 2.214520e+00 | 1.641576e+00 |
| shape:hidden_to_wide | 32 | -204.002064 | -154.734408 | 0.686641 | 359.423113 | 2.213556e+00 | 1.589236e+00 |
| shape:wide_to_hidden | 8 | -182.534985 | -178.547657 | 0.558732 | 361.641374 | 2.025109e+00 | 1.979360e+00 |

Linear overall interpretation: `paired_coordinate_coupling_likely`.
Linear 的完整 layer/role/window 结果位于 JSON 的 `decomposition.linear` 和 `case_scores.linear`；优先查看 role 与 shape 行，再定位对应 layer/case。

Attention 分解：已通过 `--no-decomposition` 关闭。

官方成绩只保留为独立历史字段，不参与本地 proxy 评分或时间换算。

## Decision: REJECTED (2026-09-03)

- Qwen compact 56 cases: mean Δgain -2.5e-2, 35 negative / 11 positive / 10 zero.
- proj 8/8 negative (mean -9.2e-2, worst -0.283), fc 14/16 negative, o 6/8
  negative. RMS smooth is a core mechanism of the v159 Linear path; disabling it
  is a catastrophic regression. Reverted.
