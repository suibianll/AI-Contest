# candidate — proxy-v2

本报告是本地 proxy；隐藏官方数据和鲲鹏硬件不可本地复制。

- evaluation scope: `ood-generalization-panel` / `overfitting-diagnosis-only-gain-in-minus-gain-ood`
- proxy ranking comparable: `False`; official-score equivalent: `False`
- Linear calibration indices: `[0, 1]`; all captured lengths: `[10, 128, 512, 1024, 1024]`
- cases: `168 Linear + 120 Attention` (stratified real-W/A panel by default)
- calibration calls: `168 weight + 24 attention` (shared state)
- input codec: `e4m3-subnormal-ceil-v1` / mode `amax6`
- test splits: `['code', 'news', 'zh']`
- source SHA256: `dfa69838d8b0cc50411addbf764acc8b3d304d51c73f59fd0e61809ce5925cc2`
- data pack: `{'code': 'fdd7637c5fb93ef5e9ac299cdf995cdb0a8526377d9154967f24a2f6201c3a94', 'news': '2b3023af207b22631f62df37e3e02d276963099a935cb6574cd1cada1eeb08a9', 'zh': '32c6bb5bf4eeb9c7caadd56a1ecdc136a0acb25f86e9700a2f0a6cbab2a92527'}`

| 指标 | 值 |
|---|---:|
| Linear mean | 0.000000000 |
| Attention mean | 0.721813793 |
| Overall mean (all captured cases) | 0.300755747 |
| Linear role macro mean | 0.000000000 |
| Attention layer macro mean | 0.721813793 |
| Candidate wall | 103.758s |
| Candidate API total | 75.173s |

## OOD 泛化摘要

- suite: `ood-suite-v1`；calibration: `in-dist WikiText calibration (shared with the base pack)`
- 定义：per-case gain = (MSE_STD - MSE_PLAYER)/MSE_STD; overfitting signal = gain_in_dist - gain_ood against a matching proxy-v2 run of the same solution

| 侧 | 域 | cases | gain mean | median | worst-quartile | 正/负/零 |
|---|---|---:|---:|---:|---:|---:|
| linear | code | 50 | 0.000000 | 0.000000 | 0.000000 | 0/0/50 |
| linear | news | 68 | 0.000000 | 0.000000 | 0.000000 | 0/0/68 |
| linear | zh | 50 | 0.000000 | 0.000000 | 0.000000 | 0/0/50 |
| linear | **overall** | 168 | 0.000000 | 0.000000 | 0.000000 | 0/0/168 |
| attention | code | 40 | 0.714984 | 0.729078 | 0.550452 | 40/0/0 |
| attention | news | 40 | 0.734079 | 0.733551 | 0.591407 | 40/0/0 |
| attention | zh | 40 | 0.716379 | 0.752420 | 0.535506 | 40/0/0 |
| attention | **overall** | 120 | 0.721814 | 0.734565 | 0.558988 | 120/0/0 |

OOD 均值不参与 proxy 排名；候选是否过拟合看 `gain_in − gain_ood`（与同 solution 的 in-dist proxy-v2 运行相减）。

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
| overall | 168 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 6.835814e-03 | 8.673785e-03 |
| family:fc | 48 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 6.726051e-03 | 8.627197e-03 |
| family:o | 24 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 6.679961e-03 | 8.207920e-03 |
| family:proj | 24 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 6.863414e-03 | 7.657928e-03 |
| family:qkv | 72 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 6.951741e-03 | 9.198751e-03 |
| role:fc_gate | 24 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 6.629023e-03 | 8.536047e-03 |
| role:fc_up | 24 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 6.823080e-03 | 8.718346e-03 |
| role:k | 24 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 6.999664e-03 | 9.224184e-03 |
| role:o | 24 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 6.679961e-03 | 8.207920e-03 |
| role:proj | 24 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 6.863414e-03 | 7.657928e-03 |
| role:q | 24 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 6.885952e-03 | 9.172487e-03 |
| role:v | 24 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 6.969608e-03 | 9.199584e-03 |
| shape:hidden_to_hidden | 48 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 6.782956e-03 | 8.690203e-03 |
| shape:hidden_to_wide | 96 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 6.855344e-03 | 8.919540e-03 |
| shape:wide_to_hidden | 24 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 6.863414e-03 | 7.657928e-03 |

Linear overall interpretation: `mixed_or_neutral`.
Linear 的完整 layer/role/window 结果位于 JSON 的 `decomposition.linear` 和 `case_scores.linear`；优先查看 role 与 shape 行，再定位对应 layer/case。

### Attention：Q / K / V / softmax

| 分组 | cases | Q-only | K-only | V-only | QK-only | Both | QK interaction | QKV interaction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 120 | -18.871744 | -39.049752 | -0.005955 | 0.716405 | 0.721814 | 58.637901 | 0.011363 |
| layer:0 | 15 | -0.659641 | -1.011847 | -0.001185 | 0.834195 | 0.836773 | 2.505683 | 0.003762 |
| layer:10 | 15 | -16.038995 | -28.292267 | -0.008184 | 0.716635 | 0.723775 | 45.047897 | 0.015324 |
| layer:13 | 15 | -21.230656 | -25.218449 | -0.015110 | 0.748904 | 0.756924 | 47.198009 | 0.023130 |
| layer:16 | 15 | -29.443875 | -16.415622 | 0.002177 | 0.566757 | 0.569337 | 46.426254 | 0.000404 |
| layer:20 | 15 | -31.915143 | -40.408941 | -0.006505 | 0.617133 | 0.622196 | 72.941217 | 0.011568 |
| layer:23 | 15 | -19.235070 | -166.928426 | 0.012902 | 0.587795 | 0.601632 | 186.751291 | 0.000934 |
| layer:3 | 15 | -17.742256 | -14.321373 | -0.026136 | 0.840754 | 0.841488 | 32.904383 | 0.026870 |
| layer:7 | 15 | -14.708313 | -19.801093 | -0.005595 | 0.819067 | 0.822386 | 35.328473 | 0.008914 |
| length:10 | 24 | -30.548139 | -56.154989 | -0.007653 | 0.740186 | 0.747970 | 87.443313 | 0.015438 |
| length:1024 | 48 | -14.444362 | -34.306502 | -0.005771 | 0.705512 | 0.709733 | 49.456376 | 0.009992 |
| length:128 | 24 | -18.738286 | -35.502105 | -0.004984 | 0.713456 | 0.720097 | 54.953846 | 0.011626 |
| length:512 | 24 | -16.183570 | -34.978664 | -0.005594 | 0.717359 | 0.721535 | 51.879593 | 0.009770 |

| Attention 中间量（overall） | standard | player |
|---|---:|---:|
| logit MSE vs reference | 7.652625e+01 | 1.550872e+04 |
| probability MSE vs reference | 9.848048e-04 | 1.140591e-04 |
| probability KL(reference || estimate) | 3.260720e-03 | 2.936095e-04 |

Attention overall interpretation: `paired_qk_coupling_likely`.
Attention 的完整 layer/length/window 结果位于 JSON 的 `decomposition.attention` 和 `case_scores.attention`。

官方成绩只保留为独立历史字段，不参与本地 proxy 评分或时间换算。
