# v186 — proxy-v2

本报告是本地 proxy；隐藏官方数据和鲲鹏硬件不可本地复制。

- evaluation scope: `attention-only-ood-generalization-panel` / `attention-only-overfitting-diagnosis-only-gain-in-minus-gain-ood`
- proxy ranking comparable: `False`; official-score equivalent: `False`
- Linear calibration indices: `[0, 1]`; all captured lengths: `[10, 128, 512, 1024, 1024]`
- cases: `0 Linear + 120 Attention` (stratified real-W/A panel by default)
- calibration calls: `0 weight + 24 attention` (shared state)
- input codec: `e4m3-subnormal-ceil-v1` / mode `amax6`
- test splits: `['code', 'news', 'zh']`
- source SHA256: `f8495dca20334acbdad16fc18ee41a4970f31e1837fdeedcee9c70aee54e7eb8`
- data pack: `{'code': 'fdd7637c5fb93ef5e9ac299cdf995cdb0a8526377d9154967f24a2f6201c3a94', 'news': '2b3023af207b22631f62df37e3e02d276963099a935cb6574cd1cada1eeb08a9', 'zh': '32c6bb5bf4eeb9c7caadd56a1ecdc136a0acb25f86e9700a2f0a6cbab2a92527'}`

| 指标 | 值 |
|---|---:|
| Linear mean | NA (not run) |
| Attention mean | 0.719497852 |
| Overall mean (all captured cases) | 0.719497852 |
| Linear role macro mean | 0.000000000 |
| Attention layer macro mean | 0.719497852 |
| Candidate wall | 73.748s |
| Candidate API total | 68.731s |

## OOD 泛化摘要

- suite: `ood-suite-v1`；calibration: `in-dist WikiText calibration (shared with the base pack)`
- 定义：per-case gain = (MSE_STD - MSE_PLAYER)/MSE_STD; overfitting signal = gain_in_dist - gain_ood against a matching proxy-v2 run of the same solution

| 侧 | 域 | cases | gain mean | median | worst-quartile | 正/负/零 |
|---|---|---:|---:|---:|---:|---:|
| linear | **overall** | 0 | 0.000000 | 0.000000 | 0.000000 | 0/0/0 |
| attention | code | 40 | 0.707819 | 0.727571 | 0.530483 | 40/0/0 |
| attention | news | 40 | 0.728266 | 0.709411 | 0.562578 | 40/0/0 |
| attention | zh | 40 | 0.722409 | 0.723825 | 0.560229 | 40/0/0 |
| attention | **overall** | 120 | 0.719498 | 0.723375 | 0.551097 | 120/0/0 |

OOD 均值不参与 proxy 排名；候选是否过拟合看 `gain_in − gain_ood`（与同 solution 的 in-dist proxy-v2 运行相减）。

## 误差源分解（evaluator-only）

控制臂只在评测器内重算，不增加六个候选 API 调用，也不改变主分数。gain 为相对标准输出误差的改善；interaction 为正表示超加性互补，负表示收益重叠或递减。

Linear 分解：已通过 `--no-decomposition` 关闭。

### Attention：Q / K / V / softmax

| 分组 | cases | Q-only | K-only | V-only | QK-only | Both | QK interaction | QKV interaction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 120 | -19.238859 | -39.684935 | -0.005955 | 0.715004 | 0.719498 | 59.638799 | 0.010448 |
| layer:0 | 15 | -0.647137 | -0.974583 | -0.001185 | 0.864002 | 0.865601 | 2.485722 | 0.002784 |
| layer:10 | 15 | -15.931711 | -29.075114 | -0.008184 | 0.713893 | 0.720409 | 45.720719 | 0.014699 |
| layer:13 | 15 | -26.751645 | -25.103393 | -0.015110 | 0.706969 | 0.710994 | 52.562007 | 0.019135 |
| layer:16 | 15 | -28.655624 | -17.446216 | 0.002177 | 0.552817 | 0.553802 | 46.654658 | -0.001192 |
| layer:20 | 15 | -30.428153 | -42.581192 | -0.006505 | 0.610209 | 0.612677 | 73.619555 | 0.008973 |
| layer:23 | 15 | -19.148883 | -167.706657 | 0.012902 | 0.610411 | 0.625601 | 187.465951 | 0.002288 |
| layer:3 | 15 | -17.568248 | -14.514770 | -0.026136 | 0.843363 | 0.845448 | 32.926381 | 0.028222 |
| layer:7 | 15 | -14.779471 | -20.077557 | -0.005595 | 0.818369 | 0.821451 | 35.675397 | 0.008677 |
| length:10 | 24 | -32.591989 | -55.066088 | -0.007653 | 0.715943 | 0.722650 | 88.374019 | 0.014361 |
| length:1024 | 48 | -14.203898 | -35.428957 | -0.005771 | 0.715320 | 0.718247 | 50.348174 | 0.008698 |
| length:128 | 24 | -18.937250 | -35.786325 | -0.004984 | 0.712456 | 0.718906 | 55.436032 | 0.011434 |
| length:512 | 24 | -16.257261 | -36.714351 | -0.005594 | 0.715982 | 0.719438 | 53.687594 | 0.009050 |

| Attention 中间量（overall） | standard | player |
|---|---:|---:|
| logit MSE vs reference | 7.652625e+01 | 1.550866e+04 |
| probability MSE vs reference | 9.848048e-04 | 1.208878e-04 |
| probability KL(reference || estimate) | 3.260720e-03 | 3.020616e-04 |

Attention overall interpretation: `paired_qk_coupling_likely`.
Attention 的完整 layer/length/window 结果位于 JSON 的 `decomposition.attention` 和 `case_scores.attention`。

官方成绩只保留为独立历史字段，不参与本地 proxy 评分或时间换算。
