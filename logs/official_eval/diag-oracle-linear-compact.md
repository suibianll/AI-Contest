# candidate — proxy-v2

本报告是本地 proxy；隐藏官方数据和鲲鹏硬件不可本地复制。

- evaluation scope: `linear-only-compact-generalization-panel` / `linear-only-low-cost-cross-holdout-mechanism-diagnosis`
- proxy ranking comparable: `False`; official-score equivalent: `False`
- Linear calibration indices: `[1, 2]`; all captured lengths: `[10, 128, 512, 1024, 1024]`
- cases: `56 Linear + 0 Attention` (stratified real-W/A panel by default)
- calibration calls: `28 weight + 0 attention` (shared state)
- input codec: `e4m3-subnormal-ceil-v1` / mode `amax6`
- test splits: `['test', 'validation']`
- source SHA256: `11a09f207df1ad35f58d09f51655486d9273fb3ef598b42e5bfd91dae4e31fbb`
- data pack: `{'train': 'e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7', 'validation': '204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c', 'test': '5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91'}`

| 指标 | 值 |
|---|---:|
| Linear mean | 0.705703425 |
| Attention mean | NA (not run) |
| Overall mean (all captured cases) | 0.705703425 |
| Linear role macro mean | 0.705703425 |
| Attention layer macro mean | 0.000000000 |
| Candidate wall | 288.941s |
| Candidate API total | 283.852s |

## Linear 泛化与尾部分析

均值只作位置统计；优先检查 median、worst-quartile、负 case、跨 split/长度和 W/A/interaction 分布。

| 分组 | cases | mean | median | q25 | worst-quartile mean | min | 正/负/零 | median player/std MSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 56 | 0.705703 | 0.685012 | 0.592082 | 0.539990 | 0.420349 | 56/0/0 | 0.314988 |
| family:fc | 16 | 0.563546 | 0.558955 | 0.543284 | 0.495499 | 0.420349 | 16/0/0 | 0.441045 |
| family:o | 8 | 0.688604 | 0.643963 | 0.618298 | 0.610842 | 0.604982 | 8/0/0 | 0.356037 |
| family:proj | 8 | 0.650717 | 0.607633 | 0.578572 | 0.527108 | 0.525886 | 8/0/0 | 0.392367 |
| family:qkv | 24 | 0.824504 | 0.824961 | 0.767744 | 0.745670 | 0.706439 | 24/0/0 | 0.175039 |
| role:fc_gate | 8 | 0.595408 | 0.585678 | 0.559508 | 0.557583 | 0.557317 | 8/0/0 | 0.414322 |
| role:fc_up | 8 | 0.531684 | 0.538838 | 0.520135 | 0.463815 | 0.420349 | 8/0/0 | 0.461162 |
| role:k | 8 | 0.833010 | 0.830678 | 0.787194 | 0.758798 | 0.756922 | 8/0/0 | 0.169322 |
| role:o | 8 | 0.688604 | 0.643963 | 0.618298 | 0.610842 | 0.604982 | 8/0/0 | 0.356037 |
| role:proj | 8 | 0.650717 | 0.607633 | 0.578572 | 0.527108 | 0.525886 | 8/0/0 | 0.392367 |
| role:q | 8 | 0.821169 | 0.820012 | 0.762774 | 0.723616 | 0.706439 | 8/0/0 | 0.179988 |
| role:v | 8 | 0.819333 | 0.823509 | 0.789031 | 0.754597 | 0.750367 | 8/0/0 | 0.176491 |

跨 layer/shape/split/test_length 的同结构统计，以及 W-only/A-only/Both/interaction 的完整分布位于 JSON `analysis.linear_generalization`。

Cross-holdout：`28/28` 对 validation/test 同号；gain gap median `0.016293`、max `0.163022`；成对 minimum-gain median `0.684431`。最不稳定 pair 位于 JSON `worst_gap_pairs`。

## 父版本配对效果

基线：`candidate`；候选：`candidate`。

| 范围 | cases | mean Δgain | median Δgain | 改善/回归/不变 | median MSE ratio | 结论 |
|---|---:|---:|---:|---:|---:|---|
| Linear overall | 56 | 0.000169 | 0.000095 | 31/25/0 | 0.999728 | mixed |
| Attention overall | 0 | 0.000000 | 0.000000 | 0/0/0 | - | no_cases |

完整 role/family、W/A 来源、最坏 case、Attention 控制臂和 API 时间差分位于 JSON `paired_effect`。

## 误差源分解（evaluator-only）

控制臂只在评测器内重算，不增加六个候选 API 调用，也不改变主分数。gain 为相对标准输出误差的改善；interaction 为正表示超加性互补，负表示收益重叠或递减。

### Linear：W / A / 交互

| 分组 | cases | W-only gain | A-only gain | Both gain | interaction | W operand rel-MSE | A operand rel-MSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| overall | 56 | -226.334855 | -151.255777 | 0.705703 | 378.296335 | 2.239703e+00 | 1.538218e+00 |
| family:fc | 16 | -190.196642 | -153.253965 | 0.563546 | 344.014152 | 2.136212e+00 | 1.642220e+00 |
| family:o | 8 | -197.793945 | -155.852057 | 0.688604 | 354.334606 | 2.099436e+00 | 1.662807e+00 |
| family:proj | 8 | -226.150255 | -153.883027 | 0.650717 | 380.684000 | 2.137150e+00 | 1.622552e+00 |
| family:qkv | 24 | -260.002166 | -147.515809 | 0.824504 | 408.342479 | 2.389637e+00 | 1.399243e+00 |
| role:fc_gate | 8 | -235.310639 | -197.503893 | 0.595408 | 433.409939 | 2.173504e+00 | 1.657345e+00 |
| role:fc_up | 8 | -145.082645 | -109.004038 | 0.531684 | 254.618366 | 2.098920e+00 | 1.627095e+00 |
| role:k | 8 | -290.350762 | -188.534427 | 0.833010 | 479.718199 | 2.388054e+00 | 1.430792e+00 |
| role:o | 8 | -197.793945 | -155.852057 | 0.688604 | 354.334606 | 2.099436e+00 | 1.662807e+00 |
| role:proj | 8 | -226.150255 | -153.883027 | 0.650717 | 380.684000 | 2.137150e+00 | 1.622552e+00 |
| role:q | 8 | -302.671237 | -153.888291 | 0.821169 | 457.380697 | 2.396237e+00 | 1.391783e+00 |
| role:v | 8 | -186.984500 | -100.124708 | 0.819333 | 287.928541 | 2.384620e+00 | 1.375152e+00 |
| shape:hidden_to_hidden | 16 | -250.232591 | -154.870174 | 0.754886 | 405.857651 | 2.247836e+00 | 1.527295e+00 |
| shape:hidden_to_wide | 32 | -214.432136 | -148.791766 | 0.694858 | 363.918761 | 2.261274e+00 | 1.522596e+00 |
| shape:wide_to_hidden | 8 | -226.150255 | -153.883027 | 0.650717 | 380.684000 | 2.137150e+00 | 1.622552e+00 |

Linear overall interpretation: `paired_coordinate_coupling_likely`.
Linear 的完整 layer/role/window 结果位于 JSON 的 `decomposition.linear` 和 `case_scores.linear`；优先查看 role 与 shape 行，再定位对应 layer/case。

Attention 分解：已通过 `--no-decomposition` 关闭。

官方成绩只保留为独立历史字段，不参与本地 proxy 评分或时间换算。
