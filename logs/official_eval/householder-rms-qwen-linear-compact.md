# householder-rms-qwen-linear-compact — proxy-v2

本报告是本地 proxy；隐藏官方数据和鲲鹏硬件不可本地复制。

- evaluation scope: `linear-only-compact-generalization-panel` / `linear-only-low-cost-cross-holdout-mechanism-diagnosis`
- proxy ranking comparable: `False`; official-score equivalent: `False`
- Linear calibration indices: `[1, 2]`; all captured lengths: `[10, 128, 512, 1024, 1024]`
- cases: `56 Linear + 0 Attention` (stratified real-W/A panel by default)
- calibration calls: `28 weight + 0 attention` (shared state)
- input codec: `e4m3-subnormal-ceil-v1` / mode `amax6`
- test splits: `['test', 'validation']`
- source SHA256: `334a01890e54ae224cf9a3d473e2afae3ce50b1ed1e4fbc2760acb42fb91415c`
- data pack: `{'train': 'e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7', 'validation': '204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c', 'test': '5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91'}`

| 指标 | 值 |
|---|---:|
| Linear mean | 0.703092965 |
| Attention mean | NA (not run) |
| Overall mean (all captured cases) | 0.703092965 |
| Linear role macro mean | 0.703092965 |
| Attention layer macro mean | 0.000000000 |
| Candidate wall | 52.474s |
| Candidate API total | 47.327s |

## Linear 泛化与尾部分析

均值只作位置统计；优先检查 median、worst-quartile、负 case、跨 split/长度和 W/A/interaction 分布。

| 分组 | cases | mean | median | q25 | worst-quartile mean | min | 正/负/零 | median player/std MSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 56 | 0.703093 | 0.687818 | 0.585586 | 0.535980 | 0.413511 | 56/0/0 | 0.312182 |
| family:fc | 16 | 0.559301 | 0.555586 | 0.544129 | 0.491908 | 0.413511 | 16/0/0 | 0.444414 |
| family:o | 8 | 0.691217 | 0.643610 | 0.623038 | 0.612565 | 0.602717 | 8/0/0 | 0.356390 |
| family:proj | 8 | 0.647238 | 0.607314 | 0.571172 | 0.523052 | 0.522528 | 8/0/0 | 0.392686 |
| family:qkv | 24 | 0.821531 | 0.825100 | 0.763808 | 0.741724 | 0.705729 | 24/0/0 | 0.174900 |
| role:fc_gate | 8 | 0.590542 | 0.581321 | 0.556396 | 0.552090 | 0.550214 | 8/0/0 | 0.418679 |
| role:fc_up | 8 | 0.528060 | 0.538131 | 0.518113 | 0.459633 | 0.413511 | 8/0/0 | 0.461869 |
| role:k | 8 | 0.828073 | 0.828597 | 0.784931 | 0.753448 | 0.748167 | 8/0/0 | 0.171403 |
| role:o | 8 | 0.691217 | 0.643610 | 0.623038 | 0.612565 | 0.602717 | 8/0/0 | 0.356390 |
| role:proj | 8 | 0.647238 | 0.607314 | 0.571172 | 0.523052 | 0.522528 | 8/0/0 | 0.392686 |
| role:q | 8 | 0.818631 | 0.817261 | 0.757823 | 0.721090 | 0.705729 | 8/0/0 | 0.182739 |
| role:v | 8 | 0.817890 | 0.820519 | 0.791415 | 0.750633 | 0.740876 | 8/0/0 | 0.179481 |

跨 layer/shape/split/test_length 的同结构统计，以及 W-only/A-only/Both/interaction 的完整分布位于 JSON `analysis.linear_generalization`。

Cross-holdout：`28/28` 对 validation/test 同号；gain gap median `0.018648`、max `0.159840`；成对 minimum-gain median `0.683742`。最不稳定 pair 位于 JSON `worst_gap_pairs`。

## 误差源分解（evaluator-only）

控制臂只在评测器内重算，不增加六个候选 API 调用，也不改变主分数。gain 为相对标准输出误差的改善；interaction 为正表示超加性互补，负表示收益重叠或递减。

### Linear：W / A / 交互

| 分组 | cases | W-only gain | A-only gain | Both gain | interaction | W operand rel-MSE | A operand rel-MSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| overall | 56 | -208.659715 | -149.793709 | 0.703093 | 359.156516 | 2.236399e+00 | 1.525980e+00 |
| family:fc | 16 | -180.271623 | -151.206927 | 0.559301 | 332.037850 | 2.132078e+00 | 1.633032e+00 |
| family:o | 8 | -193.949753 | -154.216887 | 0.691217 | 348.857857 | 2.097850e+00 | 1.659280e+00 |
| family:proj | 8 | -219.971545 | -153.774906 | 0.647238 | 374.393689 | 2.135953e+00 | 1.620257e+00 |
| family:qkv | 24 | -228.717820 | -146.050104 | 0.821531 | 375.589456 | 2.385611e+00 | 1.378753e+00 |
| role:fc_gate | 8 | -218.693402 | -193.628099 | 0.590542 | 412.912043 | 2.168086e+00 | 1.649360e+00 |
| role:fc_up | 8 | -141.849843 | -108.785755 | 0.528060 | 251.163658 | 2.096070e+00 | 1.616703e+00 |
| role:k | 8 | -242.608012 | -185.303994 | 0.828073 | 428.740079 | 2.382650e+00 | 1.403454e+00 |
| role:o | 8 | -193.949753 | -154.216887 | 0.691217 | 348.857857 | 2.097850e+00 | 1.659280e+00 |
| role:proj | 8 | -219.971545 | -153.774906 | 0.647238 | 374.393689 | 2.135953e+00 | 1.620257e+00 |
| role:q | 8 | -266.375911 | -152.724931 | 0.818631 | 419.919473 | 2.389862e+00 | 1.373958e+00 |
| role:v | 8 | -177.169537 | -100.121387 | 0.817890 | 278.108815 | 2.384321e+00 | 1.358847e+00 |
| shape:hidden_to_hidden | 16 | -230.162832 | -153.470909 | 0.754924 | 384.388665 | 2.243856e+00 | 1.516619e+00 |
| shape:hidden_to_wide | 32 | -195.080199 | -146.959809 | 0.691141 | 342.731149 | 2.257782e+00 | 1.507091e+00 |
| shape:wide_to_hidden | 8 | -219.971545 | -153.774906 | 0.647238 | 374.393689 | 2.135953e+00 | 1.620257e+00 |

Linear overall interpretation: `paired_coordinate_coupling_likely`.
Linear 的完整 layer/role/window 结果位于 JSON 的 `decomposition.linear` 和 `case_scores.linear`；优先查看 role 与 shape 行，再定位对应 layer/case。

Attention 分解：已通过 `--no-decomposition` 关闭。

官方成绩只保留为独立历史字段，不参与本地 proxy 评分或时间换算。
