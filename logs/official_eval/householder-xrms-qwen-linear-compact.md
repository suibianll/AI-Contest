# householder-xrms-qwen-linear-compact — proxy-v2

本报告是本地 proxy；隐藏官方数据和鲲鹏硬件不可本地复制。

- evaluation scope: `linear-only-compact-generalization-panel` / `linear-only-low-cost-cross-holdout-mechanism-diagnosis`
- proxy ranking comparable: `False`; official-score equivalent: `False`
- Linear calibration indices: `[1, 2]`; all captured lengths: `[10, 128, 512, 1024, 1024]`
- cases: `56 Linear + 0 Attention` (stratified real-W/A panel by default)
- calibration calls: `28 weight + 0 attention` (shared state)
- input codec: `e4m3-subnormal-ceil-v1` / mode `amax6`
- test splits: `['test', 'validation']`
- source SHA256: `7c1f18ac27a0deb1cf06e5c60ebf680e97a3c3bb0b8cd2bb2498cdffa2b60579`
- data pack: `{'train': 'e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7', 'validation': '204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c', 'test': '5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91'}`

| 指标 | 值 |
|---|---:|
| Linear mean | 0.702938525 |
| Attention mean | NA (not run) |
| Overall mean (all captured cases) | 0.702938525 |
| Linear role macro mean | 0.702938525 |
| Attention layer macro mean | 0.000000000 |
| Candidate wall | 51.030s |
| Candidate API total | 46.455s |

## Linear 泛化与尾部分析

均值只作位置统计；优先检查 median、worst-quartile、负 case、跨 split/长度和 W/A/interaction 分布。

| 分组 | cases | mean | median | q25 | worst-quartile mean | min | 正/负/零 | median player/std MSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 56 | 0.702939 | 0.681441 | 0.586346 | 0.536222 | 0.415236 | 56/0/0 | 0.318559 |
| family:fc | 16 | 0.559091 | 0.555106 | 0.541754 | 0.492334 | 0.415236 | 16/0/0 | 0.444894 |
| family:o | 8 | 0.690889 | 0.648922 | 0.625324 | 0.606528 | 0.595119 | 8/0/0 | 0.351078 |
| family:proj | 8 | 0.649355 | 0.609127 | 0.572938 | 0.523478 | 0.520661 | 8/0/0 | 0.390873 |
| family:qkv | 24 | 0.820714 | 0.824703 | 0.770950 | 0.741675 | 0.696862 | 24/0/0 | 0.175297 |
| role:fc_gate | 8 | 0.590226 | 0.583097 | 0.556522 | 0.551217 | 0.550159 | 8/0/0 | 0.416903 |
| role:fc_up | 8 | 0.527957 | 0.536193 | 0.518658 | 0.460733 | 0.415236 | 8/0/0 | 0.463807 |
| role:k | 8 | 0.826994 | 0.828857 | 0.784340 | 0.751314 | 0.742228 | 8/0/0 | 0.171143 |
| role:o | 8 | 0.690889 | 0.648922 | 0.625324 | 0.606528 | 0.595119 | 8/0/0 | 0.351078 |
| role:proj | 8 | 0.649355 | 0.609127 | 0.572938 | 0.523478 | 0.520661 | 8/0/0 | 0.390873 |
| role:q | 8 | 0.818106 | 0.817313 | 0.759350 | 0.718466 | 0.696862 | 8/0/0 | 0.182687 |
| role:v | 8 | 0.817042 | 0.820718 | 0.788812 | 0.758695 | 0.744716 | 8/0/0 | 0.179282 |

跨 layer/shape/split/test_length 的同结构统计，以及 W-only/A-only/Both/interaction 的完整分布位于 JSON `analysis.linear_generalization`。

Cross-holdout：`28/28` 对 validation/test 同号；gain gap median `0.017684`、max `0.162603`；成对 minimum-gain median `0.680368`。最不稳定 pair 位于 JSON `worst_gap_pairs`。

## 误差源分解（evaluator-only）

控制臂只在评测器内重算，不增加六个候选 API 调用，也不改变主分数。gain 为相对标准输出误差的改善；interaction 为正表示超加性互补，负表示收益重叠或递减。

### Linear：W / A / 交互

| 分组 | cases | W-only gain | A-only gain | Both gain | interaction | W operand rel-MSE | A operand rel-MSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| overall | 56 | -208.355786 | -149.847968 | 0.702939 | 358.906693 | 2.236426e+00 | 1.526274e+00 |
| family:fc | 16 | -180.535157 | -151.550779 | 0.559091 | 332.645028 | 2.132067e+00 | 1.634501e+00 |
| family:o | 8 | -196.063151 | -154.844116 | 0.690889 | 351.598155 | 2.098628e+00 | 1.661173e+00 |
| family:proj | 8 | -218.820876 | -153.797891 | 0.649355 | 373.268122 | 2.135947e+00 | 1.621432e+00 |
| family:qkv | 24 | -227.512054 | -145.730737 | 0.820714 | 374.063506 | 2.385425e+00 | 1.377438e+00 |
| role:fc_gate | 8 | -219.459600 | -194.114308 | 0.590226 | 414.164134 | 2.168107e+00 | 1.651413e+00 |
| role:fc_up | 8 | -141.610714 | -108.987251 | 0.527957 | 251.125922 | 2.096027e+00 | 1.617588e+00 |
| role:k | 8 | -245.849155 | -185.629796 | 0.826994 | 432.305946 | 2.383081e+00 | 1.405200e+00 |
| role:o | 8 | -196.063151 | -154.844116 | 0.690889 | 351.598155 | 2.098628e+00 | 1.661173e+00 |
| role:proj | 8 | -218.820876 | -153.797891 | 0.649355 | 373.268122 | 2.135947e+00 | 1.621432e+00 |
| role:q | 8 | -262.602535 | -152.441074 | 0.818106 | 415.861715 | 2.390651e+00 | 1.374720e+00 |
| role:v | 8 | -174.084473 | -99.121340 | 0.817042 | 274.022856 | 2.382544e+00 | 1.352395e+00 |
| shape:hidden_to_hidden | 16 | -229.332843 | -153.642595 | 0.754497 | 383.729935 | 2.244639e+00 | 1.517947e+00 |
| shape:hidden_to_wide | 32 | -195.250986 | -146.963174 | 0.690555 | 342.904714 | 2.257440e+00 | 1.506649e+00 |
| shape:wide_to_hidden | 8 | -218.820876 | -153.797891 | 0.649355 | 373.268122 | 2.135947e+00 | 1.621432e+00 |

Linear overall interpretation: `paired_coordinate_coupling_likely`.
Linear 的完整 layer/role/window 结果位于 JSON 的 `decomposition.linear` 和 `case_scores.linear`；优先查看 role 与 shape 行，再定位对应 layer/case。

Attention 分解：已通过 `--no-decomposition` 关闭。

官方成绩只保留为独立历史字段，不参与本地 proxy 评分或时间换算。
