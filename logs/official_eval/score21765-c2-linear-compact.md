# candidate — proxy-v2

本报告是本地 proxy；隐藏官方数据和鲲鹏硬件不可本地复制。

- evaluation scope: `linear-only-compact-generalization-panel` / `linear-only-low-cost-cross-holdout-mechanism-diagnosis`
- proxy ranking comparable: `False`; official-score equivalent: `False`
- Linear calibration indices: `[1, 2]`; all captured lengths: `[10, 128, 512, 1024, 1024]`
- cases: `56 Linear + 0 Attention` (stratified real-W/A panel by default)
- calibration calls: `28 weight + 0 attention` (shared state)
- input codec: `e4m3-subnormal-ceil-v1` / mode `amax6`
- test splits: `['test', 'validation']`
- source SHA256: `3e469337bfcdda9d53bcc288ff4a57ccdc59c107c48e3e8ecb7abb5f7256aebe`
- data pack: `{'train': 'e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7', 'validation': '204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c', 'test': '5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91'}`

| 指标 | 值 |
|---|---:|
| Linear mean | 0.616732930 |
| Attention mean | NA (not run) |
| Overall mean (all captured cases) | 0.616732930 |
| Linear role macro mean | 0.616732930 |
| Attention layer macro mean | 0.000000000 |
| Candidate wall | 120.991s |
| Candidate API total | 115.755s |

## Linear 泛化与尾部分析

均值只作位置统计；优先检查 median、worst-quartile、负 case、跨 split/长度和 W/A/interaction 分布。

| 分组 | cases | mean | median | q25 | worst-quartile mean | min | 正/负/零 | median player/std MSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 56 | 0.616733 | 0.597357 | 0.451769 | 0.378292 | 0.207798 | 56/0/0 | 0.402643 |
| family:fc | 16 | 0.422460 | 0.422441 | 0.390031 | 0.328073 | 0.207798 | 16/0/0 | 0.577559 |
| family:o | 8 | 0.607797 | 0.554952 | 0.516985 | 0.496793 | 0.483871 | 8/0/0 | 0.445048 |
| family:proj | 8 | 0.522409 | 0.464369 | 0.401706 | 0.313650 | 0.307986 | 8/0/0 | 0.535631 |
| family:qkv | 24 | 0.780668 | 0.778020 | 0.701722 | 0.669637 | 0.617081 | 24/0/0 | 0.221980 |
| role:fc_gate | 8 | 0.468648 | 0.448164 | 0.422666 | 0.417227 | 0.412462 | 8/0/0 | 0.551836 |
| role:fc_up | 8 | 0.376273 | 0.389331 | 0.360251 | 0.281068 | 0.207798 | 8/0/0 | 0.610669 |
| role:k | 8 | 0.789708 | 0.778020 | 0.726582 | 0.684512 | 0.683564 | 8/0/0 | 0.221980 |
| role:o | 8 | 0.607797 | 0.554952 | 0.516985 | 0.496793 | 0.483871 | 8/0/0 | 0.445048 |
| role:proj | 8 | 0.522409 | 0.464369 | 0.401706 | 0.313650 | 0.307986 | 8/0/0 | 0.535631 |
| role:q | 8 | 0.777585 | 0.768850 | 0.694126 | 0.643466 | 0.617081 | 8/0/0 | 0.231150 |
| role:v | 8 | 0.774712 | 0.774554 | 0.726505 | 0.680934 | 0.661633 | 8/0/0 | 0.225446 |

跨 layer/shape/split/test_length 的同结构统计，以及 W-only/A-only/Both/interaction 的完整分布位于 JSON `analysis.linear_generalization`。

Cross-holdout：`28/28` 对 validation/test 同号；gain gap median `0.024414`、max `0.260071`；成对 minimum-gain median `0.595326`。最不稳定 pair 位于 JSON `worst_gap_pairs`。

## 父版本配对效果

基线：`candidate`；候选：`candidate`。

| 范围 | cases | mean Δgain | median Δgain | 改善/回归/不变 | median MSE ratio | 结论 |
|---|---:|---:|---:|---:|---:|---|
| Linear overall | 56 | -0.088775 | -0.088583 | 4/52/0 | 1.286457 | mixed |
| Attention overall | 0 | 0.000000 | 0.000000 | 0/0/0 | - | no_cases |

完整 role/family、W/A 来源、最坏 case、Attention 控制臂和 API 时间差分位于 JSON `paired_effect`。

## 误差源分解（evaluator-only）

控制臂只在评测器内重算，不增加六个候选 API 调用，也不改变主分数。gain 为相对标准输出误差的改善；interaction 为正表示超加性互补，负表示收益重叠或递减。

### Linear：W / A / 交互

| 分组 | cases | W-only gain | A-only gain | Both gain | interaction | W operand rel-MSE | A operand rel-MSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| overall | 56 | -225.503065 | -151.194574 | 0.616733 | 377.314372 | 2.241520e+00 | 1.538021e+00 |
| family:fc | 16 | -190.248969 | -153.221800 | 0.422460 | 343.893229 | 2.137869e+00 | 1.641913e+00 |
| family:o | 8 | -194.919599 | -155.666498 | 0.607797 | 351.193895 | 2.100193e+00 | 1.661824e+00 |
| family:proj | 8 | -226.278741 | -153.837462 | 0.522409 | 380.638612 | 2.140173e+00 | 1.622267e+00 |
| family:qkv | 24 | -258.941726 | -147.471486 | 0.780668 | 407.193880 | 2.391513e+00 | 1.399410e+00 |
| role:fc_gate | 8 | -235.222739 | -197.478842 | 0.468648 | 433.170229 | 2.175236e+00 | 1.657042e+00 |
| role:fc_up | 8 | -145.275199 | -108.964757 | 0.376273 | 254.616229 | 2.100503e+00 | 1.626783e+00 |
| role:k | 8 | -289.664381 | -188.542112 | 0.789708 | 478.996201 | 2.389571e+00 | 1.431389e+00 |
| role:o | 8 | -194.919599 | -155.666498 | 0.607797 | 351.193895 | 2.100193e+00 | 1.661824e+00 |
| role:proj | 8 | -226.278741 | -153.837462 | 0.522409 | 380.638612 | 2.140173e+00 | 1.622267e+00 |
| role:q | 8 | -300.425196 | -153.801534 | 0.777585 | 455.004314 | 2.398414e+00 | 1.391875e+00 |
| role:v | 8 | -186.735601 | -100.070812 | 0.774712 | 287.581125 | 2.386552e+00 | 1.374966e+00 |
| shape:hidden_to_hidden | 16 | -247.672398 | -154.734016 | 0.692691 | 403.099104 | 2.249304e+00 | 1.526849e+00 |
| shape:hidden_to_wide | 32 | -214.224480 | -148.764131 | 0.602335 | 363.590946 | 2.262966e+00 | 1.522545e+00 |
| shape:wide_to_hidden | 8 | -226.278741 | -153.837462 | 0.522409 | 380.638612 | 2.140173e+00 | 1.622267e+00 |

Linear overall interpretation: `paired_coordinate_coupling_likely`.
Linear 的完整 layer/role/window 结果位于 JSON 的 `decomposition.linear` 和 `case_scores.linear`；优先查看 role 与 shape 行，再定位对应 layer/case。

Attention 分解：已通过 `--no-decomposition` 关闭。

官方成绩只保留为独立历史字段，不参与本地 proxy 评分或时间换算。
