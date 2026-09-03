# l3-full64-reachable — proxy-v2

本报告是本地 proxy；隐藏官方数据和鲲鹏硬件不可本地复制。

- evaluation scope: `linear-only-compact-generalization-panel` / `linear-only-low-cost-cross-holdout-mechanism-diagnosis`
- proxy ranking comparable: `False`; official-score equivalent: `False`
- Linear calibration indices: `[1, 2]`; all captured lengths: `[10, 128, 512, 1024, 1024]`
- cases: `56 Linear + 0 Attention` (stratified real-W/A panel by default)
- calibration calls: `28 weight + 0 attention` (shared state)
- input codec: `e4m3-subnormal-ceil-v1` / mode `amax6`
- test splits: `['test', 'validation']`
- source SHA256: `05dc0261000ad08c8685adda580bb5d1bbc64255b85c8c4d5569ca724dd58619`
- data pack: `{'train': 'e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7', 'validation': '204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c', 'test': '5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91'}`

| 指标 | 值 |
|---|---:|
| Linear mean | 0.687587782 |
| Attention mean | NA (not run) |
| Overall mean (all captured cases) | 0.687587782 |
| Linear role macro mean | 0.687587782 |
| Attention layer macro mean | 0.000000000 |
| Candidate wall | 55.957s |
| Candidate API total | 50.828s |

## Linear 泛化与尾部分析

均值只作位置统计；优先检查 median、worst-quartile、负 case、跨 split/长度和 W/A/interaction 分布。

| 分组 | cases | mean | median | q25 | worst-quartile mean | min | 正/负/零 | median player/std MSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 56 | 0.687588 | 0.662896 | 0.548959 | 0.517280 | 0.402634 | 56/0/0 | 0.337104 |
| family:fc | 16 | 0.542153 | 0.540010 | 0.523051 | 0.476774 | 0.402634 | 16/0/0 | 0.459990 |
| family:o | 8 | 0.631191 | 0.579239 | 0.542815 | 0.532768 | 0.523468 | 8/0/0 | 0.420761 |
| family:proj | 8 | 0.651161 | 0.608981 | 0.578198 | 0.527713 | 0.524572 | 8/0/0 | 0.391019 |
| family:qkv | 24 | 0.815485 | 0.822106 | 0.752499 | 0.725956 | 0.689030 | 24/0/0 | 0.177894 |
| role:fc_gate | 8 | 0.573492 | 0.563235 | 0.543702 | 0.531963 | 0.531301 | 8/0/0 | 0.436765 |
| role:fc_up | 8 | 0.510814 | 0.518752 | 0.502742 | 0.444307 | 0.402634 | 8/0/0 | 0.481248 |
| role:k | 8 | 0.822864 | 0.823319 | 0.768082 | 0.735328 | 0.730355 | 8/0/0 | 0.176681 |
| role:o | 8 | 0.631191 | 0.579239 | 0.542815 | 0.532768 | 0.523468 | 8/0/0 | 0.420761 |
| role:proj | 8 | 0.651161 | 0.608981 | 0.578198 | 0.527713 | 0.524572 | 8/0/0 | 0.391019 |
| role:q | 8 | 0.812521 | 0.810567 | 0.746387 | 0.707153 | 0.689030 | 8/0/0 | 0.189433 |
| role:v | 8 | 0.811071 | 0.819233 | 0.777065 | 0.735388 | 0.721051 | 8/0/0 | 0.180767 |

跨 layer/shape/split/test_length 的同结构统计，以及 W-only/A-only/Both/interaction 的完整分布位于 JSON `analysis.linear_generalization`。

Cross-holdout：`28/28` 对 validation/test 同号；gain gap median `0.016448`、max `0.161070`；成对 minimum-gain median `0.656655`。最不稳定 pair 位于 JSON `worst_gap_pairs`。

## 父版本配对效果

基线：`candidate`；候选：`l3-full64-reachable`。

| 范围 | cases | mean Δgain | median Δgain | 改善/回归/不变 | median MSE ratio | 结论 |
|---|---:|---:|---:|---:|---:|---|
| Linear overall | 56 | -0.017920 | -0.016153 | 6/42/8 | 1.052662 | mixed |
| Attention overall | 0 | 0.000000 | 0.000000 | 0/0/0 | - | no_cases |

完整 role/family、W/A 来源、最坏 case、Attention 控制臂和 API 时间差分位于 JSON `paired_effect`。

## 误差源分解（evaluator-only）

控制臂只在评测器内重算，不增加六个候选 API 调用，也不改变主分数。gain 为相对标准输出误差的改善；interaction 为正表示超加性互补，负表示收益重叠或递减。

### Linear：W / A / 交互

| 分组 | cases | W-only gain | A-only gain | Both gain | interaction | W operand rel-MSE | A operand rel-MSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| overall | 56 | -225.338024 | -151.200845 | 0.687588 | 377.226457 | 2.239392e+00 | 1.538011e+00 |
| family:fc | 16 | -189.959128 | -153.216284 | 0.542153 | 343.717565 | 2.135314e+00 | 1.641949e+00 |
| family:o | 8 | -195.302218 | -155.646247 | 0.631191 | 351.579657 | 2.100871e+00 | 1.661955e+00 |
| family:proj | 8 | -226.346384 | -153.837462 | 0.651161 | 380.835008 | 2.137051e+00 | 1.622267e+00 |
| family:qkv | 24 | -258.599769 | -147.496547 | 0.815485 | 406.911801 | 2.389064e+00 | 1.399320e+00 |
| role:fc_gate | 8 | -234.854428 | -197.462327 | 0.573492 | 432.890247 | 2.172456e+00 | 1.657028e+00 |
| role:fc_up | 8 | -145.063827 | -108.970241 | 0.510814 | 254.544882 | 2.098171e+00 | 1.626871e+00 |
| role:k | 8 | -287.898963 | -188.489693 | 0.822864 | 477.211519 | 2.386745e+00 | 1.431388e+00 |
| role:o | 8 | -195.302218 | -155.646247 | 0.631191 | 351.579657 | 2.100871e+00 | 1.661955e+00 |
| role:proj | 8 | -226.346384 | -153.837462 | 0.651161 | 380.835008 | 2.137051e+00 | 1.622267e+00 |
| role:q | 8 | -300.725733 | -153.913210 | 0.812521 | 455.451464 | 2.395535e+00 | 1.391983e+00 |
| role:v | 8 | -187.174613 | -100.086737 | 0.811071 | 288.072420 | 2.384913e+00 | 1.374589e+00 |
| shape:hidden_to_hidden | 16 | -248.013975 | -154.779729 | 0.721856 | 403.515560 | 2.248203e+00 | 1.526969e+00 |
| shape:hidden_to_wide | 32 | -213.747958 | -148.752249 | 0.679560 | 363.179767 | 2.260571e+00 | 1.522469e+00 |
| shape:wide_to_hidden | 8 | -226.346384 | -153.837462 | 0.651161 | 380.835008 | 2.137051e+00 | 1.622267e+00 |

Linear overall interpretation: `paired_coordinate_coupling_likely`.
Linear 的完整 layer/role/window 结果位于 JSON 的 `decomposition.linear` 和 `case_scores.linear`；优先查看 role 与 shape 行，再定位对应 layer/case。

Attention 分解：已通过 `--no-decomposition` 关闭。

官方成绩只保留为独立历史字段，不参与本地 proxy 评分或时间换算。

## Decision: REJECTED（2026-09-03）

目标分支真实执行：24 次 refine 共 attempted `659456` row-blocks、accepted `657540`
（99.71%），改变 `15124875` 个 code 元素。相对 parent 的 paired mean/median delta 为
`-0.017919850/-0.016153218`，`6+/42-/8=`；W-only `+0.107169` 被 A/W interaction
`-0.118818` 反转。按预注册门禁停止，不运行第二次、default、跨模型或官方评测。
