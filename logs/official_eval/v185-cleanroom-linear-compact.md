# v185-cleanroom-linear-compact — proxy-v2

本报告是本地 proxy；隐藏官方数据和鲲鹏硬件不可本地复制。

- evaluation scope: `linear-only-compact-generalization-panel` / `linear-only-low-cost-cross-holdout-mechanism-diagnosis`
- proxy ranking comparable: `False`; official-score equivalent: `False`
- Linear calibration indices: `[1, 2]`; all captured lengths: `[10, 128, 512, 1024, 1024]`
- cases: `56 Linear + 0 Attention` (stratified real-W/A panel by default)
- calibration calls: `28 weight + 0 attention` (shared state)
- input codec: `e4m3-subnormal-ceil-v1` / mode `amax6`
- test splits: `['test', 'validation']`
- source SHA256: `3ea046594fb18dd86fd8ccfd2364a391039b0112e29986c8f949f9af526c136c`
- data pack: `{'train': 'e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7', 'validation': '204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c', 'test': '5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91'}`

| 指标 | 值 |
|---|---:|
| Linear mean | 0.417230809 |
| Attention mean | NA (not run) |
| Overall mean (all captured cases) | 0.417230809 |
| Linear role macro mean | 0.417230809 |
| Attention layer macro mean | 0.000000000 |
| Candidate wall | 9.069s |
| Candidate API total | 3.860s |

## Linear 泛化与尾部分析

均值只作位置统计；优先检查 median、worst-quartile、负 case、跨 split/长度和 W/A/interaction 分布。

| 分组 | cases | mean | median | q25 | worst-quartile mean | min | 正/负/零 | median player/std MSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 56 | 0.417231 | 0.412433 | 0.321177 | 0.220563 | 0.120373 | 56/0/0 | 0.587567 |
| family:fc | 16 | 0.375963 | 0.377783 | 0.354186 | 0.324276 | 0.311163 | 16/0/0 | 0.622217 |
| family:o | 8 | 0.261669 | 0.229261 | 0.198116 | 0.195643 | 0.193469 | 8/0/0 | 0.770739 |
| family:proj | 8 | 0.314967 | 0.218422 | 0.187657 | 0.125741 | 0.120373 | 8/0/0 | 0.781578 |
| family:qkv | 24 | 0.530684 | 0.523250 | 0.491213 | 0.452057 | 0.404355 | 24/0/0 | 0.476750 |
| role:fc_gate | 8 | 0.374326 | 0.377783 | 0.351614 | 0.315007 | 0.311163 | 8/0/0 | 0.622217 |
| role:fc_up | 8 | 0.377601 | 0.383583 | 0.354186 | 0.333545 | 0.321952 | 8/0/0 | 0.616417 |
| role:k | 8 | 0.504122 | 0.519093 | 0.486837 | 0.410516 | 0.404355 | 8/0/0 | 0.480907 |
| role:o | 8 | 0.261669 | 0.229261 | 0.198116 | 0.195643 | 0.193469 | 8/0/0 | 0.770739 |
| role:proj | 8 | 0.314967 | 0.218422 | 0.187657 | 0.125741 | 0.120373 | 8/0/0 | 0.781578 |
| role:q | 8 | 0.553643 | 0.528659 | 0.507939 | 0.466148 | 0.457998 | 8/0/0 | 0.471341 |
| role:v | 8 | 0.534288 | 0.521211 | 0.491213 | 0.479509 | 0.475994 | 8/0/0 | 0.478789 |

跨 layer/shape/split/test_length 的同结构统计，以及 W-only/A-only/Both/interaction 的完整分布位于 JSON `analysis.linear_generalization`。

Cross-holdout：`28/28` 对 validation/test 同号；gain gap median `0.011771`、max `0.056343`；成对 minimum-gain median `0.408289`。最不稳定 pair 位于 JSON `worst_gap_pairs`。

## 误差源分解（evaluator-only）

控制臂只在评测器内重算，不增加六个候选 API 调用，也不改变主分数。gain 为相对标准输出误差的改善；interaction 为正表示超加性互补，负表示收益重叠或递减。

### Linear：W / A / 交互

| 分组 | cases | W-only gain | A-only gain | Both gain | interaction | W operand rel-MSE | A operand rel-MSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| overall | 56 | -20.812054 | -7.607705 | 0.417231 | 28.836990 | 2.434975e-02 | 9.481581e-02 |
| family:fc | 16 | -17.598056 | -6.402510 | 0.375963 | 24.376529 | 1.443468e-02 | 7.872659e-02 |
| family:o | 8 | -16.786109 | -7.025061 | 0.261669 | 24.072839 | 1.962810e-02 | 4.465172e-02 |
| family:proj | 8 | -31.766863 | -9.386229 | 0.314967 | 41.468058 | 2.103662e-02 | 8.169019e-02 |
| family:qkv | 24 | -20.645099 | -8.012542 | 0.530684 | 29.188325 | 3.363807e-02 | 1.266385e-01 |
| role:fc_gate | 8 | -28.186710 | -10.614411 | 0.374326 | 39.175447 | 1.508135e-02 | 7.137846e-02 |
| role:fc_up | 8 | -7.009401 | -2.190609 | 0.377601 | 9.577611 | 1.378800e-02 | 8.607472e-02 |
| role:k | 8 | -26.836065 | -10.749884 | 0.504122 | 38.090071 | 3.504572e-02 | 1.157851e-01 |
| role:o | 8 | -16.786109 | -7.025061 | 0.261669 | 24.072839 | 1.962810e-02 | 4.465172e-02 |
| role:proj | 8 | -31.766863 | -9.386229 | 0.314967 | 41.468058 | 2.103662e-02 | 8.169019e-02 |
| role:q | 8 | -20.616367 | -8.344032 | 0.553643 | 29.514042 | 3.324351e-02 | 1.253823e-01 |
| role:v | 8 | -14.482865 | -4.943709 | 0.534288 | 19.960862 | 3.262497e-02 | 1.387482e-01 |
| shape:hidden_to_hidden | 16 | -18.701238 | -7.684547 | 0.407656 | 26.793441 | 2.643581e-02 | 8.501700e-02 |
| shape:hidden_to_wide | 32 | -19.128760 | -7.124653 | 0.447584 | 26.700998 | 2.413501e-02 | 1.029966e-01 |
| shape:wide_to_hidden | 8 | -31.766863 | -9.386229 | 0.314967 | 41.468058 | 2.103662e-02 | 8.169019e-02 |

Linear overall interpretation: `paired_coordinate_coupling_likely`.
Linear 的完整 layer/role/window 结果位于 JSON 的 `decomposition.linear` 和 `case_scores.linear`；优先查看 role 与 shape 行，再定位对应 layer/case。

Attention 分解：已通过 `--no-decomposition` 关闭。

官方成绩只保留为独立历史字段，不参与本地 proxy 评分或时间换算。
