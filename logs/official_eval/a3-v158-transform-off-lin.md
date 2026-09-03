# v158-transform-off — proxy-v2

本报告是本地 proxy；隐藏官方数据和鲲鹏硬件不可本地复制。

- evaluation scope: `linear-only-default-panel` / `linear-only-proxy-ranking-within-identical-cache`
- proxy ranking comparable: `False`; official-score equivalent: `False`
- Linear calibration indices: `[0, 1]`; all captured lengths: `[10, 128, 512, 1024, 1024]`
- cases: `168 Linear + 0 Attention` (stratified real-W/A panel by default)
- calibration calls: `168 weight + 0 attention` (shared state)
- input codec: `e4m3-subnormal-ceil-v1` / mode `amax6`
- test splits: `['test', 'validation']`
- source SHA256: `831cea13036c9e2400345b82e83a4b4570c66132c577d9ae2943e0e66f77bf0b`
- data pack: `{'train': 'e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7', 'validation': '204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c', 'test': '5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91'}`

| 指标 | 值 |
|---|---:|
| Linear mean | 0.321106708 |
| Attention mean | NA (not run) |
| Overall mean (all captured cases) | 0.321106708 |
| Linear role macro mean | 0.321106708 |
| Attention layer macro mean | 0.000000000 |
| Candidate wall | 190.936s |
| Candidate API total | 169.012s |

## Linear 泛化与尾部分析

均值只作位置统计；优先检查 median、worst-quartile、负 case、跨 split/长度和 W/A/interaction 分布。

| 分组 | cases | mean | median | q25 | worst-quartile mean | min | 正/负/零 | median player/std MSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 168 | 0.321107 | 0.326395 | 0.239215 | 0.065965 | -0.867731 | 157/11/0 | 0.673605 |
| family:fc | 48 | 0.271523 | 0.267820 | 0.232076 | 0.210113 | 0.171137 | 48/0/0 | 0.732180 |
| family:o | 24 | 0.273605 | 0.265034 | 0.230538 | 0.146440 | -0.092195 | 23/1/0 | 0.734966 |
| family:proj | 24 | 0.039251 | 0.073493 | -0.110507 | -0.408351 | -0.867731 | 14/10/0 | 0.926507 |
| family:qkv | 72 | 0.463949 | 0.462604 | 0.410880 | 0.368392 | 0.333149 | 72/0/0 | 0.537396 |
| role:fc_gate | 24 | 0.299799 | 0.285169 | 0.275058 | 0.232853 | 0.171137 | 24/0/0 | 0.714831 |
| role:fc_up | 24 | 0.243246 | 0.241759 | 0.219008 | 0.210170 | 0.196686 | 24/0/0 | 0.758241 |
| role:k | 24 | 0.528485 | 0.515001 | 0.497595 | 0.474841 | 0.456333 | 24/0/0 | 0.484999 |
| role:o | 24 | 0.273605 | 0.265034 | 0.230538 | 0.146440 | -0.092195 | 23/1/0 | 0.734966 |
| role:proj | 24 | 0.039251 | 0.073493 | -0.110507 | -0.408351 | -0.867731 | 14/10/0 | 0.926507 |
| role:q | 24 | 0.398956 | 0.383960 | 0.353645 | 0.342950 | 0.333149 | 24/0/0 | 0.616040 |
| role:v | 24 | 0.464405 | 0.452512 | 0.421284 | 0.416348 | 0.411108 | 24/0/0 | 0.547488 |

跨 layer/shape/split/test_length 的同结构统计，以及 W-only/A-only/Both/interaction 的完整分布位于 JSON `analysis.linear_generalization`。

## 父版本配对效果

基线：`v158-parent`；候选：`v158-transform-off`。

| 范围 | cases | mean Δgain | median Δgain | 改善/回归/不变 | median MSE ratio | 结论 |
|---|---:|---:|---:|---:|---:|---|
| Linear overall | 168 | -0.127073 | -0.126009 | 22/146/0 | 1.235005 | mixed |
| Attention overall | 0 | 0.000000 | 0.000000 | 0/0/0 | - | no_cases |

完整 role/family、W/A 来源、最坏 case、Attention 控制臂和 API 时间差分位于 JSON `paired_effect`。

## 误差源分解（evaluator-only）

控制臂只在评测器内重算，不增加六个候选 API 调用，也不改变主分数。gain 为相对标准输出误差的改善；interaction 为正表示超加性互补，负表示收益重叠或递减。

### Linear：W / A / 交互

| 分组 | cases | W-only gain | A-only gain | Both gain | interaction | W operand rel-MSE | A operand rel-MSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| overall | 168 | 0.161989 | 0.173048 | 0.321107 | -0.013931 | 8.789374e-03 | 1.297274e-02 |
| family:fc | 48 | 0.178799 | 0.103137 | 0.271523 | -0.010413 | 7.269612e-03 | 8.421355e-03 |
| family:o | 24 | 0.128947 | 0.160377 | 0.273605 | -0.015720 | 8.751893e-03 | 9.486748e-03 |
| family:proj | 24 | -0.040378 | 0.082945 | 0.039251 | -0.003315 | 1.177006e-02 | 7.306686e-03 |
| family:qkv | 72 | 0.229253 | 0.253914 | 0.463949 | -0.019218 | 8.821482e-03 | 1.905767e-02 |
| role:fc_gate | 24 | 0.195911 | 0.117944 | 0.299799 | -0.014056 | 7.303220e-03 | 8.062733e-03 |
| role:fc_up | 24 | 0.161686 | 0.088330 | 0.243246 | -0.006769 | 7.236005e-03 | 8.779977e-03 |
| role:k | 24 | 0.239598 | 0.311593 | 0.528485 | -0.022706 | 9.045371e-03 | 1.800068e-02 |
| role:o | 24 | 0.128947 | 0.160377 | 0.273605 | -0.015720 | 8.751893e-03 | 9.486748e-03 |
| role:proj | 24 | -0.040378 | 0.082945 | 0.039251 | -0.003315 | 1.177006e-02 | 7.306686e-03 |
| role:q | 24 | 0.245573 | 0.166085 | 0.398956 | -0.012702 | 9.127762e-03 | 1.212922e-02 |
| role:v | 24 | 0.202588 | 0.284065 | 0.464405 | -0.022248 | 8.291313e-03 | 2.704313e-02 |
| shape:hidden_to_hidden | 48 | 0.187260 | 0.163231 | 0.336280 | -0.014211 | 8.939827e-03 | 1.080798e-02 |
| shape:hidden_to_wide | 96 | 0.199946 | 0.200483 | 0.383984 | -0.016445 | 7.968977e-03 | 1.547163e-02 |
| shape:wide_to_hidden | 24 | -0.040378 | 0.082945 | 0.039251 | -0.003315 | 1.177006e-02 | 7.306686e-03 |

Linear overall interpretation: `activation_dominant`.
Linear 的完整 layer/role/window 结果位于 JSON 的 `decomposition.linear` 和 `case_scores.linear`；优先查看 role 与 shape 行，再定位对应 layer/case。

Attention 分解：已通过 `--no-decomposition` 关闭。

官方成绩只保留为独立历史字段，不参与本地 proxy 评分或时间换算。
