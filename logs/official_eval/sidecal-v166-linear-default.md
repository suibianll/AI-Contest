# candidate — proxy-v2

本报告是本地 proxy；隐藏官方数据和鲲鹏硬件不可本地复制。

- evaluation scope: `linear-only-default-panel` / `linear-only-proxy-ranking-within-identical-cache`
- proxy ranking comparable: `False`; official-score equivalent: `False`
- Linear calibration indices: `[0, 1]`; all captured lengths: `[10, 128, 512, 1024, 1024]`
- cases: `168 Linear + 0 Attention` (stratified real-W/A panel by default)
- calibration calls: `168 weight + 0 attention` (shared state)
- input codec: `e4m3-subnormal-ceil-v1` / mode `amax6`
- test splits: `['test', 'validation']`
- source SHA256: `9c0eac6a7ca883a1f8962c11735744271259460f5ebbf23d530a5bbcf12b4646`
- data pack: `{'train': 'e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7', 'validation': '204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c', 'test': '5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91'}`

| 指标 | 值 |
|---|---:|
| Linear mean | 0.636589746 |
| Attention mean | NA (not run) |
| Overall mean (all captured cases) | 0.636589746 |
| Linear role macro mean | 0.636589746 |
| Attention layer macro mean | 0.000000000 |
| Candidate wall | 305.183s |
| Candidate API total | 282.760s |

## Linear 泛化与尾部分析

均值只作位置统计；优先检查 median、worst-quartile、负 case、跨 split/长度和 W/A/interaction 分布。

| 分组 | cases | mean | median | q25 | worst-quartile mean | min | 正/负/零 | median player/std MSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 168 | 0.636590 | 0.634201 | 0.536996 | 0.445645 | -0.103061 | 167/1/0 | 0.365799 |
| family:fc | 48 | 0.541669 | 0.546655 | 0.483816 | 0.429622 | 0.383455 | 48/0/0 | 0.453345 |
| family:o | 24 | 0.515175 | 0.507084 | 0.418342 | 0.296004 | -0.103061 | 23/1/0 | 0.492916 |
| family:proj | 24 | 0.559636 | 0.563967 | 0.532184 | 0.482850 | 0.335905 | 24/0/0 | 0.436033 |
| family:qkv | 72 | 0.765993 | 0.755527 | 0.732310 | 0.697524 | 0.630414 | 72/0/0 | 0.244473 |
| role:fc_gate | 24 | 0.569164 | 0.571771 | 0.528178 | 0.454427 | 0.383455 | 24/0/0 | 0.428229 |
| role:fc_up | 24 | 0.514175 | 0.515175 | 0.455475 | 0.417953 | 0.402144 | 24/0/0 | 0.484825 |
| role:k | 24 | 0.778934 | 0.770050 | 0.745021 | 0.723281 | 0.674632 | 24/0/0 | 0.229950 |
| role:o | 24 | 0.515175 | 0.507084 | 0.418342 | 0.296004 | -0.103061 | 23/1/0 | 0.492916 |
| role:proj | 24 | 0.559636 | 0.563967 | 0.532184 | 0.482850 | 0.335905 | 24/0/0 | 0.436033 |
| role:q | 24 | 0.750613 | 0.753943 | 0.694338 | 0.668233 | 0.630414 | 24/0/0 | 0.246057 |
| role:v | 24 | 0.768432 | 0.757500 | 0.735592 | 0.717407 | 0.701725 | 24/0/0 | 0.242500 |

跨 layer/shape/split/test_length 的同结构统计，以及 W-only/A-only/Both/interaction 的完整分布位于 JSON `analysis.linear_generalization`。

## 父版本配对效果

基线：`candidate`；候选：`candidate`。

| 范围 | cases | mean Δgain | median Δgain | 改善/回归/不变 | median MSE ratio | 结论 |
|---|---:|---:|---:|---:|---:|---|
| Linear overall | 168 | 0.003064 | -0.000364 | 78/90/0 | 1.000850 | mixed |
| Attention overall | 0 | 0.000000 | 0.000000 | 0/0/0 | - | no_cases |

完整 role/family、W/A 来源、最坏 case、Attention 控制臂和 API 时间差分位于 JSON `paired_effect`。

## 误差源分解（evaluator-only）

控制臂只在评测器内重算，不增加六个候选 API 调用，也不改变主分数。gain 为相对标准输出误差的改善；interaction 为正表示超加性互补，负表示收益重叠或递减。

### Linear：W / A / 交互

| 分组 | cases | W-only gain | A-only gain | Both gain | interaction | W operand rel-MSE | A operand rel-MSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| overall | 168 | -389.235503 | -130.523685 | 0.636590 | 520.395777 | 2.193280e+00 | 1.521928e+00 |
| family:fc | 48 | -157.307781 | -137.310903 | 0.541669 | 295.160353 | 2.087025e+00 | 1.563094e+00 |
| family:o | 24 | -184.658909 | -108.594871 | 0.515175 | 293.768956 | 2.179665e+00 | 1.586470e+00 |
| family:proj | 24 | -1572.370310 | -155.795074 | 0.559636 | 1728.725019 | 1.860775e+00 | 1.640490e+00 |
| family:qkv | 72 | -217.667912 | -124.884681 | 0.765993 | 343.318586 | 2.379490e+00 | 1.433448e+00 |
| role:fc_gate | 24 | -180.912528 | -177.964680 | 0.569164 | 359.446373 | 2.162901e+00 | 1.589990e+00 |
| role:fc_up | 24 | -133.703033 | -96.657125 | 0.514175 | 230.874333 | 2.011148e+00 | 1.536198e+00 |
| role:k | 24 | -275.198766 | -166.013450 | 0.778934 | 441.991150 | 2.535427e+00 | 1.461469e+00 |
| role:o | 24 | -184.658909 | -108.594871 | 0.515175 | 293.768956 | 2.179665e+00 | 1.586470e+00 |
| role:proj | 24 | -1572.370310 | -155.795074 | 0.559636 | 1728.725019 | 1.860775e+00 | 1.640490e+00 |
| role:q | 24 | -223.506288 | -115.978560 | 0.750613 | 340.235461 | 2.338979e+00 | 1.422727e+00 |
| role:v | 24 | -154.298684 | -92.662032 | 0.768432 | 247.729147 | 2.264064e+00 | 1.416149e+00 |
| shape:hidden_to_hidden | 48 | -204.082599 | -112.286716 | 0.632894 | 317.002209 | 2.259322e+00 | 1.504599e+00 |
| shape:hidden_to_wide | 96 | -186.028253 | -133.324322 | 0.657676 | 320.010251 | 2.243385e+00 | 1.500952e+00 |
| shape:wide_to_hidden | 24 | -1572.370310 | -155.795074 | 0.559636 | 1728.725019 | 1.860775e+00 | 1.640490e+00 |

Linear overall interpretation: `paired_coordinate_coupling_likely`.
Linear 的完整 layer/role/window 结果位于 JSON 的 `decomposition.linear` 和 `case_scores.linear`；优先查看 role 与 shape 行，再定位对应 layer/case。

Attention 分解：已通过 `--no-decomposition` 关闭。

官方成绩只保留为独立历史字段，不参与本地 proxy 评分或时间换算。
