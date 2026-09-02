# pre_a3_parent — proxy-v2

本报告是本地 proxy；隐藏官方数据和鲲鹏硬件不可本地复制。

- calibration lengths: `[10, 128, 512, 1024, 1024]`
- cases: `14 Linear + 1 Attention` (stratified real-W/A panel by default)
- calibration calls: `168 weight + 24 attention` (shared state)
- input codec: `e4m3-subnormal-ceil-v1` / mode `amax6`
- test splits: `['test', 'validation']`
- source SHA256: `800ca10ec3414e4fe886b93ca62bd4a350d26bba015287df7e8df2dd871ac23d`
- data pack: `{'train': 'e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7', 'validation': '204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c', 'test': '5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91'}`

| 指标 | 值 |
|---|---:|
| Linear mean | 0.582528216 |
| Attention mean | 0.942927486 |
| Overall mean (all captured cases) | 0.606554834 |
| Linear role macro mean | 0.582528216 |
| Attention layer macro mean | 0.942927486 |
| Candidate wall | 209.078s |
| Candidate API total | 201.258s |

## 误差源分解（evaluator-only）

控制臂只在评测器内重算，不增加六个候选 API 调用，也不改变主分数。gain 为相对标准输出误差的改善；interaction 为正表示超加性互补，负表示收益重叠或递减。

### Linear：W / A / 交互

| 分组 | cases | W-only gain | A-only gain | Both gain | interaction | W operand rel-MSE | A operand rel-MSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| overall | 14 | -290.589486 | -154.239278 | 0.582528 | 445.411292 | 1.830351e+00 | 1.445801e+00 |
| family:fc | 4 | -179.145107 | -135.537929 | 0.382643 | 315.065680 | 2.084748e+00 | 1.706705e+00 |
| family:o | 2 | -675.345497 | -42.325126 | 0.683818 | 718.354441 | 8.943118e-02 | 3.921639e-01 |
| family:proj | 2 | -168.118091 | -169.484881 | 0.392891 | 337.995863 | 1.995840e+00 | 1.986689e+00 |
| family:qkv | 6 | -277.457533 | -198.929693 | 0.745234 | 477.132461 | 2.185897e+00 | 1.442781e+00 |
| role:fc_gate | 2 | -212.963038 | -162.246675 | 0.396959 | 375.606672 | 2.119155e+00 | 1.702475e+00 |
| role:fc_up | 2 | -145.327176 | -108.829184 | 0.368327 | 254.524688 | 2.050342e+00 | 1.710935e+00 |
| role:k | 2 | -423.924045 | -251.705593 | 0.808461 | 676.438099 | 2.224570e+00 | 1.438977e+00 |
| role:o | 2 | -675.345497 | -42.325126 | 0.683818 | 718.354441 | 8.943118e-02 | 3.921639e-01 |
| role:proj | 2 | -168.118091 | -169.484881 | 0.392891 | 337.995863 | 1.995840e+00 | 1.986689e+00 |
| role:q | 2 | -248.905345 | -257.035592 | 0.670757 | 506.611694 | 2.271400e+00 | 1.468753e+00 |
| role:v | 2 | -159.543211 | -88.047895 | 0.756484 | 248.347590 | 2.061720e+00 | 1.420614e+00 |
| shape:hidden_to_hidden | 4 | -462.125421 | -149.680359 | 0.677287 | 612.483068 | 1.180416e+00 | 9.304582e-01 |
| shape:hidden_to_wide | 8 | -235.439368 | -152.707337 | 0.582558 | 388.729262 | 2.113947e+00 | 1.568250e+00 |
| shape:wide_to_hidden | 2 | -168.118091 | -169.484881 | 0.392891 | 337.995863 | 1.995840e+00 | 1.986689e+00 |

Linear overall interpretation: `paired_coordinate_coupling_likely`.
Linear 的完整 layer/role/window 结果位于 JSON 的 `decomposition.linear` 和 `case_scores.linear`；优先查看 role 与 shape 行，再定位对应 layer/case。

### Attention：Q / K / V / softmax

| 分组 | cases | Q-only | K-only | V-only | QK-only | Both | QK interaction | QKV interaction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 1 | -0.092814 | -0.006203 | 0.008294 | 0.938974 | 0.942927 | 1.037991 | -0.004340 |
| layer:0 | 1 | -0.092814 | -0.006203 | 0.008294 | 0.938974 | 0.942927 | 1.037991 | -0.004340 |
| length:10 | 1 | -0.092814 | -0.006203 | 0.008294 | 0.938974 | 0.942927 | 1.037991 | -0.004340 |

| Attention 中间量（overall） | standard | player |
|---|---:|---:|
| logit MSE vs reference | 6.684101e+02 | 1.299533e+05 |
| probability MSE vs reference | 3.018351e-02 | 1.530230e-03 |
| probability KL(reference || estimate) | 9.326769e-02 | 2.934108e-03 |

Attention overall interpretation: `paired_qk_coupling_likely`.
Attention 的完整 layer/length/window 结果位于 JSON 的 `decomposition.attention` 和 `case_scores.attention`。

官方成绩只保留为独立历史字段，不参与本地 proxy 评分或时间换算。
