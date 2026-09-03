# candidate — proxy-v2

本报告是本地 proxy；隐藏官方数据和鲲鹏硬件不可本地复制。

- evaluation scope: `attention-only-compact-generalization-panel` / `attention-only-low-cost-cross-holdout-mechanism-diagnosis`
- proxy ranking comparable: `False`; official-score equivalent: `False`
- Linear calibration indices: `[1, 2]`; all captured lengths: `[10, 128, 512, 1024, 1024]`
- cases: `0 Linear + 4 Attention` (stratified real-W/A panel by default)
- calibration calls: `0 weight + 4 attention` (shared state)
- input codec: `e4m3-subnormal-ceil-v1` / mode `amax6`
- test splits: `['test', 'validation']`
- source SHA256: `02511507ab836ed35570184351c3bf25a6090aa292b31c5bef12553fba6a3627`
- data pack: `{'train': 'e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7', 'validation': '204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c', 'test': '5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91'}`

| 指标 | 值 |
|---|---:|
| Linear mean | NA (not run) |
| Attention mean | 0.791003446 |
| Overall mean (all captured cases) | 0.791003446 |
| Linear role macro mean | 0.000000000 |
| Attention layer macro mean | 0.791003446 |
| Candidate wall | 11.077s |
| Candidate API total | 10.940s |

## 父版本配对效果

基线：`candidate`；候选：`candidate`。

| 范围 | cases | mean Δgain | median Δgain | 改善/回归/不变 | median MSE ratio | 结论 |
|---|---:|---:|---:|---:|---:|---|
| Linear overall | 0 | 0.000000 | 0.000000 | 0/0/0 | - | no_cases |
| Attention overall | 4 | -0.009194 | -0.002469 | 1/3/0 | 1.010897 | mixed |

完整 role/family、W/A 来源、最坏 case、Attention 控制臂和 API 时间差分位于 JSON `paired_effect`。

## 误差源分解（evaluator-only）

控制臂只在评测器内重算，不增加六个候选 API 调用，也不改变主分数。gain 为相对标准输出误差的改善；interaction 为正表示超加性互补，负表示收益重叠或递减。

Linear 分解：已通过 `--no-decomposition` 关闭。

### Attention：Q / K / V / softmax

| 分组 | cases | Q-only | K-only | V-only | QK-only | Both | QK interaction | QKV interaction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 4 | -10.108214 | -45.546022 | -0.000518 | 0.784256 | 0.791003 | 56.438492 | 0.007266 |
| layer:0 | 1 | -0.173134 | -0.147547 | -0.004158 | 0.915677 | 0.915764 | 1.236358 | 0.004245 |
| layer:15 | 1 | -22.429442 | -18.819745 | -0.000463 | 0.702449 | 0.711049 | 41.951637 | 0.009063 |
| layer:23 | 1 | -16.168585 | -161.255082 | 0.015243 | 0.593308 | 0.610066 | 178.016975 | 0.001515 |
| layer:8 | 1 | -1.661696 | -1.961714 | -0.012694 | 0.925588 | 0.927135 | 4.548999 | 0.014241 |
| length:128 | 2 | -11.301288 | -9.483646 | -0.002311 | 0.809063 | 0.813406 | 21.593997 | 0.006654 |
| length:512 | 2 | -8.915141 | -81.608398 | 0.001274 | 0.759448 | 0.768600 | 91.282987 | 0.007878 |

| Attention 中间量（overall） | standard | player |
|---|---:|---:|
| logit MSE vs reference | 2.015046e+02 | 4.388128e+04 |
| probability MSE vs reference | 5.931959e-04 | 8.863075e-05 |
| probability KL(reference || estimate) | 3.396776e-03 | 2.776425e-04 |

Attention overall interpretation: `paired_qk_coupling_likely`.
Attention 的完整 layer/length/window 结果位于 JSON 的 `decomposition.attention` 和 `case_scores.attention`。

官方成绩只保留为独立历史字段，不参与本地 proxy 评分或时间换算。
