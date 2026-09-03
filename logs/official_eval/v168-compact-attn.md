# candidate — proxy-v2

本报告是本地 proxy；隐藏官方数据和鲲鹏硬件不可本地复制。

- evaluation scope: `attention-only-compact-generalization-panel` / `attention-only-low-cost-cross-holdout-mechanism-diagnosis`
- proxy ranking comparable: `False`; official-score equivalent: `False`
- Linear calibration indices: `[1, 2]`; all captured lengths: `[10, 128, 512, 1024, 1024]`
- cases: `0 Linear + 4 Attention` (stratified real-W/A panel by default)
- calibration calls: `0 weight + 4 attention` (shared state)
- input codec: `e4m3-subnormal-ceil-v1` / mode `amax6`
- test splits: `['test', 'validation']`
- source SHA256: `5988ae47eac2e7dde7488e06b8f91939f5660a585034280a6d68a8fb6701ac79`
- data pack: `{'train': 'e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7', 'validation': '204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c', 'test': '5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91'}`

| 指标 | 值 |
|---|---:|
| Linear mean | NA (not run) |
| Attention mean | 0.797753303 |
| Overall mean (all captured cases) | 0.797753303 |
| Linear role macro mean | 0.000000000 |
| Attention layer macro mean | 0.797753303 |
| Candidate wall | 11.690s |
| Candidate API total | 11.560s |

## 父版本配对效果

基线：`candidate`；候选：`candidate`。

| 范围 | cases | mean Δgain | median Δgain | 改善/回归/不变 | median MSE ratio | 结论 |
|---|---:|---:|---:|---:|---:|---|
| Linear overall | 0 | 0.000000 | 0.000000 | 0/0/0 | - | no_cases |
| Attention overall | 4 | 0.000291 | 0.000435 | 2/2/0 | 1.000431 | mixed |

完整 role/family、W/A 来源、最坏 case、Attention 控制臂和 API 时间差分位于 JSON `paired_effect`。

## 误差源分解（evaluator-only）

控制臂只在评测器内重算，不增加六个候选 API 调用，也不改变主分数。gain 为相对标准输出误差的改善；interaction 为正表示超加性互补，负表示收益重叠或递减。

Linear 分解：已通过 `--no-decomposition` 关闭。

### Attention：Q / K / V / softmax

| 分组 | cases | Q-only | K-only | V-only | QK-only | Both | QK interaction | QKV interaction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 4 | -10.110204 | -46.279949 | -0.000518 | 0.791485 | 0.797753 | 57.181638 | 0.006787 |
| layer:0 | 1 | -0.178789 | -0.142388 | -0.004158 | 0.922874 | 0.923192 | 1.244051 | 0.004475 |
| layer:15 | 1 | -22.568976 | -19.011437 | -0.000463 | 0.730088 | 0.734099 | 42.310501 | 0.004475 |
| layer:23 | 1 | -16.088134 | -163.981698 | 0.015243 | 0.589118 | 0.607760 | 180.658950 | 0.003400 |
| layer:8 | 1 | -1.604919 | -1.984274 | -0.012694 | 0.923858 | 0.925962 | 4.513051 | 0.014798 |
| length:128 | 2 | -11.373882 | -9.576912 | -0.002311 | 0.826481 | 0.828646 | 21.777276 | 0.004475 |
| length:512 | 2 | -8.846527 | -82.982986 | 0.001274 | 0.756488 | 0.766861 | 92.586001 | 0.009099 |

| Attention 中间量（overall） | standard | player |
|---|---:|---:|
| logit MSE vs reference | 2.015046e+02 | 4.391102e+04 |
| probability MSE vs reference | 5.931959e-04 | 6.448907e-05 |
| probability KL(reference || estimate) | 3.396776e-03 | 2.032933e-04 |

Attention overall interpretation: `paired_qk_coupling_likely`.
Attention 的完整 layer/length/window 结果位于 JSON 的 `decomposition.attention` 和 `case_scores.attention`。

官方成绩只保留为独立历史字段，不参与本地 proxy 评分或时间换算。
