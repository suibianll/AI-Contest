# candidate — proxy-v2

本报告是本地 proxy；隐藏官方数据和鲲鹏硬件不可本地复制。

- evaluation scope: `attention-only-compact-generalization-panel` / `attention-only-low-cost-cross-holdout-mechanism-diagnosis`
- proxy ranking comparable: `False`; official-score equivalent: `False`
- Linear calibration indices: `[1, 2]`; all captured lengths: `[10, 128, 512, 1024, 1024]`
- cases: `0 Linear + 4 Attention` (stratified real-W/A panel by default)
- calibration calls: `0 weight + 4 attention` (shared state)
- input codec: `e4m3-subnormal-ceil-v1` / mode `amax6`
- test splits: `['test', 'validation']`
- source SHA256: `2cf06b0a5eaff8fd9ae8543809282934dc7460a713a359130d1fe2dd370bbbdb`
- data pack: `{'train': 'e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7', 'validation': '204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c', 'test': '5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91'}`

| 指标 | 值 |
|---|---:|
| Linear mean | NA (not run) |
| Attention mean | 0.779773134 |
| Overall mean (all captured cases) | 0.779773134 |
| Linear role macro mean | 0.000000000 |
| Attention layer macro mean | 0.779773134 |
| Candidate wall | 11.660s |
| Candidate API total | 11.530s |

## 父版本配对效果

基线：`candidate`；候选：`candidate`。

| 范围 | cases | mean Δgain | median Δgain | 改善/回归/不变 | median MSE ratio | 结论 |
|---|---:|---:|---:|---:|---:|---|
| Linear overall | 0 | 0.000000 | 0.000000 | 0/0/0 | - | no_cases |
| Attention overall | 4 | -0.017980 | -0.006662 | 1/3/0 | 1.086817 | mixed |

完整 role/family、W/A 来源、最坏 case、Attention 控制臂和 API 时间差分位于 JSON `paired_effect`。

## 误差源分解（evaluator-only）

控制臂只在评测器内重算，不增加六个候选 API 调用，也不改变主分数。gain 为相对标准输出误差的改善；interaction 为正表示超加性互补，负表示收益重叠或递减。

Linear 分解：已通过 `--no-decomposition` 关闭。

### Attention：Q / K / V / softmax

| 分组 | cases | Q-only | K-only | V-only | QK-only | Both | QK interaction | QKV interaction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 4 | -10.098095 | -46.051997 | 0.003967 | 0.780445 | 0.779773 | 56.930537 | -0.004640 |
| layer:0 | 1 | -0.170853 | -0.181484 | -0.000120 | 0.910152 | 0.910179 | 1.262489 | 0.000147 |
| layer:15 | 1 | -22.532269 | -18.914606 | 0.000000 | 0.742996 | 0.742996 | 42.189872 | 0.000000 |
| layer:23 | 1 | -16.089898 | -163.150830 | 0.015986 | 0.542983 | 0.540267 | 179.783711 | -0.018702 |
| layer:8 | 1 | -1.599359 | -1.961068 | 0.000004 | 0.925651 | 0.925650 | 4.486078 | -0.000005 |
| length:128 | 2 | -11.351561 | -9.548045 | -0.000060 | 0.826574 | 0.826588 | 21.726180 | 0.000074 |
| length:512 | 2 | -8.844628 | -82.555949 | 0.007995 | 0.734317 | 0.732959 | 92.134894 | -0.009353 |

| Attention 中间量（overall） | standard | player |
|---|---:|---:|
| logit MSE vs reference | 2.015046e+02 | 4.386431e+04 |
| probability MSE vs reference | 5.931959e-04 | 6.582355e-05 |
| probability KL(reference || estimate) | 3.396776e-03 | 2.087515e-04 |

Attention overall interpretation: `paired_qk_coupling_likely`.
Attention 的完整 layer/length/window 结果位于 JSON 的 `decomposition.attention` 和 `case_scores.attention`。

官方成绩只保留为独立历史字段，不参与本地 proxy 评分或时间换算。
