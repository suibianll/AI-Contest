# candidate — proxy-v2

本报告是本地 proxy；隐藏官方数据和鲲鹏硬件不可本地复制。

- evaluation scope: `attention-only-compact-generalization-panel` / `attention-only-low-cost-cross-holdout-mechanism-diagnosis`
- proxy ranking comparable: `False`; official-score equivalent: `False`
- Linear calibration indices: `[1, 2]`; all captured lengths: `[10, 128, 512, 1024, 1024]`
- cases: `0 Linear + 4 Attention` (stratified real-W/A panel by default)
- calibration calls: `0 weight + 4 attention` (shared state)
- input codec: `e4m3-subnormal-ceil-v1` / mode `amax6`
- test splits: `['test', 'validation']`
- source SHA256: `268c39651b5fb1304c233db8a5ac8ff53643c760228799bc15f52fbcb0b6aa57`
- data pack: `{'train': 'e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7', 'validation': '204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c', 'test': '5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91'}`

| 指标 | 值 |
|---|---:|
| Linear mean | NA (not run) |
| Attention mean | 0.312657859 |
| Overall mean (all captured cases) | 0.312657859 |
| Linear role macro mean | 0.000000000 |
| Attention layer macro mean | 0.312657859 |
| Candidate wall | 10.533s |
| Candidate API total | 10.409s |

## 误差源分解（evaluator-only）

控制臂只在评测器内重算，不增加六个候选 API 调用，也不改变主分数。gain 为相对标准输出误差的改善；interaction 为正表示超加性互补，负表示收益重叠或递减。

Linear 分解：已通过 `--no-decomposition` 关闭。

### Attention：Q / K / V / softmax

| 分组 | cases | Q-only | K-only | V-only | QK-only | Both | QK interaction | QKV interaction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 4 | -10.124040 | -46.898581 | -0.000518 | 0.313731 | 0.312658 | 57.336352 | -0.000554 |
| layer:0 | 1 | -0.173017 | -0.387164 | -0.004158 | 0.917405 | 0.917726 | 1.477587 | 0.004480 |
| layer:15 | 1 | -22.625090 | -19.072603 | -0.000463 | -0.538243 | -0.542388 | 41.159449 | -0.003682 |
| layer:23 | 1 | -16.093347 | -166.087284 | 0.015243 | -0.043380 | -0.044753 | 182.137251 | -0.016616 |
| layer:8 | 1 | -1.604705 | -2.047275 | -0.012694 | 0.919140 | 0.920047 | 4.571121 | 0.013600 |
| length:128 | 2 | -11.399054 | -9.729884 | -0.002311 | 0.189581 | 0.187669 | 21.318518 | 0.000399 |
| length:512 | 2 | -8.849026 | -84.067279 | 0.001274 | 0.437880 | 0.437647 | 93.354186 | -0.001508 |

| Attention 中间量（overall） | standard | player |
|---|---:|---:|
| logit MSE vs reference | 2.015046e+02 | 4.391645e+04 |
| probability MSE vs reference | 5.931959e-04 | 1.571882e-04 |
| probability KL(reference || estimate) | 3.396776e-03 | 6.163068e-04 |

Attention overall interpretation: `paired_qk_coupling_likely`.
Attention 的完整 layer/length/window 结果位于 JSON 的 `decomposition.attention` 和 `case_scores.attention`。

官方成绩只保留为独立历史字段，不参与本地 proxy 评分或时间换算。
