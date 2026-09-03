# candidate — proxy-v2

本报告是本地 proxy；隐藏官方数据和鲲鹏硬件不可本地复制。

- evaluation scope: `attention-only-compact-generalization-panel` / `attention-only-low-cost-cross-holdout-mechanism-diagnosis`
- proxy ranking comparable: `False`; official-score equivalent: `False`
- Linear calibration indices: `[1, 2]`; all captured lengths: `[10, 128, 512, 1024, 1024]`
- cases: `0 Linear + 4 Attention` (stratified real-W/A panel by default)
- calibration calls: `0 weight + 4 attention` (shared state)
- input codec: `e4m3-subnormal-ceil-v1` / mode `amax6`
- test splits: `['test', 'validation']`
- source SHA256: `70767046c70d483ab8b3b82901611c928efad3460e6a385d5b8fa83be0274d22`
- data pack: `{'train': 'e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7', 'validation': '204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c', 'test': '5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91'}`

| 指标 | 值 |
|---|---:|
| Linear mean | NA (not run) |
| Attention mean | 0.221950906 |
| Overall mean (all captured cases) | 0.221950906 |
| Linear role macro mean | 0.000000000 |
| Attention layer macro mean | 0.221950906 |
| Candidate wall | 11.121s |
| Candidate API total | 11.004s |

## 误差源分解（evaluator-only）

控制臂只在评测器内重算，不增加六个候选 API 调用，也不改变主分数。gain 为相对标准输出误差的改善；interaction 为正表示超加性互补，负表示收益重叠或递减。

Linear 分解：已通过 `--no-decomposition` 关闭。

### Attention：Q / K / V / softmax

| 分组 | cases | Q-only | K-only | V-only | QK-only | Both | QK interaction | QKV interaction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 4 | -10.124101 | -47.091737 | -0.000518 | 0.221372 | 0.221951 | 57.437210 | 0.001097 |
| layer:0 | 1 | -0.168993 | -0.398860 | -0.004158 | 0.912014 | 0.912630 | 1.479867 | 0.004773 |
| layer:15 | 1 | -22.619221 | -18.836918 | -0.000463 | -0.779509 | -0.787880 | 40.676631 | -0.007908 |
| layer:23 | 1 | -16.100218 | -167.091544 | 0.015243 | -0.159541 | -0.150072 | 183.032221 | -0.005774 |
| layer:8 | 1 | -1.607971 | -2.039626 | -0.012694 | 0.912524 | 0.913126 | 4.560122 | 0.013296 |
| length:128 | 2 | -11.394107 | -9.617889 | -0.002311 | 0.066253 | 0.062375 | 21.078249 | -0.001567 |
| length:512 | 2 | -8.854094 | -84.565585 | 0.001274 | 0.376492 | 0.381527 | 93.796171 | 0.003761 |

| Attention 中间量（overall） | standard | player |
|---|---:|---:|
| logit MSE vs reference | 2.015046e+02 | 4.391698e+04 |
| probability MSE vs reference | 5.931959e-04 | 1.894435e-04 |
| probability KL(reference || estimate) | 3.396776e-03 | 7.249643e-04 |

Attention overall interpretation: `paired_qk_coupling_likely`.
Attention 的完整 layer/length/window 结果位于 JSON 的 `decomposition.attention` 和 `case_scores.attention`。

官方成绩只保留为独立历史字段，不参与本地 proxy 评分或时间换算。
