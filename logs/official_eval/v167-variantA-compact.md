# candidate — proxy-v2

本报告是本地 proxy；隐藏官方数据和鲲鹏硬件不可本地复制。

- evaluation scope: `attention-only-compact-generalization-panel` / `attention-only-low-cost-cross-holdout-mechanism-diagnosis`
- proxy ranking comparable: `False`; official-score equivalent: `False`
- Linear calibration indices: `[1, 2]`; all captured lengths: `[10, 128, 512, 1024, 1024]`
- cases: `0 Linear + 4 Attention` (stratified real-W/A panel by default)
- calibration calls: `0 weight + 4 attention` (shared state)
- input codec: `e4m3-subnormal-ceil-v1` / mode `amax6`
- test splits: `['test', 'validation']`
- source SHA256: `8360ed80397a854bdef9a066145da3698fd6a971163b5a0176fd3d228c1078c4`
- data pack: `{'train': 'e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7', 'validation': '204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c', 'test': '5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91'}`

| 指标 | 值 |
|---|---:|
| Linear mean | NA (not run) |
| Attention mean | 0.296228828 |
| Overall mean (all captured cases) | 0.296228828 |
| Linear role macro mean | 0.000000000 |
| Attention layer macro mean | 0.296228828 |
| Candidate wall | 9.916s |
| Candidate API total | 9.791s |

## 误差源分解（evaluator-only）

控制臂只在评测器内重算，不增加六个候选 API 调用，也不改变主分数。gain 为相对标准输出误差的改善；interaction 为正表示超加性互补，负表示收益重叠或递减。

Linear 分解：已通过 `--no-decomposition` 关闭。

### Attention：Q / K / V / softmax

| 分组 | cases | Q-only | K-only | V-only | QK-only | Both | QK interaction | QKV interaction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 4 | -10.125910 | -46.896448 | -0.000518 | 0.296523 | 0.296229 | 57.318882 | 0.000224 |
| layer:0 | 1 | -0.165764 | -0.394100 | -0.004158 | 0.915635 | 0.915995 | 1.475500 | 0.004517 |
| layer:15 | 1 | -22.633795 | -18.984134 | -0.000463 | -0.593918 | -0.600532 | 41.024011 | -0.006150 |
| layer:23 | 1 | -16.100771 | -166.161886 | 0.015243 | -0.052914 | -0.049033 | 182.209742 | -0.011361 |
| layer:8 | 1 | -1.603310 | -2.045674 | -0.012694 | 0.917290 | 0.918485 | 4.566273 | 0.013890 |
| length:128 | 2 | -11.399780 | -9.689117 | -0.002311 | 0.160859 | 0.157731 | 21.249756 | -0.000817 |
| length:512 | 2 | -8.852040 | -84.103780 | 0.001274 | 0.432188 | 0.434726 | 93.388008 | 0.001264 |

| Attention 中间量（overall） | standard | player |
|---|---:|---:|
| logit MSE vs reference | 2.015046e+02 | 4.391883e+04 |
| probability MSE vs reference | 5.931959e-04 | 1.623694e-04 |
| probability KL(reference || estimate) | 3.396776e-03 | 6.404324e-04 |

Attention overall interpretation: `paired_qk_coupling_likely`.
Attention 的完整 layer/length/window 结果位于 JSON 的 `decomposition.attention` 和 `case_scores.attention`。

官方成绩只保留为独立历史字段，不参与本地 proxy 评分或时间换算。
