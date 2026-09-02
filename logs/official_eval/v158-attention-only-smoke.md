# v158-attention-only-smoke — proxy-v2

本报告是本地 proxy；隐藏官方数据和鲲鹏硬件不可本地复制。

- evaluation scope: `attention-only-effect-panel` / `attention-only-mechanism-diagnosis`
- proxy ranking comparable: `False`; official-score equivalent: `False`
- calibration lengths: `[10, 128, 512, 1024, 1024]`
- cases: `0 Linear + 5 Attention` (stratified real-W/A panel by default)
- calibration calls: `0 weight + 24 attention` (shared state)
- input codec: `e4m3-subnormal-ceil-v1` / mode `amax6`
- test splits: `['test', 'validation']`
- source SHA256: `18f9de037a29ad96ee06fb5c73095e9ad36d0d04da2953162181be3aea528277`
- data pack: `{'train': 'e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7', 'validation': '204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c', 'test': '5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91'}`

| 指标 | 值 |
|---|---:|
| Linear mean | 0.000000000 |
| Attention mean | 0.764627976 |
| Overall mean (all captured cases) | 0.764627976 |
| Linear role macro mean | 0.000000000 |
| Attention layer macro mean | 0.764627976 |
| Candidate wall | 67.465s |
| Candidate API total | 67.176s |

## 误差源分解（evaluator-only）

控制臂只在评测器内重算，不增加六个候选 API 调用，也不改变主分数。gain 为相对标准输出误差的改善；interaction 为正表示超加性互补，负表示收益重叠或递减。

Linear 分解：已通过 `--no-decomposition` 关闭。

Attention 分解：已通过 `--no-decomposition` 关闭。

官方成绩只保留为独立历史字段，不参与本地 proxy 评分或时间换算。
