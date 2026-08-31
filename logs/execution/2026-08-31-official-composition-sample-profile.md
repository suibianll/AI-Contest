# sampled-means-v2 抽样构成修订

日期：2026-08-31

## 目的

历史 `sampled-means-v1` 的固定结构是 224 Linear + 32 Attention，Attention 仅占
12.5%，只能用于历史结果复现，不能用于估计官方 250 Linear + 200 Attention
面板的总时间。将构成匹配方案合并为唯一活动 profile `sampled-means-v2`，让本地
实际执行的 Linear/Attention case 比例接近官方比例，同时不复制 case；均值和时间
从此使用同一批样本。

## 当前缓存下的计划

配置：Qwen2.5-0.5B，`seq=128`，`calib=2`，`test=4`，seed `20260831`。

| 项目 | 结果 |
|---|---:|
| Linear layer | 4 个分层 layer |
| Attention layer | 全部 24 个 layer |
| Linear case | 112 |
| Attention case | 96 |
| Linear/Attention | 1.1667:1 |
| Attention 占比 | 46.15% |
| 官方 Attention 占比 | 44.44% |

由于当前缓存只有 4 个 test window，不能在不复制 case 的前提下得到精确的
250/200；112/96 是保留全部 Linear role 且接近官方构成的可复现计划。

`sampled-means-v2` 对所需 layer 执行 calibration，并在 JSON 的
`sample_plan` 中记录 component-specific layer/window、实际比例和 calibration
layer。`timing.api_seconds` 用于 Linear/Attention 时间拆分。

## 代码与验证

- `evaluator/real_model_suite.py`：新增 profile，并让评测器支持不同组件的 layer/window
  选择；兼容旧 JSON 的 `layer_indices`/`test_indices` 字段。
- `README.md`、`README_EN.md`、`docs/current-solution-status.md` 和唯一 active plan
  已说明两个 profile 的边界。
- `.venv\\Scripts\\python.exe -m pytest -q tests/test_real_model_suite.py`：16 passed。
- 实际 cache 计划验证：历史 `sampled-means-v1=224/32`，活动
  `sampled-means-v2=112/96`。

## 使用边界

`sampled-means-v2` 同时用于精度均值和时间判断；历史 `sampled-means-v1` 仅作旧
结果复现，并且仍需单独覆盖官方变长 Attention 校准
`[10,128,512,1024,1024]`。固定 `seq=128` 的 composition profile 只能修正
case 构成，不能消除 PAWV 长序列的 (L^2) 低估。
