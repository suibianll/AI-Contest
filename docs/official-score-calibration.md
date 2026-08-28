# 本地指标到官方分数的冻结校准

## 目的与边界

`evaluator/official_score_calibration.py` 将多模型本地指标拟合为官方分数估计。它校准的是 evaluator，不是候选算法：官方分数、拟合系数和预测结果只在 `solution.py` 完成全部 HiF4 API 调用后读取，绝不传入候选 calibration state。

候选仍必须遵守第一原则：不得显式或隐式构造 `A @ W`，再利用输出、残差或拟合分数选择或反推 `Q(A)`。校准文件也不得导入、复制或硬编码到 `solution.py`。

## 当前 v0 模型

当前冻结特征是 5 个真实模型上 `linear.macro_gain` 的算术平均，使用 C21/C38/C39/C40 四个合规官方锚点做一维 OLS：

```text
predicted official score
  = 10701.888321233757
  + 9905.92863792173 × linear_macro_gain
```

诊断结果：Pearson `0.9131`、Spearman `0.8`、pairwise rank agreement `5/6`、R² `0.8338`、leave-one-anchor-out MAE `160.90`、最大留一误差 `250.84` 分。由于只有四个独立官方锚点，状态固定为 `diagnostic`；预测值和 `±2×MAE` 范围都不是官方保证或统计置信区间。

冻结文件位于 `artifacts/real_model_suite/official_score_calibration_v0.json`。文件记录训练矩阵 SHA256、候选源码 SHA256、模型 revision、WikiText revision、评测配置、每个锚点残差和留一误差。

## 1. 生成或更新校准文件

先完成固定官方锚点矩阵，再拟合：

```powershell
.\.venv\Scripts\python.exe -u evaluator\official_score_calibration.py fit `
  --input artifacts\real_model_suite\20260828_full.json `
  --output artifacts\real_model_suite\official_score_calibration_v0.json `
  --feature linear_macro_gain
```

不能用单模型结果、缺失模型的部分矩阵或不同 seq/calib/test/mode 的结果混合拟合。新增官方结果时应生成新的版本文件，例如 `official_score_calibration_v1.json`，不要覆盖 v0。

## 2. 用缓存评测当前候选

缓存必须与 v0 的 evaluator contract 完全一致：`amax6`、seq 128、2 calibration、4 test、全层和固定 5 模型面板。

```powershell
.\.venv\Scripts\python.exe -u evaluator\real_model_suite.py `
  --solution solution.py --candidate-name active `
  --cache-mode read --device cpu --algorithm-device cuda `
  --seq 128 --calib 2 --test 4 `
  --output artifacts\real_model_suite\active.json `
  --report docs\real-model-evaluator-active.md
```

`--solution` 与 `--candidates` 不能同时使用。`read` 模式不会加载 tokenizer/model、执行模型 forward 或访问网络；缓存无效时直接失败。

## 3. 预测官方分数

```powershell
.\.venv\Scripts\python.exe -u evaluator\official_score_calibration.py predict `
  --calibration artifacts\real_model_suite\official_score_calibration_v0.json `
  --input artifacts\real_model_suite\active.json `
  --output artifacts\real_model_suite\active.official-prediction.json
```

输出包含：

- `predicted_official_score`：一维 OLS 的连续预测；
- `predicted_official_score_rounded`：仅用于阅读的四舍五入值；
- `estimated_absolute_error_points`：历史留一 MAE；
- `heuristic_two_mae_range`：启发式范围，明确不是置信区间；
- `within_training_feature_range` / `extrapolation`：是否超出四个锚点的本地特征范围；
- `versus_c39`：以 C39 官方分数为固定基准的本地特征差和预测分差；
- evaluator/calibration/input SHA256，保证归档可追溯。

预测前会严格比较 mode、seq、calibration/test 数量、层数、模型列表、模型 revision、数据集/config/revision。任何字段不同都会报 `evaluator contract mismatch`，禁止静默套用旧公式。

## 使用结论

v0 可以给出本地分数到官方分数的可复现拟合，但还不能称为官方评测模拟器。实际决策时应同时查看预测分、留一误差、是否外推、5 模型逐项值以及相对 C39 的方向。积累至少 8 个独立官方锚点前，不增加回归特征；积累新锚点后按版本批量重新校准，不针对同一个候选逐次调参。
