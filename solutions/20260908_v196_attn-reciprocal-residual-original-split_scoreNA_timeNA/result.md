# v196 — attn-reciprocal-residual-original-split

## 状态

- 机制：按历史正向 A22-2 的原始配置移植 Q/K 互逆残差——从
  `solutions/continuous_attention_anchor22-a2/solution.py` 完整移植 `_a21_exp`、
  `_a21_exp_backward`、`_a21_project`、`_a21_scale_loss_grad`、`_a21_matrix_grad`、
  `_a22b_train`、`_a21_gate_loss`。父状态由当前根原 `hif4_calibration_attention` 产生；
  `fit = calib_qkv_list[:-1]`（4 窗训练）、`gate = calib_qkv_list[-1]`（1 窗选择），
  部署 `Q_new = Q_parent @ exp(S)`、`K_new = K_parent @ exp(-S)`、`c_new = c_parent @ exp(-S)`。
  与 v192 的唯一差异：恢复原始 4 窗训练 + 1 窗单门严格改善（v192 为 3 窗训练 + 后 2 窗双门）。
- 父 SHA256：`12352efdd4e23cc5e1e17953008664fbaa5ea5d693373635fdafc4d28ce4e24e`
- 候选 SHA256：`0889a3a5ce639d9eb44dadf197d78538105585931972ecb03b0958a0793824f9`
- 固定配置：32 步、学习率 `0.01`、梯度裁剪 `1.0`、正则 `0.001`、对称零迹 `S`、
  谱范围 `±log(2)/2`；候选数 `1`。
- 官方状态：`TIMEOUT`（用户于 2026-09-09 回传超时）。

## 检查

- `verify.py` / `check_math_and_import.py`：PASS。互逆关系 `exp(S)@exp(-S)≈I`
  （最大误差 2.52e-05）；K-center 同步编译 `c_new == c @ exp(-S)` 精确成立；
  合成 spike 数据上真实 `_a22b_train` + 真实 `_a21_gate_loss` 门真实接受
  （`a22b_attempted=1`、`a22b_accepted=1`、`a22b_fit_windows=4`，gate 仅最后窗口），
  fallback 分支正确回退父；输出有限、合法 state、脱离仓库单文件六 API 导入通过。

## 4B shard0

- eval-v3、Qwen3.5-4B proxy-v2、CUDA、Attention-only、shard 0、calibration-cache-mode write：12 cases。
- candidate mean `0.5702418426766733`，父 mean `0.5702418426766733`，paired delta `0`
  （12/12 case 全零，mse ratio 1.0），`reasonableness_issues=0`。
- calibration API：父 `6.246814s` → 候选 `8.602498s`（+37.7%）——32 步全矩阵训练真实执行，
  但 gate 在 shard0 全部 case 拒绝，最终逐位回退父状态。

## 证据位置

- 归档：`solutions/20260908_v196_attn-reciprocal-residual-original-split_scoreNA_timeNA/`
- 评测：`artifacts/proxy_v3/full_solution/attn-reciprocal-residual-original-split-shard0/candidate/manifest.json`
- 工作脚本：`workbench/full_solution/attn-reciprocal-residual-original-split/`

## 官方结果

- **`TIMEOUT`（`>300s`，无分数），不替换当前 v195 根。** 依据：
  1. 本地 shard0 最终逐位回退父状态——按 2026-09-08 AGENTS.md"本地最终回退父状态的候选不提交"；
  2. 同计算量的 v192（32 步全矩阵互逆残差）官方已 `TIMEOUT`，本候选校准 API 实测 +37.7%，
     当前根官方余量仅 11s；
  3. v192 归档已记录"恢复原始窗口划分不能消除同级计算量，不再直接提交 v196"——本次 shard0
     证实窗口划分恢复不改变本地结论（gate 仍全拒绝）。
- 该全矩阵残差实现在当前门控下无可达正向路径；按计划，不缩窗、不减步数重试。
