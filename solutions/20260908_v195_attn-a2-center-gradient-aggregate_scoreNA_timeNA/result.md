# v195 — attn-a2-center-gradient-aggregate

## 状态

- 机制：修复 `_a2_train_rotation` 的 K-center 梯度聚合 bug。根实现中 `grad_theta` 对全部
  训练窗口累加，但 `grad_center = dk3.sum(dim=0)` 在窗口循环内反复覆盖，center 的 Adam
  更新实际只用了最后一个训练窗口的梯度。修复为 step 开始时 `grad_center = zeros_like(center)`、
  循环内累加（仅两行差异），其余训练代码（32 步、学习率 0.01、Cayley rotation、最终输出
  MSE、最后窗口选择）完全不变。
- 父 SHA256：`12352efdd4e23cc5e1e17953008664fbaa5ea5d693373635fdafc4d28ce4e24e`
- 候选 SHA256：`839adb1e617c3115c6b55071a34b281c5db0ff2aa070adbbc71fd1549e761d7f`
- 固定配置：同根，无新增超参。
- 官方状态：`RETAINED`；用户于 2026-09-09 回传 `18053 / 289s`。

## 检查

- `verify.py`：PASS。spy 包装记录逐窗口 `dk3.sum(dim=0)` 后独立 Adam 回放：候选 center 与
  "全部窗口梯度求和"回放逐元素吻合（atol 1e-5），父与"仅最后窗口"回放吻合；输出有限、
  rotation 正交误差 3.6e-7；完整小合成校准下候选与父的 train/gate loss 不同（修复生效）；
  合法 state、动态输出契约、脱离仓库单文件六 API 导入均通过。

## 4B shard0

- eval-v3、Qwen3.5-4B proxy-v2、CUDA、Attention-only、shard 0、calibration-cache-mode write：12 cases。
- candidate mean `0.5721770801604279`，父 mean `0.5702418426766733`；
  paired delta mean `+0.0019352375`、median `+0.0011072358`、min `−0.0127937746`、
  max `+0.0230149876`，+/-/0 = `6/6/0`，L1 `0.007082`；`reasonableness_issues=0`。
  候选非 no-op、无灾难性错误；本地小幅正负不用于挑参数，交官方裁决。
- API total：父 `6.741033s` → 候选 `5.283496s`（相邻运行噪声水平，只作记录；
  修复本身每 step 每窗口仅多一次小向量加法）。

## 证据位置

- 归档：`solutions/20260908_v195_attn-a2-center-gradient-aggregate_scoreNA_timeNA/`
- 评测：`artifacts/proxy_v3/full_solution/attn-a2-center-gradient-aggregate-shard0/candidate/manifest.json`
- 工作脚本：`workbench/full_solution/attn-a2-center-gradient-aggregate/`

## 官方结果

- **`18053 / 289s`，RETAINED。** 相对提交时当前根 `18032 / 280s`，分数 `+21`、时间
  `+9s`，仍低于官方 `300s` 硬限。该候选替换当前根；标准 Linear 诊断版仍是独立归因候选，
  不与本完整官方结果混淆。
