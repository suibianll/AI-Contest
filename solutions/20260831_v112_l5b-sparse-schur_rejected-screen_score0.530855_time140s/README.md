# v112 L5b sparse cross-block Schur proposal — rejected at screen

## Status

`rejected screen`。本候选实现了 L5b 稀疏跨 block Schur/LDLQ proposal，但五层×七
role screen 未超过当前 v111 precision parent，因此不运行 24 层 full-layer，也不替换
根 `solution.py`。官方评分不可用，screen 时间只作记录。

## Implementation

每次 calibration 从最多 1024 个通道中按归一化 block off-diagonal coupling 选择最多
两对互不重叠的 64-channel blocks。候选使用阻尼 PSD Schur block，并在选中的
128 维坐标上做一次固定 hierarchy denominator 的离散坐标下降；weight-side 以两折
交叉验证 gate，activation-side 只把校准侧 `AᵀA` 统计写入 state，运行时再用真实
部署权重 `G_q=W_qᵀW_q` 逐行选择。没有把 evaluator output、test/holdout 或模型/role
标识写入在线状态。

## Screen

- 命令：
  `python evaluator/linear_candidate_screen.py --cache artifacts/real_model_suite/cache/qwen2.5-0.5b__seq128__calib2__test4__layersall__schema1.pt --solution solution.py --layers 0 5 11 17 23 --roles q k v o fc_gate fc_up proj --output artifacts/real_model_suite/l5b-sparse-schur-stratified-qwen.json --report logs/execution/2026-08-31-l5b-sparse-schur-stratified.md`
- 35 cases，Qwen2.5-0.5B，cache read。
- Linear mean：`0.5308551015775216`。
- 当前 v111 screen：`0.5318869456762372`；差值：`-0.0010318440987156`。
- Weight-perfect mean：`0.7140353246697475`；Activation-perfect mean：
  `0.8177746053380867`。
- Elapsed：`140.300s`。
- Candidate source LF SHA256：`94a06fcce29b3e6639c4dab4d8c96e4e37f4f74947adec6e1f57b87512e0bc9`。

## Decision

按 active L5 规则，screen 必须相对当前 precision parent 有正向信号才允许 full-layer；
该候选没有达到门槛，故归档为 rejected。根主线恢复/保持 v111；L5c 统计元路由是
下一步唯一方向。

完整 screen JSON 与日志已复制到本目录；源代码快照为本候选实际运行版本。
