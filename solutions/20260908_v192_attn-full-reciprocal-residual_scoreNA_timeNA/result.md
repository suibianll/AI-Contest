# v192 — attn-full-reciprocal-residual

## 状态

- 机制：在当前根的 Attention Q/K 坐标中训练一个共享的对称、零迹全矩阵残差，部署为
  `Q_parent@exp(S)`、`K_parent@exp(-S)`，并同步编译 K learned center。
- 父 SHA256：`12352efdd4e23cc5e1e17953008664fbaa5ea5d693373635fdafc4d28ce4e24e`
- 候选 SHA256：`dab718bc1835d391eef5212dda869d9deac3db71dc072e08ec7ab3d6c4c6ba07`
- 固定配置：32 步 Adam，学习率 `0.01`，梯度裁剪 `1.0`，正则 `0.001`，谱界
  `+/-log(2)/2`；fit windows `0,1,2`，gate windows `3,4`；候选数 `1`。
- 官方状态：`unregistered/NA`；根 `solution.py` 未替换。

## 检查

- `check_math_and_import.py`：PASS（矩阵指数互逆、谱界/零迹/对称性、六 API 公共契约、
  隔离导入）。
- `verify.py`：PASS（固定 32 步训练、互逆编译、零残差父控、强制可达、父回退、状态和输出契约）。

## 4B shard0

- eval-v3、Qwen3.5-4B proxy-v2、CUDA、Attention-only、shard 0：12 cases。
- candidate mean `0.5702418426766733`，父 mean `0.5702418426766733`，paired delta `0`，
  `reasonableness_issues=0`；因 gate 拒绝，候选最终逐位回退父状态。
- candidate API total `9.135397s`，其中 calibration API `8.732055s`；本地时间和 proxy 只作
  诊断，不换算官方分数/时间，也不构成本地晋级门。

## 校准审计

- 全矩阵训练实际执行：`a22b_attempted=1`、`a22b_accepted=0`、`a22b_arm=parent`，32 步，
  `a22b_initial_loss=2.0`，`a22b_final_loss=1.22369574`；`s_norm=9.36543655`、
  `s_max_abs=0.0770087`，Q/K 互逆误差 `0.0002223`。
- 两个 gate 均拒绝：window 3 的 parent/candidate loss 为
  `0.0003881380/0.0003995409`，window 4 为 `0.0004327810/0.0004465908`。因此保留父状态，
  证明训练分支实际可达但当前固定门控不接受该残差。

## 证据位置

- 归档：`solutions/20260908_v192_attn-full-reciprocal-residual_scoreNA_timeNA/`
- 评测：`artifacts/proxy_v3/full_solution/attn-full-reciprocal-residual-shard0/candidate/manifest.json`
- 工作脚本：`workbench/full_solution/attn-full-reciprocal-residual/`
