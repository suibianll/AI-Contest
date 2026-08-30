# v113 L5c operand-local meta-router — rejected/no-op at screen

## Status

`rejected screen / no-op`。本候选实现了每次 calibration call 的八维 operand-local
统计特征、两折 leave-one-fold-out 一层决策树和 v107/v109/v110 三路线 proposal；
只有跨折严格优于当前 v111 路径时才把标量 `meta_route` 写入 state，否则为 `-1`。
screen 没有产生任何总体精度增益，因此不运行 full-layer，根保持 v111。

## Features and gate

特征为 shape ratio、weight/activation RMS、两侧 kurtosis、64-block condition estimate、
部署权重 Gram 的 off-diagonal ratio 和合法 hierarchy codec gap。标签只由两折
activation-local codec MSE 产生；每个 held-out fold 复核后，最终 route 还必须在两折都不
差且累计 gain 超过门槛。标签探测不构造输出，也不使用 test/holdout/官方分数。

## Screen

- 命令：
  `python evaluator/linear_candidate_screen.py --cache artifacts/real_model_suite/cache/qwen2.5-0.5b__seq128__calib2__test4__layersall__schema1.pt --solution solution.py --layers 0 5 11 17 23 --roles q k v o fc_gate fc_up proj --output artifacts/real_model_suite/l5c-meta-router-stratified-qwen.json --report logs/execution/2026-08-31-l5c-meta-router-stratified.md`
- 35 cases，Qwen2.5-0.5B，cache read。
- Linear mean：`0.5318869456762372`。
- 当前 v111 screen：`0.5318869456762372`；差值：`0`。
- Weight-perfect mean：`0.7140714612323843`；Activation-perfect mean：
  `0.8188904985846687`。
- Elapsed：`168.982s`。
- Candidate source LF SHA256：`65e4ad45808e8a4e24bb688f369a0606786344d5470ad6d334cbad436f0b0699`。

## Decision

screen 与 v111 逐 case 相同，说明两折门禁在该固定 cache 上没有选择非默认路线；
元路由没有可验证的精度贡献。按 active L5 规则归档为 rejected/no-op，不进入
full-layer。L5d 外部实现逐组件差异审计是下一步唯一方向。

完整 screen JSON、日志、测试和源代码快照均已复制到本目录。
