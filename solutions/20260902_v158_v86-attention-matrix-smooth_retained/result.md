# v158 — exact v86 + analytic Attention Matrix-Smooth

- **Status:** `RETAINED`（官方通过，新可复现基线）
- **Parent:** v86，官方 `16744 / 222.7s`
- **唯一算法变化:** Linear 与 V 冻结；在 v86 最终 Q/K 坐标后，对每个 GQA KV 组、每个相邻
  2 通道解析求解 `S A S = B`，Q 使用 `M=sqrt(S)`，K 使用 `M^-T`。连续 QK logits 不变；
  偶数 calibration 窗拟合，奇数窗通过完整部署 attention 输出门控，无参数 sweep。
- **协议:** `proxy-v2`，Qwen2.5-0.5B，cache
  `artifacts/official_eval/cache/qwen2.5-0.5b-proxy-v2.pt`，algorithm device `cuda`。
- **Effect command:**
  `.venv\Scripts\python.exe -u evaluator\official_eval.py --solution solutions\20260902_v158_v86-attention-matrix-smooth_retained\solution.py --name v158-v86-attention-matrix-smooth --cache artifacts\official_eval\cache\qwen2.5-0.5b-proxy-v2.pt --cache-mode read --effect-panel --algorithm-device cuda --baseline-json artifacts\official_eval\v086-proxy-v2-effect.json --output artifacts\official_eval\v158-v86-attention-matrix-smooth-effect.json --report logs\official_eval\v158-v86-attention-matrix-smooth-effect.md`
- **Effect result:** Linear `0.480787684`，Attention `0.764627976`；配对 Linear
  `0/0/56`、Attention `1/0/4`，mean delta `+0.007194699`。API `293.102s`，wall
  `307.247s`；相对父版本 Attention calibration `+3.031s`，Q/K 动态合计约 `+0.0013s`。
- **Default command:** 同上去掉 `--effect-panel`，父 JSON 改为
  `artifacts/official_eval/v086-proxy-v2-panel.json`。
- **Default result:** Linear `0.448179673`，Attention `0.735752195`；配对 Linear
  `0/0/168`、Attention `49/16/55`，mean delta `+0.011017609`。API `295.069s`，wall
  `325.896s`。这是本地机制证据，不映射官方总分或官方时间。
- **合法性:** `PYTHONPATH=<repo> pytest -q` 为 `35 passed`；连续 GQA QK 点积最大误差
  `9.54e-7`；Linear calibration、Linear dynamic、V dynamic 与 v86 逐字段一致；真实 Qwen
  Attention smoke 合法；单文件可编译。
- **SHA256:** `18F9DE037A29AD96EE06FB5C73095E9AD36D0D04DA2953162181BE3AEA528277`
- **Official:** score `16861`；time `223s`；status `pass`（用户 2026-09-02 回传）。
- **Official delta vs v86:** `+117` 分，`+0.3s`。
- **Decision:** 正式晋级为仓库内最高可复现官方基线。该结果同时证明本地 default 的 mixed
  标签不能否定官方 Attention 方向。根 `solution.py` 未切换。
