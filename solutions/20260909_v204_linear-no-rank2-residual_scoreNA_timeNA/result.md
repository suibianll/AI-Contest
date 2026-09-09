# v204 — linear-no-rank2-residual（定价用 ablation）

## 状态

- 机制：减法诊断——关闭 L-R2 rank-2 残差段（`_WEIGHT_RESIDUAL_RANK` 2→1，单行开关，
  `solution.py:78`），保留 v166 rank-1 残差与其余全部组件。目的：官方实测 rank-2 段的
  时间成本（其官方分数贡献已知仅 +1；本地审计占 Linear 残差块主成本，rank-2 ≈ rank-1
  的 1.5–2.7×）。
- 父（当前根 v202+v195）SHA256：`56dc805d6e5a3aef896db8021045740292735725d688b48e3d4393e55efcb2bd`
- 候选 SHA256：`11d7bbcd049092e75aa26336c46eeb995c1a9c9d7fcbb05feed63f3d8e041ca2`
- 官方状态：`unregistered/NA`。

## 检查

- `verify.py` 全 PASS：候选无 `[L-R2]` 输出、state 残差对形状 [1024,2]→[1024,1]（rank-2
  分支零调用、rank-1 保留），非 no-op；合法 state、有限输出、脱离仓库六 API 导入通过。

## 4B shard0

- eval-v3、Qwen3.5-4B proxy-v2、CUDA、Linear-only、shard 0、calibration-cache-mode auto：56 cases。
- candidate mean `0.5174823055`，父 mean `0.5173485923`；paired delta mean `+0.0001337`、
  median `+0.0002206`、min −0.0048549、max +0.0040042，+/-/0 = `30/26/0`；
  `reasonableness_issues=0`。本地看 rank-2 段净贡献≈0（逐 case 有正有负）。
- 时序：baseline 命中校准缓存（auto），candidate 重建，api_total 不可比（本次仅接口检查）。

## 证据位置

- 归档：`solutions/20260909_v204_linear-no-rank2-residual_scoreNA_timeNA/`
- 评测：`artifacts/proxy_v3/full_solution/linear-no-rank2-residual-shard0/candidate/manifest.json`
- 工作脚本：`workbench/full_solution/linear-no-rank2-residual/`

## 官方结果

- `unregistered/NA`。提交目的：官方实测"砍 rank-2"的时间节省与分数损失，
  为是否用该余量回收 v192 类机制或 Linear A@W v2 提供定价。
