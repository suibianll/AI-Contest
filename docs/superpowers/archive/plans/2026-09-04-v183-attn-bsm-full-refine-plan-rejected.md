# v183 计划：attention block-smooth 搜索 refine 全覆盖（2026-09-04）

> 状态：**COMPLETE/REJECTED — v183 官方 `17598/279.7s`（2026-09-04 回传），
> 与父 v182 分数持平（step_gain 0）→ 按预注册规则 REJECTED，覆盖率族关闭；
> 官方父保持 v182（17598/273s）。计划归档。**

来源：榜首差距机制假设审计（[`诊断日志`](../../../logs/execution/2026-09-04-gate-audit-and-coverage-diag.md)）。

## 假设

榜首 290s vs 我方 242-273s 的时间差可转化为校准覆盖精度。方向 1 诊断
（COV-A 变体）证明 Linear 侧 caps 全部不 binding（0/0/168 no_effect），
唯一有正余量的 cap 是 attention block-smooth 搜索的 final-quantizer
refine 覆盖（0.50→1.00：Qwen default +0.000511、校准时间中性）。

## 代码入口

v182 归档（SHA `F3E39E99...A438`）仅改 2 常量：
`_ATTN_BLOCK_SMOOTH_REFINE_RATIO 0.50→1.00`、
`_ATTN_BLOCK_SMOOTH_REFINE_BLOCKS 24_576→131_072`。
单预注册配置，不扫 ratio/blocks 邻域；Linear 与在线路径未动。

## 验收（官方）

`S(v183) > 17598` 且 `<300s` → RETAINED 新父；`≤17598` → REJECTED
（覆盖率族关闭）；`>300s` → TIMEOUT。任何正增益保留，不设晋级门槛。

## 产物

- `solutions/20260904_v183_attn-bsm-full-refine_rejected/`（solution.py + result.md）
- `artifacts/official_eval/v183-{compact,attn}-default.json`、`v183-gpt2-attn{,-default}.json`、`v183-opt-attn-default.json` + 父 baseline
- 对应 `logs/official_eval/*.md`
