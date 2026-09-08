# Attention mask-aligned output selector 执行记录

日期：2026-09-06
计划：[`Attention mask-aligned output selector`](../../docs/superpowers/archive/plans/2026-09-06-attention-noncausal-selector-plan.md)
父：v189 `static-actorder-hdiag-recovered`，根 SHA256
`261202248a0146a2ee45f3df60bd1979bb8171b7c162921013b0024c848617af`

## 裁决

候选只把 A1 校准输出误差选择器和终验门的 causal/non-causal 主次对调；候选生成、
状态、在线六 API、Linear 路径和 V 路径均未改。R0、R1、R2 的接口、有限性、case 唯一性
和 OOD 门通过，但 fresh default 的 Overall 低于当前本地最高，且时间模型超门，因此
候选 **REJECTED**，不分配 v190，不生成 `solution.zip`，不提交官方；根目录保持 v189。

候选源码：`workbench/attention_noncausal_selector_solution.py`
候选 SHA256：`de1e07c2515062298a575f5e2f6a1748a60bfab948fae67f4b27d32bc97fd1fb`

## R0：接口与实现边界

- `py_compile` 和六个正式 API 导入通过。
- 只覆盖 A1 的 primary/safety 选择角色；未增加候选数量、动态算子或 state 字段语义。

## R1：eval-v3 Attention 前两片

固定 `proxy-v2` cache、CUDA、v189 baseline，命令口径为
`evaluator/eval.py --attention-only --shards 0,1`。

| shard | delta mean | median delta | L1 | 正/负/零 |
|---:|---:|---:|---:|---:|
| 0 | `+0.013737235` | `+0.000141079` | `0.013737235` | `4/0/4` |
| 1 | `+0.013085379` | `0` | `0.013085379` | `2/0/6` |

两片输出均有限、case identity 唯一、expected coverage 通过，未发现未修改 control
变化或接口错误。

## R2：eval-v3 六片与 OOD

六片 Attention 合计 48 cases：candidate mean `0.7547572534031689`，parent mean
`0.7527723548408157`，delta `+0.0019848985623532`。这是 eval-v3 的分片集成诊断，
不等价于兼容后端的 120-case default-panel。

| shard | delta mean | median delta | L1 | 正/负/零 | 最小 delta |
|---:|---:|---:|---:|---:|---:|
| 0 | `+0.013737235` | `+0.000141079` | `0.013737235` | `4/0/4` | — |
| 1 | `+0.013085379` | `0` | `0.013085379` | `2/0/6` | — |
| 2 | `-0.004704340` | `0` | `0.011387434` | `3/3/2` | `-0.030244621` |
| 3 | `0` | `0` | `0` | `0/0/8` | `0` |
| 4 | `-0.007772224` | `0` | `0.011079147` | `1/1/6` | `-0.075405484` |
| 5 | `-0.002436659` | `0` | `0.010601326` | `3/1/4` | `-0.052151936` |

OOD 六片 candidate mean `0.751437947313`、parent mean `0.751857367979`，
in-dist delta `+0.001984898562`、OOD delta `-0.000419420666`，所以
`Δ(gain_in-gain_ood)=+0.002404319228`，满足 `|Δgap|<=0.01`。OOD 输出有限、case
唯一、coverage 通过；该结果仅作拟合诊断。

## Fresh default gate

使用兼容后端 `evaluator/official_eval.py` 的同一 `proxy-v2` default-panel（168 Linear
`+` 120 Attention，fresh calibration，输入 NVFP4 cache 命中）。该 scope 是本计划的
default gate，不能与上面的 48-case eval-v3 分片结果混排。

| 侧 | candidate | v189 parent | delta |
|---|---:|---:|---:|
| Linear | `0.640258324430` | `0.640258324430` | `0` |
| Attention | `0.748924596433` | `0.752173407020` | `-0.003248810587` |
| Overall | `0.685535937765` | `0.686889608842` | `-0.001353671078` |

当前本地最高完整 default 为 `0.687776303`，候选相差 `-0.002240365235`，未满足严格
超越条件。

Fresh default API 分解：

- `W_calib=274.630146s`
- `A_calib=60.748621s`
- `dyn_act=60.692522s`
- `dyn_qkv=3.064692s`
- 时间模型预测 `283.748107s`，未满足 `<280s`
- API total `399.135981s`，wall `423.829521s`；均为本地参考测量，不是官方时间。

证据：

- eval-v3 ID：`artifacts/proxy_v3/attention-noncausal-selector-20260906/`
- default JSON：`artifacts/official_eval/attention-noncausal-selector-fresh-default-r1.json`
- default report：`logs/official_eval/attention-noncausal-selector-fresh-default-r1.md`

## 后续边界

该固定 selector 机制已关闭；不扫描 causal/non-causal 混合权重、case 选择、gate
阈值、fold、seed 或其它邻域。根 `solution.py` 保持 v189 官方源码，SHA 与 v189 归档一致。
