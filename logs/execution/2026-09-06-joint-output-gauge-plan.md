# 联合输出坐标规范化计划执行记录

日期：2026-09-06。计划：
[`联合输出坐标规范化诊断计划`](../../docs/superpowers/archive/plans/2026-09-06-joint-output-gauge-plan-rejected.md)。

## 执行范围

J0 只运行一个预注册的 paired diagonal group-gauge oracle，不改根 `solution.py`：

- Linear 固定层 `0/8/15/23`，7 个 role（重点 `fc_gate/fc_up/proj`，其余作 control）；
  每窗采样 32 token、32 个权重行、4 个 64-block；每个 block 切为 8 个连续 8-channel
  group。
- Attention 固定 Q head 0 与最后 KV head，每层每窗采样 32 token；Q/K 各使用同一组
  8-channel paired gauge。
- gauge 集合固定为 `{2^-1/4, 1, 2^1/4}`。Linear 使用 `A_g=A G^-1, W_g=W G`；
  Attention 使用 `Q_g=QG, K_g=KG^-T`。
- 每个 fold 只用其余 4 个 calibration window 选择 group gauge，再在 holdout window
  上用 `evaluator/reference_hif4.py` 的合法 exact solver 计算 output MSE；不读取 test。

固定输入与配置：

- 父源码：`solution.py` v186，SHA256
  `f8495dca20334acbdad16fc18ee41a4970f31e1837fdeedcee9c70aee54e7eb8`。
- cache：`artifacts/official_eval/cache/qwen2.5-0.5b-proxy-v2.pt`，SHA256
  `ef278e9dbbf72670d54a502648b055833e081d9ea8097b2ba5b9269eb6f2824b`。
- 工具：`workbench/joint_output_gauge_probe.py`，SHA256
  `7e2c377f81e0e55e0db341e239ef510a0f243fa55e7671eddb998fd753341487`。
- 配置：`workbench/joint_output_gauge_probe_config.json`；结果目录：
  `artifacts/proxy_v3/joint-output-gauge-20260906/j0/`。

## 结果与固定门

运行完成状态为 `REJECTED`，gate 为 `NO_SUPPORTED_JOINT_OUTPUT_GAUGE_ORACLE`，耗时
`323.9033 s`。结果和 manifest：

- [`result.json`](../../artifacts/proxy_v3/joint-output-gauge-20260906/j0/result.json)
- [`report.md`](../../artifacts/proxy_v3/joint-output-gauge-20260906/j0/report.md)
- [`run_manifest.json`](../../artifacts/proxy_v3/joint-output-gauge-20260906/j0/run_manifest.json)

Linear 28 条记录中，重点层的 fold-median output gain 为：

| layer | focus median | 是否超过 1% |
|---:|---:|:---:|
| 0 | `0` | 否 |
| 8 | `0` | 否 |
| 15 | `0` | 否 |
| 23 | `0` | 否 |

重点正层 `0/4`，control 中位 gain `0`；Linear J0 门不通过。

Attention 4 层的 joint fold-median output/logit/probability gain 为：

| layer | output | logits | probability |
|---:|---:|---:|---:|
| 0 | `-0.269746` | `+0.000718` | `-0.387712` |
| 8 | `0` | `0` | `0` |
| 15 | `0` | `0` | `0` |
| 23 | `-0.079147` | `-0.000676` | `-0.106814` |

joint 正层 `0/4`；Attention J0 门不通过。

## 裁决

J0 两侧均未达到预注册材料门，计划关闭为 `CLOSED / J0_REJECTED`。不扫描 gauge、fold、
group-size 或 gauge 指数邻域，不创建 J1，不修改根 `solution.py`，不归档新版本，也不提交
官方。后续若继续优化，必须从新的、未关闭且有独立证据的机制计划开始。
