# Linear 校准内 carrier-energy 块序执行记录

日期：2026-09-06  
父版本：v189（官方 `17616/275s`）  
候选 SHA256：`a30e05b7fb71a64121908bc10d102dd50133a23e32f6568ed7044dbc45321730`  
状态：**CLOSED / R3_REJECTED_TIME**

## R0

候选从根 v189 复制。`py_compile`、六 API 导入通过；校准临时样本在构造最终
state 前被消费，不会进入部署 state。168 个 Linear state 的 carrier-energy order
均可达，有限输出检查通过。

## R1/R2

使用固定 `proxy-v2` cache、CUDA 和 v189 baseline 的 eval-v3 Linear 配对：

| 范围 | candidate mean | v189 mean | delta mean | L1 | 正/负/零 |
|---|---:|---:|---:|---:|---:|
| shard 0 | — | — | `+0.000207` | `0.001452` | `36/20/0` |
| shard 1 | — | — | `+0.000201` | `0.001084` | `31/23/2` |
| 六 shard | `0.637750315` | `0.636799489` | `+0.000950827` | — | aggregate 正向 |

六片 reasonableness issues 为 0，所有输出有限。固定 OOD 面板 candidate mean
为 `0.649639183`，v189 为 `0.648734547`，相对父版本的 in−OOD gap 变化约
`+0.000046`，在 `0.01` 门内。

## R3 fresh default

兼容后端 `official_eval.py` 的新鲜 default-panel 结果与上一 carrier-energy 版本
逐位一致：Linear `0.641778372`、Attention `0.752173407`、Overall
`0.687776303`。时间分解如下：

| API | seconds |
|---|---:|
| `hif4_calibration_and_quantize_weight` | `274.721588` |
| `hif4_calibration_attention` | `61.435543` |
| `hif4_dynamic_quantize_activation` | `60.547017` |
| Q/K/V dynamic sum | `2.961587` |

按固定六 API 模型，`T_pred=284.291453s`，超过 `<280s`；API total 为
`399.665734s`，wall 为 `426.211627s`。候选分数也没有超过当前本地最高
`0.687776303363468`，故不分配版本、不生成 zip、不提交官方，根保持 v189。

证据归档于
[`solutions/20260906_linear-integrated-carrier-energy_time-rejected`](../../solutions/20260906_linear-integrated-carrier-energy_time-rejected/)。
