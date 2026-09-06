# Attention 输出质量加权 2×2 Q/K 变换执行记录

日期：2026-09-06  
计划：`2026-09-06-attention-output-mass-pair-plan`  
状态：**CLOSED / M2_TIME_REJECTED**  
父版本：v189，SHA `261202248a0146a2ee45f3df60bd1979bb8171b7c162921013b0024c848617af`  
候选 SHA-256：`10154C7EF5FAC891433687BA34B88BCECF4E60C6C8BCE0B12A7753BB32D44EBF`

## M0

py_compile、六 API、有限输出、合法 2×2 reciprocal pair state、连续 Q·K 不变量和
causal `p²` mass capture 通过；root `solution.py` 未修改。

## M1：eval-v3

固定 proxy-v2 cache、CUDA、Attention 六 shard、v189 baseline。48 case 候选 mean
`0.752881086113999`，baseline `0.752772354840816`，delta `+0.000108731273184`。
shard delta 为 `0`、`+0.001415867840441`、`-0.000763480201340`、`0`、`0`、`0`。

## M2：OOD 与 fresh default

OOD 候选 `0.751526904172546`，父 `0.751857367978909`；相对父 gap 变化约
`+0.0004392`，通过 `|Δgap|<=0.01`。fresh default 结果：

| 指标 | 候选 | v189 | delta |
|---|---:|---:|---:|
| Linear | `0.640258324429894` | `0.640258324429894` | `0` |
| Attention | `0.752399782610857` | `0.752173407020070` | `+0.000226375590787` |
| Overall | `0.686983932005295` | `0.686889608842467` | `+0.000094323162828` |

API 分解：`W_calib=271.393797799712s`、`A_calib=60.488939199597s`、
`dyn_act=59.605983000249s`、`dyn_qkv=3.135593600005s`；时间模型预测
`282.286164185503s`。

## 决策

Attention 的正向变化过小且时间预测超过 `<280s`，故机制关闭，不分配 v190、不提交
官方、不扫描参数邻域。候选归档于
`solutions/20260906_attention-output-mass-pair_time-rejected/`。
