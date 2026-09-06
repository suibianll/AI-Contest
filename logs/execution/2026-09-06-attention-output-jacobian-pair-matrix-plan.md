# Attention 输出 Jacobian 加权 2×2 Q/K 变换执行记录

日期：2026-09-06  
计划：`2026-09-06-attention-output-jacobian-pair-matrix-plan`  
状态：**CLOSED / P2_TIME_REJECTED**  
父版本：v189，SHA `261202248a0146a2ee45f3df60bd1979bb8171b7c162921013b0024c848617af`  
候选 SHA-256：`cf99ca3ec54754bd3d2a6d8bd7d79b129e71e601d9d8a37e8b130f88d69cf7e7`

## P0

候选从 v189 单文件复制。`py_compile`、六个 API 导入、有限 Jacobian 权重、合法
2×2 state shape 通过；固定 reciprocal pair transform 的连续 Q·K 不变量最大误差
约 `1.9e-6`。root `solution.py` 未修改。

## P1：eval-v3 Attention 六 shard

固定 proxy-v2 dense cache、CUDA、`--attention-only --shards 0,1,2,3,4,5`，baseline
为 v189。48 个配对 case 的结果：

| 指标 | 候选 | v189 baseline | delta |
|---|---:|---:|---:|
| Attention mean | `0.7569128773065631` | `0.7527723548408157` | `+0.00414052246574736` |

shard delta mean 为：

`0.013525590458857192`、`0.005133423918639052`、`0.00561201929731793`、
`0.002732973706860764`、`-0.002160872587190721`、`0`。

候选 Attention calibration API total 为 `60.48816199996509s`（eval-v3 诊断）。
前四 shard 正向，后两 shard 为轻微负/零；未修改 Linear control 未调用。

## P2：OOD 与 fresh default 时间

- OOD mean：候选 `0.75700131700196`，父 `0.751857367978909`；相对父的
  `Δ(in−OOD)≈-0.0010034`，通过 `|Δgap|<=0.01`；
- fresh default（proxy-v2 兼容后端，168 Linear + 120 Attention）：Linear
  `0.6402583244298936`，与 v189 逐位相同；Attention `0.7535836669915871`，
  相对 v189 `0.75217340702007` 为 `+0.0014102599715171`；Overall
  `0.6874772171639325`；
- API 分解：`W_calib=274.717107299715s`、`A_calib=59.7907665004022s`、
  `dyn_act=59.7201452993322s`、`dyn_qkv=2.930944901650608s`；代入已校准模型
  `T_pred=282.59095457584823s`，超过 `<280s` 时间门。

## 决策

输出 Jacobian 加权 pair transform 的准确率/OOD 信号为正，但固定实现不满足官方
时间资格；关闭该配置，不分配 v190、不提交官方、不扫描其参数邻域。候选源码和
manifest/result 归档于
`solutions/20260906_attention-output-jacobian-pair-matrix_time-rejected/`。

证据：

- eval-v3 P1：`artifacts/proxy_v3/attention-output-jacobian-pair-matrix-20260906/p1b/candidate/manifest.json`
- eval-v3 P2 OOD：`artifacts/proxy_v3/attention-output-jacobian-pair-matrix-20260906/p2-ood/candidate/manifest.json`
- fresh default：`artifacts/official_eval/attention-output-jacobian-pair-matrix-fresh-default.json`
