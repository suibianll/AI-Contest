# Attention 输出 Jacobian 加权 pair transform 融合实现计划

> 创建：2026-09-06  
> 状态：**CLOSED / F2_TIME_REJECTED**  
> 父版本：v189（local Linear `0.640258324430`、Attention `0.752173407020`、Overall
> `0.686889608842`，官方 `unregistered/NA`）  
> 参考候选：输出 Jacobian 加权 2×2 版本（local positive，因时间门关闭）

## 1. 目的与唯一变更

上一候选的输出 Jacobian 加权 2×2 Q/K 变换在 eval-v3 和 default-panel 上提高了
Attention，但 `T_pred=282.59s` 超过 `<280s`。本计划只做实现级融合，不改变数学
规则或 state 表达：

1. 复用 base Attention calibration 已经建立的 `a1_context`、Q/K 状态、V state 和
   calibration 前缀，去除候选 wrapper 的重复 NVFP4 解码与 reference 重建；
2. 将 causal/non-causal 两轨 Jacobian Frobenius 标量敏感度在一次 batched 小张量
   计算中求出，再取与上一候选相同的归一化平均权重；
3. 保持同一奇偶 fit/validation、同一 SPD reciprocal 2×2 pair transform、同一
   deployment gate；不改变 ridge、权重、token 数、head 路由或候选数量。

## 2. 固定执行顺序

### F0：等价性与状态 smoke

从 v189 复制单文件，检查 `py_compile`、六 API、有限输出、合法 pair-transform shape、
连续 Q·K 不变量，并在合成输入上确认融合前后 Jacobian 权重有限。root `solution.py`
保持不变。

### F1：Attention eval-v3 六 shard 配对

使用固定 proxy-v2 cache、CUDA、`--attention-only --shards 0,1,2,3,4,5`，baseline
为 v189；记录与时间拒绝候选的 Attention gain、L1、Q/K/V、长度/层尾部和 API 时间。
若相对 v189 不再正向、出现 no-op 或 state/control 异常，立即关闭，不调参数。

### F2：default/OOD 与时间

只有 F1 正向才运行 OOD 和 proxy-v2 default-panel fresh audit；要求 Attention 严格
高于 v189、Linear control 逐位不变、OOD `|Δ(gain_in-gain_ood)|<=0.01`，并用
`T≈170.3+0.115·W_calib+0.694·A_calib+0.734·dyn_act−1.58·dyn_qkv` 预测
`<280s`。

### F3：归档与官方

全部门禁通过才分配 v190、保存源码/SHA/manifest/result/zip，并准备官方提交包；官方
网页上传未通过工具确认前记 `unregistered/NA`，root 不自动切换。失败或官方负向后
关闭融合实现，不扫描数学参数邻域。

## 4. 执行结论（2026-09-06）

F0 通过：融合 estimator 与原 prototype 的 causal/non-causal 权重最大差约 `2.4e-7`，
连续 Q·K 不变量保持，root 未修改。

F1 六个 eval-v3 Attention shard（48 个 case）保持正向：候选 mean
`0.756914389304788`，v189 baseline `0.752772354840816`，delta
`+0.00414203446397265`；`0.013525590458857192`、`0.005133423918639052`、
`0.005621091286669697`、`0.002732973706860764`、`-0.002160872587190721`、
`0` 为六个 shard delta mean。Linear 未修改且未调用。

F2 OOD 保持在先前输出-Jacobian候选的门内；fresh default 结果为 Linear
`0.640258324429894`、Attention `0.753580573564532`、Overall `0.687475928235993`。
API 分解 `W_calib=269.289149999153s`、`A_calib=60.1939524004702s`、
`dyn_act=59.312791100936s`、`dyn_qkv=2.903339898446577s`，时间模型预测
`T_pred=281.99116684437035s`，仍不满足 `<280s`。

结论：融合只降低约 `0.60s` 的预测时间，未回到提交门；不分配 v190、不提交官方，
关闭该实现级计划。
