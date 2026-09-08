# 2026-09-08 `v192 / attn-full-reciprocal-residual`

## 结论

候选从当前完整根构建，六 API、状态合法性、矩阵指数互逆关系、固定训练和实际可达性均通过。
4B eval-v3 Attention shard0 未出现接口或有限性问题；32 步全矩阵训练实际运行，但两个独立 gate
窗口均回归，候选按计划回退父状态。候选已归档，官方状态为 `unregistered/NA`，根方案保持不变。

## 固定配置与源码

- 父 SHA256：`12352efdd4e23cc5e1e17953008664fbaa5ea5d693373635fdafc4d28ce4e24e`
- 候选 SHA256：`dab718bc1835d391eef5212dda869d9deac3db71dc072e08ec7ab3d6c4c6ba07`
- 32 步 Adam，学习率 `0.01`，梯度裁剪 `1.0`，正则 `0.001`，谱界 `+/-log(2)/2`；fit
  `0,1,2`，gate `3,4`。
- 公式：`Q_new=Q_parent@exp(S)`、`K_new=K_parent@exp(-S)`，`trace(S)=0`；K center 同步
  折叠 `exp(-S)`。

## 检查结果

```text
check_math_and_import.py: PASS
verify.py: PASS
```

## 4B shard0

候选与当前根均为 12 个 Attention cases，mean `0.5702418426766733`，paired delta `0`，
`reasonableness_issues=0`。candidate API total `9.135397s`，父 API total `0.743173s`；本地时间
只作风险记录，不换算官方时间。

## 机制审计

`a22b_attempted=1`、`a22b_accepted=0`、`a22b_arm=parent`。训练 loss 从 `2.0` 降至
`1.22369574`，但 window 3 与 4 的 gate loss 分别从 `0.0003881380` 增至 `0.0003995409`、
从 `0.0004327810` 增至 `0.0004465908`，故保留父状态。该结果关闭本配置，不扩展邻域扫描。

## 证据位置

- 归档：`solutions/20260908_v192_attn-full-reciprocal-residual_scoreNA_timeNA/`
- 评测：`artifacts/proxy_v3/full_solution/attn-full-reciprocal-residual-shard0/candidate/manifest.json`
- 工作脚本：`workbench/full_solution/attn-full-reciprocal-residual/`
