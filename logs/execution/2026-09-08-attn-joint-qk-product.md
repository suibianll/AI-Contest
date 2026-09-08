# 2026-09-08 `v193 / attn-joint-qk-product`

## 结论

候选从当前完整根构建，六 API、状态合法性、矩阵指数互逆关系、联合乘积手工梯度、固定训练和
实际可达性均通过。4B eval-v3 Attention shard0 未出现接口或有限性问题；联合 Q/K block-scale
训练实际运行，但两个独立真实输出 gate 窗口均回归，候选按计划回退父状态。候选已归档，官方状态
为 `unregistered/NA`，根方案保持不变。

## 固定配置与源码

- 父 SHA256：`12352efdd4e23cc5e1e17953008664fbaa5ea5d693373635fdafc4d28ce4e24e`
- 候选 SHA256：`5c2dc19347741d3f2368eabb8b97a008eaec3064fe1f7da47bf20f62d5f299e0`
- 32 步 Adam，学习率 `0.01`，梯度裁剪 `1.0`，正则 `0.001`，谱界 `+/-log(2)/2`；fit
  `0,1,2`，gate `3,4`。
- 目标：`mean(aQ_new*aK_new/(aQ_parent*aK_parent+1e-12)) + 0.001*mean(S^2)`。
- 部署：`Q_parent@exp(S)`、`K_parent@exp(-S)`，K center 折叠 `exp(-S)`。

## 检查结果

```text
check_math_and_import.py: PASS
verify.py: PASS
```

## 4B shard0

候选与当前根均为 12 个 Attention cases，mean `0.5702418426766733`，paired delta `0`，
`reasonableness_issues=0`。candidate API total `8.926325s`，父 API total `0.883214s`；本地时间
只作风险记录，不换算官方时间。

## 机制审计

`a23_attempted=1`、`a23_accepted=0`、`a23_arm=parent`。训练目标从 `1.0` 降至
`0.37283017`，父/候选 QK block 乘积均值为 `49.97351011/21.11840280`，但 window 3 与 4
的 gate MSE 分别从 `0.0003881380` 增至 `0.0004023325`、从 `0.0004327810` 增至
`0.0004487973`，故保留父状态；不扫描步数或参数邻域。

## 证据位置

- 归档：`solutions/20260908_v193_attn-joint-qk-product_scoreNA_timeNA/`
- 评测：`artifacts/proxy_v3/full_solution/attn-joint-qk-product-shard0/candidate/manifest.json`
- 工作脚本：`workbench/full_solution/attn-joint-qk-product/`
