# Linear 动态块能量块序执行计划

> 创建：2026-09-06
> 状态：**CLOSED / R3_REJECTED_TIME**
> 父版本：v189 `17616/275s`，根源码 SHA256
> `261202248a0146a2ee45f3df60bd1979bb8171b7c162921013b0024c848617af`

## 1. 唯一假设

上一动态样本能量候选在本地 default 提升至 Overall `0.688994941`，但时间预测
`284.775756s`。本候选保留“每次 activation call 按当前样本决定 64-block 访问顺序”
的机制，固定把排序分数压缩为
`mean_tokens(block_energy) × mean_channel_importance(block)`；不使用逐通道乘积，
不增加 layer/role 路由、不改变 GPTQ/HiF4 编码和 Attention。

## 2. 固定门禁

1. R0：单文件导入、`py_compile`、六 API、有限输出和动态顺序 reachability。
2. R1：v189 baseline 的 eval-v3 Linear shard 0/1；若 no-op、系统性回退或非法状态，
   立即关闭。
3. R2：R1 通过后跑 Linear 六 shard 与 OOD，检查 median、L1、尾部、control 和 gap。
4. R3：fresh default proxy-v2 时间审计。只有 Overall 严格超过当前本地最高
   `0.687776303363468` 且预测 `<280s`，才归档、生成提交包并推送；否则记
   `REJECTED_TIME`/`REJECTED`，根保持 v189，不提交官方。

本计划只验证一个块级统计，不扫描采样长度、权重混合、阈值、角色路由或排序邻域。

## 3. 执行记录

执行结果写入 `logs/execution/2026-09-06-linear-dynamic-block-energy-plan.md`。

## 4. 裁决

R0、R1、R2 均通过：六 shard Linear delta mean 为 `+0.003011996`，OOD delta mean
为 `+0.003559695`，`Δ(in−ood)` gap change 为约 `-0.000548`，输出有限且动态块序
可达。fresh default 的 Linear/Attention/Overall 为
`0.643820278430/0.752173407020/0.688967415343`，超过本地最高
`0.687776303363`；但时间模型预测 `286.022476s`，未通过 `<280s` 提交门。
候选标记为 **CLOSED / R3_REJECTED_TIME**，归档于
`solutions/20260906_linear-dynamic-block-energy_time-rejected/`，未提交官方，根版本保持
v189。
