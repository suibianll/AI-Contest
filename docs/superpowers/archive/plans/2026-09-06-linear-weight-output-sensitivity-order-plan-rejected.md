# Linear 权重输出敏感度 GPTQ 块序计划

> 创建：2026-09-06  
> 状态：**CLOSED / W2_REJECTED**  
> 父版本：v189（Linear `0.640258324430`、Attention `0.752173407020`、Overall
> `0.686889608842`，官方 `unregistered/NA`）  
> 根正式版本：v186（官方 `17599/272s`，SHA
> `F8495DCA20334ACBDAD16FC18EE41A4970F31E1837FDEEDCEE9C70AEE54E7EB8`）

## 1. 目的与唯一机制

现有 v189 的 activation-GPTQ 已经改变了**激活侧**的完整 64-channel 处理顺序；本计划
只测一个尚未执行的正交方向：在权重校准阶段，按最终部署坐标中的输出敏感度固定
weight-GPTQ 块序。对每个 64-channel 权重块使用一次固定统计量：

```text
score(block) = Σ_{j in block} diag(Gram_X)_j · ||W_smooth[:, j]||²
```

其中 `Gram_X` 是现有 weight-GPTQ 已使用的 transformed activation Gram，`W_smooth`
是进入最终 HiF4 编码器的连续权重。分数高的块先量化，GPTQ 误差补偿仍使用同一
`H^{-1}`，并将结果按原始块位置恢复；因此只改变离线权重 GPTQ 的访问顺序，不改变
HiF4 字段、连续域乘积、在线 activation state、Attention、候选数量或动态路径。

它与已关闭的 activation-side `diag(H)`/carrier-energy 块序不同：本机制排序的是
**weight-GPTQ 的误差传播顺序**，统计量来自权重列的输出二次型；也不使用已关闭的
`diag(H^{-1})` 条件曲率、cross-block 更新或其参数邻域。配置只允许上述一个公式，禁止
扫描权重能量/Gram 权重、升降序、块大小、阈值、seed、层/role 路由或混合分数。

## 2. 固定执行顺序

### W0：单文件与顺序 smoke

从 v189 研究源码构造候选，确认单文件导入、`py_compile`、六 API、合法 state 和
`score` 有限；用合成 128-channel Gram/weight 检查块序为合法排列、非自然序时结果
按自然块布局输出，连续输入乘积与父实现一致。

### W1：Linear eval-v3 双 shard

使用固定 `proxy-v2` cache、CUDA、`--linear-only --shards 0,1`，以 v189 为
`--baseline-solution` 做逐 case 配对。记录 mean/median、L1、负 case、worst
quartile、层/role 分布、reachability 和未修改 Attention control；若无真实变化或
前两 shard 系统性回归，立即关闭，不运行后续 shard。

### W2：Linear 六 shard 与 OOD

只有 W1 有实际正向信号才运行六 shard Linear 和 OOD。要求六 shard 的输出有限、
块序 reachability 完整、`L1 < 0.02`、未修改 control 逐位一致；OOD 仅作过拟合诊断，
按 `|Δ(gain_in-gain_ood)| <= 0.01` 记录，不把本地增益换算为官方分数。

### W3：default、时间、归档与官方资格

运行一次 fresh default。候选必须严格超过 v189 的本地 Overall `0.686889608842`，
且六 API 分解时间预测 `<280s`；满足后才归档候选源码、SHA、manifest、result 和
仅含 `solution.py` 的 zip，并登记官方 `unregistered/NA`。官方平台上传不由本地工具
自动完成，根 `solution.py` 只有收到官方正裁决后才切换。若只分数正向但时间超门，归档
为 `REJECTED_TIME`，不提交；若官方负向则关闭本机制且不扫邻域。

## 3. 证据与回滚边界

- 根 `solution.py` 保持 v186，不在官方回传前修改。
- 统一使用 `evaluator/eval.py` 的 `eval-v3`、同一 cache/panel/device；父版本不重复
  运行，局部配对结果只做机制诊断。
- 该计划不重开 activation-side act-order、conditional-curvature、cross-block、
  carrier-energy 或任何已关闭参数族；若 W0/W1 失败，直接归档并登记关闭原因。

## 4. 执行结果（2026-09-06）

- W0：通过。六 API、`py_compile`、合法 state、非自然块序和自然布局恢复 smoke 均通过；
  候选 SHA256 为
  `dd16211e798d83c03d044b9102d4b42013dbbc7212ceafaf29039d1ad7abd2ce`。
- W1：通过。Linear shard 0/1 相对 v189 的 delta mean 为
  `+0.001381/+0.001197`，L1 为 `0.003430/0.001993`，112 cases 输出全有限。
- W2：连续跑到 shard 4 后自动停止。五个已完成 shard 的 delta mean 为
  `+0.001381、+0.001197、+0.000726、-0.001063、-0.000992`；负向集中在
  `o` role 的深层桶（shard 3/4），因此未运行第 5 shard、OOD 或 fresh default。

## 5. 裁决

权重侧输出敏感度块序在浅/中层有小幅正向，但跨深度不稳定，连续两个 shard 回归；按
预注册规则 **CLOSED / W2_REJECTED**。不扫描升降序、分数混合、块大小、层/role 路由
或其它阈值邻域。根 `solution.py` 保持 v186。

证据目录：
`artifacts/proxy_v3/linear-weight-output-sensitivity-order-20260906/w1/` 与
`artifacts/proxy_v3/linear-weight-output-sensitivity-order-20260906/full/`。
