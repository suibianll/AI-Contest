# Linear 静态 activation-GPTQ 条件曲率块序计划

> 创建：2026-09-06  
> 状态：**CLOSED / C1_REJECTED**  
> 父版本：v189（Linear `0.640258324430`、Attention `0.752173407020`、Overall
> `0.686889608842`，官方 `unregistered/NA`）  
> 根正式版本：v186（官方 `17599/272s`，SHA
> `F8495DCA20334ACBDAD16FC18EE41A4970F31E1837FDEEDCEE9C70AEE54E7EB8`）

## 1. 目的与唯一机制

v189 的静态 activation-GPTQ 使用部署权重列能量 `diag(H)` 对完整 64-channel block
排序。这个计划只替换排序统计为同一已存 Hessian inverse 的条件曲率：

```text
score(block) = sum_i 1 / diag(H^{-1})_i
```

其中 `H` 是现有 activation GPTQ 的 ridge weight Gram；`H^{-1}` 已在父 state 中生成，
不新增矩阵求逆、不新增校准样本、不改变合法编码。条件曲率衡量在其余坐标可补偿时，
该 block 的残差代价，和 v189 的边际能量排序是数学上不同的单一块序规则。

除块序外，Smooth/CAT/L-R2、权重编码、Attention、state 字段、在线 activation GPTQ
候选池和恢复布局全部保持 v189。禁止扫描 `diag(H)`/`diag(H^{-1})` 混合权重、分位数、
反向排序、block ratio、seed、layer/role 路由或任何候选邻域。

## 2. 固定执行顺序

### C0：单文件与 state smoke

从 v189 研究源码构造候选，检查单文件导入、`py_compile`、六 API、条件曲率有限且
state 形状合法；确认除固定块序索引外不增加 state，Attention 与父逐位一致。

### C1：Linear eval-v3 六 shard

使用固定 `proxy-v2` cache、CUDA、`--linear-only --shards 0,1,2,3,4,5`，与 v189
逐 case 配对。记录 mean/median、L1、负 case、worst quartile、角色/层分布、
validation/test 同号率、块序 reachability 和未修改 Attention control。无真实变化或
出现系统性回归立即关闭。

### C2：default 与 OOD

只有 C1 有实际变化才运行完整 Linear default 和 OOD 六 shard；要求 default Overall
严格高于 v189、`L1 < 0.02`、control 逐位不变，且
`|Δ(gain_in-gain_ood)| <= 0.01`。本地 proxy 仅作风险判读，不换算官方分数。

### C3：时间、归档与官方

运行一次 fresh default 时间审计；候选必须满足六 API 分解预测 `<280s`。只有 Overall
严格超过 `0.686889608842` 且所有门禁通过时，才归档 v190 源码、manifest、result 和
仅含 `solution.py` 的 zip；官方网页上传未确认前记 `unregistered/NA`，根 `solution.py`
不自动切换。官方负向或超时后关闭本机制，不扫块序邻域。

## 4. 执行结果（2026-09-06）

- C0：py_compile、单文件导入、条件曲率排序合成 smoke 均通过；排序状态真实
  reachability 已由候选日志确认。
- C1：使用 v189 baseline、固定 proxy-v2 cache、CUDA、Linear shard 0/1；因负向
  门禁停止，未运行 shard 2–5、default/OOD 或官方提交。
- 配对 Linear delta mean：shard 0 `-0.000491`，shard 1 `-0.000125`；L1 分别
  `0.001316`、`0.001128`。主要回归集中在 `o`、`fc_gate`。
- 候选源码已归档于
  `solutions/20260906_linear-static-gptq-conditional-curvature_rejected/`，SHA256
  `54638e1146a07558b019c098f1496b3fa5d33f6805026e08a5ff855338e1d5ec`。
- 证据目录：
  `artifacts/proxy_v3/linear-static-gptq-conditional-curvature-20260906/c1/`。

结论：该单一条件曲率块序规则关闭；不得在本计划内继续扫描块序邻域。
