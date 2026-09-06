# Linear 冻结激活状态的输出感知静态权重编译计划

> 创建：2026-09-06  
> 状态：**CLOSED / J1_REJECTED**  
> 父版本：v189（Linear `0.640258324430`、Attention `0.752173407020`、Overall
> `0.686889608842`，官方 `unregistered/NA`）  
> 根正式版本：v186（官方 `17599/272s`，SHA
> `F8495DCA20334ACBDAD16FC18EE41A4970F31E1837FDEEDCEE9C70AEE54E7EB8`）

## 1. 目的与唯一机制

当前 v189 的 Linear 调用图包含成熟的合法权重/激活量化，但历史源码中的 JDRQ
输出感知静态权重选择器没有接入当前根版本的校准返回路径。本计划只把这一选择器
接入 v189，并严格冻结已经完成的 activation state（包括 v189 静态 activation-GPTQ
块序），使它针对真实部署的 `Q(A) @ Q(W)^T` 目标编译静态权重载荷。

固定目标为宽下投影（`in_features >= 2048` 且 `out_features < in_features`）：

1. 先由父校准完整生成 v189 的 activation state；
2. 用 exact transformed activation 与冻结的实际部署 activation quantizer 构造
   calibration-only teacher/product；
3. 用现有固定 JDRQ hierarchy/rowwise 规则在合法 E6M2、lv2/lv3、signed mantissa
   网格上做一次静态权重更新；
4. 只在固定 robust product loss 更低时写回 `weight_params`，不改变 state 或在线 API。

这不是重新做 cross-fold Hessian、block order、层级分区或 Attention source-scale；
也不扫描 lambda、eta、ratio、fold、layer/role 路由。JDRQ 代码中的既定常量保持原值，
其多候选是一个预先固定的编译器内部规则，而非根据评测结果扩展的搜索。

## 2. 固定执行顺序

### J0：单文件与冻结状态 smoke

从 v189 研究源码复制候选，检查 `py_compile`、六 API、合法 state、有限输出；确认
JDRQ 的 frozen activation 与 v189 静态块序路径逐位对齐，Attention state/API 不被
修改，root `solution.py` 不变。

### J1：Linear eval-v3 六 shard 配对

使用固定 proxy-v2 cache、CUDA、`--linear-only --shards 0,1,2,3,4,5`，baseline 为
v189。记录 default 调用图下的逐 case mean/median、L1、负 case、尾部、role/layer、
reachability、API 分解和未修改 control；前两 shard 出现明确负向即关闭，不调参数。

### J2：default/OOD 与时间

只有 J1 有实际正向变化才运行 Linear default/OOD 六 shard；要求相对 v189 Overall
严格上升、`L1 < 0.02`、control 逐位不变、OOD `|Δ(gain_in-gain_ood)| <= 0.01`，
并用 fresh default 的六 API 分解预测官方时间 `<280s`。本地分数不能换算官方绝对分。

### J3：归档与官方

通过全部门禁才分配 v190，归档候选源码、SHA、manifest、result 和仅含 `solution.py`
的 zip；官方网页上传未通过可用工具确认前记 `unregistered/NA`，根 `solution.py`
不自动切换。失败、超时或官方负向后关闭该固定 JDRQ 接入，不扫描其邻域。

## 4. 执行结论（2026-09-06）

J0 通过：候选可单文件编译并导入，六个 API 可见，JDRQ 仅作用于冻结后的宽下投影，
且 root `solution.py` 未修改。

J1 在前两个 Linear shard（112 个配对 case）已足够否决，按计划提前停止，不运行 J2/J3：

- 候选 local mean `0.6343498783403304`，v189 baseline `0.634575771204753`；
- shard 0 delta mean `-0.0003604818362111497`，median `0`，L1
  `0.0003604818362111497`，正/负/零 `0/8/48`；
- shard 1 delta mean `-0.00009130389263410246`，median `0`，L1
  `0.0002452943856971499`，正/负/零 `3/5/48`；
- candidate API total `133.8395642 s`（weight calibration `99.9741984 s`、dynamic
  activation `33.8653658 s`）。

JDRQ 的 reachability 已确认，但在当前 eval-v3、v189 冻结 activation state 和实际
`Q(A) @ Q(W)^T` 目标下为负，不能分配 v190、不能提交官方，也不扫描 JDRQ 邻域。
