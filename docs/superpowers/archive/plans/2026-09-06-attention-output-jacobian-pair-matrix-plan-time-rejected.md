# Attention 输出 Jacobian 加权 2×2 Q/K 变换计划

> 创建：2026-09-06  
> 状态：**CLOSED / P2_TIME_REJECTED**  
> 父版本：v189（local Linear `0.640258324430`、Attention `0.752173407020`、Overall
> `0.686889608842`，官方 `unregistered/NA`）  
> 根正式版本：v186（官方 `17599/272s`，SHA
> `F8495DCA20334ACBDAD16FC18EE41A4970F31E1837FDEEDCEE9C70AEE54E7EB8`）

## 1. 目的与唯一机制

当前 Attention 已有的 pair-matrix smoothing 用 Q/K 二阶协方差平衡 2×2 坐标，
但其拟合权重与最终 attention 输出无关。本计划只增加一个固定的 calibration-only
变体：在 v189 已完成的 Q/K 状态之后，按 causal 与 non-causal attention 的局部
logit-to-output Jacobian Frobenius 质量，为每个 token/head 加权 Q/K 2×2 协方差，
再求同一个 SPD reciprocal pair transform。连续 Q·K 点积仍保持不变，在线路径只
携带合法 `pair_transform`、importance 和既有 refinement state。

固定规则：

1. 使用 v189 base calibration 完成后的 Q/K/V state，不改变 Linear state；
2. 在奇偶隔离的 calibration windows 中用奇数位拟合、偶数位验证；
3. Jacobian 权重固定为 causal/non-causal 两轨的平方敏感度平均，使用现有 v-state
   的实际部署输出作为 V；
4. 只生成一个 output-weighted 2×2 matrix candidate，使用父版本相同的真实部署
   attention gate；拒绝则逐位保留父 state；不扫权重温度、ridge、blend、head 或
   block 邻域。

这不是重新开启 source-scale、Fisher channel importance、head scale、rotation、
pair-matrix 的参数邻域或 V 侧方向；它只检验“同一合法 2×2 坐标自由度是否应按输出
敏感度拟合”的单一新目标。

## 2. 固定执行顺序

### P0：单文件与合法 state smoke

从 v189 研究源码复制候选，检查 `py_compile`、六 API、有限输出、Q/K pair-transform
形状、连续 Q·K 不变量和 root `solution.py` 不变。确认候选没有在线新增 Jacobian
或矩阵求解。

### P1：Attention eval-v3 六 shard 配对

使用固定 proxy-v2 cache、CUDA、`--attention-only --shards 0,1,2,3,4,5`，baseline
为 v189；记录 Q/K/V、QK/QKV interaction、causal/non-causal 输出误差、长度/层尾部、
API 分解和未修改 Linear control。前两个 shard 出现明确负向或全程 no-op 即关闭，
不调参数。

### P2：default/OOD 与时间

只有 P1 有实质正向变化才运行 Attention default/OOD 六 shard；要求相对 v189
Attention 严格上升、`L1 < 0.02`、Linear control 逐位不变、OOD
`|Δ(gain_in-gain_ood)| <= 0.01`，并用 fresh default 的六 API 分解预测官方时间
`<280s`。本地分数不能换算官方绝对分。

### P3：归档与官方

通过全部门禁才分配 v190，归档候选源码、SHA、manifest、result 和只含 `solution.py`
的 zip；官方网页上传未通过可用工具确认前记 `unregistered/NA`，根 `solution.py`
不自动切换。失败、超时或官方负向后关闭该固定输出加权机制，不扫描其邻域。

## 4. 执行结论（2026-09-06）

P0 通过。P1 使用 eval-v3、固定 proxy-v2 cache、CUDA 和六个 Attention shard 完成
48 个配对 case：候选 mean `0.7569128773065631`，v189 mean `0.7527723548408157`，
delta `+0.00414052246574736`；前四个 shard 的 delta mean 分别为
`+0.013525590458857192`、`+0.005133423918639052`、`+0.00561201929731793`、
`+0.002732973706860764`，shard 4 为 `-0.002160872587190721`，shard 5 为 `0`。

P2 OOD mean 为候选 `0.75700131700196`、父 `0.751857367978909`，相对父的
`Δ(in−OOD)=-0.0010034`（绝对值小于 `0.01`），通过 OOD 门。

P2 fresh default（兼容后端的 168 Linear + 120 Attention 时间审计）结果：

- Linear `0.6402583244298936`，与 v189 逐位相同；
- Attention `0.7535836669915871`，相对 v189 `0.75217340702007` 增加
  `+0.0014102599715171`；Overall `0.6874772171639325`；
- `W_calib=274.717107299715s`、`A_calib=59.7907665004022s`、
  `dyn_act=59.7201452993322s`、`dyn_qkv=2.930944901650608s`，时间模型
  `T_pred=282.59095457584823s`，不满足 `<280s`。

结论：机制的本地准确率信号和 OOD 均为正，但固定实现超出官方时间门；不分配 v190、
不提交官方。候选源码保留在
`solutions/20260906_attention-output-jacobian-pair-matrix_time-rejected/`，不扫描
其权重、ridge、blend 或 token 邻域。
