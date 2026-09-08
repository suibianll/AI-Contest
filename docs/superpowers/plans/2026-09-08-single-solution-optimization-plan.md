# 标准 Linear 承载的 Attention 有效性验证计划

> ACTIVE，2026-09-08。当前完整根官方 `18032/280s`。v190、v191、v192 在当前 Linear 上均
> `TIMEOUT`，因此先移除当前 Linear 的时间占用，用标准 Linear 单独测出 Attention 算法的官方效果。

## 1. 这轮要回答两个问题

1. v190、v191、v192、v195 的 Attention 修改究竟提高还是降低官方分数。
2. 当前最高分 Linear 在 `18032` 中贡献多少。

固定三个已有官方锚点，不重复提交：

| 组合 | 官方分数 / 时间 | 用途 |
|---|---:|---|
| 标准 Linear + 标准 Attention | `1001 / 146s` | 全标准零点 |
| 标准 Linear + R3 Attention | `14405 / 238s` | 本轮 Attention 对照 |
| 当前 Linear + R3 Attention | `18032 / 280s` | 当前完整根 |

当前 Linear 的净贡献已经可以直接计算：

```text
C_linear = 18032 - 14405 = 3627
当前 Linear 的侧等价分 = 1001 + 3627 = 4628
```

交叉核对使用上一完整根 `17636` 和同 Attention 的标准 Linear 组合 `14009`：

```text
17636 - 14009 = 3627
```

两组独立组合得到相同结果。因此当前证据下，当前最高分 Linear 的官方净贡献为 **3627**，
等价侧分为 **4628**；它占当前高于全标准基线部分 `18032-1001` 的约 **21.3%**。这是组合差分
得到的贡献，不登记成一次独立官方 Linear 提交结果。

## 2. 如何构造候选

每个候选保留原 Attention 源码和四个 Attention API，只把两个 Linear API 替换成 v162 已验证的
标准 Linear 实现。具体复用
`solutions/v162_attention_r1-v189-attnstack-recovery_officialNA_timeNA/solution.py` 中已经使用过的
标准 Linear 尾部覆盖方式，不重新实现 codec。

统一在 `workbench/standard_linear_attention_probes/` 放一个构建脚本和一个核对脚本。构建脚本固定读取
以下五个已归档源码，不修改原归档：

```text
solutions/20260908_v190_attn-diag-reciprocal-balance_scoreNA_timeNA/solution.py
solutions/20260908_v191_attn-block-triangular-transport_scoreNA_timeNA/solution.py
solutions/20260908_v192_attn-full-reciprocal-residual_scoreNA_timeNA/solution.py
solutions/20260908_v194_attn-a2-calibration-fused_scoreNA_timeNA/solution.py
solutions/20260908_v195_attn-a2-center-gradient-aggregate_scoreNA_timeNA/solution.py
```

构造四个计分候选：

| 候选 | Attention 来源 | 要判断的问题 |
|---|---|---|
| `standard-linear_v190-attn` | v190 逐通道 Q/K 互逆平衡 | 去掉当前 Linear 后是否有官方增益 |
| `standard-linear_v191-attn` | v191 稀疏三角搬运 | 已部署修改是否提高 Attention 分数 |
| `standard-linear_v192-attn` | v192 全矩阵互逆残差 | 隐藏官方校准上是否被接受并产生收益 |
| `standard-linear_v195-attn` | v195 K-center 多窗口梯度修复 | bug 修复是否优于 R3 |

v194 是 R3 输出等价提速，不属于新计分算法。另构造 `standard-linear_v194-attn-speed`，只用于比较
官方时间是否低于 R3 的 `238s`。v196 与 v192 属于同一32步全矩阵机制，而且本地最终回退 R3，
不再重复提交。

## 3. 执行顺序

### 第一步：生成标准 Linear 组合

为上述五个候选各生成一个单文件 `solution.py`。生成后只做一次组合核对：

- 两个 Linear API 与 v162 标准 Linear 逐位一致；
- 四个 Attention API 与各自原候选逐位一致；
- 六 API 能脱离仓库导入，state 合法，输出有限。

执行命令固定为：

```powershell
.venv\Scripts\python.exe workbench/standard_linear_attention_probes/build.py
.venv\Scripts\python.exe workbench/standard_linear_attention_probes/verify.py
```

`build.py` 一次生成五个候选，`verify.py` 一次输出五行 PASS/FAIL；不为每个候选再创建一套检查脚本。

已有 v190–v195 的本地 shard0 和 reachability 结果直接复用，不重跑六 shard、OOD、跨模型、参数扫描
或分数门禁。这里的目的就是取得官方侧分；v190/v192 在本地 shard0 回退 R3 不阻止这一次标准 Linear
诊断提交，因为官方隐藏校准可能作出不同选择。

### 第二步：先提交 v194 提速对照

提交 `标准 Linear + v194`：

- 分数应为 `14405`；不同则说明所谓输出等价没有在官方输入上成立，停止使用 v194。
- 分数相同且时间 `<238s`，记录实际节省秒数，并把 v194 作为后续完整组合的 Attention 实现。
- 分数相同但时间没有下降，v194 不替换实现。

### 第三步：逐个提交四个 Attention 算法

按 `v195 → v191 → v190 → v192` 提交，每个只提交一次。统一计算：

```text
attention_step_gain = 官方候选分数 - 14405
```

- `> 0`：算法有效，进入完整根回装。
- `= 0`：对官方计分无效果，不继续调参。
- `< 0`：算法负向，关闭该实现。
- 标准 Linear 载体仍 `TIMEOUT`：该实现本身超时，关闭该实现。

不使用本地 mean、L1 或运行时间替代上述官方判断，也不因小幅正负结果扫描学习率、步数、窗口或
clamp。v192 放最后，因为它的32步全矩阵训练成本最高。

### 第四步：只回装官方最优 Attention

四个候选全部回传后，只选择官方分数最高且高于 `14405` 的一个。把它的四个 Attention API 回装到
当时最快的当前 Linear 完整根：若 v194 官方提速成立，先带上 v194 的等价实现，再合入该算法修改。

只提交这一个完整组合：

- 分数高于 `18032` 且时间 `<300s`：替换当前根。
- 分数没有提高：保留 `18032/280s` 根，说明该 Attention 与当前 Linear 存在负交互。
- `TIMEOUT`：Attention 算法的分数结论仍保留，但必须先做 Linear 等价降时，释放时间后才能再次组合。

若四个计分候选均不高于 `14405`，不做完整组合提交，下一轮直接转 Linear 等价降时或新的算法机制。

## 4. 归档方式

每个标准 Linear 组合独立归档：

```text
solutions/<standard-linear_attention-candidate>/
  solution.py
  result.md
  official-result.json
```

`result.md`只记录原 Attention 来源、组合 SHA、官方分数/时间、相对 `14405/238s` 的差值和结论。
本地已有结果只链接原候选证据，不复制成长篇检查报告。全部回传后在本计划末尾追加一张结果表，
并在 `docs/current-solution-status.md` 与 `solutions/README.md` 更新最终结论。

## 5. 已有候选状态

- **v194 `attn-a2-calibration-fused`：本地完成，待标准 Linear 官方提速对照。** 候选 SHA
  `1e1d9846...229dce`；合成与真实 4B shard0 均逐位一致，calibration API
  `6.021s → 4.669s`（−22.5%）。
- **v195 `attn-a2-center-gradient-aggregate`：本地完成，待标准 Linear 官方计分。** 候选 SHA
  `839adb1e...761d7f`；已确认梯度包含全部训练窗口，shard0 delta mean `+0.0019352`，非 no-op。
- **v190、v191、v192：** 完整根官方均为 `TIMEOUT`，原候选源码与结果归档直接复用；不重新实现算法。
- **v196：** shard0 全部回退父且与 v192 同属32步全矩阵残差，不纳入本轮。
