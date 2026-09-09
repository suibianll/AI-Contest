# 2026-09-09 L-RB1 执行记录

活动计划第一张卡：L-RB1 静态权重有符号输出残差舍入边界。本地六 shard 全部负向，按 v219/v220/v221
的本地负向实践归档为 `REJECTED`，未提交官方；根保持 v202 Linear + v195 Attention `18053/281s`。

- 工作目录：`workbench/full_solution/linear-lrb1-residual-rounding/`
- 归档：`solutions/20260909_v226_linear-lrb1-residual-rounding_rejected_scoreNA_timeNA/`
- parent SHA256：`56dc805d6e5a3aef896db8021045740292735725d688b48e3d4393e55efcb2bd`
- candidate SHA256：`31d08ce90140e1527b5ffbff9e92e27d87b9d25d6239b0560daf65f6ed468e3d`
- 运行：`artifacts/proxy_v3/lrb1-shard0/`、`artifacts/proxy_v3/lrb1-sixshard/`、
  `artifacts/proxy_v3/lrb1-sixshard-full/`、`artifacts/proxy_v3/lrb1-parent-fresh-shard0/`

## 实现

候选 = 完整根 + 尾部追加的一个 `hif4_calibration_and_quantize_weight` 后处理包装，沿用 v220/AW13
既有模式，不修改任何 `solutions/` 归档源码，六 API 与动态路径不变。每个 calibration 调用学习
12 个共享边界 `tau[s,m]`，`s∈{-1,+1}`、`m∈{1..6}`，规则为 `c = m if frac(u) < tau[s,m] else m+1`，
`m = floor(u)`。只有 `父码 == clamp(round(u),0,7)` 的元素进入拟合，因此 `tau=0.5` 逐位恢复父五字段。
求解用 64 个固定 `frac` 桶 + 后缀和解析取 argmin（并列取最接近 0.5），整表合并后只做一次精确
二次型接受 `ΔL = 2⟨G,ΔW⟩ + Σ_f ω_f‖X̂_f ΔWᵀ‖²`，严格 `< 0` 才接受。

**计划与实现的一处必要修正**：`tau[m]` 必须按 **floor 码**索引，而不是父码。父码为 m 的 eligible
元素里既有"向下舍入到 m"（`frac<0.5`）也有"向上舍入到 m"（`frac>0.5`）的；若按父码索引，后者会在
`tau=0.5` 时被翻到 `m+1`，破坏父回退。`verify.py` 的 control A 抓到了这一点。

## 验证

`workbench/full_solution/linear-lrb1-residual-rounding/verify.py`（随机张量、CUDA）：

- 六 API 可独立导入；脱离仓库单文件导入检查通过（在 `/tmp` 下加载，模块路径不来自仓库）。
- control A：全 0.5 表逐位恢复父码，`changed = 0`。
- control B：合成 `tau[+1,3]=0.25` 精确命中预期 eligible 集合（316/316），并把目标码抬到 4。
- 接受的表只产生 `+0.25` mantissa 步进，`changed` 计数与 mantissa 差值一致，`ΔL < 0`。

## 六 shard 结果

Linear-only，336 例，六 shard 全部 `reject`：

| shard | delta mean | median | 正/负/零 |
|---:|---:|---:|---|
| 0 | -0.000143631 | 0.0 | 7 / 12 / 37 |
| 1 | -0.000593319 | 0.0 | 5 / 19 / 32 |
| 2 | -0.000339417 | 0.0 | 7 / 17 / 32 |
| 3 | -0.000439196 | 0.0 | 6 / 13 / 37 |
| 4 | -0.000599656 | 0.0 | 4 / 16 / 36 |
| 5 | -0.000346331 | 0.0 | 4 / 18 / 34 |

等权 shard 均值 `-0.000410258`，合计 `33 / 95 / 208`。

每层诊断（168 次 calibration 调用）：65 层 accepted / 103 层回退，334 个边界提案，
改动 10,381,928 个 mantissa 码；接受层校准 `ΔL` 合计 `-2.519897e-05`（每层 ≈ -5e-7），
回退层 `ΔL` 合计 `+8.270461e-04` 且全部 `≥ 0`。接受层集中在宽形态
（`2560x9216`、`2560x4096`、`9216x2560`、`4096x2560`），也正是配对结果的最差形态。

时间（仅诊断）：候选六 shard fresh `api_total` 1289.2s / 校准 1018.0s；父 shard0 fresh 对照
179.3s / 134.0s 对候选 204.6s / 164.2s（约 1.14x `api_total`、1.23x 校准），低于指引的 1.5x 时间风险线。

## 结论

机制可达、非等价，但**校准收益在数值噪声底**（每接受层 ≈ 5e-7）而扰动极大（10.4M 个硬码）。
这正是计划第 1 节判断的"高自由度逐码修改在校准窗口过拟合"形态，而不是本卡要验证的
"少量共享边界决定大量真实硬码"。按计划 §3.5 关闭该机制，不改 per-row/per-block 表、
不拆 sign、不加桶数、不改粒度重试。下一卡进入 A-RB1。
