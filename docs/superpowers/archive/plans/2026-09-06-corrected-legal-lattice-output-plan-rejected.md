# 修正版合法离散网格与输出目标优化计划

> 创建：2026-09-06  
> 状态：**CLOSED / R1_NO_SUPPORTED_MECHANISM**  
> 父版本：根 `solution.py` v186，官方 `17599/272s`，SHA
> `F8495DCA20334ACBDAD16FC18EE41A4970F31E1837FDEEDCEE9C70AEE54E7EB8`

## 1. 目的与边界

本计划修复并重测上一轮审计指出无效的合法离散编码诊断。目标是用参考 HiF4 的真实
五字段语义，检查 NVFP4 carrier、E6M2 scale、lv2/lv3 共享分组和 mantissa/sign 的
联合离散余量；不把旧 cb1/cb2 的自定义解码、错误计数或 operand 松弛结果当作结论。

联合搜索只在离线研究器中执行；没有把搜索、`A@W` 反推激活或不受限候选循环带入在线
路径。Linear 与 Attention 的输出目标也没有混用。

## 2. 执行与裁决

### R0：参考 codec 与合法性回归 — PASS

`workbench/legal_codec_output_probe.py` 的 R0 检查了 signed E2M1 carrier、scale 输入、
数据相关 E6M2 seed、官方层级 shape、非共享 lv3 拒绝、operand/output 目标区分以及
真实 NVFP4 cache 加载。126 个 E4M3 scale、15 个 signed E2M1 carrier 和 1890 个原始
乘积均通过；BF16 原始码本差异为 0，独立变换 dense 的差异为 100/128。

### R1：真实输入的合法联合 output oracle — NO_SUPPORTED_MECHANISM

固定 28 个 `(layer, role)` state，在 112 个独立 holdout case 上评估 64-block 内的
合法 scale×lv2×lv3×mantissa 联合模式。每块保持一个共享 E6M2 scale，模式数为
14,622,720；固定父部署路径与实际输出目标均来自同一 NVFP4 cache。

结果：

- LOO mean/median `-0.0031395047/-0.0002933227`，worst `-0.0782907179`；
- holdout mean/median `-0.0003231619/-0.0000542296`，L1 `0.0004041152`；
- 正 case `10/112`，changed values `1770`；
- layer 0/8/15/23 的 holdout mean 分别为 `-0.00012385/-0.00058242/-0.00024327/-0.00034311`，
  四层 median 均为负。

按计划停止条件，R2 不执行，不注册部署规则，不创建官方候选。该结果关闭的是本次
修正版固定联合 oracle；它不恢复旧 cb1/cb2 的错误证明，也不宣称整个合法格式空间
已经被穷尽。

## 3. 产物与复现

- 执行记录：[`2026-09-06 执行记录`](../../../logs/execution/2026-09-06-corrected-legal-lattice-output-plan.md)
- R0：`artifacts/proxy_v3/corrected-legal-lattice-20260906/r0/result.json`
- R1：`artifacts/proxy_v3/corrected-legal-lattice-20260906/r1-fixed-panel/result.json`
- 研究器：`workbench/corrected_legal_lattice_oracle.py`
- 回归：`tests/test_legal_codec_output_probe.py`，`6 passed`

根 `solution.py` 未修改，官方状态未变化。下一活动计划为
[`Attention source-scale proposal`](../../plans/2026-09-06-attention-source-scale-proposal-plan.md)。
