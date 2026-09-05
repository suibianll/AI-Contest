# 下一步计划的证据审计（2026-09-05）

状态：代码审读完成；新数值复测未执行。依据 git `cc73b83` 附近工作树，根 v186 SHA256
`F8495DCA20334ACBDAD16FC18EE41A4970F31E1837FDEEDCEE9C70AEE54E7EB8` 未改。
本说明不覆盖旧实验数字，修正其用于下一步计划的解释。

## 1. cb1/cb2 的具体实现问题

| 源文件/函数 | 代码事实 | 对结论的影响 |
|---|---|---|
| cb1 `hybrid_encode_block` | 从 `dense_block` 得到 abs_q 后，与裸 E2M1 CODES 比较计数 | 不能据 exact_count=0 证明 BF16 破坏码本 |
| cb1 `hybrid_encode_block` | 计算 E_v，但 exact_present 仅乘指数合法性 ok，没有乘 E_v | 所报精确数不等于兼容表判定的精确数 |
| cb2 `count_codes` | 用 q==正 cv，没有 abs | 负码被漏计 |
| cb2 `exact_encode_block` | E_v 未用于 exact_present；sf exponent seed 固定 0 | 精确计数及输入量级适配均未实现所述机制 |
| cb2 `exact_encode_block` | mant 使用 abs(quant_block)/sf，没有乘 scale_sub | 编码目标不是输入 NVFP4 dense，负 MSE 不是机制证伪 |
| cb1/cb2 层级字段 | 自定义结构按 16 元素展开 lv3，并由自定义解码器计算误差 | 没有按参考 8/4 共享结构给出标准五字段合法性证明 |
| cb1/cb2 主评估 | 与 `_dense_to_hif4(refine=0)` 比同输入张量 MSE | 不等同真实 v186 pipeline 的 A@W 或 Attention 输出评测 |

cb1 的 58–60×、cb2 的 51–26566× 是旧程序输出事实，但只反映这些已实现程序。
旧计划没有完成其声明的完整乘积/attention 输出对比，不能用上述数值关闭整个合法编码空间。
这也不意味着修好后必有收益；新计划只授权按固定配置修复验证，不重新扫旧参数族。

P0 的 S 是单元素可表示值并集。逐元素存在某个 lv2/lv3 不保证共享 4/8 组内同时可实现。
原 P0 精确率可留作放宽域统计，不能写成已构造出的合法精确率。必须返回五字段并用参考解码复核。
是否存在 BF16 snap 误差，需直接核对 carrier×scale；经过连续变换丢失码本与 BF16 舍入是两件事。

## 2. 早先诊断的解释边界

- 连续域 B² 小说明当前变换保语义，不能推出可选等价变换已穷尽。
- `coordinate_diagnostics.py::single_relax_substitution_output` 用其他 float 操作数，
  目标是 O_t；不能直接当完整量化后的部署收益。R2/R3 是 operand 拟合松弛，非输出全局最优。
- P2 报告表内 Q R3-lv3 `1.109e-3 → 0.885e-3`、K `1.252e-3 → 0.978e-3` 有约
  20%/22% 的均值差；这些还是报告近似数、基准目标需配对复核，不能表述“全部无余量”。
- 饱和元素约 4.6% 是数量占比，不是裁剪误差能量占比；不能仅据数量判定能量贡献次要。
- 非法 mantissa 放宽结果既不是合法上界构造，也不能与不同操作数残差直接相加后换算官方分。
- 官方 QK 贡献约 85%、W2/W3 承载已实现增益是有用的优先级信息，不能识别榜首差距分配。
- 不同算法 ±4 分或 <100 分没有证明随机噪声；旧官方结果及版本裁决保留，不用于重复提交。

受影响说明包括：同日 coordinate/probes、P2、targeted autopsy、cb1/cb2 日志和
`docs/next-research-direction-analysis-2026-09-05.md` 的“天花板/全部饱和”推断。
原文件不重写；以后读取须同时阅读本说明与 stale inventory。

## 3. 后续执行

见[合法编码复核与最终输出优化计划](../../docs/superpowers/plans/2026-09-05-legal-codec-and-output-objective-plan.md)。
本次仅注册计划、修正导航和证据解释；未改算法、未跑新评测、未提交官方。
