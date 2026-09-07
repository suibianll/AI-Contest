# Qwen3.5审查收尾：并发修复状态更正

本文件是[本轮完整审查](2026-09-07-dual-results-and-qwen35-evaluator-review.md)的后续更正，
当前代码状态以本文件为准。原始审查日志不覆盖。

用户在审查期间继续修改并提交了评测器。结束前重新读取确认：

| 早期发现 | 最新代码状态 |
|---|---|
| 跨层hook键覆盖 | 已加入panel_index前缀，注册与读取使用对应键；hook及时转CPU |
| 裁剪test QKV槽位与加载器冲突 | validate_qkv_bank新增required_samples，test传入panel_test_slots |
| raw路径缺total_layers导致coverage回退 | 主metadata已写total_layers；读取还可由panel_model_layers推导 |
| RoPE CPU/GPU位置不一致 | q/k与cos/sin当前均在CPU计算，device冲突已消除；仍需验证与实际forward一致 |

以上为**静态代码修复确认，不是完整捕获与独立数值闭环通过**。
完整审查末尾列出的SHA是在用户修改推进之后采集，因此不可将这些SHA称作上述早期bug的快照。
早期发现以当时读取代码为证据；当前修复版本中不再宣称前三项仍阻断。
本代理没有修改评测器，也没有执行新模型捕获或GPU算法实验。

仍需处理：

1. 独立核验各层真实模块输入/输出与抓取值，特别是CPU重算RoPE的数值、QKV排列和q_proj切片。
2. capture fingerprint贯通dense/carrier/state/result，包含真实层映射、dtype、模型/权重/代码身份，
   修复后的数据不能继续命中修复前的旧缓存。
3. metadata声明实际dtype；4B dense FFN不代表35B MoE专家，GQA也有差异，不能以同族替代真实数据保证。
4. 4B新面板成本不能代入旧0.5B六API公式；门禁继续区分平均负向L1与最坏case。
5. Linear原始56case重算negative L1=0.018546532828210376，未超0.02；
   纠正报告误拦、fold1训练/验证混用以及低维候选与原逐列卡登记不一致。

建议顺序仍是：确认捕获正确性与缓存隔离→固定官方提取边界→新面板父版本标定→两侧机制继续。
Attention官方A22-2为14424/271s，较R3+19；Linear只完成局部验证，不能将两侧统称完整计划均通过。
