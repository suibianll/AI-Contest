# proxy-v3 分片评测与诊断工具实施计划

状态：DONE（2026-09-04）

## 目标

在不修改 `evaluator/official_eval.py` 的前提下新增 `proxy-v3`，并将
`evaluator/eval.py` 切为稳定用户入口，降低坏候选的平均淘汰成本，让每次父子对比直接
输出可行动的退化定位，不再依赖人工翻查大型 JSON。

## 固定边界

- 不修改根 `solution.py`，不注册算法候选，不消耗官方提交配额。
- 不修改 `evaluator/official_eval.py`；其他 session 正在优化该文件，v3 通过独立入口读取其现有 dense cache 和参考 codec。
- 本地结果仍不得换算官方绝对分；保留 `delta_mean > 0 && L1 < 0.02`、OOD `|delta gap| <= 0.01` 和官方时间预测 `<280s` 的当前纪律。
- `proxy-v2` 历史 JSON 保持不可变；`proxy-v3` 使用独立 cache/profile，禁止混排。
- 校准产物缓存只服务准确率/OOD 诊断；正式时间审计必须关闭缓存并真实执行六 API。

## 实现与验收

1. `evaluator/proxy_v3_eval.py` 提供六个确定性平衡 shard；每个 shard 覆盖 4 个深度分散层、全部 7 个 Linear role、跨 validation/test 的成对窗口，以及对应 Attention 层/多窗口 case。
2. 校准 artifact 按 solution SHA、校准输入身份、state keys、场景和运行环境严格校验；命中时跳过 Weight/Attention calibration，并在 timing 中明确标记为非正式时间。
3. `evaluator/proxy_v3_analyze.py` 输出 paired delta、L1、tail、role/layer/shape/length 热点、组件方向、focus/control、OOD gap、API 成本分解和可执行原因；禁止官方分数预测。
4. `evaluator/proxy_v3_runner.py` 顺序运行 shard，单进程只加载一次 multi-GB dense cache，支持连续非正向提前停止、两侧独立累计统计、每 shard 诊断产物和已有 JSON 重放。
5. 单测覆盖 shard 完整性/不重复性、cache round-trip/身份拒绝、分析器故障定位和时间模型。

## 实施结果

- 新增 v3 评测、分析、runner、稳定入口、使用文档和测试；没有修改 `evaluator/official_eval.py` 或根 `solution.py`。
- timing 分离 cache-load、calibration、scoring；缓存结果不会进入官方时间模型。
- 定向回归：`44 passed`；v3 文件和测试 `py_compile` 通过；本次文件 `git diff --check` 通过；`official_eval.py` 无 diff。
- 执行记录：[`2026-09-04-proxy-v3-implementation.md`](../../../../logs/execution/2026-09-04-proxy-v3-implementation.md)。

## 失败处理

当前合法 state 均可安全序列化并在加载时重新校验；若未来出现不支持的 state 类型，应保持缓存功能关闭，不使用宽松反序列化或静默回退。
