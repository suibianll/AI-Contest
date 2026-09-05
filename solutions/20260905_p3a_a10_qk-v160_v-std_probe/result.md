# A10 probe（P3-A）

- 状态：**PROBE_COMPLETE（本地 control 通过，待官方回传）**
- 构造：Q/K 用 v160 路径、V 用 standard codec、Linear standard。从 v164 归档构造，仅重定义 hif4_dynamic_quantize_v 为 standard（_ref codec），Q/K 与校准保持不变。
- 源码：`./20260905_p3a_a10_qk-v160_v-std_probe/solution.py`，SHA256 `3D613D035026DD53...`
- 归档基：v164（官方 13945/204s）→ A10/A01；v163（官方 4587/202s）→ W0-W3。
- 端点（复用官方，不重提）：A00 = v162 (1001/146s)；A11 = v164 (13945/204s)。
- 本地 control：逐位：Q/K == v164(v160 路径)；V == v162(standard)。CONTROL FAILURES 0（shard0 全部 attention case）。
- 时间论证：探针只把部分 API 替换为更便宜的 standard codec（或对目标桶保持原路径），最坏工作量 <= 归档基（A: v164 官方 204s；W: v163 官方 202s），官方时间预测余量 > 20s，安全（`T_pred < 280s` 满足）。
- 目的（计划 §7）：P3-A 官方分块贡献辨识；正负均保存，不自动替换根。
