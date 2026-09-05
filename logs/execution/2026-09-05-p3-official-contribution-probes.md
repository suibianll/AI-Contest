# P3 官方贡献探针：构造、本地 control 与待提交清单

> 计划 §7。2026-09-05。生成器 `workbench/p3_build_probes.py`，逐位控制
> `workbench/p3_local_control.py`。6 个单文件探针位于 `solutions/20260905_p3*`。
> P3-B（A1 长度桶）记 **DESIGN_BLOCKED**，理由见 §3。官方提交待平台入口执行。

## 1. P3-A 配对 QK / V 路径效应（2 个新提交）

固定 standard Linear；端点复用：A00 = v162（官方 1001/146s）、A11 = v164（官方 13945/204s）。

| 探针 | Q/K | V | 构造（从 v164 = v160 Attention + std Linear） | 本地逐位 control |
|---|---|---|---|---|
| A10 | v160 | **std** | 仅追加重定义 `hif4_dynamic_quantize_v` → standard | Q/K == v164、V == v162；0 失败 |
| A01 | **std** | v160 | 仅追加重定义 `hif4_dynamic_quantize_q/k` → standard | Q/K == v162、V == v164；0 失败 |

端点自检：A10 的未改侧（Q/K）与 v164 逐位一致；A01 的未改侧（V）与 v164 逐位一致；
standard 侧与 v162 逐位一致（shard0 全部 attention case）。

## 2. P3-C Linear 形状桶（4 个新提交）

固定 standard Attention；端点：非目标桶 = v162 standard，目标桶 = v163 v160 Linear。
从 v163 构造（= v160 Linear + standard Attention，官方 4587/202s）。桶互斥完备已用本地
全部 weight shapes 审计：`{(128,896),(896,896),(896,4864),(4864,896)}` 覆盖无重叠。

| 探针 | 桶条件 | 本地落入（shard0） | control |
|---|---|---|---|
| W0 | rows<=256 | k/v（rows 128） | 目标 8 state 非空 / std 20 空且 == v162 |
| W1 | rows>256 且 0.75<=rows/cols<=1.33 | q/o（896/896） | 同上 |
| W2 | rows>256 且 rows/cols>1.33 | fc_gate/fc_up（4864/896） | 同上 |
| W3 | 其余（rows/cols<0.75） | proj（896/4864） | 目标 4 / std 24 |

全部 CONTROL FAILURES 0。

## 3. P3-B A1 长度桶 —— DESIGN_BLOCKED

官方调用合同（`赛事说明书.txt`）：`hif4_dynamic_quantize_q/k/v` 是三次**相互独立**的 API
调用，入参仅 `(quant, scale, num_heads, head_dim, state)`；K 的 shape 甚至只写"接口类似"。
**没有任何跨调用场景键**（无 layer_id / case_id / 通信通道）。长度桶若按各自
`tensor.shape[0]` 独立切换，只有当官方保证 Q_len==K_len 才可能对齐；合同未排除 Q_len≠K_len
（decode 场景）。在无共同键下按本地长度切换不满足"同逻辑场景同一桶"要求 → 按计划
§7.3 记 `DESIGN_BLOCKED`。核心矩阵相应为 6 个新提交（2 + 4），已全部构造。
补充：A1 官方正向（v168 相对 v164 +60）已在整体层面确认；长度维度内部分配留待官方数据或
其他信息渠道。

## 4. 时间安全性

每个探针只把部分 API **替换为更便宜/等价的 standard codec**（或对目标桶保持原 v160 路径，
与归档基相同），未新增任何在线算子或校准量：最坏工作量 <= 归档基官方实测
（A 组 <= v164 = 204s；W 组 <= v163 = 202s）。父版本既有官方计时直接复用作为上界，
`T_pred < 280s` 满足；不提交前不需要额外 default 新鲜计时（无新增计算路径）。

## 5. 待办（用户/平台侧）

- 将 6 个探针依次提交官方评测（建议一次一批，独立结果记账）。
- 官方回传后按总账 §9.2 判定：PROBE_COMPLETE 正负均保存；C_QK = S10 − S00、
  C_V = S01 − S00、I = S11 − S10 − S01 + S00；C_Wj = S_Wj − 1001、R = 3586 − ΣC。
