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

## 6. 官方回传与判定（2026-09-05 收到）

| 探针 | 官方 | C（相对 S00=1001） | 判定 |
|---|---|---|---|
| A10（Q/K v160，V std） | **12010 / 203s** | C_QK = **11009** | PROBE_COMPLETE |
| A01（Q/K std，V v160） | **2974 / 175s** | C_V = **1973** | PROBE_COMPLETE |
| W0（rows≤256：k/v） | **1001 / 149s** | C_W0 = **0** | PROBE_COMPLETE（空桶效应） |
| W1（方形：q/o） | **1001 / 147s** | C_W1 = **0** | PROBE_COMPLETE（空桶效应） |
| W2（fc 大矩阵） | **2819 / 195s** | C_W2 = **1818** | PROBE_COMPLETE |
| W3（proj wide） | **2768 / 162s** | C_W3 = **1767** | PROBE_COMPLETE |

- P3-A 交互：`I = S11 − A10 − A01 + S00 = 13945 − 12010 − 2974 + 1001 = −38`（≈0；
  Q/K 与 V 路径近似可加）。C_QK+C_V+I = 12944 = v160 Attention 总贡献，闭合。
- P3-C 残差：`R = 3586 − (0+0+1818+1767) = +1`；形状桶近似可加且**全覆盖**。
- 全部时间 < 210s，无超时，低于归档基（v164 204s / v163 202s）上界成立。

### 解读

1. **Attention 官方增益高度集中在 Q/K 路径**（11009 / 12944 ≈ 85%），V 路径仅 1973
   （≈15%），且二者交互 −38≈0。与 P1 同坐标诊断完全一致（Q/K 单侧量化误差 ≈ V 的
   2.5–2.8× → 量化收益也主要落在 Q/K）。
2. **v160 Linear 的全部官方增益落在 W2(fc) + W3(proj) 两个大形状桶**（1818+1767 =
   3585 ≈ C_L 3586）；rows≤256（k/v）与方形（q/o）桶为**精确零**。这证明 v160 Linear
   机制对 hidden_to_hidden 与小输出权重不带来任何官方改进——即使这些桶在 P1 的
   "输出 MSE 绝对值"上不小（qkv 2.5e-3），standard 相对 v160 已无可改进去空间。
3. 方向性结论（供未来机制选择）：Attention 侧新机制若存在，只应作用于 Q/K（V 无关紧要）；
   Linear 侧只应在 fc/proj 这类 expansive/wide 形状上找增量，q/k/v/o 与窄小权重桶不再
   注册任何候选。

## 7. P4/P5 状态

P2 已判 `NO_SUPPORTED_MECHANISM`；P3 官方证据没有产生新的可编译合法机制构想，只给出
"收益形状/路径分布"结构性结论。因此 P4 保持 CONDITIONAL 且不启动新候选，P5 无对象；
本轮计划以诊断与探针证据结束，根 `solution.py` 保持 v186（`F8495DCA…7EB8`）。

