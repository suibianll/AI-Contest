# v194 — attn-a2-calibration-fused

## 状态

- 机制：A2/R3 校准等价提速，输出与当前根逐位一致，只消除三处重复计算：
  1. `_a2_train_rotation` 每个 step 的 `_m_cayley_pair(theta)` 与 `rotation = base @ cayley`
     从窗口循环内移到循环外（theta 在 step 内不变）；
  2. 窗口预处理中 `std_v` 与 `v_hat` 共用一次 `_dense_to_hif4(v_sub)` 编码（`v_hat = std_v`）；
  3. `_a2_true_path_gate_loss` 融合为一次调用同时返回 `parent_mse` 与 `candidate_mse`，
     父状态的 Q/K/V 编码与 Attention 前向只执行一次。
- 父 SHA256：`12352efdd4e23cc5e1e17953008664fbaa5ea5d693373635fdafc4d28ce4e24e`
- 候选 SHA256：`1e1d984619d9660a5ce65d0abb738d8fc6e32df4669aafb0252e89a057229dce`
- 固定配置：无任何算法配置变化（训练步数、采样 token、loss、优化器、state 字段、动态 API 全部不变）。
- 官方状态：`REJECTED_TIME`，用户于 2026-09-09 回传 `18032 / 285s`。
- 标准 Linear 侧隔离官方分（2026-09-09 回传）：`14405 / 234s`，相对基线（标准 Linear + R3
  `14405/238s`）**Δscore = 0**（符合"输出逐位等价"的承诺）。但**侧隔离 −4s、完整包 +5s**（285s
  vs 父 280s）方向相反 → 再一次证明侧隔离时间**不能**外推到完整根；该等价提速路线不成立。

## 检查

- `verify_equivalence.py`：PASS。固定 seed 小合成输入下父子 `hif4_calibration_attention`
  返回 state 逐字段逐字节一致（identity 臂与 rotation 接受臂均覆盖），
  `hif4_dynamic_quantize_q/k/v` 输出逐位一致；根全文件无 RNG 调用，融合不改变随机序列；
  脱离仓库单文件六 API 导入通过；`reference_hif4` 合法 state 检查通过。

## 4B shard0

- eval-v3、Qwen3.5-4B proxy-v2、CUDA、Attention-only、shard 0、calibration-cache-mode write：12 cases。
- candidate mean `0.5702418426766733`，父 mean `0.5702418426766733`，paired delta `0`
  （12/12 case 全零，mse ratio 1.0），`reasonableness_issues=0`——真实 4B 数据上逐位等价成立。
- calibration API：父 `6.021268s` → 候选 `4.669476s`（−22.46%）；API total
  `6.398457s` → `5.006078s`。本地时间只作诊断，不换算官方分数/时间。

## 证据位置

- 归档：`solutions/20260908_v194_attn-a2-calibration-fused_scoreNA_timeNA/`
- 评测：`artifacts/proxy_v3/full_solution/attn-a2-calibration-fused-shard0/candidate/manifest.json`
- 工作脚本：`workbench/full_solution/attn-a2-calibration-fused/`

## 官方结果

- **`18032 / 285s`，REJECTED_TIME。** 分数与当前根 `18032 / 280s` 完全相同，但官方时间
  增加 `5s`。本地 calibration API 虽下降 22.46%，没有转化成官方端到端提速，因此 v194 不替换
  当前根，也不作为后续组合的提速实现。
