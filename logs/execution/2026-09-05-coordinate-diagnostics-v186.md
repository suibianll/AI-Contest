# P1 同坐标系误差分解报告（v186，全 24 层）

> 计划 §5。工具：`evaluator/coordinate_diagnostics.py`（新增，独立 CLI，不改官方评测器）。
> 输入：v186 根 `solution.py` SHA `F8495DCA…7EB8`，dense cache `qwen2.5-0.5b-proxy-v2.pt`，
> eval-v3 六 shard（layers 0–23 × both），共 **48 Attention case + 336 Linear case**。
> 产物：`artifacts/proxy_v3/coordinate-diagnostics-v186/run-001/`（shard0）与
> `run-all-1-5/`（shard1–5）。测试：`tests/test_coordinate_diagnostics.py`（5 passed）。

## 1. 正确性验证（金标准）

| 检查 | 结果 |
|---|---|
| Linear `X_tW_t`（同坐标连续乘积）vs 原始 `XWᵀ` | **0.0（全 336 case，机器精度内）**——权重/激活连续镜像精确复现乘积 |
| Attention B/E 分解闭合 | `MSE(O_h,O_ref) == mean(B²)+mean(E²)+2mean(BE)`，最大残差 2.3e-10 |
| 111 臂 / player gain 复现 | 与 `official-audit-smoke3` 已记录 v186 shard0 逐 case gain **逐位一致**（如 layer0 case0 gain 0.9233616316569982） |
| 小代数测试 | FP64 恒等式 + GQA/causal 语义 + 镜像一致性，`5 passed` |

## 2. 主要结果

### 2.1 连续偏差 vs 量化扰动：量化占绝对主导

- **Attention**：`mean(B²) = 7e-7`，`mean(E²) = 2.7e-3`。**量化扰动约为连续 state
  变换输出偏差的 4000 倍**；逐层 B² 在 5e-10～5e-6 区间（见 §2.3），任何层都不超过 E² 的
  1e-3 倍。v186 的 multiplier（含 A1/D1 logits gain）/permutation/rotation/block_smooth/
  pair/K-center 在最终输出空间近似等价于原坐标（连续域几乎零偏差）。
- **Linear**：`X_tW_t == ref` 全 case → 连续项严格保持（解析变换族本身无偏差），
  全部输出误差来自 HiF4 量化。**不存在"误差抵消来自连续坐标改变"的空间**。

### 2.2 单操作数（固定坐标）量化影响

**Attention（mse_to_ref 均值）**：

| 臂 | 含义 | MSE |
|---|---|---|
| 000 | 连续（全浮点） | 7.2e-7 |
| 100 | 仅 Q 量化 | 1.11e-3 |
| 010 | 仅 K 量化 | 1.25e-3 |
| 001 | 仅 V 量化 | 4.50e-4 |
| 110 | Q+K 量化 | 2.32e-3 |
| 111 | Q+K+V（player） | 2.74e-3 |

Q、K 各自量化影响约为 **V 的 2.5~2.8 倍**；`110 ≈ 100 + 010`（交互 ≈ 0，无抵消也无放大）。

**Linear（mse_to_ref 均值，336 case）**：

| 臂 | MSE | 说明 |
|---|---|---|
| X_hW_t（仅激活量化） | 1.08e-3 | |
| X_tW_h（仅权重量化） | **2.03e-3** | 权重静态编码误差约为激活的 1.9 倍 |
| X_hW_h（player） | 3.10e-3 | |

按 role family：proj 权重侧误差是激活侧的 **4.4×**（1.81e-3 vs 0.42e-3）；
qkv 2.4×；o 2.0×；fc 1.2×。shape：hidden_to_wide（fc 类）绝对误差最大（3.9e-3）。
**交互项**（`X_hW_h − X_hW_t − X_tW_h`）mean −8.8e-6 → Linear 两侧量化误差基本**可加、无抵消**。

### 2.3 逐层分布（Attention，量化 E² 随层增大）

浅层（L0–1）E² ≈ 1e-5～7e-4；深层（L21–23）E² ≈ 4e-3～8e-3。量化误差从浅到深
单调增长约 2 个数量级；最差层 L21（7.5e-3）。B² 无同类趋势（始终 ≪ E²）。

## 3. P1 结论（供 P2/P4 使用）

1. **同坐标量化误差分解成立**：诊断链正确（X_tW_t==ref、B/E 闭合、player 复现）。
2. **误差几乎全部是纯量化扰动**：v186 的连续变换族已经饱和到输出零偏差；官方差的
   4166 分如需缩小，只能通过减少**量化扰动**，不是继续找连续坐标等价变换。
3. Attention 内部分配：**Q/K 是 V 的 2.5× 主战场**；Q/K 与 V 可加（无遮挡交互）。
   V 的量化几乎免费 → 符合 v186 V state 不做变换的现状。
4. Linear 内部分配：**权重静态编码误差 > 激活动态编码（1.9×）**，其中 proj（wide 权重
   编码后重）与 qkv 最重；两侧误差近似可加。与"ms_E_W 元素级很小、但输出级误差大"
   一致：静态权重在 64 维点积上放大了编码残差，属表示质量问题而非 scale 尺度问题。
5. 纵深：深层（≥L16）量化误差约浅层的 10–100 倍 → 任何新机制若想官方可见，优先瞄准
   深层 Q/K 与深层权重侧；浅层余量已经很低。

## 4. 边界与后续

- 本诊断使用 eval-v3 分片窗口（每层 2 个 test window / 5 个长度中的 2 个），不是 default
  120/168 panel 的全窗口；结论是分布性质，不是官方分数预测。
- 诊断计时未记录（API 时间仍由正常评测流程承担）；不代入官方时间模型。
- P2 将基于同一坐标系放宽格式约束（R0–R3），定位"量化扰动中哪些字段约束贡献最大"。
