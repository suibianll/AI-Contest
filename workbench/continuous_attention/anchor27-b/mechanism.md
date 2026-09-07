# A27-B 机制卡（8 字段，R3 出卡，预注册后不再调整）

1. **ID / 侧 / 父 SHA / 创建轮次**：A27-B（实例化 R-6 预注册 next_card）/ Attention /
   A23 `8714ac2a044779465c5e406ef0768be7071ac626f3a2171cdc5083350be92dbf`（官方 14437/276s，
   侧父；候选部署结构沿 A26 线 = v189 基座单主干 + learned_rotation）/ R1 续（循环 R2 候选）
2. **靶点**：F2 = 0.192（OPEN，占 F4=0.468 的 41%）。次大格 F5-V=0.275 为 FAMILY_CLOSED
   （AGENTS §7）已上报用户决策，本轮不取。F1−F2≈−0.002 表明变换选择近零损失，
   QK 侧误差几乎全部在纯码分配 → 靶 F2 正确。
3. **改变什么**：放弃解析残差代理（A26-A 证伪）；S 限制为**低维谱系数参数**——
   每 KV 组一个 8 维系数向量 c_g，S_g = U·diag(band_expand(c_g))·Uᵀ（U = 归一化
   Sylvester/Kronecker 序 Hadamard 256×256，8 个连续谱带各 32 维），exp(±S_g) 闭式
   （U 为共同特征基，无需 eigh）；训练信号 = **真实五字段读出 MSE 的中心差分符号梯度**
   （组切片上完整复现部署链：`_a1_stack_transform` → `_a2_apply_group_rotation` →
   `_dense_to_hif4`（含 state 全部 refine 参数）→ `_dequantize_hif4` → 组内 attention MSE，
   v 固定父 v_state 解码；校准期复杂计算合法，v165 边界只限动态 API）。
   调用图差异仅在校准函数内部：`hif4_calibration_attention` 的训练器替换，部署 API 与
   gate 结构（`_a21_gate_loss`，末折真实读出，严格小于接受，平局保 B）与 A26-A 逐位同构。
4. **为什么可能有效**：A26-A 证明解析独立残差代理与真实自适应层级编码器
   （MSE-optimal lv2/lv3）**系统性反向**（模型 loss 降 20-64% 但 logits qk_mse 恶化 10×），
   同时解释 A25 官方失败；A24 证明全维 STE 梯度恒零（互逆恒等直传）。剩余唯一可信
   梯度源 = 真实编码器读出的差分。低维参数（16-32 个）使差分成本可行（成本探针：
   组切片编码 6.5-8.6ms/折，单层 FD 全程 ≈1.8s proxy；官方 2 KV 组减半）。
   读出 MSE 对 c 是分片常数（码字离散），中心差分在码边界处给出真实方向信号；
   符号梯度对该地形鲁棒（幅度信息本就不可靠）。
5. **固定配置**（全部冻结，不扫参）：
   - 基 U：Sylvester 递归 Hadamard 256（Kronecker 序），归一化 1/16；8 谱带 = 连续 32 维索引
   - ε=0.05（中心差分步长，系数单位）；η=0.05 符号梯度（FD=0 不动）；
     8 步坐标轮换（步 j 更新每组第 j 个系数；组间损失可分，互不影响）
   - 投影 `_a27_project`：平移二分（32 迭代）使 Σc=0（零迹）+ clamp ±log2/2（逐系数，
     语义同 `_a21_project` 的特征值钳制）
   - 训练数据：折 0-3（末折 gate）；每折均匀稀疏索引 token 上限 256
     （`_a2_even_indices`，与旧训练器同源；Q/K 同索引；
     折 0=10、折 1=128 低于上限不变，折 2=512→256、折 3=1024→256）
   - 读出：state 的 importance（_ACTIVATION_SAMPLE_IMPORTANCE=False，按组切片）、
     offsets/error_threshold/accept_margin/max_refine_ratio/max_refine_blocks 全部照传
   - **冻结不动**：V 侧、center、B 全部 state、门（严格小于）、末折、基 U、带数 8、
     ε、η、步数、token 上限；S=0 起步（候选 ≡ B 逐位）
6. **证伪判据**（写入后不调整）：6 层 gate 接受数 < 3/6，或 4B 72 例目标侧
   attention Δmean(vs R3) 未改善（≤0）→ 关闭本卡。gate 全拒时 72 例运行被
   先验截断（gate 拒绝 ⇒ 部署回退 B ⇒ 面板 ≡ 父，no-op）。
7. **去重声明**（四项比对：目标 / 变量 / 插入点 / 编码）：
   - vs **A24**（互逆+STE，REJECTED_BEFORE_EVALUATION）：目标同为 QK 码误差，但 A24
     变量=全维 S 矩阵、梯度=STE（恒等直传恒零）；本卡变量=8 维谱系数、梯度=真实读出
     中心差分（非零）。插入点同（learned_rotation）；编码同（真实编码器）。
     **不等价**（梯度来源+参数维度不同）。
   - vs **A26-A**（锯齿网格残差代理，LOCAL_REJECTED）：目标同（F2），变量=全维 S+
     解析一阶代理 vs 本卡低维+真实读出差分；**A26-A 根因（代理反向）恰好被本卡机制
     排除**。不等价。
   - vs **A23**（scale 乘积代理，官方正向 14437）/ **A25**（amax scale 代理，官方负）：
     二者训练信号为 amax 族解析代理；本卡无 amax 目标、读出即真值。不等价。
   - vs **A2 full-KV 训练**（官方 14440）：A2 训练目标是全长度 KV 的 old-trainer 目标，
     无谱带限制、无差分。不等价。
   - vs **per-call 动态 Gram/自适应精化族、+4 scale 窗口、block-smooth**：插入点与
     变量均不同（本卡不改 sweep/窗口/覆盖率）。不等价。
   - vs **V 侧已关闭族**：本卡不触碰 V。不冲突。
8. **关闭粒度**：失败只关闭"**Kronecker 序 Hadamard 8 谱带系数 + 中心差分符号梯度**
   这一实现"；低维+真实读出差分的其余参数化（如数据驱动特征基、块级对角、前向差分、
   SPSA）与新谱带划分需新证据才可出卡。F2 格本身维持 OPEN。

## 成本与时间风险（预注册时记录，非门禁）

- 成本探针（`probe_cost.json`，L0）：q 切片编码 8.6ms / k 切片 6.5ms（fold3 1024 token）、
  完整 gate 评估 42ms、基座校准 3.2s/层。proxy 4 组 64 次FD 评估/层 ≈ +1.8s/层（256 cap 后）。
- 官方模型 2 KV 组 → 16 参数、32 次评估/层，q 切片 2048 宽 ≈2×；估计净增量
  ≈ +1-2s/FA 层 × ~10 层，相对 A25 结构基线（官方 254s）预计 < 285s。若 TIMEOUT，
  按循环框架只关闭该复杂度实现，机制仍 OPEN。

## 部署状态与回退

- 候选 state = B state + `q_state/k_state["learned_rotation"] = E_q/E_k`（(kh,256,256)，
  E_q(g)=U·diag(exp(+d_g))·Uᵀ，E_k(g)=U·diag(exp(−d_g))·Uᵀ，float32 CPU）。
- gate 拒绝或折数不足 ⇒ 返回 B states（与父逐位一致）；c 全零起步保证 tie 保 B 路径
  在无信号时自动回退（attempted=64、moved=0、code_change=0 会如实记录）。
