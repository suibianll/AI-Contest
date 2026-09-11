# APG-1：C76.4 的择优粒度本可以**逐 KV 组**，而现在是整层一个（2026-09-11）

**结论：6 层里有 4 层的最优 block 在 KV 组之间不同；逐组择优**不增加任何候选评估**，
预计面板增益 `+0.004 ~ +0.006`。这是本轮 Attention 侧最有价值的发现。**

---

## 为什么逐组 MSE 是**精确且免费**的

评分是逐元素 MSE（`(player − reference)²` 的均值）。attention 输出形状是
`(tokens, q_heads * head_dim)`，**每个元素恰好属于一个 head**，所以

```
mse = (1/(T·qh·hd)) Σ_h Σ_t Σ_d (out − ref)²
```

**按 head 精确可加**，因而是**按 KV 组精确可加**。同一次前向就同时得到整层标量与
逐组分解 —— **分解本身零成本**。

## 读数（`apg1_probe.py`，本地 GPU，gate 窗）

| layer | 各组 argmin block | 当前选中的 block | 整层唯一 block 的组均值 MSE | 逐组最小 | 改善 |
|---|---|---|---|---|---|
| 0 | [32, 64, 64, 16] | 32 | 0.052552 | 0.051507 | **+1.99%** |
| 1 | [32, 64, 32, 16] | 64 | 0.049004 | 0.047707 | **+2.65%** |
| 5 | [16, 64, 64, 32] | 64 | 0.059095 | 0.058111 | **+1.67%** |
| 8 | [64, 64, 64, 64] | 64 | 0.429970 | 0.429970 | +0.00% |
| 15 | [32, 32, 32, 32] | 32 | 0.116500 | 0.116500 | +0.00% |
| 22 | [32, 64, 16, 64] | 32 | 0.171458 | 0.168650 | **+1.64%** |

**六层平均相对 MSE 削减 `+1.32%`。**

## 折算成面板增益

`gain = 1 − mse_player/mse_std`；某层 MSE 相对削减 `r` 会把该层 gain 抬高 `r·(1−gain)`。
以 `gain = 0.5340` 计：

- 若六层都受影响：`0.0132 × 0.466 =` **`+0.0062`**
- 若只有 4/6 层受影响：**`+0.0041`**

**对比**：本轮 Attention 侧我发出的最大增益是 VK `+0.0046`（且官方 `−307`）。
**这一条的同量级、但代价为零。**

## 为什么代价为零

当前搜索**已经**为 6 个候选各跑了一次 `_attention_deployed_mse`；
逐组择优**不新增任何一次前向**，只是把 argmin 从"整层一个"改成"每组一个"。
按 **#43 的决策规则**（成本是确定的、收益是不确定的；面板无符号预测力时只看成本），
**零成本改动进候选池**。

## 实现要点（尚未实现）

1. `_apply_attention_rotation(dense, heads, head_dim, signs, block_size)` 现在吃**单个** `block`
   （`blocks = head_dim // block` 后 reshape）。逐组需要**每 KV 组一个 block**：
   按组切片、各自用本组的 block 做 Hadamard，再拼回。
2. `_build_qk_states(..., rotation_block=...)` 需要能接收逐组 block。
3. `q_state["rotation_block"]` / `k_state["rotation_block"]` 现在是 **int**，
   需改为**长度 kv_heads 的 list**（state 允许 list）。
4. 读它的地方：`solution.py:2975`、`10767`（→ `_nvfp4_to_hif4` 的 `attention_rotation_block`）
   —— 每一处都要能处理逐组形式，且**int 形式必须保持向后兼容**（旧 state 仍要能用）。
5. 择优逻辑：候选通过整层 gate 后，**每组取该组 MSE 最小的候选**；
   整层的 `rotation_mean` 用各组最小值之均值。

## 口径

- 读数为本地 GPU、gate 窗（`calib_qkv_list[-1]`）。**注意**：择优实际发生在
  `_attention_deployed_mse` 的 causal 列表上，本卡读的就是那一列。
- **本地面板对 Attention 无符号预测力（#43）**，所以 `+0.004~0.006` 是**预期**不是保证；
  但**代价为零**这一条与面板无关，是确定的。
- 未实现、未跑面板、未发货。

---

## 实现尝试（2026-09-11，**未发货**）

按上述要点实现了六个字节级 hunk（`apg1_build.py`，每处带唯一性断言与 head/tail 核验）：

1. `_apply_attention_rotation` 支持逐组 block（int 路径逐字节不变，向后兼容）；
2. 新增 `_attention_deployed_mse_grouped`（同一次前向附带逐组 causal MSE）；
3. 择优循环里跟踪 `group_best_mse` / `group_best_row`；
4. 用分组版替换原调用；
5. 末尾按逐组选择组装 state；
6. `_build_qk_states` 的 `int(rotation_block)` 改为接受 list。

`AST OK`，六处 hunk 全部落在目标循环（Loop B，A-TR1 的 block×seed 循环）上。

**面板读数：`delta_mean = 0.0`，12/12 恰好为零 —— 与探针预测不符**（探针说层 0 各组应选
`[32, 64, 64, 16]`）。

**第一层诊断**：直接跑校准并打印 `q_state["rotation_block"]`，三层（0/1/22）**全是 `None`**。
即 `_build_qk_states` 里 `rotation_state is None` —— **C76.4 的旋转没有走 `state["rotation"]` /
`state["rotation_block"]` 这条路径**。所以本实现整条链没有被触达，12/12 零变化由此解释。

**下一步（未做，留给接手者）**：先查清 C76.4 选择的 `signs` 与 `block` 最终**存进了哪个 state 字段、
由哪个函数消费**（线索：`state["rotation"]`、`state["block_smooth_signs"]`、以及 A2 包装器写的
`state["learned_rotation"]` 三者之间谁在部署路径上真正生效）。**在查清之前不要重做本实现** ——
否则会再写一遍一条不生效的链。

**结论：`solution.py` 已回退到 v250（K=6）并核对 SHA。本卡不发货（零增益）。**
