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

---

## 前置问题已查清：C76.4 的选择**不走** `state["rotation"]`

这是上一节要求的"查清之后再做"的那件事。做法：直接跑校准，转储 `q_state` 的全部字段。

**三层（0 / 1 / 22）一致：**

| 字段 | 形状 | 是什么 |
|---|---|---|
| `learned_rotation` | (4, 256, 256) | **A2 包装器**训练出的正交矩阵 |
| **`block_smooth_signs`** | **(4, 256)** | **正是 `_attention_rotation_signs` 的输出形状** |
| `logit_gain` | (4,) | A1（v168）机制 |
| `pair_transform` | (16, 128, 2, 2) | 成对变换 |
| **`rotation`** | **`None`** | ← 未启用 |
| **`rotation_block`** | **`None`** | ← 未启用 |

**结论：C76.4 选的 `signs` 与 `block` 走的是 `block_smooth_size` / `block_smooth_signs`，
由 `_block_hadamard_transform` 应用；`state["rotation"]` / `state["rotation_block"]`
（即 `_apply_attention_rotation` 那条路，`solution.py:3026`）在部署路径上**是空转的旧路**。**

所以上一节的六处 hunk **打在了不生效的链上** —— 12/12 零变化由此**完全解释**，
不需要再怀疑实现细节。

## 重做的正确做法（未做）

1. `_block_hadamard_transform(dense, block_size: int, seed=0)` 现在吃**单个** size；
   逐组需要**每 KV 组一个 size**，或按组切片分别调用后拼回。
2. state 的 `block_smooth_size` 由 **int 改为长度 kv_heads 的 list**（`block_smooth_signs`
   已是 (kv_heads, head_dim)，逐组 signs 天然支持）。
3. 消费点：`solution.py:2978`（动态端读 `block_smooth_signs`）、`10768`、`10802`
   （`attention_block_signs=state.get("block_smooth_signs")`）。
4. **仍未验证的一环**：C76.4 的择优循环里，`signs` 与 `block` 究竟在**哪一行**被写入
   `block_smooth_size` / `block_smooth_signs`（候选是每层一个 (block, seed)，
   而 state 里两项都在 `_build_qk_states` 内写入）。**重做前先把这一行找到。**

**`solution.py` 保持 v250（K=6），未改动。**

---

## 插桩结果：两个事实互相矛盾（**未解决，不要据此改代码**）

对 `_apply_attention_rotation` 与 `_block_hadamard_transform` 同时插桩，跑 layer 0：

```
during CALIBRATION (layer 0): {'apply_rotation': 358}      block_hadamard: 一次未调用
  apply_rotation 的 block 实参样例: [4, 4, 4, 4, 4, 4]
during DYNAMIC q/k:            {'apply_rotation': 2}       block_hadamard: 一次未调用
  apply_rotation 的 block 实参样例: [8, 8]
```

**事实 A（支持"打对了函数"）**：部署路径**确实**调用 `_apply_attention_rotation`，
`_block_hadamard_transform` 在部署端**一次都没被调用**。
所以上一节"应该改 `_block_hadamard_transform`"的结论**是错的**，撤回。

**事实 B（与转储矛盾）**：转储 `q_state` 显示 `rotation_block = None`；
而插桩显示动态调用时 block 实参是 **8**（即 `state.get("rotation_block")` 非 None）。
**同一个字段、同一层、两次运行给出不同结果** —— 这个矛盾**没有解决**。

**可能的解释（都未验证，不要采信）**：
- 转储那次与插桩那次读的不是同一个 state 对象（例如包装器在返回后又写了一轮）；
- 或 `block_size` 的 8 来自 `_apply_attention_rotation` 内部对 `_ATTN_H64_BLOCK` 的默认，
  而我的插桩把默认值也记录了（**最可疑**：插桩打印的是"实参"，`None` 会被打印成 `None`，
  但打印出的是 8，所以这条解释站不住，除非另有调用点**）；
- 或两次运行的 `hif4_calibration_attention` 走了不同分支（本地面板对 Attention 有已知的
  设备/路径敏感性，见缺陷 #38）。

## 结论：**停下来，先解决矛盾**

**不解决这个矛盾就继续改代码，等于在没有确认目标的情况下再写一遍。**
本轮已经出现过一次"打在空转的链上"（12/12 零变化），**第二次同类错误应当避免**。

**下一步（唯一一件事）**：写一个探针，在**同一次运行**里
①调用 `hif4_calibration_attention`，②立刻转储 `q_state` 的 `rotation` / `rotation_block`，
③再调用 `hif4_dynamic_quantize_q` 并打印传进 `_apply_attention_rotation` 的 block。
**三者必须在同一次运行里读取**，才能判断 A/B 矛盾是"两次运行不同"还是"读取点不同"。

**`solution.py` 未改动（v250 / K=6）。本卡仍未发货。**
