# A-C76.5 残差定向 C76.4 正交候选 — 关闭记录（2026-09-09）

- run_id: `attention-c765-residual-directed-rotation`
- parent: 根 `solution.py` SHA256 `56dc805d6e5a3aef896db8021045740292735725d688b48e3d4393e55efcb2bd`（未修改）
- candidate: `candidate/solution.py` SHA256 `98c41860cc79cd691ba1c1daa3d245223280d4afd164a0cc923f31e8f450cb83`
- 结果：**`CLOSED / NO_EFFECT`（NOOP）。六层全部与根逐位相同，残差候选从未被 deployed-MSE 选择选中。未分配正式版本，未提交官方。**

## 机制与实现

- C76.4 的 H16/H32、原候选和部署逻辑保持不变（v205 定价 +84 分，必须保留）。
- 新增：在 C76.4 搜索的父状态用现有 Attention backward 得到 `dQ_hat/dK_hat`，按 KV group 累计
  `G_g = Σ(Q_g^T dQ_g + K_g^T dK_g)`（Q_g/K_g 为 C76.4 符号插入点的 dense，case 等权平均）。
- 对 `B∈{16,32}`，把 `G_g` 切成连续 `B×B` 对角块，`C=(G_blk+G_blk^T)/2`，取 `|eigenvalue|`
  最大的特征向量（并列取较小索引，首非零元固定为正），符号 `s_j=+1 (u_j>=0) / -1`，拼成
  `[kv_heads, head_dim]` signs；用现有 `x -> (x*signs)@H_B` 生成候选，无 permutation、无 seed/阈值搜索。
- 两个新候选与既有 C76.4 候选一起走完整 deployed-MSE 选择。

## 验证（`verify.py`，CUDA）

- 残差 signs 合法（±1）、shape 正确；合成耦合上不与 4 个 seed 逐位重复。
- 签名 Hadamard 保持浮点 QK 内积不变：max 误差 `1.91e-06`。
- 六 API 导入、`reference_hif4.validate_state`、动态 Q/K/V 有限输出全部通过。

## 六 shard 结果（eval-v3 / 4B / attention-only / CUDA，`--stop-after-nonpositive 6`）

| shard | 残差候选 duplicate | 选中 | paired delta_mean |
|---:|---|---:|---:|
| 0 | block16/32 = 0/0 | 否 | `0` |
| 1 | 0/0 | 否 | `0` |
| 2 | 0/0 | 否 | `0` |
| 3 | 0/0 | 否 | `0` |
| 4 | 0/0 | 否 | `0` |
| 5 | 0/0 | 否 | `0` |

- 候选 attention mean `0.533998` = 基线 `0.533998`；六 shard 全部逐位相同。
- 残差候选在每一层都**可达且非重复**，但从未在 deployed-MSE 选择中胜过既有 seed 候选，
  因此部署输出零变化。

## 关闭依据

计划 §5 预注册规则：「六层全部重复或全部不被选择时关闭该机制，不继续换 eigensolver、seed、
block size 或符号规则」。六层全部未被选择 → 关闭“残差定向 C76.4”机制。

## 产物

- `c765_code.py`、`build.py`、`verify.py`、`config.json`、`verification.json`、`candidate/solution.py`
- 六 shard：`artifacts/proxy_v3/c765-sixshard/`；shard0：`artifacts/proxy_v3/c765-shard0/`
