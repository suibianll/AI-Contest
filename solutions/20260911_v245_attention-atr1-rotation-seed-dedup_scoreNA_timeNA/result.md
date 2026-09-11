# v245（A-TR1）旋转候选循环的重复 seed 去重

**类型**：等价 + 提速。**不改变任何输出**，只跳过**重算同一个候选**的迭代。
**父**：v244 根，SHA `cdb9a823c6fe754e1d66ab0cfa7c9638dbf804efdc4b33e41fcbb02432a5686f`（518506 B）。
**候选**：SHA `738d2b3617a3fa287524c9f9558a0a43b3af11076bc94a00d0e38a112e45521f`（519334 B，+828 B）。
**官方**：`unregistered/NA`，未提交。

---

## 之前不知道的一件事：这张卡是从剖面里找出来的

本轮先做了校准的耗时剖面（`workbench/full_solution/attention-base-profile/abp1_profile.py`）：

| 量 | 读数（CPU 路径，layer 0） |
|---|---|
| 整个 `hif4_calibration_attention` | 17.27 s |
| `_dense_to_hif4` 合计 | **13.10 s（75.9%）** |
| 其中被 `hif4_dynamic_quantize_q/k` 调用 | **10.33 s（181 次）** |
| 其中 `_attention_deployed_mse:2891` 一处 | **6.95 s（74 次）** |

再往下一层的**决定性检验**：给每次 `_attention_deployed_mse` 调用做
**`(q_state, k_state)` 内容哈希**并记录调用行号：

```
distinct state pairs and the lines that evaluate them:
   bbacfce4 c8c883db  x3   lines [10513]  <<< REDUNDANT
   cf05eb22 cf510541  x3   lines [10513]  <<< REDUNDANT
   a2d94074 91ff715e  x3   lines [10513]  <<< REDUNDANT
   7c3d7161 86c2dcde  x2   lines [10217, 10647]  <<< REDUNDANT
   ...
REDUNDANT calls (same state pair recomputed): 7 of 16
```

**16 次调用里 7 次是在重算已经算过的状态对。**

## 根因

```python
for block_size in _ATTN_ROTATION_BLOCKS:      # (16, 32, 64)
    ...
    for seed in _ATTN_ROTATION_SEEDS:         # (0, 1, 2, 3)
        signs = _attention_rotation_signs(kv_num_heads, head_dim, int(seed))
```

**`_attention_rotation_signs` 只吃 seed，不吃 block_size**，却被放在 `block × seed` 的双重循环体里。
实测该形状下：

```
seed -> signs sha256 前 8 位:  {0: '6c4b415a', 1: '6c4b415a', 2: '6c4b415a', 3: '5bb2a9c6'}
```

**seed 0/1/2 产生完全相同的 sign 向量**，只有 seed 3 不同。于是每个 block 把同一个候选
**重算了三遍**。

## 为什么跳过是精确的

1. **候选 state 是 `(signs, block)` 的函数，不是 seed 的函数**——
   `_build_qk_states(..., rotation=signs, rotation_block=block)` 的入参里没有 seed。
   相同 signs + 相同 block ⇒ **逐位相同**的 `rotation_q_state/rotation_k_state`。
2. 择优是**严格 `<`**：`if rotation_mean < best_rotation_mean`。
   重复迭代的分数**完全相同**，因此**永远不可能赢**，跳过不改变 `best_rotation_states`。

## 改动

在 `for block_size` 内维护 `distinct_signs`，signs 与已见者相同则 `continue`。
`(16,32,64) × (0,1,2,3)` 的 12 次迭代降为 **6 次**。

## 等价性（唯一的放行条件，逐字节、不设容差）

**(1) GPU 路径 —— 6/6 逐字节相同**（`verification-gpu-equiv.out` / `verification-gpu-timing.out`）

```
layer   0: arm=rotation state diffs=0 dynamic diffs=0  IDENTICAL
layer   1: arm=rotation state diffs=0 dynamic diffs=0  IDENTICAL
layer   5: arm=rotation state diffs=0 dynamic diffs=0  IDENTICAL
layer   8: arm=identity state diffs=0 dynamic diffs=0  IDENTICAL
layer  15: arm=rotation state diffs=0 dynamic diffs=0  IDENTICAL
layer  22: arm=rotation state diffs=0 dynamic diffs=0  IDENTICAL
equivalence: 6/6 layers byte-identical
```

**(2) CPU 张量路径 —— 同样 6/6 逐字节相同**，`arm` 分布相同（层 8 = identity）。

动态侧比较 `hif4_dynamic_quantize_q/k/v` 的全部输出字段，同样全零。

## 时间（配对三臂，含同字节 null）

**GPU 路径**（面板口径）：

| 臂 | median | min | mean |
|---|---:|---:|---:|
| parent | 4.4156 s | 4.2428 s | 4.3976 s |
| null（同字节） | 4.3920 s | 4.2669 s | 4.3838 s |
| **candidate** | **3.7047 s** | 3.6732 s | 3.7147 s |

- null 跨度 **0.0236 s**；**候选 − 父 = −0.7108 s（−16.10%）**，是噪声底的 **30.2 倍**。

**CPU 张量路径**：见 `verification-cpu-timing.out`（同一次作业的第二段）。

## 口径（沿用缺陷 #38 的纪律）

- 两条设备路径都测，**相对量随设备不同，两个都真实**。
- 官方是鲲鹏 920B CPU 且该机 `cuda.is_available()` 为假；本地两条路径**都不完全等于**官方路径。
- **不预测官方秒数、不换算系数。** 只主张：同机同路径下候选更快且超过同字节 null。

## 三张卡的合计（各自相对自己的父，不可相加）

| 卡 | 省的东西 | GPU 路径 | CPU 张量路径 |
|---|---|---:|---:|
| v243（A-TF1） | 循环不变量重算（Cayley solve） | −6.17% | −1.18% |
| v244（A-TG1） | gate 的两个冗余全窗臂 | −1.03% | −6.31% |
| **v245（A-TR1）** | 旋转候选的重复 seed | **−16.10%** | 见输出 |

## 纯追加核验

单一 hunk，区间外逐字节相同：

```
outside-the-hunk segments identical: True
segment lengths parent: [429381, 88920]   cand: [429381, 88920]
parent 518506 B -> candidate 519334 B (+828)
```

## 未做

- 未提交官方。
- 未做面板 0-delta 复核（v244 做过；本卡尚未）。
- 未做纯 CPU（`cuda.is_available() == False`）仿真。
- **第二个冗余点未处理**：lines `[10217, 10647]`（winner 与 parent）是同一个状态对，算了两次。
  量级比本卡小，留作下一张。
