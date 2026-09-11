# v243（A-TF1）A2 训练窗口循环里的循环不变量外提

**类型**：等价 + 提速。**不改变任何输出**，只删掉重复计算。
**父**：v237 根，SHA `ecb1f9e510b5507e1a2dc8b95f8a84e9b51864a420b3828537a328813e2ce554`（516697 B）。
**候选**：SHA `86169fd57b417e73cba307c1d5c8ecb326d480e25cc447f643da2c79cc8d82ea`（517135 B，+438 B）。
**官方**：`unregistered/NA`，未提交。

---

## 改动（单一 hunk）

`_a2_train_rotation` 的窗口循环原为：

```python
for item in prepared:
    c, _right = _m_cayley_pair(theta)                    # (groups, dim, dim) linalg.solve
    rotation = torch.einsum("dk,gkl->gdl", base, c)
    q_rot = _a2_apply_group_rotation(item["q"], q_heads, rotation)
    ...
```

改为把前两行提到循环外。**`theta` 在窗口循环内只读**——它只在本步末尾的 Adam 更新里被写
（`theta = theta - _A2_TRAIN_LR * (...)`，在 `for item in prepared` 退出之后）。因此
`c` 与 `rotation` 对同一 step 内的每个窗口**取值相同**。

每层调用从 `len(prepared) × _A2_TRAIN_STEPS` 次（4×32 = 128）降为 `_A2_TRAIN_STEPS` 次（32）。

## 等价性（先于任何计时，且是唯一的放行条件）

`workbench/full_solution/attention-calib-hoist/atf1_verify.py`（GPU）。

判据是**逐字节相等**，不是容差：state 与动态输出都必须完全相同。
改动只是把两次计算移到输入已固定的位置，所以任何非零都是缺陷。

```
attention layers: [0, 1, 5, 8, 15, 22]   (24 层中其余 18 层是 Linear-only，qkv 槽为 None)
layer   0: state diffs=0 dynamic diffs=0  IDENTICAL
layer   1: state diffs=0 dynamic diffs=0  IDENTICAL
layer   5: state diffs=0 dynamic diffs=0  IDENTICAL
layer   8: state diffs=0 dynamic diffs=0  IDENTICAL
layer  15: state diffs=0 dynamic diffs=0  IDENTICAL
layer  22: state diffs=0 dynamic diffs=0  IDENTICAL

equivalence: 6/6 layers byte-identical
```

动态侧对同一窗口比较 `hif4_dynamic_quantize_q/k/v` 的全部输出字段，同样全零。

## 时间（配对三臂，含同字节 null）

`ATF1_DEVICE=cuda ATF1_ROUNDS=3`，三臂各 18 次调用（6 层 × 3 轮）：

> **2026-09-11 更正（缺陷 #38）**：本卡初版只测了一条**未固定设备**的路径，
> 报 `−1.18%`。补测面板口径（GPU）后，**同一改动是 `−6.17%`**。
> 两个数都真实，省的不是同一种成本（见下）。**下表两条都列。**

**GPU 路径**（张量搬到算法设备，即 `proxy_v3` 面板口径；`atf1-verify-gpu.out`）：

| 臂 | median | min | mean |
|---|---:|---:|---:|
| parent | 4.7823 s | 4.6514 s | 4.7662 s |
| **null**（父模块第二次加载，同字节） | 4.7772 s | 4.6450 s | 4.7539 s |
| **candidate** | **4.4874 s** | 4.3424 s | 4.4817 s |

- null 跨度 **0.0052 s**；候选 − 父 **−0.2950 s（−6.17%）** → **可分辨**

**CPU 张量路径**（初版口径；`atf1-verify.out` 为初测，`atg1-verify.out` 同构）：

| 臂 | median |
|---|---:|
| parent | 18.2882 s |
| null | 18.2342 s |
| **candidate** | **18.0733 s** |

- null 跨度 **0.0540 s**；候选 − 父 **−0.2149 s（−1.18%）** → 可分辨

## 必须写明的口径限制（已按 #38 重写）

1. **相对量随设备翻转，两个数都真实。** A-TF1 省的是 **Python 分发**（循环不变量重算），
   GPU 上占比大 → GPU 路径收益大；A-TG1 省的是**注意力前向**，CPU 上占比大 → CPU 路径收益大。
   **两者互补，且在两条路径上的排序相反。**
2. **官方判题是鲲鹏 920B（CPU），且官方机器上 `torch.cuda.is_available()` 为假**，
   所以基准栈、训练器全部在 CPU 上。本地这两条路径**都不完全等于**官方路径
   （本地跑 CPU 张量路径时，训练器仍会选到 GPU）。
3. **绝对秒数不预测、不外推官方系数。** 本卡只主张：同一台机器、同一条设备路径上，
   候选快于父且差值超过同字节 null。
4. 本卡的定位是**余量**（离线诊断 §6.3 的"先存时间"），不是分数。它自身不产生任何 gain。

## 为什么仍然归档

与 v233（L-TF1）、v237（L-TF2）同类：**逐位等价 + 更快**，且等价性有独立于结论的检查
（逐字节比较）。这两张都拿到官方 `RETAINED`（同分更快）。本卡是这个模式的 Attention 侧
第一个成员。

## 纯追加性核验

`head identical: True` / `tail identical: True` —— 候选 = 父在**单一连续区间**上的替换，
区间外逐字节相同，+438 B。

> **过程中自查出的一个缺陷**：第一版是用编辑器改的，它在写回时把文件尾部 713 行的
> 换行符从 LF 改成了 CRLF（父文件是混合换行）。这违反"父字节一个未动"。
> 已用**字节级替换**重做，并加了 head/tail 核验。见缺陷台账 **#37**。

## 未做

- 未做官方提交。
- 未重跑 Linear 面板（本卡不触碰任何 Linear 路径；上表的 `diff` 显示改动区在
  `_a2_train_rotation`，Linear 侧字节未动）。
- 未测 CPU 时间。
