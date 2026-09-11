# v244（A-TG1）A2 gate 的 standard 臂复用

**类型**：等价 + 提速。**不改变任何输出**，只删掉重复计算。
**父**：v243 根，SHA `86169fd57b417e73cba307c1d5c8ecb326d480e25cc447f643da2c79cc8d82ea`（517135 B）。
**候选**：SHA `cdb9a823c6fe754e1d66ab0cfa7c9638dbf804efdc4b33e41fcbb02432a5686f`（518506 B，+1371 B）。
**官方**：`unregistered/NA`，未提交。

---

## 改动

`hif4_calibration_attention` 的 gate 段原本调用 `_a2_true_path_gate_loss` **两次**，
每次内部各跑**两个**全窗臂（player + standard），合计 **4 个全窗臂**：

```python
loss_identity = _a2_true_path_gate_loss(gw, states, None,     device)                    # 2 臂
loss_rotation = _a2_true_path_gate_loss(gw, states, rotation, device, center)            # 2 臂
```

两件事使其中 **2 个臂是多余的**，且都可证明：

1. **identity 臂的 player 与 standard 是同一个计算。** `rotation=None` 时
   `player_k_state = dict(k_state, learned_rotation=None)`，而判据是
   `if learned_rotation is not None`（`solution.py:4269`）——显式 None 与缺键同义。
2. 因此 `loss_identity == x / max(x, 1e-12)`，`x` 即 standard 臂。

**改动**：把内层 `_run` 闭包提为模块级 `_a2_gate_arm_mse`，`_a2_true_path_gate_loss` 增加
可选参数 `standard_mse`；gate 段只算 **2 个臂**（standard 一次 + rotation 一次）。

## 先决测量（改之前就做了，`atg1_probe.py`）

| 量 | 读数 |
|---|---|
| 六层的 `loss_identity` | **全部恰好 `1.0`**（bit-equal） |
| identity 臂 player_mse vs standard_mse | **六层逐位相同** |
| gate 占 A2 块的比例（CPU 路径剖面） | **65.9%** |
| 层 8（rotation 输掉 gate 的那层） | `loss_rotation=1.0286 > 1.0` → identity |

层 8 是唯一会让比较翻转的层，所以它是等价性的**关键检验点**。

## 等价性（唯一的放行条件，逐字节、不设容差）

三处独立检查：

**(1) 进程内逐字节比较 —— GPU 路径**（`atg1-verify-gpu.out`）

```
layer   0: arm=rotation state diffs=0 dynamic diffs=0  IDENTICAL
layer   1: arm=rotation state diffs=0 dynamic diffs=0  IDENTICAL
layer   5: arm=rotation state diffs=0 dynamic diffs=0  IDENTICAL
layer   8: arm=identity state diffs=0 dynamic diffs=0  IDENTICAL
layer  15: arm=rotation state diffs=0 dynamic diffs=0  IDENTICAL
layer  22: arm=rotation state diffs=0 dynamic diffs=0  IDENTICAL
equivalence: 6/6 layers byte-identical
```

**(2) 同一比较 —— CPU 张量路径**（`verification-cpu-path-timing.out`）：同样 6/6 逐字节相同，
`arm` 分布相同（层 8 = identity）。

**(3) 面板 shard0**（`panel-shard0-manifest.json`，`--attention-only --shards 0` 对 v243 根）：
**12/12 逐例 delta 恰好为 0**（`positive_cases 0 / negative_cases 0 / zero_cases 12`）。

动态侧对同一窗口比较 `hif4_dynamic_quantize_q/k/v` 的全部输出字段，同样全零。

## 时间（配对三臂，含同字节 null）

**关键口径问题：相对量随设备翻转，两个数都真实。** 详见下方"口径"。

| 路径 | parent | null（同字节） | candidate | null 跨度 | Δ | 倍数 |
|---|---:|---:|---:|---:|---:|---:|
| **GPU（面板口径）** | 4.4667 s | 4.4772 s | **4.4208 s** | 0.0105 s | **−0.0459 s（−1.03%）** | 4.4× |
| **CPU 张量路径** | 18.3155 s | 18.3088 s | **17.1598 s** | 0.0067 s | **−1.1557 s（−6.31%）** | 171.9× |

两者都"可分辨于噪声底"。**官方判题是鲲鹏 920B（CPU），所以 CPU 列是更相关的一列。**

## 口径（必须写明，这一条本卡踩过坑）

**本地"GPU 路径"与"CPU 张量路径"给出的相对量差了 6 倍，方向还相反（见 v243 的对照）。**
原因是两者省的**不是同一种成本**：

- A-TG1 省的是**全窗注意力前向** → CPU 上占比大 → CPU 路径收益大；
- A-TF1 省的是**Python 分发** → GPU 上占比大 → GPU 路径收益大。

**这张卡第一版把 CPU 路径的数（−6.31%）当成"虚高"去更正，那是错的**：
存在两个都真实的配置，原探针的问题只是**没有固定设备**，量出的是一个
**既不是面板的、也不是官方的混合路径**。已记入缺陷台账 **#38**。

**官方机器上 `torch.cuda.is_available()` 为假**，所以基准栈、训练器、gate **全部**在 CPU 上。
本地这两条路径都**不完全等于**官方路径：本地跑 CPU 张量路径时，训练器仍会因
`torch.cuda.is_available()` 为真而选到 GPU。因此：

- **不预测官方秒数、不换算本地→官方系数。**
- 本卡只主张：**在同一台机器、同一条设备路径上，候选与父的差可分辨，且方向为更快。**

## 三处替换的纯追加核验

`build_v244.py` 用**字节级替换**构建，每处都带"父字节中唯一匹配"断言与 head/tail 逐字节核验；
另有切段比较证明**三处 hunk 之外逐字节相同**：

```
outside-the-hunks segments identical: True
segment lengths (parent): [469792, 79, 1507, 43516]
segment lengths (cand)  : [469792, 79, 1507, 43516]
parent 517135 B -> candidate 518506 B (+1371)
```

## 与 v243 的关系

两张是**互补**的两刀，都在 Attention 校准路径上、都逐字节等价：

| | 省的东西 | GPU 路径 | CPU 张量路径 |
|---|---|---:|---:|
| v243（A-TF1） | 循环不变量重算（Cayley solve） | **−6.17%** | −1.18% |
| v244（A-TG1） | gate 的两个冗余全窗臂 | −1.03% | **−6.31%** |

**不可相加**（各自相对自己的父测的）；但方向一致：**两条路径上都更快**。
v244 的父是 v243。

## 未做

- 未提交官方。
- **未做纯 CPU（`torch.cuda.is_available() == False`）的仿真测量** —— 那是最接近官方的一条路径，
  本机尚未测。**这是本卡最该补的一条。**
- 未测 Linear 侧（本卡不触碰任何 Linear 路径）。
