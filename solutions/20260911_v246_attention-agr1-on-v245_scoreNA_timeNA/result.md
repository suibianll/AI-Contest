# v246（A-GR1 ∘ A-TF1/TG1/TR1）把官方唯一正分的互逆族挂到带余量的根上

**类型**：**分数候选**（不是余量卡）。机制是 A-GR1，父是 v245。
**父**：v245 根，SHA `738d2b3617a3fa287524c9f9558a0a43b3af11076bc94a00d0e38a112e45521f`（519334 B）。
**候选**：SHA `0cc6119a2149b343008ffa8f4302ab8b262b92967d03c9432b2f696bfe0f1edc`（534530 B，+15196 B）。
**官方**：`unregistered/NA`，**待提交**。

---

## 为什么做这一张

A-GR1 是本项目 **Attention 侧唯一有官方正分记录、且只输在时间上**的机制：

| 记录 | 值 |
|---|---|
| 侧隔离官方（`standard-linear_v234-attn`） | **`14455 / 263.7s`**，相对 R3 基线 `14405` 为 **+50**；相对 v195 `14426` 为 **+29** |
| 完整包 v234 | **TIMEOUT**（父 v230 官方 292 s，余量 8 s） |
| 完整包 v236（重挂 v231 根） | **TIMEOUT**（父余量 9 s） |

v236 的记录写明："父差仅 1s，故两次超时构成双向封闭区间——**不存在'换个 Linear 父就能过'的空间**"。
**那条结论的前提是"父的成本不变"。** 本轮的三张卡改变了这个前提：它们在**不改变任何输出**的
前提下把一次 Attention 校准调用压下 ~22%（GPU 路径）/ ~29%（CPU 张量路径）。

**所以这张卡就是在测那个前提还成不成立。**

## 合成方式（机械的）

`diff -u v231 v236` 是**单一 hunk** `@@ -12323,3 +12323,409 @@` —— A-GR1 是一次**纯末尾追加**：
只定义 `_AGR1_*` 常量与函数，然后

```python
_AGR1_PARENT_CALIBRATION = hif4_calibration_attention   # 别名捕获"最后定义"
...
def hif4_calibration_attention(...):                    # 覆盖式包装
    states = _AGR1_PARENT_CALIBRATION(...)              # 先跑根自己的校准
    ...                                                 # 再做 A-GR1
```

所以把它接到 v245 末尾即可，别名会捕获 **v245 的 A2 包装器**（内含 A-TF1/A-TG1/A-TR1），
A-GR1 就在优化过的根之上训练。

**核验**：候选 = v245 的全部字节 + 追加块，**其余逐字节未动**（公共前缀 519333 B）。

## 等价性：合成后的 Attention state 必须与 v236 逐字节相同

**这是本卡最强的检验，而且可证伪。** A-GR1 的父在 v236 是 v231 的 A2 包装器、在 v246 是 v245 的
A2 包装器，而后者经三张卡验证与前者**逐字节等价**。所以 A-GR1 在两边看到**完全相同的父**，
合成后的 state 必须逐字节相同 —— 任何差异都是合成缺陷。

```
layer   0: agr1_arm=parent    state diffs=0 dynamic diffs=0  IDENTICAL to v236
layer   1: agr1_arm=parent    state diffs=0 dynamic diffs=0  IDENTICAL to v236
layer   5: agr1_arm=parent    state diffs=0 dynamic diffs=0  IDENTICAL to v236
layer   8: agr1_arm=parent    state diffs=0 dynamic diffs=0  IDENTICAL to v236
layer  15: agr1_arm=accepted  state diffs=0 dynamic diffs=0  IDENTICAL to v236
layer  22: agr1_arm=accepted  state diffs=0 dynamic diffs=0  IDENTICAL to v236

composed states identical to v236: 6/6
```

`agr1_arm` 分布（层 15/22 接受、其余保持父）与 v234/v236 的记录一致。
动态侧 `hif4_dynamic_quantize_q/k/v` 的全部输出字段同样全零。

## 时间：三张卡还回来多少

`v246_timing.py`，**CPU 张量路径**，`rounds=1`，每臂 6 次（每层一次）：

| 臂 | median | min |
|---|---:|---:|
| **v236**（同 A-GR1 块，父无三张卡） | 22.5724 s | 22.5032 s |
| null（v236 同字节二次加载） | 22.6659 s | 22.5443 s |
| **v246**（父含三张卡） | **17.0688 s** | 16.9679 s |

- null 跨度 **0.0934 s**
- **v246 − v236 = −5.5036 s（−24.38%）**，是噪声底的 **58.9 倍**
- 两者跑**同一个 A-GR1 块**、产出**逐字节相同的 state**，所以这个差**就是三张卡还回来的余量**
- **每层 5.5 s × 6 层 ≈ 33 s**

## 与 A-GR1 已知代价的对照（**必须看清口径不同**）

| 量 | 值 | 口径 |
|---|---|---|
| A-GR1 的代价 | **+20.7 s** | 侧隔离包（标准 Linear + 候选 Attention），基线 `14426/243s` |
| A-GR1 的完整包后果 | **> 8 s**（超时，父余量 9 s） | 完整包 v236 |
| 三张卡还回 | **本地 ≈ 33 s** | 本地 CPU 张量路径，每层 5.5 s × 6 |

**方向是有利的，但三方口径各不相同，不能直接相减。** 尤其：

- A-GR1 的 +20.7 s 是**侧隔离包**上的数，不是完整包上的；
- 三张卡的数字是**本地机器**上的，官方是**鲲鹏 920B CPU**；
- **不预测官方秒数、不换算系数。**

**这张卡的正是与非，只能由官方提交回答。**

## 已知风险（不隐瞒）

1. **可能仍然超时。** 本地回收 ~24% 不等于官方回收 ~24%；官方 CPU 上各部分的相对占比与本地不同
   （缺陷 #38：相对量随设备变）。
2. **v246 的 Linear 侧是 v245 的**（= v237 的 L-TF2，官方 `RETAINED 18518/289s`），
   与 v236 的 Linear 侧（v231）不同。这一差异是官方认可的（同分 −2s），但确实是一个变量。
3. A-GR1 的校准代价与 Linear 父无关（v236 已实测），所以这一点不构成新风险。

## 未做

- **未提交官方。**
- 未跑面板（本卡不改变 A-GR1 的精度，只改变其成本；面板读数应由官方裁决）。
- 未测 GPU 路径（CPU 路径是更接近官方的配置）。
