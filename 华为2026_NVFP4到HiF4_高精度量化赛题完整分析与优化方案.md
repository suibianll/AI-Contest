# 2026 华为算法大赛：NVFP4 → HiF4 高精度 4Bit 转换赛题深度分析与完整优化方案

> 版本：2026-09-02  
> ⚠️ **口径失效标注（2026-09-02）**：本文第 22 章及多数预算讨论按 **420s（7 分钟）** 与
> **250 Linear + 200 Attention** 面板设计，均为旧官方口径。官方 2026-08-31 已将端到端限制
> 收紧为 **300s**，并降低 Linear 评分权重（新权重未公开）；当前唯一有效协议为本地 `proxy-v2`
> + 官方回传（见 `docs/current-solution-status.md` 与 `docs/stale-information-inventory-2026-09-02.md`）。
> 本文算法机制与候选设计仍可参考，但所有时间/面板/目标数值须按新口径重算。
> 目标：在 **7 分钟总评测时间、250 个 Linear 样例 + 200 个 Attention 样例**约束下，最大化相对标准 HiF4 的 MSE 改善总分。  
> 重要规则更新：根据最新官方口径，**A@W 可用于离线校准与拟合，不再限制基于 A@W 的输出重构优化**。因此本方案把 Linear 问题重新定义为“HiF4 约束下的算子输出重构”，而非单纯的逐张量格式转换。

---

## 1. 赛题本质

### 1.1 不是普通的 NVFP4 → HiF4 数值转换

题目表面上要求把 NVFP4 转换为 HiF4，但真正参与评分的是最终算子输出：

Linear：

\[
Y = A_{FP32}W_{FP32}^T
\]

量化后：

\[
\hat Y = \hat A_{HiF4}\hat W_{HiF4}^T
\]

目标：

\[
\min MSE(Y,\hat Y)
\]

Attention：

\[
O = Softmax(QK^T/\sqrt d)V
\]

目标：

\[
\min MSE(O,\hat O)
\]

因此真正需要优化的不是：

\[
\|X-Q(X)\|^2
\]

而是：

\[
\boxed{\text{operator-level reconstruction error}}
\]

这一区别决定了整个算法设计：允许某个张量自身的量化误差略微增大，只要它与另一侧的量化误差能够在 MatMul / Attention 中抵消，最终得分就可能更高。

---

## 2. 评分函数带来的竞赛策略

每个 case 的得分为：

\[
Score_i =
\frac{MSE_{STD,i}-MSE_{PLAYER,i}}
{MSE_{STD,i}}
=
1-\frac{MSE_{PLAYER,i}}{MSE_{STD,i}}
\]

总分是所有测试用例得分之和。

因此目标不是简单最小化总 MSE，而更接近：

\[
\min \sum_i \frac{MSE_{PLAYER,i}}{MSE_{STD,i}}
\]

### 2.1 直接影响

1. **不能只追求平均 MSE**。某些 baseline 本来就很好的 case，如果出现轻微退化，归一化后可能产生较大负分。
2. **需要控制 worst-case regression**。激进策略必须带安全回退。
3. **校准目标最好与正式评分保持一致**。如果本地能模拟标准 HiF4，可把相对改善率作为模型选择依据；如果不能，则直接 operator MSE 最接近。
4. **模型选择优先于复杂优化**。在 450 组总样例下，少量高质量候选 + 精确 rerank 通常比大规模搜索更划算。

---

# 3. HiF4 格式为什么难

HiF4 每 64 个元素共享一个三级层级：

\[
x_i \approx s_{E6M2}
\cdot 2^{e^{(2)}_{g(i)}}
\cdot 2^{e^{(3)}_{h(i)}}
\cdot q_i
\]

其中：

- Level-1：1 个 E6M2，全 64 元素共享；
- Level-2：8 个 1-bit micro exponent，每 8 个元素共享；
- Level-3：16 个 1-bit micro exponent，每 4 个元素共享；
- payload：S1P2，取值为 `0, ±0.25, ±0.5, ..., ±1.75`；
- 总平均存储约 4.5 bit/value。

HiF4 官方算法采用三级最大值归约，从 64 个值中估计全局 scale，再依次决定 E1_8 和 E1_16。官方格式说明见：

- HiFloat4 论文：https://arxiv.org/abs/2602.11287
- HiFloat 官方说明：https://hifloat.gccorg.com/docs/en/hifloat4/white_paper/hifloat4_format_for_language_model_inference.html
- 官方代码：https://github.com/global-computing-consortium/HiFloat4

### 3.1 NVFP4 → HiF4 的结构错位

NVFP4 的 block size 是 16，而 HiF4 是 64。转换时每 4 个 NVFP4 block 会被重新组织进一个 HiF4 group。主要困难：

1. **Scale 层级错位**：4 个独立 NVFP4 scale 需要压缩成 1 个 E6M2 + 两级微指数。
2. **局部动态范围耦合**：64 个值中的少数 outlier 会影响全局 scale。
3. **值集不同**：NVFP4 与 S1P2 的离散表示空间不一致。
4. **E6M2 也是离散量化值**：scale 自身存在量化误差。
5. **层级 exponent 存在组合优化空间**：官方直接 cast 并不一定是 MSE 最优。
6. **最终指标不是张量 MSE**：最佳 HiF4 表示甚至可能故意偏离原始 W/A，以换取更小的算子输出误差。

---

# 4. 相关论文与可迁移技术

## 4.1 最重要的工作总览

| 方法 | 核心思想 | 对本赛题的价值 | 推荐优先级 |
|---|---|---|---|
| HiFloat4 / HiGPTQ (2026) | HiF4 专用分层格式；官方还提出 HiGPTQ | 与赛题格式直接一致 | S |
| MR-GPTQ, ICLR 2026 | microscaling FP4 专用 block rotation + GPTQ | 证明 FP4 需要格式专用二阶优化 | S |
| GPTQ, ICLR 2023 | Hessian/二阶误差补偿 | Linear 权重量化核心 | S |
| ScaleSweep, 2026 | 扫描可行 scale，最小化 MSE/WMSE | 直接适配 E6M2 scale 搜索 | S |
| SmoothQuant, ICML 2023 | A/W 之间迁移量化难度 | 适合作为快速初始化 | A |
| AWQ, MLSys 2024 Best Paper | activation-aware weight scaling | activation 统计指导 W | A |
| OmniQuant, ICLR 2024 | 可学习 clipping + 等价变换 | 可压缩成小规模校准 | A |
| AffineQuant, ICLR 2024 | 学习 affine transform | 可启发非严格等价补偿 | A |
| FlatQuant, ICML 2025 | learnable affine flattening | 高精度，但完整算法太重 | A |
| RPTQ, 2023 | 按 range 重排 channel | 很适合 HiF4 64-group 重组 | A |
| QuaRot, NeurIPS 2024 | Hadamard rotation 去 outlier | 可作为 block rotation | A |
| SpinQuant, 2024 | 学习旋转矩阵 | 精度潜力大，但 7 分钟下偏重 | B |
| QuIP# / QTIP | incoherence + VQ/TCQ | 原理重要，直接迁移价值有限 | B |
| KVQuant, NeurIPS 2024 | K per-channel、NUQ、outlier isolation | Attention/K 的重要启发 | S |
| KIVI, ICML 2024 | K per-channel、V per-token | 说明 Q/K/V 应非对称设计 | S |
| QServe / SmoothAttention | W4A8KV4 + attention smoothing | Q/K 联合缩放值得借鉴 | A |
| Atom, MLSys 2024 | W4A4、reorder、clip、GPTQ | 多技术组合参考 | A |

### 4.2 关键参考链接

- GPTQ：https://arxiv.org/abs/2210.17323  
  Code：https://github.com/IST-DASLab/gptq
- SmoothQuant：https://arxiv.org/abs/2211.10438  
  Code：https://github.com/mit-han-lab/smoothquant
- AWQ：https://github.com/mit-han-lab/llm-awq
- OmniQuant：https://github.com/OpenGVLab/OmniQuant
- AffineQuant：https://github.com/bytedance/AffineQuant
- FlatQuant：https://proceedings.mlr.press/v267/sun25l.html  
  Code：https://github.com/ruikangliu/FlatQuant
- RPTQ：https://arxiv.org/abs/2304.01089
- QuaRot：https://github.com/spcl/QuaRot
- SpinQuant：https://github.com/facebookresearch/SpinQuant
- QuIP#：https://arxiv.org/abs/2402.04396
- QTIP：https://arxiv.org/abs/2406.11235
- KVQuant：https://arxiv.org/abs/2401.18079  
  Code：https://github.com/SqueezeAILab/KVQuant
- KIVI：https://proceedings.mlr.press/v235/liu24bz.html
- QServe：https://arxiv.org/abs/2405.04532
- Atom：https://github.com/efeslab/Atom
- MR-GPTQ / microscaling FP4：https://proceedings.iclr.cc/paper_files/paper/2026/hash/b87bb4f6346d727b265088235e5bc389-Abstract-Conference.html
- ScaleSweep：https://arxiv.org/abs/2606.07618
- NVFP4 pretraining：https://arxiv.org/abs/2509.25149

---

# 5. 从已有研究得到的关键结论

## 5.1 结论一：必须做“格式专用”优化

MR-GPTQ 对 MXFP4/NVFP4 的研究表明：把 INT4 中有效的 outlier mitigation 直接迁移到 microscaling FP4 往往收益有限。原因是 scale 粒度和格式约束决定了量化误差结构。

对 HiF4 更是如此：其 64-value + 三级 micro exponent 是全新的组合空间。因此最优方法应围绕：

\[
E6M2 + E1_8 + E1_{16} + S1P2
\]

专门设计，而不是把 HiF4 当成普通 uniform INT4。

## 5.2 结论二：scale 初始化不是小问题

ScaleSweep 的核心观点是：AbsMax scale 并非最优，有限范围内直接 sweep 候选 scale 可以明显降低误差。

这对 HiF4 尤其重要，因为一级 E6M2 决定整个 64-group 的基准量级。建议从官方：

\[
s_0 \approx \frac{\max |x|}{7}
\]

附近产生离散 E6M2 候选，而不是固定直接 round。

## 5.3 结论三：rotation 应是“小块、格式感知”的

QuaRot/SpinQuant 表明 rotation 能抑制 outlier；MR-GPTQ 进一步说明对于 FP4，block-wise Hadamard 比全局大旋转更匹配 microscaling block。

因此本赛题更适合：

- 64/128 维 Hadamard；
- sign-Hadamard；
- 小候选随机 sign；
- 与 HiF4 group 对齐。

不建议首先实现大规模 learned dense rotation。

## 5.4 结论四：Attention 必须 Q/K/V 非对称处理

KIVI、KVQuant 的共同观察：

- Key 更适合沿 channel 方向考虑统计；
- Value 更接近 per-token / per-vector 动态量化；
- K 的 outlier 结构与 V 不同；
- Attention 的真正敏感点是 logits / probability，而不是 Q/K/V 单独 MSE。

因此 Q、K、V 三个函数绝不能用完全相同的量化策略。

---

# 6. 推荐的总体算法：HiF4-ORQ

建议把最终算法定义为：

> **HiF4-ORQ：HiF4-aware Operator Reconstruction Quantization**

核心由四个模块构成：

1. **Format-aware hierarchical search**：直接优化 E6M2 + E1_8 + E1_16；
2. **Operator-aware reconstruction**：Linear 用 A@W，Attention 用真实 Attention 输出或低成本 surrogate；
3. **Transformation search**：scaling / permutation / block rotation；
4. **Error compensation**：GPTQ/Hessian + activation-weight 联合补偿。

总体流程：

```text
NVFP4
  │
  ▼
一次反量化 BF16/FP32
  │
  ├──────── Linear ────────────────────────┐
  │                                        │
  │   Acalib, W                            │
  │       │                                │
  │       ├─ Yref = A @ W.T                │
  │       ├─ statistics / Hessian          │
  │       ├─ transform candidate search    │
  │       ├─ HiF4 hierarchical search      │
  │       ├─ GPTQ-style compensation       │
  │       └─ learn activation_state        │
  │                                        │
  └──────── Attention ─────────────────────┤
          Q,K,V                            │
            │                              │
            ├─ head statistics             │
            ├─ Q/K joint transform         │
            ├─ K centering candidate       │
            ├─ hierarchical search          │
            └─ Attention-output rerank      │
                                             ▼
                                   HiF4合法参数
```

---

# 7. Linear：完整高精度设计

## 7.1 第一步：一次性解码与缓存

校准函数中：

\[
W = Dequant_{NVFP4}(W_q,W_s)
\]

\[
A_j = Dequant_{NVFP4}(A^j_q,A^j_s)
\]

建议：

- 统一转 `float32` 计算统计；
- 不要重复 NVFP4 dequant；
- calibration list 先拼接/采样；
- 保留 sampled rows 与 full rows 两套视图。

### 行采样

为了减少 A@W 成本：

- `A_fast`：128～512 行，用于搜索候选；
- `A_full`：全部校准行，仅对 top-1/top-2 做最终确认。

如果样例很大，可做 norm-aware sampling，而不是纯随机采样。

---

## 7.2 第二步：直接构造 reference output

既然官方已不限制 A@W：

\[
Y = AW^T
\]

必须直接利用。

这是整个 Linear 方案最重要的变化。

量化方案 \(\theta\) 的真正 calibration loss：

\[
L(\theta)
=
MSE\left(
Y,
Q_A(A;\theta_A)
Q_W(W;\theta_W)^T
\right)
\]

不再依赖纯 tensor-MSE proxy 做最终判断。

---

# 8. Linear：Hierarchical HiF4 Quantizer

## 8.1 Level-1 E6M2 candidate sweep

对每个 64-group，先由：

\[
m=\max |x|
\]

得到官方初始化 \(s_0\)。

不要只取一个 scale，而是构造附近 E6M2 representable 候选：

\[
\mathcal S =
\{s_{-2},s_{-1},s_0,s_{+1},s_{+2}\}
\]

其中 \(s_k\) 是 E6M2 编码域中相邻 scale。

也可以使用相对候选：

\[
s = \text{Proj}_{E6M2}
(c \cdot s_0),\quad
c\in\{0.70,0.80,0.90,1.00,1.10,1.20,1.35\}
\]

推荐最终只保留 3～5 个唯一 E6M2 候选。

### 目标

快速阶段：

\[
L_{tensor} =
\sum_i w_i(x_i-\hat x_i)^2
\]

最终阶段：

\[
L_{op}
=
MSE(Y,\hat A\hat W^T)
\]

---

## 8.2 Level-2 / Level-3 coordinate descent

24 个 micro exponent 若全枚举：

\[
2^{24}
\]

不可行。

使用坐标下降：

```text
初始化：按官方阈值生成 E1_8 / E1_16

repeat 1~2:
    固定 E1_8
        对 16 个 E1_16 独立比较 0/1
    固定 E1_16
        对 8 个 E1_8 独立比较 0/1

最后重新计算 S1P2 nearest / biased rounding
```

由于每个 E1_16 只影响连续 4 个元素，每个 E1_8 只影响连续 8 个元素，局部损失可向量化计算。

### 建议

第一版只做 1 轮 coordinate refinement，观察性价比。通常比纯官方 threshold 更有收益，同时成本可控。

---

# 9. Linear：Output-aware Weight Quantization

## 9.1 GPTQ 形式

对于一行 weight \(w\)：

\[
L_w =
\|Aw-A\hat w\|^2
\]

有：

\[
L_w =
(w-\hat w)^T H (w-\hat w)
\]

其中：

\[
H=A^TA
\]

因此天然适合 GPTQ。

### HiF4-GPTQ

传统 GPTQ 是 scalar/group quantization；本赛题需要把 quantizer 替换成：

\[
Q_{HiF4}^{64}(w_g)
\]

即每次处理与 HiF4 64-group 对齐的块。

推荐：

1. 计算近似 Hessian；
2. 按 input-channel block 处理；
3. 每个 block 调用 HiF4 hierarchical quantizer；
4. 把量化残差按 \(H^{-1}\) 传播给后续未量化部分。

---

## 9.2 Hessian 成本控制

完整：

\[
H=A^TA\in \mathbb R^{K\times K}
\]

如果 K 很大，CPU 成本和内存过高。

使用三级策略：

### H0：Diagonal

\[
H_d[k]=\sum_n A_{nk}^2
\]

最便宜。

### H1：Block-diagonal

按 64 或 128 channel：

\[
H_b = A_b^TA_b
\]

推荐作为主版本。

### H2：Full GPTQ

只有在 K 不大或 profile 证明可接受时启用。

**推荐默认 H1。**

---

# 10. Linear：Activation-aware Scaling

SmoothQuant：

\[
A'=AD,\quad W'=WD^{-1}
\]

其中：

\[
d_k =
\frac{
(\max |A_k|)^\alpha
}{
(\max |W_k|)^{1-\alpha}
}
\]

但本赛题允许 A@W 拟合，因此不必拘泥于固定等价形式。

## 10.1 第一阶段：等价 scaling 初始化

搜索：

\[
\alpha\in\{0,0.25,0.5,0.75,1\}
\]

快速 operator MSE 选 top-2。

## 10.2 第二阶段：非严格等价 compensation

进一步允许：

\[
A'=AD_A,\quad W'=WD_W
\]

并不要求：

\[
D_AD_W=I
\]

直接最小化：

\[
\|AW^T-Q(AD_A)Q(WD_W)^T\|^2
\]

为了避免过拟合与状态过大，只学习非常低维的参数：

- per-64-group scalar；
- per-128-group scalar；
- global \(\alpha_A,\alpha_W\)；
- 限幅在例如 `[0.5, 2]`。

这比完整 AffineQuant/FlatQuant 更适合比赛时间预算。

---

# 11. Linear：Channel Reordering

RPTQ 的关键思想非常适合 HiF4。

定义每个输入 channel 的难度：

\[
r_k =
\log(\max |W_{:,k}|+\epsilon)
+
\lambda\log(\max |A_{:,k}|+\epsilon)
\]

或者：

\[
r_k =
\max|W_{:,k}|\sqrt{E[A_k^2]}
\]

然后构造 3 类 permutation：

1. **sort-range**：按 \(r_k\) 排序；
2. **balanced**：把大、小 channel 交错放进 64-group；
3. **cluster**：按 log-range 分桶，再连续排布。

每种 permutation 都做真实 operator-loss rerank。

### 为什么值得做

HiF4 group 64 是固定的。通过 permutation 可以主动决定：

> 哪 64 个 feature 共享一个 E6M2。

这是赛题中非常强的自由度。

---

# 12. Linear：Block Rotation

### 12.1 直接借鉴 MR-GPTQ，而不是大 SpinQuant

对 64 或 128 维 feature block：

\[
A'=AR,\quad W'=WR
\]

其中 \(R\) 为正交矩阵：

\[
RR^T=I
\]

则：

\[
A'W'^T=AW^T
\]

首选：

\[
R = H D
\]

其中 H 是 Hadamard，D 是随机 ±1 diagonal。

建议只尝试：

- identity；
- Hadamard；
- 2～4 个不同 random sign Hadamard。

再用 calibration output 直接选最优。

这样能吸收 QuaRot、QuIP#、MR-GPTQ 的收益，但不会引入 SpinQuant 那种昂贵的 learned rotation。

---

# 13. Linear：Learned / Biased Rounding

S1P2 nearest rounding 未必是 operator-optimal。

可以引入非常轻量的 rounding bias：

\[
q=Round_{S1P2}(z+b)
\]

其中 b 可以是：

- global；
- per-level；
- per-64-group class；
- per-output-channel 小标量。

搜索：

\[
b\in\{-0.125,-0.0625,0,0.0625,0.125\}
\]

只在 top candidate 上做。

目的不是让 \(\hat W\) 更接近 W，而是：

\[
\hat A\hat W^T
\]

更接近 Y。

---

# 14. Linear：Activation Error Compensation

当 W 已经量化成 \(\hat W\) 后，理想 activation 并不一定是 A。

定义：

\[
A^* = \arg\min_X \|AW^T-X\hat W^T\|^2
\]

其最小二乘解可写为伪逆形式。

完整计算过重，也容易过拟合，因此不建议直接保存 \(A^*\)。但它可用于推导**低维补偿目标**。

推荐拟合：

\[
Q_A(A;D,b)
\]

其中只开放：

- feature/group scale D；
- clipping ratio；
- rounding bias。

目标：

\[
\min_{D,b}
\|Y-Q_A(A;D,b)\hat W^T\|^2
\]

这是当前规则下极值得做的方向。

---

# 15. Linear 最终候选搜索框架

不做全组合搜索，而采用 beam search。

## Stage 0：统计

一次计算：

- absmax A/W；
- RMS A/W；
- per-channel energy；
- sampled \(Y=A W^T\)；
- diagonal / block Hessian。

## Stage 1：Transform 候选

生成 8～12 个候选：

```text
C0 vanilla
C1 scale α=0.25
C2 scale α=0.5
C3 scale α=0.75
C4 sort permutation
C5 balanced permutation
C6 scale + sort
C7 scale + balanced
C8 Hadamard
C9 scale + Hadamard
```

在 `A_fast` 上真实 MatMul MSE。

保留 Top-2/3。

## Stage 2：HiF4 refine

对 Top-2：

- E6M2 sweep；
- E1 coordinate descent；
- clipping candidate；
- rounding bias。

保留 Top-1。

## Stage 3：GPTQ refinement

只对 Top-1：

- block-Hessian；
- HiF4 block GPTQ；
- activation compensation。

## Stage 4：full calibration confirm

如果 refined 方案在完整 calibration 上反而差，则回退 Stage 2 best。

---

# 16. Attention：误差结构

Attention：

\[
S = QK^T/\sqrt d
\]

\[
P=Softmax(S)
\]

\[
O=PV
\]

误差来源：

\[
\Delta O
\approx
J_{softmax}(S)\Delta S V
+
P\Delta V
\]

而：

\[
\Delta S
\approx
\Delta QK^T
+
Q\Delta K^T
\]

所以：

- Q/K 的误差首先被放大/压缩成 logits 误差；
- softmax 对接近决策边界的 logits 很敏感；
- V 的误差直接按 attention probability 加权进入输出。

这也是为什么 Q、K、V 必须分别设计。

---

# 17. Attention：Q/K 联合缩放

严格等价：

\[
Q'=QD,\quad K'=KD^{-1}
\]

则：

\[
Q'K'^T=QK^T
\]

因此这是第一优先级。

### 搜索目标

初筛：

\[
L_{logit}
=
\|QK^T-\hat Q\hat K^T\|^2
\]

最终：

\[
L_{attn}
=
\|Attn(Q,K,V)-Attn(\hat Q,\hat K,\hat V)\|^2
\]

### scaling 统计

per-head/per-dim：

\[
d_k =
\left(
\frac{s_K(k)}
{s_Q(k)+\epsilon}
\right)^\alpha
\]

搜索少量 \(\alpha\)。

对于 GQA/MQA，多个 Q head 共享同一个 K head，因此 D 必须按共享关系保持兼容。

---

# 18. Attention：K Centering

标准 row-wise softmax 存在平移不变性：

若对所有 key token 减去同一个向量 \(\mu\)：

\[
K_j'=K_j-\mu
\]

则：

\[
q_iK_j'^T
=
q_iK_j^T-q_i\mu^T
\]

对固定 query i，后一项对所有 j 相同，因此：

\[
Softmax(QK'^T)
=
Softmax(QK^T)
\]

因此 per-head 公共 K mean：

\[
\mu_h=E_{token}[K_h]
\]

可以作为候选中心化操作。

### 注意

仅允许：

- 每 head 一个公共向量；
- calibration 估计、online 固定应用。

不能：

- 每 token 一个不同 mean；
- 量化后再忘记保持数学一致性。

应在真实官方 Attention evaluator 下严格验证。

---

# 19. Attention：K 与 V 不同的量化策略

借鉴 KIVI / KVQuant：

## K

重点：

- per-head / per-channel range；
- shared Q/K transform；
- K centering；
- rotation；
- logits-aware scale。

## V

重点：

- per-token/per-vector dynamic range；
- robust clipping；
- output probability sensitivity；
- 不宜随意做无法补偿的 feature transform。

推荐：

\[
L_V =
\sum_t \omega_t
\|V_t-\hat V_t\|^2
\]

其中 \(\omega_t\) 可由 calibration attention probability 的平均重要性估计。

---

# 20. Attention：真实输出 rerank

Attention 计算是：

\[
O(S^2d)
\]

所以不能像 Linear 一样大量试 candidate。

建议三级损失：

### Level 0

tensor WMSE：

\[
L_Q,L_K,L_V
\]

### Level 1

logits reconstruction：

\[
L_S=\|QK^T-\hat Q\hat K^T\|^2
\]

可加 softmax-weighted：

\[
L_P=\|P-\hat P\|^2
\]

### Level 2

只对 top-2：

\[
L_O=\|PV-\hat P\hat V\|^2
\]

真实 Attention 只做最后 rerank。

---

# 21. Attention 推荐候选

每个 case 只构造少量候选：

```text
A0 vanilla
A1 Q/K shared scaling
A2 K centering
A3 shared scaling + K centering
A4 shared Hadamard Q/K
A5 scaling + Hadamard
A6 V robust clip
A7 best Q/K + robust V
```

通过：

1. tensor/logit loss 快筛；
2. top-2 true attention loss；
3. 保存 q_state/k_state/v_state。

---

# 22. 时间预算设计

总时间：

\[
420s
\]

总 case：

\[
250+200=450
\]

平均：

\[
\approx0.93s/case
\]

但各阶段成本不均，因此不能真的每组做完整搜索。

## 推荐总预算

| 模块 | 目标占比 |
|---|---:|
| NVFP4 dequant / 基础 HiF4 | 10–15% |
| Linear calibration search | 35–40% |
| Linear online quant | 10–15% |
| Attention calibration | 15–20% |
| Attention online quant | 10–15% |
| Python/框架余量 | ≥10% |

### 设计原则

**复杂工作尽量挪到 offline calibration，online 只执行固定变换 + 动态量化。**

---

# 23. CPU 工程优化

判题器是鲲鹏 CPU，最危险的是 Python 小循环。

## 必做

1. 所有 64-group 用 reshape：
   ```python
   x = x.reshape(..., -1, 64)
   ```
2. candidate scale 增加一维 batch：
   ```python
   [G, C, 64]
   ```
3. E1_8：
   ```python
   [..., 8, 8]
   ```
4. E1_16：
   ```python
   [..., 16, 4]
   ```
5. 使用 `torch.where / amin / amax / gather / argmin`。
6. 避免逐元素、逐 group Python for。
7. 预生成：
   - S1P2 codebook；
   - E6M2 representable table；
   - reciprocal table；
   - exponent factor table。
8. 禁止重复：
   - NVFP4 dequant；
   - `torch.tensor(constant)`；
   - reshape metadata。
9. 避免在 hot path 做 `.clone()` / `.contiguous()`，除非确有必要。
10. online state 用连续 tensor，避免巨型 dict。

---

# 24. activation_state / q_state 设计

推荐：

```python
activation_state = {
    "version": 3,
    "transform": 1,
    "perm": torch.int32 tensor,      # optional
    "scale": torch.float16 tensor,   # group/channel
    "rot_sign": torch.int8 tensor,   # optional
    "clip": float,
    "round_bias": float,
    "policy": torch.int8 tensor,
}
```

Attention：

```python
q_state = {
    "scale": ...,
    "perm": ...,
    "rot_sign": ...,
}

k_state = {
    "scale": ...,
    "center": ...,
    "perm": ...,
    "rot_sign": ...,
}

v_state = {
    "clip": ...,
    "scale_policy": ...,
}
```

状态尽量使用 tensor 打包，避免节点数超过 4096。

---

# 25. 安全回退机制

评分允许负分，所以每个 case 都应该有：

\[
\text{best\_candidate}
=
\arg\min_{\theta\in\Theta}L_{calib}(\theta)
\]

同时保留 vanilla candidate。

如果：

\[
L_{best}
>
(1-\delta)L_{vanilla}
\]

收益很小，则不使用复杂方案。

推荐 \(\delta\) 由本地仿真调，例如 0.5%～2%。

可以额外做泛化风险指标：

\[
R =
\frac{
|L_{fast}-L_{full}|
}{
L_{full}+\epsilon
}
\]

R 太大时回退更保守策略。

---

# 26. 推荐最终算法版本

## V1：稳定高性价比版本

Linear：

- E6M2 multi-candidate；
- E1 coordinate refinement；
- SmoothQuant α search；
- permutation shortlist；
- direct A@W rerank；
- diagonal WMSE。

Attention：

- Q/K shared scaling；
- K centering；
- robust V；
- logits rerank；
- top-2 true Attention。

这是第一版应该完成的方案。

## V2：高分版本

Linear 增加：

- block-Hessian GPTQ；
- block Hadamard；
- activation compensation；
- biased rounding。

Attention 增加：

- Q/K shared Hadamard；
- head sensitivity；
- output-aware scale search。

## V3：极限冲分版本

如果仍有足够时间：

- non-invariant \(D_A,D_W\) joint compensation；
- per-group low-dimensional learned scaling；
- HiF4 group-level operator-aware coordinate descent；
- adaptive beam width；
- case type classifier，按数据分布选择算法。

---

# 27. 推荐实现结构

```text
solution.py
├── constants
│   ├── S1P2_TABLE
│   ├── E6M2_TABLE
│   └── inverse tables
│
├── nvfp4
│   └── dequant_fast()
│
├── hif4_core
│   ├── e6m2_candidates()
│   ├── encode_payload()
│   ├── hierarchical_quantize()
│   ├── hierarchical_quantize_candidates()
│   └── dequant_hif4()
│
├── linear_calib
│   ├── sample_rows()
│   ├── collect_stats()
│   ├── generate_transforms()
│   ├── eval_operator_loss()
│   ├── block_gptq()
│   └── fit_activation_policy()
│
├── attention_calib
│   ├── collect_head_stats()
│   ├── qk_scale_candidates()
│   ├── k_center()
│   ├── logit_loss()
│   └── attn_loss()
│
└── official APIs
    ├── hif4_calibration_and_quantize_weight()
    ├── hif4_dynamic_quantize_activation()
    ├── hif4_calibration_attention()
    ├── hif4_dynamic_quantize_q()
    ├── hif4_dynamic_quantize_k()
    └── hif4_dynamic_quantize_v()
```

比赛要求只有 solution.py 时，可以逻辑分区，不一定拆文件。

---

# 28. 关键伪代码

## 28.1 Linear calibration

```python
def calibrate_linear(W_nv, A_calib_nv):
    W = dequant_nvfp4_once(W_nv)
    A = concat_or_sample(dequant_nvfp4_once(A_calib_nv))

    A_fast = sample_rows(A)
    Y_fast = A_fast @ W.T

    stats = collect_stats(A_fast, W)

    candidates = generate_transform_candidates(stats)

    beam = []
    for transform in candidates:
        Wt = apply_w_transform(W, transform)
        At = apply_a_transform(A_fast, transform)

        qw = hif4_quant_fast(Wt)
        qa = hif4_quant_fast(At)

        loss = mse(Y_fast, deq(qa) @ deq(qw).T)
        beam.append((loss, transform, qw))

    top = select_topk(beam, 2)

    refined = []
    for cand in top:
        qw = hierarchical_refine(cand.W)
        qw = block_gptq_if_profitable(A_fast, cand.W, qw)
        policy = fit_activation_policy(A_fast, Y_fast, qw, cand.transform)
        refined.append(...)

    best = full_calibration_confirm(refined)

    return best.weight_params, best.activation_state
```

## 28.2 Dynamic Activation

```python
def dynamic_activation(aq, asc, state):
    a = dequant_nvfp4(aq, asc)

    a = apply_transform(a, state)
    a = apply_clip(a, state.clip)

    return hif4_quantize_fast(
        a,
        policy=state.policy,
        round_bias=state.round_bias,
    )
```

## 28.3 Attention calibration

```python
def calibrate_attention(calib_qkv, q_heads, kv_heads, d):
    samples = decode_samples(calib_qkv)

    stats = collect_attention_stats(samples)

    cands = build_qk_candidates(stats)
    cands += build_k_center_candidates(stats)

    # cheap stage
    ranked = []
    for c in cands:
        qh, kh = transform_qk(samples, c)
        l = logit_reconstruction_loss(qh, kh)
        ranked.append((l, c))

    top = topk(ranked, 2)

    # true attention stage
    best = None
    for c in top:
        qh, kh, vh = quantize_all(samples, c)
        l = true_attention_mse(samples, qh, kh, vh)
        best = min(best, ...)

    return states(best)
```

---

# 29. 必须做的消融实验

## Linear

| 实验 | 目的 |
|---|---|
| Baseline direct HiF4 | 建基准 |
| + E6M2 sweep | 衡量 scale 搜索 |
| + E1 coordinate | 衡量层级优化 |
| + scaling | SmoothQuant 类收益 |
| + permutation | 评估 group 重组 |
| + Hadamard | 评估 outlier flatten |
| + diagonal WMSE | 评估 sensitivity |
| + block GPTQ | 二阶补偿收益 |
| + activation compensation | 误差抵消收益 |
| + biased rounding | 最终细化 |

## Attention

| 实验 | 目的 |
|---|---|
| Vanilla Q/K/V | 基准 |
| Q/K scaling | 联合等价变换 |
| K centering | softmax 不变性收益 |
| Q/K Hadamard | outlier flatten |
| robust V | V 动态范围 |
| logit rerank | surrogate 有效性 |
| true Attention rerank | 最终收益 |

---

# 30. 本地评估必须记录的指标

每个 case：

```text
case_id
shape
calibration_rows
baseline_tensor_mse
player_tensor_mse
baseline_operator_mse
player_operator_mse
relative_score
algorithm_variant
scale_candidate
permutation_type
rotation_type
gptq_enabled
calibration_time
online_time
```

额外汇总：

- 平均 score；
- 中位数；
- P10/P90；
- 最差 10 case；
- 负分 case 数；
- Linear/Attention 分开；
- 不同 shape bucket 分开；
- 不同算法 variant 的收益分布。

**最重要指标不是平均提升，而是“负分 case 数量 + 最差 case”。**

---

# 31. 超参数建议

初始建议：

```text
Linear
------
fast_rows              = 128~512
transform_candidates    = 6~10
beam_width              = 2
e6m2_candidates         = 3~5
e1_coordinate_rounds    = 1
alpha_candidates        = [0.25, 0.5, 0.75]
clip_candidates         = [1.00, 0.995, 0.99, 0.98]
hadamard_candidates     = 1~3
hessian_block           = 64 or 128
round_bias_candidates   = 3~5

Attention
---------
qk_alpha_candidates     = 3
k_center                = {off,on}
qk_rotation             = {identity,hadamard}
v_clip_candidates       = 2~3
true_attn_topk           = 2
```

不要一次把所有参数打开；否则组合数量会爆炸。

---

# 32. 预计最有价值的技术排序

## Linear

1. **真实 A@W output-aware candidate rerank**
2. **E6M2 candidate sweep**
3. **HiF4 hierarchical E1 coordinate optimization**
4. **block-Hessian HiF4-GPTQ**
5. **scaling + permutation**
6. **activation-weight error compensation**
7. **block Hadamard**
8. **biased rounding**

## Attention

1. **Q/K joint scaling**
2. **K centering**
3. **Q/K/V 非对称量化**
4. **logit-aware candidate selection**
5. **true Attention top-k rerank**
6. **robust V**
7. **Q/K Hadamard**
8. **head sensitivity weighting**

---

# 33. 哪些论文不建议完整照搬

### SpinQuant

学习 rotation 的训练成本太高，7 分钟、450 case 不适合逐 case 做 Cayley optimization。

**借鉴：** rotation 重要。  
**不要照搬：** learned full rotation。

### FlatQuant / AffineQuant

完整 affine matrix 搜索成本高。

**借鉴：** 不要局限于单纯 diagonal smooth。  
**赛题版：** group-wise scale + permutation + Hadamard + 少量 affine scalar。

### QuIP# / QTIP

代码本与 lattice/trellis codebook 不符合 HiF4 固定 payload。

**借鉴：** incoherence / rotation。  
**不要照搬：** codebook。

### SqueezeLLM/KVQuant 的 sparse outlier

HiF4 输出格式没有额外 sparse side channel，无法直接保存高精度 outlier。

**借鉴：** 识别敏感值、设计 clipping/transform。  
**不要照搬：** dense+sparse 存储。

---

# 34. 一个更专业的算法命名

建议最终方案对外描述为：

## HiF4-ORQ

**HiF4-aware Operator Reconstruction Quantization**

中文：

> **面向算子输出重构的 HiF4 层级协同量化**

核心技术点可以写成：

1. **层级微指数联合搜索**  
   Hierarchical Micro-Exponent Joint Optimization

2. **二阶敏感度驱动的 HiF4 重构量化**  
   Hessian-Guided HiF4 Reconstruction Quantization

3. **算子感知的激活-权重误差协同补偿**  
   Operator-Aware Activation-Weight Error Compensation

4. **面向 HiF4 Group 的动态范围重整**  
   HiF4-Group-Oriented Dynamic-Range Reshaping

5. **Attention Logit 保真的 Q/K/V 非对称量化**  
   Logit-Preserving Asymmetric Q/K/V Quantization

---

# 35. 最终建议

当前规则下，最值得投入的不是继续优化单张量 quantizer，而是把比赛彻底当成：

\[
\boxed{
\text{HiF4 constrained operator reconstruction}
}
\]

Linear 的主路线应当是：

\[
\boxed{
A@W\ \text{直接监督}
+
\text{HiF4 scale/E1 专用搜索}
+
\text{block GPTQ}
+
\text{低维 activation compensation}
}
\]

Attention 的主路线应当是：

\[
\boxed{
Q/K\ \text{联合变换}
+
\text{logit reconstruction}
+
\text{K centering}
+
\text{Q/K/V 非对称量化}
+
\text{真实 Attention top-k rerank}
}
\]

如果只选择一个最有可能继续明显拉分的创新点，优先实现：

> **Block-Hessian HiF4-GPTQ + Output-aware Activation Compensation**

因为它同时利用了：
- 最新官方规则允许 A@W 拟合；
- GPTQ 的二阶误差补偿优势；
- HiF4 的固定 64-group 结构；
- 当前评分直接看 MatMul 输出 MSE 的特点。

而对 Attention，最优先实现：

> **Q/K joint scaling + K centering + true-Attention rerank**

这三项计算代价小、理论依据强，也最容易在 7 分钟预算内形成稳定收益。

---

# 参考资料

1. Luo et al., **HiFloat4 Format for Language Model Inference**, 2026.  
   https://arxiv.org/abs/2602.11287  
   https://github.com/global-computing-consortium/HiFloat4

2. Egiazarian et al., **Bridging the Gap Between Promise and Performance for Microscaling FP4 Quantization**, ICLR 2026.  
   https://proceedings.iclr.cc/paper_files/paper/2026/hash/b87bb4f6346d727b265088235e5bc389-Abstract-Conference.html

3. Frantar et al., **GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers**, ICLR 2023.  
   https://arxiv.org/abs/2210.17323  
   https://github.com/IST-DASLab/gptq

4. Xiao et al., **SmoothQuant**, ICML 2023.  
   https://arxiv.org/abs/2211.10438  
   https://github.com/mit-han-lab/smoothquant

5. Lin et al., **AWQ: Activation-aware Weight Quantization**, MLSys 2024 Best Paper.  
   https://github.com/mit-han-lab/llm-awq

6. Shao et al., **OmniQuant**, ICLR 2024 Spotlight.  
   https://github.com/OpenGVLab/OmniQuant

7. Ma et al., **AffineQuant**, ICLR 2024.  
   https://github.com/bytedance/AffineQuant

8. Sun et al., **FlatQuant: Flatness Matters for LLM Quantization**, ICML 2025.  
   https://proceedings.mlr.press/v267/sun25l.html  
   https://github.com/ruikangliu/FlatQuant

9. Yuan et al., **RPTQ: Reorder-based Post-training Quantization for Large Language Models**.  
   https://arxiv.org/abs/2304.01089

10. Ashkboos et al., **QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs**, NeurIPS 2024.  
    https://github.com/spcl/QuaRot

11. Liu et al., **SpinQuant: LLM Quantization with Learned Rotations**.  
    https://github.com/facebookresearch/SpinQuant

12. Tseng et al., **QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks**, ICML 2024.  
    https://arxiv.org/abs/2402.04396

13. Tseng et al., **QTIP: Quantization with Trellises and Incoherence Processing**, NeurIPS 2024 Spotlight.  
    https://arxiv.org/abs/2406.11235

14. Hooper et al., **KVQuant**, NeurIPS 2024.  
    https://arxiv.org/abs/2401.18079  
    https://github.com/SqueezeAILab/KVQuant

15. Liu et al., **KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache**, ICML 2024.  
    https://proceedings.mlr.press/v235/liu24bz.html

16. Lin et al., **QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving**, 2024.  
    https://arxiv.org/abs/2405.04532

17. Zhao et al., **Atom: Low-bit Quantization for Efficient and Accurate LLM Serving**, MLSys 2024.  
    https://github.com/efeslab/Atom

18. NVIDIA et al., **Pretraining Large Language Models with NVFP4**, 2025.  
    https://arxiv.org/abs/2509.25149

19. **ScaleSweep: Accurate NVFP4 Post-Training Quantization of LLMs via Block Scale Initialization**, 2026.  
    https://arxiv.org/abs/2606.07618
