# HiF4 量化算法实施方案手册（含完整文献引用）

> 目标：NVFP4 → HiF4 转换，逐 case 击败官方基线量化器的输出 MSE。
> 本手册按实施步骤（Step 0–6）组织，每步含：目标、具体流程、伪代码、理论依据（带论文链接）、验收标准。
> 合规前提（2026-08-31 修订后）：官方不再限制任何 `A@W` 拟合用法，`Q(W)`、`Q(A)`
> 均可自由使用校准输出/残差/`A@W` 信息优化；输出信息流入 `Q(A)` 数值生成已合法。
> 仍有效的硬约束：①端到端运行时间 `<300s`（任何在线输出域路径都计入预算）；
> ②`activation_state` 满足赛事说明书格式约束（合法五字段、CPU tensor、深度/节点数上限）。

> **官方口径更新（2026-08-29）**：评测集现为 250 个 Linear case 与 200 个
> Attention case，官方总时间限制为 7 分钟（420 秒）。样例数增加会抬高逐 case
> 求和分数和端到端时间；以下旧版 5 分钟字样仅属于历史规划，当前提交以 420 秒为准。
>
> **官方口径再次更新（2026-08-31）**：官方时间限制已从 420s 收紧为 **`300s`（5 分钟）**，
> 且不再限制任何 `A@W` 拟合用法（`Q(W)`、`Q(A)` 均可自由使用），只限制端到端运行时间。
> 上方 08-29 的 420s 表述仅属历史口径；v98 已在最新限制下官方判为 timeout（本地
> API `406.24s`），v107 官方保持 Attention `wrong answer`（非 timeout）。
>
> **官方评分权重第三次更新（2026-08-31 晚）**：官方**减少 Linear 样例的评分权重**，
> 官方总分据此大幅下降；新权重下已确认 **v84 官方通过 `16517 / 252.563s`（< 300s）**、
> **v86 官方通过 `16744 / 222.7s`（< 300s，新权重下分数最高且最快）**。
> 旧权重分数（v66/v72/v74、外部 24153 等）与新权重不可互相换算，官方未提供权重系数，
> 本地不复制 case 拟合官方绝对分。

---

## 总体架构

```
                    ┌──────────────────────────────────────────────┐
                    │       当前 root solution.py（clean path）     │
                    ├──────────────────────────────────────────────┤
                    │ codec: NVFP4 反量化 + 合法 HiF4 五字段编码      │
                    │ BOAT: RMS 对角平衡 + 4/8/16/64 signed-Hadamard │
                    │ W-HSDQ: cross-fold AᵀA + 15 levels + top-2 block│
                    │ A-HSDQ: 静态 WᵀW Gram + hierarchy + 2 sweeps  │
                    │ Attn: Q/K 不变量候选，前 4 个部署路径复评      │
                    │ API: 六个赛事正式接口                            │
                    └──────────────────────────────────────────────┘

离线 Linear: W ─▶ NVFP4 反量化 ─▶ BOAT ─▶ HiF4 ─▶ cross-fold W-HSDQ ─▶ weight_params
                         └─▶ 静态 WᵀW Gram + BOAT 逆缩放 ─▶ activation_state
在线 Linear: A ─▶ NVFP4 反量化 ─▶ 应用 state ─▶ HiF4 + A-HSDQ ─▶ activation_params

离线 Attention: Q/K/V ─▶ 不变量候选 ─▶ 真实输出复评 ─▶ Q/K state（V 独立编码）
在线 Attention: Q/K/V ─▶ 应用静态 state ─▶ 合法 HiF4 参数
```

### 当前主版本实测快照（2026-08-30）

固定 Qwen2.5-0.5B 全 24 层、`seq=128`、`calib=2`、`test=4`、CPU 缓存评测：

| 指标 | 当前值 |
|---|---:|
| Linear native mean | 0.501558 |
| Attention native mean | 0.841829 |
| Qwen shaped panel total | **293.755106** |
| official-flow native total（诊断） | 417.862253 |
| 六 API 累计时间 | **382.153528s** |

这不是官方分数；完整结果见 [`docs/current-solution-status.md`](docs/current-solution-status.md)。
后续实现仍应参考下方 Step 0–6 的理论候选，但只有已经落入根 `solution.py` 的
机制才是当前线上行为；历史 C86/GPTQ/AdaRound/BRECQ 分支不再隐式启用。

外部 [`youxilee/hif4`](https://github.com/youxilee/hif4) v2.7 的本地 CPU 复测中，
最高单模型 Qwen native 为 `369.527269`，按本地 250/200 面板投影的最高同口径
基准为 `250.327102`；五模型 raw sum `1085.743597` 仅作诊断，不能作为排名分或
官方 `24153 / 239s` 的线性换算。当前根分别领先这两条本地线 `48.334984`
（`13.08%`）和 `43.428004`（`17.35%`）。

---

## Step 0：合法基线与兜底框架（入场券，最高优先级）

**目标**：保证"任何情况下不判负、不拿负分"。

**流程**：
1. 实现直通量化器 `quantize_naive`：amax → 最小覆盖 E6M2 → 规则填 e2/e3 → 最近邻舍入；
2. 复现 `dequantize_hif4`（按任务书 §1.3 公式，HiF4 格式定义见官方论文 [^2^]），与 self_check.py 对齐校验所有输出字段；
3. 实现**内部 MSE 估计器**：对任意量化输出，计算表示 MSE `‖x̂−x‖²`；
4. 实现**兜底逻辑**：每 case 同时产出 naive 解与优化解，内部估计不优则回退 naive；
5. 边角 case 分支：amax=0（scale 取最小档、mant 全 0）、全同值、NaN（E6M2=NaN 时全组 NaN）。

**验收**：self_check 全绿；随机数据上优化解的表示 MSE ≤ naive 解（逐 case）。

---

## Step 1：核心量化器 `hif4_quantize_core`（P0，所有收益的地基）

**优化对象**：64 元素 group 的三级参数 (s, e2[8], e3[16], sign[64], mant[64])。

**理论依据**：
- 最优标量量化的 **Lloyd-Max 条件**（量化点=区间质心）[^36^][^37^]；
- **Panter-Dite / Bennett 高分辨率量化理论**（失真 ≈ 步长²/12，点密度应匹配 p^(1/3)）[^38^][^39^]；
- **Gish-Pierce 渐近理论**（高码率失真界 ~ 2^(−2R)）[^40^]；
- 可学习 clip/步长的 **PACT / LSQ / OmniQuant**（证明"允许 clip 换细步长"优于 amax 覆盖）[^26^][^27^][^23^]；
- micro-exp 分配 = **可分离组合优化**（每个 8 子块 3 bit 独立），混合精度位分配思想见 **HAWQ** [^25^]。

**具体流程**：

```
输入: x [..., 64] (已按最后维分组成 64 倍数), 可选逐元素权重 w (默认1)
输出: 合法 HiF4Params

1. amax_g = |x|.max(dim=-1)                       # 每组幅值
2. 候选 scale 集: S = E6M2网格 ∩ [amax/16, amax/1.0]   # 约 12~17 档
   （可选: 用解析式定搜索中心收缩候选, 参考 SOAR 思路 [^12^]）
3. 向量化主循环（无 Python 循环, 全部展成大张量）:
   for s in S:                                    # 或一次性 broadcast
     d = s · 2^(e2⊗ones4 + e3)                    # 8 组合 × 8 子块的除数表
     m = clamp_round(x / d, 到 {0,.25,...,1.75})  # 最近邻投影 + clip
     x̂ = d · sign(x) · m
     sse = Σ w·(x−x̂)²                             # 支持加权(Step 3/4 复用)
   每子块保留 8 组合中最小 SSE 的 (e2,e3a,e3b)
4. 每组保留 sse 最小的 s → 组装参数
5. 合法性投影: 强制 scale∈E6M2网格、lv2/lv3∈{1,2}、mant∈八值
```

**关键实现细节**：
- 8 组合的除数表预先计算成常量张量，避免重复 2 的幂运算；
- round 用 `searchsorted` 在 8 个 mant 值上查表；
- 加权版接口预留 `w` 参数（Step 3 的 H 加权、Step 4 的 p̄ 加权复用同一核）；
- 舍入坚持最近邻——随机舍入对前向/推理张量有害，见 NVFP4 论文 [^14^]。

**验收**：随机 + 重尾 + outlier 三类合成数据上，组 MSE 一致低于 naive（预期 −30%~60%）。

---

## Step 2：分布整形（P1，增益主力）

**理论依据**：
- **Kashin 分裂**（1977）：存在正交变换使任意向量各坐标"平坦化"——旋转消 outlier 的存在性证明 [^41^]；
- **Johnson-Lindenstrauss 引理**：随机投影保距 [^42^]；
- **DuQuant++** 关键结论：旋转块尺寸应与量化 group 尺寸对齐（→ 取 64）[^21^]；
- **MR-GPTQ** 警告：旋转+朴素舍入对 NVFP4 系格式可能有害（小 group 抵消 outlier 抑制收益），大 group 转正 → **必须门控** [^1^]；
- 旋转应用框架：QuaRot [^6^]、SpinQuant [^7^]、DuQuant [^20^]、FlatQuant [^22^]、OSTQuant [^34^]。

### 2a. SmoothQuant 缩放 [^5^]

```
离线:
  c_k = max over calib & rows of |X[:,k]|        # channel 峰值
  s_k = c_k^α / max(c^α),  α ∈ {0, 0.3, 0.5, 0.8}  # 候选集, 门控选定
  W' = W · diag(s)
在线:
  X' = X / s
```

相关变体：AWQ 的 salient channel 保护 [^4^]、Outlier Suppression+ [^28^]、OmniQuant 的 clip+scale 联合优化 [^23^]、LLM.int8() 的 outlier 隔离思想 [^24^]。

### 2b. 块对齐 Hadamard 旋转

```
H64 = hadamard(64)                               # 与 HiF4 group 对齐 [^21^]
旋转方式: 最后维 reshape(-1,64) @ H64 → reshape 回   # FWHT O(K log 64)
Linear:  X' = X·rot,  W' = W·rot                 # 同一 rot, 乘积不变
Q/K:     同一 head_dim 维旋转（或 head 内分块64）, 共用
V:       禁止旋转
```

### 2c. 门控（与 Step 5 联动）

每个变换都带开关：离线在校准集上比较 {直通, +smooth, +rot, +smooth+rot} 的（真值或表示）MSE，选最优组合写入 state。**不优不启用。**

---

## Step 3：权重输出感知优化（P2，离线榨干，官方放行后收益最大）

**理论谱系**（35 年 Hessian 补偿线）：OBD（LeCun 1989）[^43^] → OBS（Hassibi 1993）[^44^] → OBQ/OBC → GPTQ [^3^] → QuIP/LDLQ（证明损失 ηᵀDη 的理论最优性）[^15^] → QuIP#（E8 格码本 + Hadamard 不相干性）[^16^] → GPTQ 等价于格 CVP 的 Babai 最近平面算法 [^32^]；输出重建家族：AdaRound [^8^]、BRECQ [^9^]、QDrop [^10^]、QuantEase [^11^]；MR-GPTQ 给出 FP4 格式的专用适配范式 [^1^]。

### 3a. GPTQ 补偿（基础版）[^3^][^15^]

```
H = X_calibᵀ X_calib + λI        # λ = 0.01·mean(diag(H)) 阻尼
按 LDLQ 形式: 对 H 做 LDL 分解 → 逐列量化, 误差按 D 补偿后续列
（固定列序即可; 可选 ActOrder 按 diag(H) 降序 [^1^]）
```

### 3b. AdaRound 式坐标下降（放行后启用，用真实输出）[^8^][^11^]

```
y_ref = X_calib @ W_fpᵀ                           # 官方放行的离线真值裁判
Ŵ = gptq 输出
残差 r = y_ref − X_calib @ Ŵᵀ
for 轮次 t in 1..3:
  for 列 k（按 |误差贡献| 降序）:
    候选 = Ŵ[:,k] 在 HiF4 值集上的相邻 2~3 档
    对每个候选: Δerr = 2·c·(X[:,k]ᵀ·r) + H[k,k]·c²   # 秩-1 O(M) 评估
    接受最优候选, 秩-1 更新 r −= X[:,k]·Δc
```

### 3c. BRECQ 组间接力（与 3b 融合）[^9^]

按 group 顺序量化，每组的目标 = 当前输出残差，而非本组孤立误差。

### 3d. 真值选型（裁判角色）

```
for config in {3a, 3a+3b, 3a+3b+3c} × {±旋转, α 各档}:
    err[config] = ‖X_calib @ quantize(W,config)ᵀ − y_ref‖²
选用 argmin config
```

**验收**：Linear 校准集上输出 MSE 较 Step 1 单独使用再降 ≥15%；耗时在离线预算内。

---

## Step 4：Attention 特化（P2）

**理论依据**：
- **KIVI**：K 有 per-channel outlier、V 没有 → K per-channel、V per-token 的不对称处理 [^17^]；
- **KVQuant**：校准驱动的非均匀量化点、敏感度加权、dense-and-sparse outlier 隔离、attention-sink 感知 [^18^]；
- **StreamingLLM**：attention sink 是 KV 量化崩溃主因 [^19^]；
- 低秩/稀疏残差分解：SpQR [^29^]、SVDQuant [^30^]；
- 系统协同：QServe（W4A8KV4）[^35^]。

### 4a. Q/K 联合旋转 + per-head 门控

```
离线:
  对每个 head h:
    管道候选 = {不旋转, 旋转64对齐, 旋转head_dim}
    用校准真实 Attn 输出 MSE 逐 head 投票（官方放行）
  q_state = {per-head 旋转掩码, H_d}
在线:
  Q/K reshape 按 head → 按掩码选择性旋转 → 核心量化
```

### 4b. V 的注意力加权量化

```
离线:
  P = softmax(Q_calib K_calibᵀ/√d)                # 校准注意力矩阵 [^18^][^19^]
  p̄_i = mean over queries & calib of P[:,i]²      # 位置被关注强度(sink自然获得极大权重)
  v_state = {p̄}
在线:
  hif4_quantize_core(V, w = broadcast(p̄))         # 复用 Step 1 加权核
```

### 4c. 搜索预算分配

离线测每 head 的 score 方差（敏感度代理，参考 KIVI 的分布研究法 [^17^]）：敏感 head → 加密 E6M2 候选；钝感 head → 快扫。总在线时间恒定下重分配。

---

## Step 5：元算法层——门控与预算分配（P3，多架构稳健性）

**理论依据**：统计模型选择/交叉验证（控制有限校准样本的选择偏差）；QDrop 的随机子集防过拟合思想 [^10^]；minimax（worst-case 优先，对应负分惩罚）；率失真理论上界（判断剩余可榨空间，量化信息论视角见 [^36^][^40^]）。

```
离线门控协议（每组数据独立执行）:
  1. 校准集随机二分: A 半用于生成 state, B 半用于验证   # QDrop 式防过拟合 [^10^]
  2. 枚举管道组合 Π = {变换开关 × 量化器档位 × 补偿开关}
  3. 用 B 半（真值 MSE, 放行后）给 Π 排序
  4. 若最优管道的 B 半误差 ≥ naive 基线的 95% → 回退 naive   # 负分防线
  5. 将选定配置固化进 state
时间预算协议:
  - 离线: 每组数据预算按 case 大小比例分配, 预留 20% 余量
  - 在线: 核心量化器候选数随张量大小自适应降档
```

---

## Step 6：工程化与验收（贯穿全程）

1. **向量化**：核心量化器候选×group×组合×元素全展平；禁止逐元素 Python 循环；
2. **计时框架**：每个接口独立计时打表，总预算 5 分钟（300 秒，2026-08-31 修订的官方端到端限制）；
3. **自检闭环**：提交前 self_check.py 全量 + 本地评分预测器（复现判题流程估分，保护每日 30 次额度）；
4. **审核可解释性**：state 内容写注释（注明各张量的格式合法性与用途）；`A@W` 用法自
   2026-08-31 起不受限制（不再限于裁判函数），但任何在线 `A@W` 路径都要计入 `<300s` 预算。

---

## solution.py 接口骨架

```python
import torch

# ---------- 工具层 ----------
def dequantize_nvfp4(q, s, blk=16): ...      # 题目给定
def dequantize_hif4(params): ...             # 本地复现 [^2^], 用于内部估计/裁判
def s1p2_round(v): ...                       # 最近邻投影到8值网格+clip
def hadamard_fwht(x): ...                    # O(K log K) 无乘法旋转 [^6^][^21^]

# ---------- 核心量化器 (Step 1) ----------
def hif4_quantize_core(x, w=None, n_cand=15):
    """x: [...,L] L%64==0; w: 可选逐元素权重. 返回合法 HiF4Params dict"""
    ...

# ---------- 整形与补偿 (Step 2/3) ----------
def fit_smooth_alpha(W, Xc_list): ...        # [^5^]
def gptq_quantize(W, H): ...                 # [^3^][^15^]
def adaround_cd(W_hat, Xc, y_ref, iters=3): ...  # [^8^][^11^], 放行后启用

# ---------- 判题接口 ----------
def hif4_calibration_and_quantize_weight(w_q, w_s, calib_list):
    # Step 0→2→3→5: 反量化→统计→整形→GPTQ→AdaRound→真值选型→兜底
    return {"weight_params": ..., "activation_state": ...}

def hif4_dynamic_quantize_activation(a_q, a_s, state):
    # 反量化→应用state(÷s,旋转)→核心量化(快档)
    ...

def hif4_calibration_attention(calib_qkv_list, q_h, kv_h, d):
    # per-head 旋转门控 + p̄ 统计 + head 敏感度
    return {"q_state":..., "k_state":..., "v_state":...}

def hif4_dynamic_quantize_q(q_q, q_s, q_h, d, q_state): ...
def hif4_dynamic_quantize_k(k_q, k_s, kv_h, d, k_state): ...
def hif4_dynamic_quantize_v(v_q, v_s, kv_h, d, v_state):
    # 核心量化(加权核, w=p̄)
    ...
```

---

## 优先级与预期收益总表

| 步骤 | 内容 | 场景 | 预期收益 | 风险 | 依赖 | 关键文献 |
|---|---|---|---|---|---|---|
| 0 | 基线+兜底+自检 | 两者 | 防守（锁 0 分下界） | 无 | — | [^2^] |
| 1 | 核心量化器搜索 | 两者 | ★★★（组MSE −30~60%） | 低 | — | [^36^][^37^][^12^] |
| 2 | SmoothQuant+旋转(门控) | Linear, Q/K | ★★★ | 中(门控控制) | Step 1 | [^5^][^6^][^21^][^1^] |
| 3a | GPTQ(H=XᵀX) | Linear W | ★★ | 低 | Step 1 | [^3^][^15^] |
| 3b/3c | AdaRound+BRECQ(真值) | Linear W | ★★~★★★ | 低(W侧合法) | 官方放行 | [^8^][^9^][^11^] |
| 4a | Q/K per-head 旋转门控 | Attention | ★★★ | 中 | 官方放行 | [^17^][^6^] |
| 4b | V 注意力加权 | Attention | ★★ | 低 | — | [^18^][^19^] |
| 4c | head 敏感度预算分配 | Attention | ★ | 低 | — | [^17^][^35^] |
| 5 | 门控协议+预算协议 | 两者 | 稳健性(防过拟合) | 低 | 全部 | [^10^] |

---

## 参考文献

### 赛题指定与格式定义

- [^2^]: HiFloat4 Format for Language Model Pre-training on Ascend NPUs（HiF4 格式原始论文，赛题指定参考文献）. arXiv, 2026. https://arxiv.org/abs/2604.08826
- [^14^]: NVIDIA. Pretraining Large Language Models with NVFP4（NVFP4 训练配方：RHT + 随机舍入 + 2D 量化，赛题指定参考文献）. arXiv, 2025. https://arxiv.org/abs/2509.25149

### FP4/微缩放格式专项（最直接相关）

- [^1^]: Egiazarian, V., et al. Bridging the Gap Between Promise and Performance for Microscaling FP4 Quantization（MR-GPTQ：FP4 解析 MSE 理论、块级 Hadamard、格式专用 scale 优化）. arXiv, 2025. https://arxiv.org/abs/2509.23202
- [^21^]: Fine-grained Rotation Enhances Microscaling FP4 Quantization（旋转块尺寸与量化 group 对齐的结论）. arXiv, 2026. https://arxiv.org/abs/2604.17789
- [^12^]: SOAR: Scale Optimization for Accurate Reconstruction in NVFP4 Quantization（解析确定最优 scale）. arXiv 检索. https://arxiv.org/search/?searchtype=title&query=SOAR+Scale+Optimization+NVFP4

### Hessian 补偿谱系

- [^43^]: LeCun, Y., Denker, J., Solla, S. Optimal Brain Damage（OBD）. NeurIPS 1989. https://proceedings.neurips.cc/paper/1989/hash/6c9882bbac1c7093bd25041881277658-Abstract.html
- [^44^]: Hassibi, B., Stork, D. Optimal Brain Surgeon（OBS）. NeurIPS 1992. https://proceedings.neurips.cc/paper/1992/hash/303ed4c69846ab36c2904d3ba8573050-Abstract.html
- [^3^]: Frantar, E., et al. GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers. ICLR 2023. https://arxiv.org/abs/2210.17323
- [^15^]: Chee, J., et al. QuIP: 2-Bit Quantization of LLMs with Guarantees（LDLQ 理论）. NeurIPS 2023. https://arxiv.org/abs/2307.13304
- [^16^]: Tseng, A., et al. QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks（E8 格码本）. 2024. https://arxiv.org/abs/2402.04396
- [^32^]: GPTQ as Babai's Nearest Plane Algorithm（量化的格 CVP 视角）. arXiv 检索. https://arxiv.org/search/?searchtype=title&query=GPTQ+Babai+nearest+plane

### 输出重建家族

- [^8^]: Nagel, M., et al. Up or Down? Adaptive Rounding for Post-Training Quantization（AdaRound）. ICML 2020. https://arxiv.org/abs/2004.10568
- [^9^]: Li, Y., et al. BRECQ: Pushing the Limit of Post-Training Quantization by Block Reconstruction. ICLR 2021. https://arxiv.org/abs/2102.05426
- [^10^]: Wei, X., et al. QDrop: Randomly Dropping Quantization for Extremely Low-bit Post-Training Quantization. ICLR 2022. https://arxiv.org/abs/2203.05740
- [^11^]: Behdin, K., et al. QuantEase: Optimization-based Quantization for Language Models（坐标下降）. NeurIPS 2023. https://arxiv.org/abs/2309.01885

### 分布整形（缩放/旋转/置换）

- [^5^]: Xiao, G., et al. SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models. ICML 2023. https://arxiv.org/abs/2211.10438
- [^4^]: Lin, J., et al. AWQ: Activation-aware Weight Quantization for On-Device LLM Compression. MLSys 2024 最佳论文. https://arxiv.org/abs/2306.00978
- [^23^]: Shao, W., et al. OmniQuant: Omnidirectionally Calibrated Quantization for Large Language Models. ICLR 2024. https://arxiv.org/abs/2308.13137
- [^28^]: Wei, X., et al. Outlier Suppression+: Accurate Quantization of LLMs by Equivalent and Optimal Shifting and Scaling. EMNLP 2023. https://arxiv.org/abs/2304.09145
- [^24^]: Dettmers, T., et al. LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale. NeurIPS 2022. https://arxiv.org/abs/2208.07339
- [^6^]: Ashkboos, S., et al. QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs. NeurIPS 2024. https://arxiv.org/abs/2404.00456
- [^7^]: Liu, Z., et al. SpinQuant: LLM Quantization with Learned Rotations. Meta, 2024. https://arxiv.org/abs/2405.16406
- [^20^]: Lin, H., et al. DuQuant: Distributing Outliers via Dual Transformation Makes Stronger Quantized LLMs. NeurIPS 2024. https://arxiv.org/abs/2406.01721
- [^22^]: Sun, Y., et al. FlatQuant: Flatness Matters for LLM Quantization. 2024. https://arxiv.org/abs/2410.09426
- [^34^]: Hu, X., et al. OSTQuant: Refining LLM Quantization with Orthogonal and Scaling Transformations. ICLR 2025. https://openreview.net/forum?id=rAcgDBdKnP

### KV Cache / Attention 量化

- [^17^]: Liu, Z., et al. KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache（K per-channel / V per-token 的结构发现）. ICML 2024. https://arxiv.org/abs/2402.02750
- [^18^]: Hooper, C., et al. KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization（敏感度加权非均匀量化、sink 感知）. NeurIPS 2024. https://arxiv.org/abs/2401.18079
- [^19^]: Xiao, G., et al. StreamingLLM: Efficient Streaming Language Models with Attention Sinks. ICLR 2024. https://arxiv.org/abs/2309.17453
- [^29^]: Dettmers, T., et al. SpQR: A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression. ICLR 2024. https://arxiv.org/abs/2306.03078
- [^30^]: Li, M., et al. SVDQuant: Absorbing Outliers by Low-Rank Components for 4-Bit Diffusion Models（低秩吸收 outlier 的思想可迁移）. ICLR 2025. https://arxiv.org/abs/2411.05007
- [^35^]: Lin, Y., et al. QServe: W4A8KV4 Quantization and System Co-Design for Efficient LLM Serving. MLSys 2025. https://arxiv.org/abs/2405.04532

### 量化数学理论基础

- [^36^]: Lloyd, S. Least Squares Quantization in PCM（Lloyd-Max 最优标量量化）. IEEE Trans. Information Theory, 1982. https://doi.org/10.1109/TIT.1982.1056489
- [^37^]: Max, J. Quantizing for Minimum Distortion. IRE Trans. Information Theory, 1960. https://doi.org/10.1109/TIT.1960.1057548
- [^38^]: Panter, P.F., Dite, W. Quantizing Distortion in Pulse-Count Modulation（高分辨率失真 ≈ 步长²/12 的源头）. Proc. IRE, 1951. https://doi.org/10.1109/JRPROC.1951.273679
- [^39^]: Bennett, W.R. Spectra of Quantized Signals. Bell System Technical Journal, 1948. https://doi.org/10.1002/j.1538-7305.1948.tb01340.x
- [^40^]: Gish, H., Pierce, J. Asymptotically Efficient Quantizing（高码率失真界）. IEEE Trans. Information Theory, 1968. https://doi.org/10.1109/TIT.1968.1054193
- [^41^]: Kashin, B.S. Sections of Some Finite-Dimensional Sets and Classes of Smooth Functions（Kashin 分裂：正交变换平坦化的存在性）. Izv. RAN, 1977. 背景介绍: https://en.wikipedia.org/wiki/Kashin%27s_splitting_theorem
- [^42^]: Johnson, W.B., Lindenstrauss, J. Extensions of Lipschitz Mappings into a Hilbert Space（JL 引理）. Contemporary Mathematics, 1984. https://doi.org/10.1090/conm/026/737400
- [^25^]: Dong, Z., et al. HAWQ: Hessian AWare Quantization of Neural Networks with Mixed-Precision（Hessian 位宽分配）. ICCV 2019. https://arxiv.org/abs/1911.03852
- [^26^]: Choi, J., et al. PACT: Parameterized Clipping Activation for Quantized Neural Networks. 2018. https://arxiv.org/abs/1805.06085
- [^27^]: Esser, S., et al. Learned Step Size Quantization（LSQ）. ICLR 2020. https://arxiv.org/abs/1902.08153
