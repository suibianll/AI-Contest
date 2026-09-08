# A25：单主干互逆变换训练 + 四维度机制诊断 + 三方对照

登记于 2026-09-07；一个机制、一个配置，见 config.json。官方状态 NA。
隶属 [21071 机制证据驱动下一轮计划](../../../../docs/superpowers/archive/plans/21071-evidence-driven-research.md)。
用户指令（2026-09-07 23:36）：停止"保留旧训练+叠加残余+改代理目标"的复杂度堆积路线；
以一套互逆变换训练为主干，先检查机制是否充分生效，不用已发现无效的梯度，
把"替换旧训练"与"新机制收益"分开验证。

## 路线问题回顾

A21→A22→A23→A24 四代的共同模式：保留 R3 旧训练（Cayley rotation+center）→
在其上叠加残余 exp(±S) → 改 scale 代理目标。复杂度持续增加（每代加一层训练），
但没有确认核心互逆变换到底解决了多少量化误差。A24 发现 STE 输出误差梯度
在互逆参数化下精确为零（float64 FD 仲裁）——说明"直接优化输出误差"这条
梯度路径走不通，但 scale 代理目标（A22-2 可加 +19、A23 乘积 +13 官方正向）
确实有效。问题在于：没有诊断清楚互逆变换改变了什么。

## 机制设计

### 主干（单套训练，不保留旧训练）

- 父 = 基础栈 B（`_V189_CALIBRATION_ATTENTION` 输出，无 learned_rotation/
  learned_center）。**不执行 R3 旧训练**——新训练独立承担优化。
- 互逆变换参数化：T = exp(S)，Q′ = Q·T，K′ = K·T⁻ᵀ（S 对称零迹、谱约束
  ±log2/2、cond≤2），与 A21/A22 同族但**无 R0 Hadamard 初始化**（S=0 起步，
  纯学习而非 Hadamard 预旋转后学习）。
- 训练目标：A22-2 官方正向验证过的**可加 scale 代理**——每 role 按实际连续
  64 块 amax 比值平方、Q/K 分别块均值后相加、等权 fold、epsilon/amax 并列
  极值次梯度。**不用 STE 输出误差目标**（A24 已证互逆参数化下精确为零）。
- 32 步 Adam/lr0.01/clip1、手工矩阵指数梯度、谱投影——优化器与 A22-2 相同。
- gate：对 B 真实 readout MSE（原最后校准窗口），严格小于才接受，平局保留 B。
  gate 是真实误差选择器（不是 scale 代理）——即使 scale 代理训练的提案，
  最终部署与否由真实输出决定。

### 与 A21-1 的关键区别

A21-1 也是"B + 新 scale 训练"结构，但：① 用 R0=Hadamard 预旋转
（learned_rotation = R0·exp(±S)），本卡不预旋转；② A21-1 没有四维度诊断；
③ A21-1 没有三方对照；④ A21-1 官方 14199（−206），本卡的目标是先诊断
再决定是否提交。

### 四维度机制诊断（本卡核心产出）

对每个 FA 层、每个校准窗口，在 S=0（变换前）和 S=trained（变换后）分别记录：

| 维度 | 记录内容 | 判读 |
|---|---|---|
| **① scale** | 每 role/块 amax 比值（变换后/变换前） | scale 是否真降？降多少？ |
| **② 量化码** | 变换前后合法 HiF4 五字段码字变化计数 | 合法编码是否真改变？变多少？ |
| **③ QK** | Q·Kᵀ 的 MSE（变换后量化 vs 未量化参考） | QK 误差是否改善？ |
| **④ Attention 输出** | softmax(QK)·V 的 MSE（变换后量化 vs NVFP4 参考） | 最终输出是否改善？ |

四种状态判读：
- **没学动**：S_norm≈0、scale 比值≈1、码字零变化 → 训练失败，机制未生效
- **scale 降但码没变**：scale 代理降但量化码字未变 → 代理与真实编码脱节
- **码变了但输出没改善**：码字变化但 QK/输出 MSE 不降 → 变换方向不对
- **输出改善**：四维度全链条正 → 机制有效

### 三方对照（分开删除损失与新训练收益）

| 方案 | 构成 | 测量 |
|---|---|---|
| **B**（基础栈） | v189 校准，无任何 learned 变换 | 基线 |
| **P_old**（R3 旧训练） | B + R3 Cayley rotation+center | 旧训练收益 = P_old − B |
| **C**（新训练） | B + 新互逆 exp(±S) scale 代理训练 | 新训练收益 = C − B；净增量 = C − P_old |

三方对照明确回答：删除旧训练损失多少（P_old−B）、新训练贡献多少（C−B）、
新训练是否优于旧训练（C−P_old）。最终由完整官方结果决定。

## 去重审读

| 对照 | 与 A25 的区别 |
|---|---|
| A21-1 | R0 Hadamard 预旋转 + 无诊断 + 无三方对照 |
| A22-2 | 保留 R3 旧训练 + 残余叠加（两套训练）|
| A23 | 保留 R3 旧训练 + 残余叠加 + 乘积目标 |
| A24 | STE 输出误差目标（梯度精确为零，已关闭）|
| A2 旧训练 | 同形输出误差目标但 Cayley 正交（非互逆对称）+ B 坐标 |

## 验证清单

- 手工 exp/scale 梯度 vs autograd（float64 参考，不用 float32 matrix_exp autograd——A24 发现数值损坏）。
- S=0 强制回退 = B 逐位（合成 + 真实 4B 窗口）。
- 互逆转置（T·T⁻ᵀ=I 误差）、GQA 组映射、inference_mode、fuzz 合同、隔离导入。
- 四维度诊断脚本在真实 4B 窗口上运行（6 FA 层）。
- V 与两个 Linear API 逐位冻结。

## 评测与官方

4B 面板 paired：双对照（vs B 和 vs R3），72 case。判读门：Δmean>0、
validation/test 各正、mean(max(−Δgain,0))<0.02。无本地时间门；官方 300s。
**本地 paired 仅风险记录**（A23 实证：4B paired 负向但官方 +13）；
最终由完整官方结果决定。

## 复现入口

CUDA venv：`build.py` → `verify.py` → `check_math_and_import.py` →
`fuzz_check.py` → `diagnose.py`（四维度诊断）→ `run.py screen` → `run.py full`。
结果目录 `artifacts/proxy_v3/continuous/attention/anchor25-a1/`。
