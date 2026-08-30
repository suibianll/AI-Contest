# 当前主版本算法效果与评测状态

> 更新日期：2026-08-30
> 适用文件：根目录 [`solution.py`](../solution.py)
> 文档性质：本地可复现实测记录，不是官方成绩承诺。

## 一句话结论

根目录已经从 C86 实验集合重写为单一路径的 BOAT + HSDQ 实现。固定 Qwen2.5-0.5B
缓存、`seq=128`、`calib=2`、`test=4`、全 24 层、CPU 的完整运行中，当前主版本
的 Qwen shaped panel 为 **293.755106**，正式 API 累计 **382.153528 s**，低于
`420 s` 限制；该数值用于本地 A/B 排序，不能线性换算为官方排行榜分数。

## 当前实现

`solution.py` 只保留六个正式 API 和必要的 codec/优化原语：

1. **BOAT（Block Output-Alignment Transform）**：用激活/权重各自的 RMS 构造
   对角平衡 `D`，再搜索 4/8/16/64 维 signed-Hadamard 块和两个确定性 seed。连续
   乘积保持不变：

   $$X'=XD^{-1}R,\qquad W'=W D R,\qquad X'W'^T=XW^T.$$

   候选只依赖两侧 operand-local HiF4 重建误差，不构造 Linear 输出，因此固定参数
   可以安全写入 `activation_state`。
2. **Cross-fold Weight-HSDQ**：对满足宽度/形状条件的权重块使用校准激活
   `A_f^T A_f` 的低秩 Hessian，对 HiF4 的 15 个 signed levels 做精确二次增量
   搜索；fold 1 生成的候选必须改善 fold 2，最终只改变离线 `weight_params`。
3. **Gram-hierarchy Activation-HSDQ**：从静态变换后权重计算 64 维 Gram block，
   先按二次型选择层级和 E6M2 offset，再做最多 128 个 block、2 轮坐标扫描。状态
   只保存 CPU 上的静态 `gram64`、BOAT 逆缩放和整数/符号配置。
4. **Attention 输出感知 shortlist**：搜索 reciprocal RMS 平衡、K-centering 和
   16/32/64 维共享 signed-Hadamard；先用便宜代理排序，再对前 4 个候选用真实
   non-causal Attention 输出和部署侧 Gram-HSDQ 复评。V 保持独立合法 HiF4 编码。

## 最新全层实测

报告文件：[`clean-gram-hierarchy-full.md`](../logs/evaluations/clean-gram-hierarchy-full.md)；
原始 JSON：[`clean-gram-hierarchy-full.json`](../artifacts/real_model_suite/clean-gram-hierarchy-full.json)。

固定输入为 Qwen2.5-0.5B（24 层、hidden 896、14 Q heads、2 KV heads、head dim 64），
calibration 使用 train 的 2 个窗口，test 使用 validation 的 4 个不重叠窗口。

| 指标 | 当前主版本 | 旧 C86 | 变化 |
|---|---:|---:|---:|
| Linear native mean | 0.501558 | 0.477821 | +0.023737 |
| Attention native mean | 0.841829 | 0.739264 | +0.102565 |
| Qwen panel Linear | 125.389403 | 119.455153 | +5.934250 |
| Qwen panel Attention | 168.365703 | 147.852757 | +20.512947 |
| Qwen panel total | **293.755106** | 267.307909 | **+26.447197（+9.89%）** |
| official-flow native total | 417.862253 | 392.064774 | +25.797479 |
| six-API time | **382.153528 s** | 313.577669 s | +68.575859 s |
| wall time | 414.025852 s | — | `<420 s` |

当前版本的官方流程诊断由 672 个 Linear case 和 96 个 Attention case 求和得到：

| 组件 | case 数 | gain sum | gain mean | global gain |
|---|---:|---:|---:|---:|
| Linear | 672 | 337.046716 | 0.501558 | 0.436952 |
| Attention | 96 | 80.815538 | 0.841829 | 0.857768 |
| 合计 | 768 | **417.862253** | — | — |

`panel_score` 不是把 768 个 case 复制成 450 个，而是保留组件均值后投影：

$$P_L=250\times0.5015576125=125.389403,$$

$$P_A=200\times0.8418285164=168.365703,$$

$$P_{total}=P_L+P_A=293.755106.$$

因此 `official_flow_total` 与 `panel_score.total` 同时出现是设计结果，不是计算冲突。

## Linear 角色归因

| 角色 | native mean |
|---|---:|
| q | 0.616561 |
| k | 0.620526 |
| v | 0.563596 |
| o | 0.483463 |
| fc_gate | 0.375126 |
| fc_up | 0.430255 |
| proj | 0.421376 |

`v` 与 `fc_gate` 是当前最弱角色；但从总体收益看，扩张 FFN 和输出投影仍受跨
64-block 相关性、校准 fold 数量和运行时约束共同限制，不能仅靠增加 offset 或 sweep
解决。

## 距离 Linear 0.9 与 36,000

当前 Linear native mean 为 `0.5015576125`。若以本地 panel 的 mean 作为诊断目标：

$$\Delta g_L=0.9-0.5015576125=0.3984423875,$$

$$\frac{\Delta g_L}{1-g_L}=\frac{0.3984423875}{0.4984423875}=79.94\%.$$

也就是还要消除当前 Linear 剩余归一化误差的约 **79.94%**，对应 250-case panel
仍差 **99.6106** 分。这个数轴不是官方排行榜的绝对分数。

官方历史合规锚点为 C66：`22557 / 217.2s`；外部参考 `youxilee/hif4` 为用户提供的
`24153 / 239s`。从 C66 到 `36000` 的官方分差是 **13443**，但当前本地 panel
不做官方绝对分回归，因此不能声称当前版本对应某个官方分数或已经接近 `36000`。

## 时间与合规

- 六个 API 累计：calibration `216.530669 s` + dynamic `165.622859 s`
  = `382.153528 s < 420 s`。
- 调用次数：weight calibration 168；attention calibration 24；dynamic activation
  672；Q/K/V 各 96。
- `activation_state` 只包含 BOAT 逆缩放、静态 Gram、旋转整数配置和符号；输出
  监督只用于离线 Attention 候选和权重侧选择，不进入在线 `Q(A)`。
- 当前源码 SHA256：
  `5d1128cc79fef58154da2f600ec4b472ff95030e1f1e61b96593d06fd9aac94f`。
- 发布前检查：`34 passed`，包含 reference HiF4、Linear 合规和真实模型套件测试。

历史 C86 源码和每次实验结果仍在 `solutions/`、`logs/` 与 Git 历史中，本文只描述
当前根目录实现；历史记录不应被当作当前代码行为。
