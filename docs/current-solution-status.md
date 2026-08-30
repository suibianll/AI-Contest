# 当前主版本算法效果与评测状态

> 更新日期：2026-08-30
> 适用文件：根目录 [`solution.py`](../solution.py)
> 文档性质：本地可复现实测记录，不是官方成绩承诺。

## 一句话结论

根目录已经从 C86 实验集合重写为单一路径的 BOAT + HSDQ 实现。固定 Qwen2.5-0.5B
缓存、`seq=128`、`calib=2`、`test=4`、全 24 层、CPU 的完整运行中，当前主版本
的 Qwen shaped panel 为 **293.755106**，正式 API 累计 **382.153528 s**，低于
`420 s` 限制；该数值用于本地 A/B 排序，不能线性换算为官方排行榜分数。

2026-08-30 已按执行计划完成 E1→A5 的连续实验，并继续执行真正跨 block LRH-r8；
所有候选均已归档并回退，根目录没有留下未经验证的算法。当前最高可信版本仍是
上述 stable parent；官方评测不可用期间，以固定 Qwen panel 继续验证后续独立路线。

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

## 外部代码本地复测与最高基准

外部 [`youxilee/hif4`](https://github.com/youxilee/hif4) 的 v2.7 提交
`dd5ee6515323169dbd4133b3d4fd1ff1cb7be646` 已在本地用未修改源码复测。直接走
CUDA 会在外部代码的 CPU state / CUDA activation 混用处触发 device mismatch，因而
以下结果采用 `device=cpu, algorithm_device=cpu`，固定
`amax6 / seq=128 / calib=2 / test=4 / cache=read`。这是一组代理诊断，不是官方
评测复现；逐模型证据见
[`外部 v2.7 本地差距审计`](../logs/candidates/2026-08-29-external-hif4-gap-analysis.md)。

| 外部本地模型 | Linear | Attention | native total | API(s) | 基准含义 |
|---|---:|---:|---:|---:|---|
| gpt2-small | 150.431816 | 24.250501 | 174.682317 | 108.66 | 软 guardrail |
| gpt2-medium | 258.627694 | 48.719212 | 307.346907 | 320.94 | 软 guardrail |
| opt-125m | 28.245796 | 19.574717 | 47.820513 | 106.06 | 软 guardrail |
| pythia-160m | 145.543961 | 40.822630 | 186.366591 | 108.08 | 软 guardrail |
| **qwen2.5-0.5b** | **303.581186** | **65.946083** | **369.527269** | **357.67** | **最高单模型 native 基准** |
| 五模型相加 | 886.430453 | 199.313144 | 1085.743597 | — | **仅诊断，禁止作排名分** |

本地排名必须使用最高的**同模型、同面板口径**，而不是把不同模型层数直接相加：

1. **最高单模型 native**：外部 Qwen `369.527269`；它是五个代理中最高值，且
   结构上最接近新增的 GQA/RoPE/SwiGLU 压力用例。
2. **最高同口径 panel（主比较基准）**：外部 Qwen `250.327102`，由
   `250 × (303.581186 / 672) + 200 × (65.946083 / 96)` 得到；五模型合计不参与
   这个投影。
3. **最高官方基准**：用户确认的外部官方 `24153 / 239s`。本地 native/panel
   没有官方缩放因子，不能把 `369.527269` 线性换算成 `24153`。

当前根与外部最高本地基准的差值如下：

| 比较口径 | 当前根 | 外部最高基准 | 当前根领先 |
|---|---:|---:|---:|
| Qwen native total | 417.862253 | 369.527269 | **+48.334984（+13.08%）** |
| Qwen shaped panel | 293.755106 | 250.327102 | **+43.428004（+17.35%）** |
| panel Linear | 125.389403 | 112.939429 | **+12.449974** |
| panel Attention | 168.365703 | 137.387673 | **+30.978030** |

因此，后续本地算法 A/B 应以外部 Qwen `250.327102` 作为第一比较线，
以外部 Qwen native `369.527269` 作为第二诊断线；`1085.743597` 只能用于检查
跨模型结构性回退，不能作为“外部最高分”或与官方 `24153` 做差值。

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

## 2026-08-30 执行计划结果

以下均为同一 Qwen2.5-0.5B、24 层、`cache=read` 的本地 shaped panel；parent
永远保留，候选失败后已恢复。详细 fold 与角色数据见各 execution log。

| 实验 | panel | Linear mean | API time | 裁决 |
|---|---:|---:|---:|---|
| E1 progressive full-hierarchy | 290.923906 | 0.490233 | 693.21s | 拒绝，跨层回退且超时 |
| A2 expansive sparse-row | 292.831952 | 0.497865 | 385.48s | 拒绝 |
| A3 rowwise block-leverage | 293.250467 | 0.499539 | 384.83s | 拒绝 |
| A4 blockwise BOAT-2 | 292.978009 | 0.498449 | 368.23s | 拒绝 |
| A5 joint-fold offline A@W | 284.595177 | 0.464918 | 358.24s | 拒绝 |
| A3 true cross-block LRH-r8 | 292.426982 | 0.496245 | 381.84s | 拒绝 |
| A4 full CAT-inspired BOAT-2 | 283.159693 | 0.459176 | 600.61s | 拒绝，超时 |
| A5 frozen-Q(A) ridge/Qronos | 293.755106 | 0.501558 | 455.73s | 持平但超时 |
| A6 Global Activation-LRH | 282.616646 | 0.457010 | 373.97s | 拒绝 |
| stable parent（当前根） | **293.755106** | **0.501558** | **382.15s** | **active** |

归档目录：`solutions/20260830_v087...` 至 `solutions/20260830_v095...`；
执行日志：`logs/execution/2026-08-30-e1-progressive-hsdq.md`、
`2026-08-30-a2-expansive-sparse-hsdq.md`、
`2026-08-30-a3-rowwise-block-hsdq.md`、
`2026-08-30-a4-blockwise-boat.md`、
`2026-08-30-a5-joint-aw.md`、
`2026-08-30-a3-lrh-r8.md`、
`2026-08-30-a4-cat-boat2.md`、
`2026-08-30-a5-frozen-qronos.md`、
`2026-08-30-a6-global-activation-lrh.md`。

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
