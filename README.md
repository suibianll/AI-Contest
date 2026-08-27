# HiF4 量化竞赛工程

华为 2026 算法竞赛赛道（NVFP4 → HiF4）的开发工作区。输入是 NVFP4 数据
（E2M1 载荷与块 scale），输出是 HiF4 表示。目标是在保持计算格式合法的前提下，
使反量化结果尽可能接近 NVFP4 reference。评测同时覆盖 Linear 层与 Attention
投影路径，分数是相对标准 HiF4 转换的 MSE 改善比例。

官方 B0 基线是 `youxilee/hif4` v2.0，官方结果为 `15313 / 137s`
（用户于 2026-08-27 确认）。当前官方记录是 v024（C21，`16043 / 173.8s`，
用户于 2026-08-27 确认）；根目录 `solution.py` 与该归档逐字节一致。本地增益不得
换算成官方分数声明。

英文版：[README_EN.md](README_EN.md)

## 工程结构

```text
solution.py                         唯一活跃、可提交的算法文件
evaluator/
  nvfp4_sim.py                      权威 NVFP4 编解码模拟器
  real_data_eval.py                 真实 GPT-2 评测器（默认 models/gpt2）
  synthetic_attention_eval.py       E1 合成 Attention 安全评测器
  requirements.txt                  评测依赖
models/gpt2/                        本地 GPT-2 权重（约 525MB，git 忽略）
solutions/
  README.md                         版本、分数、运行时间总表
  YYYYMMDD_vNNN_.../solution.py     不可变的历史算法源码
  YYYYMMDD_vNNN_.../result.md       各版本来源、结果与结论
tests/test_release_candidate.py     发布候选检查（含 E1 子集）
artifacts/                          本地运行的原始证据输出
docs/superpowers/                   设计规格、计划和执行日志
```

`solution.py` 是唯一活跃文件。`solutions/` 保存已经提交或明确记录的版本，
从不作为运行时依赖。`docs/superpowers/` 保存完整设计过程，不属于竞赛提交内容。

## 算法概览

当前 v2.0 系列方案的处理流程：

1. 按官方 NVFP4 scale 规则和 E2M1 载荷重建浮点 reference。
2. 为每个 Linear 层收集校准激活，搜索 SmoothQuant scale、通道置换以及
   Weight/Activation 误差重要度；宽层使用更细的 alpha 网格。
3. 校准通过安全门时，尝试 4/8/16 维确定性 signed Hadamard 块对角变换；
   未达到改善阈值则回退到对角路径。
4. 对 Weight、Activation 和 Q/K/V 的高误差 HiF4 块执行有预算的层级 scale 精修，
   使用绝对误差排序、二次统计和边缘扩展提高收益。
5. 在真实 Q/K/V Tensor 上校准 Attention 路径，支持 MHA/GQA head 分组，并通过
   同一动态量化 API 输出可提交的 HiF4 参数。

所有 state 都是普通 CPU Tensor 或标量。动态量化阶段只依赖校准 state，不依赖
评测器内部信息。

## 当前算法运行详解

本节按真实代码路径描述当前 `solution.py`（C21/v024）如何运行，所有函数名、
执行顺序与常量均与源码一一对应。

### 1. 评测器与六个官方 API 的调用时序

评测器 `evaluator/real_data_eval.py` 用 GPT-2 前向 2 个校准 batch 与 2 个测试
batch，hook 捕获每层激活，然后按以下时序调用 solution：

```text
校准阶段（每层一次）
  hif4_calibration_and_quantize_weight(weight_quant, weight_scale, calib_act_list)
      → {weight_params(五字段), activation_state}
  hif4_calibration_attention(calib_qkv_list, q_heads, kv_heads, head_dim)
      → {q_state, k_state, v_state}

测试阶段（每层每个 batch）
  hif4_dynamic_quantize_activation(act_quant, act_scale, activation_state) → 五字段
  hif4_dynamic_quantize_q / _k / _v(...) → 五字段
```

Linear 与 Attention 使用同一评分公式：

```text
score = (MSE(standard, reference) - MSE(candidate, reference))
        / MSE(standard, reference)
```

其中 standard 是“朴素 HiF4”（amax/7、阈值 lv2/lv3、round mantissa、无精修），
candidate 是本方案动态量化结果，reference 是 NVFP4 反量化浮点值。

### 2. HiF4 目标格式（五个合法字段）

输入是 NVFP4（E2M1 载荷与每 16 通道块 scale）。第一步
`_dequantize_nvfp4_float32` 重建 float32 稠密 Tensor。输出端把最后 64 个通道
视为一个顶层块，按层级分解：

```text
x = sign * mant * scale_lv3 * scale_lv2 * scale_factor

[... , 64] → reshape [..., blocks, 8, 2, 4]
scale_factor  每个顶层 64 块一个，E6M2 浮点 scale（标准取 amax/7 的最近 E6M2 码）
scale_lv2     每个 8 通道组一个，取值 {1, 2}（组级 ×2 指数）
scale_lv3     每个 4 通道子组一个，取值 {1, 2}
sign          ±1（mantissa=0 时规范为 0）
mant          尾数码 {0, 0.25, ..., 1.75}（即 code×0.25，code∈0..7）
```

最大可表示内层值为 `1.75 × 2 × 2 = 7 × scale`
（`_HIF4_MAX_INNER = 7.0`）。反量化按上式逐元素相乘
（`_dequantize_hif4`）。

### 3. Weight 校准（`hif4_calibration_and_quantize_weight`）

每个 Linear 层依次执行：

1. **统计收集**：反量化校准激活，累计每通道二阶矩 `sum_square`、amax，以及
   `_WEIGHT_QUADRATIC` 开启时的全协方差 `cov_sum = X^T X`。
2. **SmoothQuant 候选**：`d = act_amax^α / w_amax^(1-α)`，alpha 网格为
   `(0.25, 0.5, 0.75)`；宽层（in/out ≥ 2048）使用 5 档细网格。每档同时生成
   amax 与 RMS 版本，并除以几何均值防止整体漂移。
3. **通道置换候选**：`_hierarchy_aware_permutation` 把幅值相近的通道排入同一
   64 块，使块内 amax 更均匀、scale 利用率更高；然后扩展 Weight/Activation 的
   amax/rms 四种单侧 range 排序。
4. **候选评分与门控**：每个 `(d, perm)` 候选用采样行（不超过 256 个 Weight 行、
   128 个 token 激活样本）评分；`_candidate_is_safe` 要求均值改善超过门槛且最差
   样本不退化，否则拒绝。
5. **块 Hadamard 变换**（Matrix SmoothQuant 扩展）：在选定的 `d/perm` 上继续尝试
   4/8/16 维 signed Hadamard。变换正交，state 只存
   `block_smooth_size/seed` 两个整数。此步骤使用
   `_linear_output_candidate_metrics`（真实 Linear 输出 oracle）评分，是官方
   `A @ W` 禁令下的已知违规点。
6. **一次性全量变换**：胜出组合对完整 Weight 执行严格等价变换
   `W_t = W · D · P · R`；与 Activation 侧
   `X_t = X · D⁻¹ · P · R` 严格配对，代数上保证
   `X_t · W_t^T = X · W^T` 不变。
7. **Weight 编码**：调用 `_dense_to_hif4(weight_t, importance=H_x 对角,
   gram=4×4 块对角协方差, search_offsets, 预算)`，细节见第 4 节。
8. **8/16 组二阶精修**：`_refine_weight_groups8/16` 对 top-K 高损失的
   8/16 通道组执行坐标级精修，使用增量 `H·e` 公式。
9. **生成 activation_state**：保存 `smooth_inv = 1/d`、`permutation`、
   `block_smooth_*`、`importance`（按 `weight_hat` 列能量）、
   `gram/gram8/cross8`（Weight 空间 Gram 与交叉项，后者为违规点）、offset 集和
   精修预算；数据驱动时按“损失捕获比例”决定 ratio。

### 4. 核心编码器（`_dense_to_hif4`）

这是所有 Tensor 共用的量化路径：

1. reshape 为 `[blocks, 8, 2, 4]`，提取 sign 与 abs。
2. 将标准 scale `amax/7` 编码成最近 E6M2 码再解码，保证格式合法。
3. 阈值层级：`max8 ≥ 4·scale → lv2=2`；
   `max4 ≥ 2·scale·lv2 → lv3=2`；mantissa 为
   `round(|x|·4/denominator)`，clamp 到 0..7 后乘 0.25。
4. **hard 块筛选**：归一化误差大于 `1e-7` 的块进入精修池，按绝对加权损失
   top-K 截断到 `max_refine_ratio × 总块数`，同时受最大块数限制。
5. **offset 批量搜索**：对 hard 块将 `standard_code + offsets` 沿 offset 维展开为
   `[K, N]`，一次调用 `_solve_exact_hierarchy` 精确求解。每个候选 scale 枚举总指数
   `2^e (e=0,1,2)`，生成三张损失表，使用对角 importance 或
   `Δ^T G Δ` 二次型，精确选择 lv2/lv3/mantissa，再逐块 argmin 选择 offset。
6. **边缘扩展**：若胜出 offset 位于搜索集边缘，Weight 侧 `(-2,…,3)`、
   Activation 侧 `(-1,…,3)` 的块继续向外最多搜索 2 步。
7. **L1 数据驱动 scale**：当前 `_L1_DATA_DRIVEN_SCALE = False`，默认关闭；
   启用时对当前胜者生成最小二乘 scale 与分位数 trim 候选，再经过精确求解和逐块回退。
8. **接受门**：只有 `best_loss ≤ (1-margin)·standard_loss` 才写回精修结果，
   否则保持标准参数；任何精修只允许变好。

### 5. 动态 Activation 量化（`hif4_dynamic_quantize_activation`）

每个测试 batch 逐层调用，全程无梯度：

```text
反量化 NVFP4 → × smooth_inv → 通道置换 → 块 Hadamard
→ _dense_to_hif4(importance, gram4, offsets, 预算)
→ _refine_weight_groups8(gram8, 可选 cross8)   # top-K 8 通道组二阶精修
```

Activation 侧 gram/gram8 来自 Weight 空间
`weight_smooth^T · weight_smooth` 的块对角，cross8 来自
`(W_hat − W)·W_hat^T` 交叉项。这些 state 字段属于 C18–C21 cross 机制，也是
官方禁令下待删除的违规路径。

### 6. Attention 校准（`hif4_calibration_attention`）

Q/K/V 使用独立通道。核心思想是利用 `Q·K^T` 点积的严格等价变换：
`d_kv` 按 head 对齐，Q 侧乘 `d`，K 侧乘 `1/d`，因此点积不变。

1. **统计**：计算 per-head 二阶矩与峰值；K 还计算 midrange 居中后的版本。
2. **A1 上下文**：使用校准前缀的真实 Q/K/V 计算 reference Attention，包含 causal
   与 non-causal 双轨；固定 V 量化结果，隔离 Q/K 变换选择。
3. **Smooth-QK**：`d = k_peak^α / q_peak^(1-α)`，GQA 对齐到 kv-head 粒度。
4. **K 居中**：使用 softmax 平移不变性的精确 midrange centering。
5. **headwise 置换**：Q/K 共享同一 head 内置换，保证点积不变。
6. **双轨选择与终验门**：A1 轨按真实输出误差选择变换，proxy 轨按 B0 式重建
   proxy 选择变换；各自产生 winner 后，用完整部署路径
   `hif4_dynamic_quantize_q/k/v` 重算输出误差。A1 无明确优势或安全轨退化时回退
   proxy winner。
7. **A3 V importance**：Q/K 定稿后，比较 head 级
   `E[A²]`、`E[A]`、`E[A²]+E[A]²` 三种重要性，同样经过真实输出门控。
8. 输出 `q_state/k_state/v_state`，包含 multiplier、permutation、importance、offsets
   与精修预算；A2 H64 旋转当前默认关闭。

动态 Q：`× d_q → 置换 → 编码(importance=h_k, offset 搜索)`；
动态 K：`居中 → × 1/d → 置换 → 编码(importance=h_q)`；
动态 V：`编码(importance=head 级重要性)`。

### 7. 流程总结

```text
                校准（每层一次，CPU state）                 动态（每 batch）
  ┌─────────────────────────────────────────────┐   ┌──────────────────────┐
  │ NVFP4 → float32 → 统计(矩/amax/协方差)        │   │ NVFP4 → float32      │
  │ → Smooth d / 置换 P / Hadamard R 搜索+门控    │   │ → ×D⁻¹ → P → R       │
  │ → W_t = W·D·P·R (严格等价)                    │   │ → _dense_to_hif4     │
  │ → _dense_to_hif4 + 8/16 组二阶精修 → W 参数    │   │   (offset 搜索+精确层级)│
  │ → activation_state {D⁻¹,P,R,importance,      │   │ → 8×8 组二阶精修      │
  │    gram/gram8/cross8, 预算}                  │   │ → HiF4 五字段         │
  └─────────────────────────────────────────────┘   └──────────────────────┘
  Attention: Smooth-QK + K 居中 + headwise 置换（点积严格不变），A1 真实输出门控
```

设计要点：所有“等价变换”只利用缩放对、置换对、正交旋转对等代数恒等式，保证
`X·W^T` 与 `Q·K^T` 严格不变；量化误差削减发生在变换后的坐标空间。所有 state
均为普通 CPU Tensor，动态阶段不依赖评测器内部。

合规提示：第 3 节第 5 步、第 5 节 cross8，以及 Weight 校准中的
`_linear_output_candidate_metrics`、`_activation8_gate_decisions` 输出监督路径，
在官方 `A @ W` 禁令口径下属于待删除项。整改方案见
`docs/superpowers/plans/2026-08-27-hif4-26000-algorithm-implementation-plan.md`
Phase 0。

## 最新已验证算法

官方最优（合规锚点）是 v025（候选 C21-C，提交 `3c1366b`）：官方结果
`14437 / 166.6s`。根目录 `solution.py` 当前的 SHA256 为
`648a27b3560ef7f5d939cd409301e445e5065047cbd5438c1a73a013730e467f`
（C38 超参组合，本地 Linear mean 0.5695 / CPU 99s，官方结果待回填）。

下表列出官方闭环锚点与本地已实现机制链。每个机制都是独立候选并单独归档；
主效应来自 candidate ledger 的 offset-0 记录（本地配对，非官方绝对分数）。

官方闭环锚点：

| 版本 | 机制 | 官方分数 | 时间 |
|---|---|---:|---:|
| v000 | v9 基线 | ~9000+ | NA |
| v001 | 旧基线 | 10250 | 127s |
| v002 (B0) | youxilee/hif4 v2.0 | 15313 | 137s |
| v013 (C10) | 宽层 Activation 二次精修 | 15799 | 144s |
| v024 (C21) | 门控精确交叉选择 | 16043 | 173.8s（含违规灰路径） |
| v025 (C21-C) | 合规基线 | 14437 | 166.6s |

C21 机制链：

| # | 机制 | 候选 | 验证 | 主效应（offset 0） |
|---|---|---|---|---|
| 1 | 输出感知 Attention 选择器 | C1 / v003 | 本地 | causal Attention +7.12pp |
| 2 | top-K 8×8 Weight 二阶精修 | C3 / v006 | 本地 6/6 | Linear +1.10pp |
| 3 | top-K 16×16 Weight 二阶精修 | C5 / v008 | 本地 6/6 | Linear +0.23pp |
| 4 | 宽层（3072 FFN）Activation 二次精修 | C10 / v013 | 官方 | proj +0.54pp |
| 5 | 宽层 Activation 8×8 残差 | C11 / v014 | 本地 6/6 | proj +0.31pp |
| 6 | 校准门控全宽度 Activation 8×8 | C14 / v017 | 本地 6/6，全分项安全 | Linear +0.45pp |
| 7 | 门控 Activation 8×8 覆盖 8% | C17 / v020 | 本地 6/6，36/36 分项 | Linear +0.29pp |
| 8 | 校准门控精确交叉选择 | C21 / v024 | 官方 | Linear +0.15pp；修复 C20 pow2 回退 |

累计效应：Attention 路径保留 A1 的 `+7.12pp` causal 增益；Linear mean 从 C1 的
`0.5668` 提高到 C21 的 `0.5930`，约 `+2.62pp`。所有候选均通过
`evaluator/real_data_eval.py` 固定回归矩阵（amax6/amax4/pow2 × MHA/GQA ×
causal/non-causal，offset 0/97/193/389）和
`evaluator/synthetic_attention_eval.py` 冻结合成矩阵（8 个场景、576 case）。

合规提示：官方已明确 Linear 校准不得使用 `A @ W` 或数学等价的输出监督拟合
`Q(A)`。C21 的 Linear 校准包含 `_linear_output_candidate_metrics`、
`group_cross8` 等输出监督路径，按新口径属于不合规实现。下一条主线 HiF4-OSQ
将先删除这些路径并建立合规基线 C21-C，再依次引入 64 维 Hadamard 旋转、full-64
GPTQ Weight 精修、top-K full-64 Activation 求解和可学习等价 scale。详见
`docs/superpowers/plans/2026-08-27-hif4-26000-algorithm-implementation-plan.md`。
主目标为官方 `22000~25000`，`26000` 为 stretch 目标，官方时间上限为 `300s`。

## 评测方法

评测器加载真实 GPT-2 Weight 和文本前向 Activation，并捕获每层 Q/K/V、Attention
投影及 FFN 输入。每个样本生成：

- NVFP4 反量化结果（reference）；
- 标准 HiF4 结果（baseline）；
- 当前 `solution.py` 的校准/动态量化结果（candidate）。

每个 Linear 或 Attention 样本均按以下公式评分：

```text
score = (MSE(standard, reference) - MSE(candidate, reference))
        / MSE(standard, reference)
```

然后在各层和测试 batch 上取平均。评测器与远程 hif4 工程共享核心评分路径，另增加
可配置的 `--solution` 与 `--model` 参数。

`evaluator/synthetic_attention_eval.py`（E1）运行冻结的 8 场景、576 case 合成
Attention 矩阵，覆盖 saturated logits、near-uniform、V outliers、heavy tails 等，
作为 Attention 路径修改的安全门，用于预筛真实数据评测无法暴露的退化。

## 环境与使用

使用工程虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r evaluator\requirements.txt
```

GPT-2 Weight 位于 `models/gpt2/`（约 525MB，已被 `.gitignore` 排除）。评测器默认
加载该目录，无需联网。默认运行 GPT-2 12 层、2 个校准 batch 与 2 个测试 batch：

```powershell
.\.venv\Scripts\python evaluator\real_data_eval.py
```

GPU 加速使用 `--device cuda`，默认设备为 `cpu`。`--model` 可接受 Hugging Face 名称
或其他本地模型目录：

```powershell
.\.venv\Scripts\python evaluator\real_data_eval.py `
  --solution solution.py --model gpt2 --device cuda
```

开发阶段的快速方向比较：

```powershell
.\.venv\Scripts\python evaluator\real_data_eval.py `
  --layers 1 --seq 16 --calib 1 --test 1
```

运行全部 576 个冻结合成 Attention case：

```powershell
.\.venv\Scripts\python evaluator\synthetic_attention_eval.py `
  --solution solution.py
```

发布检查，包括 state 合法性、参数字段、feature-off 等价性和合成子集：

```powershell
.\.venv\Scripts\python -m pytest tests\test_release_candidate.py -q
```

关键参数包括 `--layers`、`--seq`、`--calib`/`--test`、
`--mode`（`amax6`/`amax4`/`pow2`）、`--kv-heads`（GQA smoke）、
`--token-offset`（固定本地测试窗口）。输出包含六个 Linear 分项
（q/k/v/o/fc/proj）、causal/non-causal Attention 分数，以及统一边界的
algorithm-stage/API 计时。

## 本地评测与归档流程

官方评测器并非持续可用，因此使用已知 B0 官方结果作为基线锚点。后续候选根据可复现
的本地配对结果晋级；不等待新官方分数，也不从本地指标推算官方绝对分数：

1. B0 与候选必须使用完全相同的模型、设备、mask、mode、token offset 和 batch 数量
   进行配对评测。
2. offset `0` 是开发集；`97`、`193`、`389` 是固定本地回归窗口。它们已经用于 A1
   仲裁，不再声称为 blind set，也禁止针对它们调参。
3. 开发筛选覆盖 `amax6/amax4/pow2`、MHA/GQA 和 causal/non-causal；head_dim 128
   与 saturated-logit 区域由冻结合成安全矩阵覆盖。
4. 晋级必须同时满足目标均值、逐层 tail、state 合法性、E1 合成安全轨和 CPU 时间门。
5. 晋级后创建本地结果归档，记录精确源码 SHA256、完整配置、分项分数和运行时间；
   `Official Score/Time` 保持 `NA`，禁止填入本地估计。

版本历史：

- v000：旧 v9 基线，官方约 9000+；
- v001：旧活跃基线，`10250 / 127s`；
- v002：`youxilee/hif4` v2.0，官方 B0，`15313 / 137s`，已关闭；
- v013：C10 宽层 Activation 二次精修，官方 `15799 / 144s`；
- v024：C21 门控精确交叉选择，当前官方记录 `16043 / 173.8s`；根目录
  `solution.py` 与该归档逐字节一致。完整机制链、源码 SHA 和固定矩阵结果见
  `solutions/README.md` 与 progressive candidate ledger。

本地指标与官方分数的四锚点校准记录在
`docs/superpowers/logs/2026-08-27-evaluator-calibration-report.md`：本地 Linear mean
每提高 1pp，约对应 297 个官方分数，关系相对稳定；本地 Attention 增益的兑换较弱。
因此本地 Linear 是高杠杆指标，合成矩阵是安全轨而非得分杠杆。

`.venv/`、Python cache 和其他本地产物均被 `.gitignore` 排除，不进入算法归档或
竞赛提交。
