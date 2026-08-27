# HiF4 Quantization Competition Project

Development workspace for the Huawei 2026 algorithm competition track
(NVFP4 → HiF4). The input is NVFP4 data (E2M1 carrier + block scale);
the output is a HiF4 representation. The goal is to make the dequantized
result as close as possible to the NVFP4 reference while keeping the
compute format legal. Evaluation covers both Linear layers and the
Attention projection path; the score is the MSE improvement relative to
the standard HiF4 conversion.

The official B0 baseline is `youxilee/hif4` v2.0, officially closed at
`15313 / 137s` (user-confirmed 2026-08-27). The current official record
is v024 (C21, `16043 / 173.8s`, user-confirmed 2026-08-27); the root
`solution.py` is byte-identical to that archive. Local gains must not
be converted into official-score claims.

## Project Structure

```text
solution.py                         The only active, submission-ready algorithm file
evaluator/
  nvfp4_sim.py                      Authoritative NVFP4 encode/decode simulator
  real_data_eval.py                 Real GPT-2 evaluator (defaults to models/gpt2)
  synthetic_attention_eval.py       E1 synthetic attention safety evaluator
  requirements.txt                  Evaluation dependencies
models/gpt2/                        Local GPT-2 weights (~525MB, git-ignored)
solutions/
  README.md                         Master table of versions, scores, runtimes
  YYYYMMDD_vNNN_.../solution.py     Immutable archived algorithm sources
  YYYYMMDD_vNNN_.../result.md       Origin, results, conclusions per version
tests/test_release_candidate.py     Release-candidate checks (incl. E1 subset)
artifacts/                          Raw evidence outputs of local runs
docs/superpowers/                   Design specs, plans, execution logs
```

`solution.py` is the only active file. `solutions/` holds versions that
were submitted or explicitly recorded; it is never a runtime dependency.
`docs/superpowers/` preserves the full design process and is not part of
the competition submission.

## Algorithm Overview

The current v2.0-line solution pipeline:

1. Reconstruct the floating-point reference using the official NVFP4
   scale rules and the E2M1 carrier.
2. Collect calibration activations per Linear layer and search
   SmoothQuant scaling, channel permutations, and weight/activation
   error importance; wide layers use a finer alpha grid.
3. When calibration passes the safety gate, try block-diagonal
   Hadamard transforms, enumerating block sizes 4/8/16 with
   deterministic sign seeds; fall back to the diagonal path if the
   improvement threshold is not met.
4. Apply budgeted hierarchical scale refinement to weights, activations,
   and high-error HiF4 blocks of Q/K/V, using absolute-error ordering,
   quadratic statistics, and boundary extension to increase the gain.
5. Calibrate the attention path on real Q/K/V tensors with MHA/GQA head
   grouping and produce submittable HiF4 parameters through the same
   dynamic-quantization API.

All state lives in plain CPU tensors/scalars. The dynamic-quantization
stage only depends on the calibration state, never on evaluator internals.

## 算法运行详解（How the Current Pipeline Actually Runs）

本节按真实代码路径描述当前 `solution.py`（C21/v024）究竟如何运行，
所有函数名、执行顺序与常量均与源码一一对应。

### 1. 评测器与六个官方 API 的调用时序

评测器（`evaluator/real_data_eval.py`）用 GPT-2 前向 2 个校准 batch +
2 个测试 batch，hook 捕获每层的激活，然后按以下时序调用 solution：

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

评分公式（Linear 与 Attention 同式）：
`score = (MSE(standard, reference) − MSE(candidate, reference)) / MSE(standard, reference)`。
其中 standard 是"朴素 HiF4"（amax/7、阈值 lv2/lv3、round mantissa、无精修），
candidate 是本方案动态量化结果，reference 是 NVFP4 反量化浮点值。

### 2. HiF4 目标格式（五个合法字段）

输入是 NVFP4（E2M1 载荷 + 每 16 通道块 scale），第一步
`_dequantize_nvfp4_float32` 先重建 float32 稠密张量。输出端把最后 64
个通道视为一个顶层块，按层级分解：

```text
x = sign * mant * scale_lv3 * scale_lv2 * scale_factor

[... , 64] → reshape [..., blocks, 8, 2, 4]
scale_factor  每个顶层 64 块一个，E6M2 浮点 scale（标准取 amax/7 的最近 E6M2 码）
scale_lv2     每个 8 通道组一个，取值 {1, 2}（组级 ×2 指数）
scale_lv3     每个 4 通道子组一个，取值 {1, 2}
sign          ±1（mantissa=0 时规范为 0）
mant          尾数码 {0, 0.25, ..., 1.75}（即 code×0.25，code∈0..7）
```

可表示最大内层值为 `1.75 × 2 × 2 = 7 × scale`（`_HIF4_MAX_INNER = 7.0`）。
反量化即按上式逐元素相乘（`_dequantize_hif4`）。

### 3. Weight 校准（`hif4_calibration_and_quantize_weight`）

对每个 Linear 层，按顺序执行：

1. **统计收集**：反量化校准激活，累计每通道二阶矩 `sum_square`、
   amax、以及（`_WEIGHT_QUADRATIC` 开启时）全协方差 `cov_sum = X^T X`。
2. **SmoothQuant 候选**：`d = act_amax^α / w_amax^(1-α)`，α 网格
   `(0.25, 0.5, 0.75)`；宽层（in/out ≥ 2048）用更细网格（5 档）。
   每档同时给 amax 版与 RMS 版，全部除以几何均值防整体漂移。
3. **通道置换候选**：`_hierarchy_aware_permutation` 把幅值相近的通道
   排进同一 64 块（使块内 amax 分布更均匀、scale 利用率更高）；再扩展
   单侧排序基（w/x 的 amax/rms 四种 range 排序）。
4. **候选评分与门控**：每个 `(d, perm)` 候选用采样行（≤256 权重行 +
   ≤128 token 激活样本）打分；`_candidate_is_safe` 要求均值改善超过
   门槛且最差样本不退化，否则拒绝。
5. **块 Hadamard 变换**（Matrix SmoothQuant 扩展）：在选定 `d/perm`
   之上再试 4/8/16 维 signed Hadamard（正交，state 只存
   `block_smooth_size/seed` 两个整数）。注意：此步用
   `_linear_output_candidate_metrics`（真实 Linear 输出 oracle）评分，
   是官方 `A @ W` 禁令下的已知违规点。
6. **一次性全量变换**：胜出组合对完整权重做严格等价变换
   `W_t = W · D · P · R`（与激活侧 `X_t = X · D⁻¹ · P · R` 严格配对，
   代数上保证 `X_t · W_t^T = X · W^T` 不变）。
7. **权重编码**：`_dense_to_hif4(weight_t, importance=H_x 对角,
   gram=4×4 块对角协方差, search_offsets, 预算)`（见第 4 节）。
8. **8/16 组二阶精修**：`_refine_weight_groups8/16` 对 top-K 高损失
   的 8/16 通道组做坐标级精修（增量 `H·e` 公式）。
9. **生成 activation_state**：`smooth_inv = 1/d`、`permutation`、
   `block_smooth_*`、`importance`（按 `weight_hat` 列能量）、
   `gram/gram8/cross8`（权重空间 Gram 及交叉项——后者为违规点）、
   offset 集与精修预算（数据驱动时按"损失捕获比例"定 ratio）。

### 4. 核心编码器（`_dense_to_hif4`）——所有张量共用的量化路径

1. reshape 成 `[blocks, 8, 2, 4]`，取 sign / abs；
2. 标准 scale：`amax/7` 编码为最近 E6M2 码再解码（保证合法）；
3. 阈值式层级：`max8 ≥ 4·scale → lv2=2`；`max4 ≥ 2·scale·lv2 → lv3=2`；
   mantissa = `round(|x|·4/denominator)` clamp 到 0..7 再 ×0.25；
4. **hard 块筛选**：归一化误差 `> 1e-7` 的块进入精修池，按**绝对**加权
   损失 top-K 截断到预算（`max_refine_ratio × 总块数`，另有块数上限）；
5. **offset 批量搜索**：对 hard 块把 `standard_code + offsets` 沿 offset
   维展开成 `[K, N]`，一次性调用 `_solve_exact_hierarchy` **精确求解**——
   对每个候选 scale 枚举总指数 `2^e (e=0,1,2)` 生成三张损失表（对角
   importance 加权或 `Δ^T G Δ` 二次型），精确选最优 lv2/lv3/mantissa，
   再按块 argmin 选 offset；
6. **边缘扩展**：胜出 offset 落在搜索集边缘（权重侧 `(-2,…,3)`、
   激活侧 `(-1,…,3)`）的块继续向外最多试 2 步；
7. **L1 数据驱动 scale**（当前 `_L1_DATA_DRIVEN_SCALE = False` 关闭）：
   对当前胜者做最小二乘 scale + 分位数 trim 候选，再过一次精确求解
   与逐块回退；
8. **接受门**：最终损失须满足 `best_loss ≤ (1−margin)·standard_loss`
   才写回，否则该块保持标准参数——任何精修只允许变好。

### 5. 动态激活量化（`hif4_dynamic_quantize_activation`）

每个测试 batch 逐层调用，全部无梯度：

```text
反量化 NVFP4 → × smooth_inv → 通道置换 → 块 Hadamard
→ _dense_to_hif4(importance, gram4, offsets, 预算)
→ _refine_weight_groups8(gram8, 可选 cross8)   # top-K 8 通道组二阶精修
```

激活侧的 gram/gram8 来自**权重空间**（`weight_smooth^T · weight_smooth`
的块对角），cross8 来自 `(W_hat − W)·W_hat^T` 交叉项——这两个 state 字段
是 C18–C21 的 cross 机制，也是官方禁令下待删除的违规路径。

### 6. Attention 校准（`hif4_calibration_attention`）

Q/K/V 走独立通道，核心思想是利用 **Q·K^T 点积的严格等价变换**
（`d_kv` 按 head 对齐：Q 侧乘 `d`，K 侧乘 `1/d`，点积不变）：

1. **统计**：per-head 二阶矩与峰值（K 另算 midrange 居中后的版本）；
2. **A1 上下文**：用校准前缀的真实 Q/K/V 跑参考 attention（causal +
   non-causal 双轨），V 量化结果固定，隔离 Q/K 变换选择；
3. **Smooth-QK**：`d = k_peak^α / q_peak^(1-α)`（GQA 对齐到
   kv-head 粒度）；
4. **K 居中**：midrange centering（softmax 平移不变性的精确利用）；
5. **headwise 置换**：Q/K 共享同一 head 内置换，保证点积不变；
6. **双轨选择 + 终验门**：A1 轨（真实输出误差选变换）与 proxy 轨
   （B0 式重建 proxy）各选一个 winner，再用**完整部署路径**
   （真实的 `hif4_dynamic_quantize_q/k/v`）重算输出误差，A1 无明确
   优势或安全轨退化时回退 proxy 选择；
7. **A3 V importance**：在 Q/K 定稿后，比较 head 级
   `E[A²] / E[A] / E[A²]+E[A]²` 三种重要性，同样过真实输出门控；
8. 输出 `q_state / k_state / v_state`（含 multiplier、permutation、
   importance、offsets、精修预算；A2 H64 旋转当前默认关闭）。

动态 Q：`× d_q → 置换 → 编码(importance=h_k, offset 搜索)`；
动态 K：`居中 → × 1/d → 置换 → 编码(importance=h_q)`；
动态 V：`编码(importance=head 级重要性)`。

### 7. 一图总结

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

设计要点：所有"等价变换"只利用代数恒等式（缩放对、置换对、正交旋转对）
保证 `X·W^T` 与 `Q·K^T` 严格不变，量化误差的削减发生在变换后的坐标空间；
所有 state 均为 CPU 普通 tensor，动态阶段不依赖评测器内部。

合规提示：第 3 节第 5 步、第 5 节 cross8、以及 Weight 校准中的
`_linear_output_candidate_metrics` / `_activation8_gate_decisions`
输出监督路径，在官方 `A @ W` 禁令口径下属于待删除项（见
`docs/superpowers/plans/2026-08-27-hif4-26000-algorithm-implementation-plan.md`
Phase 0）。

## Latest Verified Algorithms（最新已验证算法）

**中文**：当前官方最优为 v024（候选 C21，提交 `23d1cf7`）：官方 `16043 / 173.8s`，
较 B0（v002，`15313 / 137s`）累计 `+730`，较 v013（`15799 / 144s`）`+244`。根目录
`solution.py`（SHA256 `40F4D17C12F976F83856B9641BE9A3951867BC8979992D773C60C0C1C3E8066A`）
经 git blob 校验与 v024 归档字节一致，即为官方评测字节。下表为官方闭环锚点与
C21 的已验证机制链：每个机制单独成候选、逐个归档，主效应取自候选 ledger 的
offset-0 记录。

**English**: The current official champion is v024 (candidate C21, commit
`23d1cf7`): `16043 / 173.8s` official, `+730` over B0 (v002,
`15313 / 137s`) and `+244` over v013 (`15799 / 144s`). The root
`solution.py` (SHA256
`40F4D17C12F976F83856B9641BE9A3951867BC8979992D773C60C0C1C3E8066A`) is
byte-identical to the v024 archive — exactly the bytes officially
evaluated. The tables below list the closed official anchors and the
verified mechanism chain of C21: each mechanism is a single-mechanism
candidate, individually archived; main effects are the offset-0 records
from the candidate ledger.

官方闭环锚点 / Closed official anchors:

| Version / 版本 | Mechanism / 机制 | Official / 官方分数 | Time / 时间 |
|---|---|---:|---:|
| v000 | v9 baseline / v9 基线 | ~9000+ | NA |
| v001 | former baseline / 旧基线 | 10250 | 127s |
| v002 (B0) | youxilee/hif4 v2.0 | 15313 | 137s |
| v013 (C10) | wide-layer activation quadratic / 宽层激活二次精修 | 15799 | 144s |
| v024 (C21) | gated exact cross selection / 门控精确交叉选择 | 16043 | 173.8s |

C21 机制链 / C21 mechanism chain:

| # | Mechanism / 机制 | Candidate / 候选 | Verification / 验证 | Main effect (offset 0) / 主效应 |
|---|---|---|---|---|
| 1 | output-aware Attention selector / 输出感知 Attention 选择器 | C1 / v003 | local / 本地 | causal Attention +7.12pp |
| 2 | top-K 8×8 weight quadratic refinement / top-K 8×8 Weight 二阶精修 | C3 / v006 | local 6/6 | Linear +1.10pp |
| 3 | top-K 16×16 weight quadratic refinement / top-K 16×16 Weight 二阶精修 | C5 / v008 | local 6/6 | Linear +0.23pp |
| 4 | wide (3072 FFN) activation quadratic / 宽层激活二次精修 | C10 / v013 | official / 官方 | proj +0.54pp |
| 5 | wide activation 8×8 residual / 宽层激活 8×8 残差 | C11 / v014 | local 6/6 | proj +0.31pp |
| 6 | calibration-gated all-width activation 8×8 / 校准门控全宽度激活 8×8 | C14 / v017 | local 6/6, all components safe / 全分项安全 | Linear +0.45pp |
| 7 | gated activation 8×8 coverage 8% / 门控激活 8×8 覆盖 8% | C17 / v020 | local 6/6, 36/36 components / 36/36 分项 | Linear +0.29pp |
| 8 | calibration-gated exact cross selection / 校准门控精确交叉选择 | C21 / v024 | official / 官方 | Linear +0.15pp; fixes C20 pow2 regression / 修复 C20 pow2 回退 |

**中文**：累计效应——Attention 保留 A1 的 `+7.12pp` causal 增益；Linear mean 由
C1 的 `0.5668` 升至 `0.5930`（约 `+2.62pp`）。所有候选均通过
`evaluator/real_data_eval.py` 固定回归（amax6/amax4/pow2 × MHA/GQA ×
causal/non-causal，offset 0/97/193/389）与 `evaluator/synthetic_attention_eval.py`
冻结合成矩阵（8 场景 576 case）双重验证。

**English**: Cumulative effect — the attention path keeps A1's `+7.12pp`
causal gain; Linear mean rose from `0.5668` (C1) to `0.5930` (C21),
about `+2.62pp`. Every candidate passed both the fixed regression
matrix of `evaluator/real_data_eval.py` (amax6/amax4/pow2 × MHA/GQA ×
causal/non-causal, offsets 0/97/193/389) and the frozen synthetic
matrix of `evaluator/synthetic_attention_eval.py` (8 scenarios,
576 cases).

**中文（合规提示）**：官方已明确 Linear 校准禁令——不得以 `A @ W` 或数学等价的
输出监督拟合 `Q(A)`。C21 的 Linear 校准包含输出监督路径
（`_linear_output_candidate_metrics`、`group_cross8` 等），按新口径属于不合规实现。
下一条主线 HiF4-OSQ（见
`docs/superpowers/plans/2026-08-27-hif4-26000-algorithm-implementation-plan.md`）
将先删除这些路径建立合规基线 C21-C，再依次引入 64 维 Hadamard 旋转、full-64 GPTQ
Weight 精修、top-K full-64 激活求解与可学习 scale。主目标为官方 `22000~25000`，
`26000` 为 stretch 目标，官方时间上限 `300s`。

**English (compliance note)**: The competition has clarified that Linear
calibration must not fit `Q(A)` from `A @ W` or mathematically equivalent
output supervision. C21's Linear calibration contains output-supervised
paths (`_linear_output_candidate_metrics`, `group_cross8`, ...), which
are non-compliant under the clarified rule. The next mainline, HiF4-OSQ
(see
`docs/superpowers/plans/2026-08-27-hif4-26000-algorithm-implementation-plan.md`),
first removes those paths to build the compliant baseline C21-C, then
adds a 64-dim Hadamard rotation, full-64 GPTQ weight refinement, top-K
full-64 activation solving, and a learnable equivalent scale. The
primary target is `22000~25000` official; `26000` is a stretch goal;
the official time limit is `300s`.

## Evaluation Method

The evaluator loads real GPT-2 weights and text-forward activations and
captures per-layer Q/K/V, attention projections, and FFN inputs. For
every sample it produces:

- the NVFP4 dequantized result (reference),
- the standard HiF4 result (baseline),
- the current `solution.py` calibration/dynamic-quantization result
  (candidate).

Each Linear or attention sample is scored with the same formula:

```text
score = (MSE(standard, reference) - MSE(candidate, reference))
        / MSE(standard, reference)
```

then averaged over layers and test batches. The evaluator shares the
core scoring path with the remote hif4 project but adds configurable
`--solution` and `--model` options.

In addition, `evaluator/synthetic_attention_eval.py` (E1) runs a frozen
8-scenario / 576-case synthetic attention matrix (saturated logits,
near-uniform, V outliers, heavy tails, ...) as a safety gate for
attention-path changes — it pre-screens regressions that real-data
evaluation alone cannot expose.

## Environment and Usage

Use the project virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r evaluator\requirements.txt
```

GPT-2 weights live in `models/gpt2/` (~525MB, excluded by `.gitignore`);
the evaluator loads this directory by default, no network needed.
Default run (GPT-2, 12 layers, 2 calibration + 2 test batches):

```powershell
.\.venv\Scripts\python evaluator\real_data_eval.py
```

GPU acceleration: `--device cuda` (default `cpu`); `--model` accepts a
Hugging Face name or another local model directory:

```powershell
.\.venv\Scripts\python evaluator\real_data_eval.py `
  --solution solution.py --model gpt2 --device cuda
```

Fast directional comparison during development:

```powershell
.\.venv\Scripts\python evaluator\real_data_eval.py `
  --layers 1 --seq 16 --calib 1 --test 1
```

Synthetic attention safety matrix (all 576 frozen cases):

```powershell
.\.venv\Scripts\python evaluator\synthetic_attention_eval.py `
  --solution solution.py
```

Release checks (state legality, param fields, feature-off equivalence,
synthetic subset):

```powershell
.\.venv\Scripts\python -m pytest tests\test_release_candidate.py -q
```

Key options: `--layers`, `--seq`, `--calib`/`--test`, `--mode`
(`amax6`/`amax4`/`pow2`), `--kv-heads` (GQA smoke runs), `--token-offset`
(pinned local test windows). Output includes the six Linear component
scores (q/k/v/o/fc/proj), causal/non-causal attention scores, and
uniformly bounded algorithm-stage/API timings.

## Local Evaluation and Archival Workflow

The official evaluator is not continuously reachable; the known B0
official result serves as the baseline anchor. Later candidates are
promoted on reproducible local paired results — we do not wait for new
official scores, nor infer official absolute scores from local metrics:

1. B0 and candidates must run paired with identical model, device, mask,
   mode, token offset, and batch counts.
2. Offset `0` is the development set; `97`/`193`/`389` are pinned local
   regression windows (already consumed in the A1 arbitration — no
   tuning against them, and they are no longer claimed as blind sets).
3. Development screening covers `amax6/amax4/pow2`, MHA/GQA, and
   causal/non-causal; head_dim 128 and saturated-logit regimes are
   covered by the frozen synthetic safety matrix.
4. Promotion requires all of: target mean, per-layer tails, state
   legality, E1 synthetic safety track, and the CPU time gate.
5. On promotion, create a local result archive recording the exact
   source SHA256, full configuration, component scores, and timings;
   `Official Score/Time` stays `NA` — never fill in local estimates.

Version history:

- v000: legacy v9 baseline, ~9000+ official;
- v001: former active baseline, `10250 / 127s`;
- v002: `youxilee/hif4` v2.0, official B0, `15313 / 137s`, closed;
- v013: C10 wide-activation quadratic, official `15799 / 144s`;
- v024: C21 gated exact cross selection, official champion
  `16043 / 173.8s`; root `solution.py` is byte-identical to this
  archive; see `solutions/README.md` and the progressive candidate
  ledger for the full chain, source SHAs, and fixed-matrix numbers.

A four-anchor calibration of local metrics against official scores
(`docs/superpowers/logs/2026-08-27-evaluator-calibration-report.md`)
shows ~297 official points per 1pp of local Linear mean (stable) but a
weak conversion for local attention gains — local Linear is the
high-leverage metric, while the synthetic matrix is a safety track, not
a scoring lever.

The `.venv/` directory, Python caches, and other local artifacts are
excluded by `.gitignore` and never enter algorithm archives or
competition submissions.
