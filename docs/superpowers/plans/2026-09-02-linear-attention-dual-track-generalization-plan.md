# HiF4 双线优化与跨模型泛化计划

> 状态：**ACTIVE**
>
> 更新：2026-09-02
>
> 当前代码与 v159 归档 SHA256：
> `13C9CF0BFCF2277F0828D8CC1A18A8F7414DB183F3E27DD898D52597ACC5EC79`
>
> 官方事实：原始 v159 SHA `0508045A...4242` 为 `17532 / timeNA`；v158 为
> `16861 / 223s`。当前归档含数学等价 GPU 修复与中间量复用，尚未官方复测。

## 1. 总目标与硬边界

同时推进两个独立方向：

- **Linear 线**：固定 v158 Attention，只优化 Weight calibration 与 Activation dynamic。
- **Attention 线**：固定当前 v159 Linear，只优化 Attention calibration 与 Q/K/V dynamic。

两线不得在一个实验中同时改变。每个候选只允许一个数学机制；只有两线分别通过各自门禁后，
才进行一次集成审计。正式提交仍是根目录 `solution.py` 的六个 API，单文件、自包含。

17816 完整源码无法提供，只保留为不可复现锚点；不等待、不重建、不围绕 284 分差拟合。本地
Qwen、GPT-2、Pythia/OPT 都不是官方模型，任何本地结果只用于机制、复杂度和泛化诊断。

## 2. 防过拟合规则

### 2.1 三层数据门禁

1. **Qwen compact：开发集**。只筛接口、机制、尾部和父子配对，不作晋级结论。
2. **Qwen default：主本地审计**。目标侧完整 panel 只运行一次，确认全层、全 role/length 泛化。
3. **跨模型真实前向：封存验证集**。Qwen default 通过后，必须使用另一模型真实前向捕获的
   W/A/Q/K/V 运行父子配对。跨模型通过前不得提交官方评测。

跨模型结果不能反过来成为参数调优集。一个机制在 GPT-2/Pythia 上失败后，必须回到数学假设或
统一规则重新设计，禁止为模型名、layer、role 或具体 case 增加阈值、路由和例外。新设计必须
重新从 Qwen compact 开始，不能连续查询跨模型 holdout 来做网格搜索。

### 2.2 泛化判定

父子比较必须使用同一模型、cache、device、panel 和 case identity。依次检查：

- focus mean 与 median 是否同向；
- q25、worst-quartile、最差 case 和 negative case 是否恶化；
- validation/test、浅层/深层、短/长序列是否同向；
- 未修改 control 是否保持不变；
- Linear 的 W-only/A-only/W+A/interaction，或 Attention 的 Q/K/V/QK/QKV 来源是否可解释；
- Qwen 与跨模型的父子方向是否一致。

若 Qwen 正向、跨模型整体负向，结论为 `model-specific / REJECTED`；不得用 Qwen mean 覆盖。
若均值正向但 median、尾部或主要 shape/length 系统性负向，结论为 `mixed / not promotable`。

### 2.3 跨模型实现顺序

扩展 `evaluator/cross_model_eval.py`。当前 CLI（行 377–390）只有
`--model/--solution/--name/--cache/--cache-mode/--linear-cases/--attention-cases/--full-cases/
--capture-device/--algorithm-device/--no-decomposition/--output/--report`；`_load_gpt2`
（行 55）在行 71 硬编码 `model_type == "gpt2"`。

1. **场景隔离**：argparse 增加 `--linear-only` 与 `--attention-only`，取值转发到已复用的
   `official_eval.prepare_pack(evaluation_scenario=...)`（`official_eval.py` 行 899，已支持
   `both/linear/attention`）；禁用侧不得调用任何候选 API。验收：GPT-2 linear-only 运行的
   JSON 中 attention API 调用数为 0，反之亦然；
2. **父子配对与 replay**：argparse 增加 `--baseline-json/--candidate-json`，语义与
   `official_eval` 一致（`--candidate-json` 必须配 `--baseline-json`，replay 不重跑候选
   API）。配对诊断直接调用 `official_eval._paired_effect_diagnostics`（行 2018，签名
   `(baseline, candidate, focus_linear_roles)`，内含 case identity 与标准臂一致性检查），
   禁止另写宽松比较。父版本 cache/JSON 只生成一次（`--name parent`），候选同 case 配对；
3. **GPT-2 门禁**：`gpt2` 作为每个通过 Qwen default 的强制真实前向验证，走 1–2 的新 CLI
   运行 linear/attention 单侧配对；
4. **Pythia/OPT 二次验证**：最终候选上增加一个不同架构。优先本地 `pythia-160m`
   （GPTNeoX：rotary + fused-QKV）。捕获时点决策：hook 挂在 **rotary 之后**
   （`GPTNeoXAttention` 内 rotary 应用返回处），使捕获 Q/K 与真实前向输入逐元素一致——
   评测侧 `_attention_forward` 不施加 rotary，因此捕获前必须已完成 rotary。QKV 按
   `query/key/value` 真实 module mapping 记录，禁止伪造 Qwen role。验收：任取一个
   calib batch，hook 捕获值与手工切片参考前向的 max abs diff < 1e-5（float32）。若适配
   成本过高改用 `opt-125m`（无 rotary、标准 QKV），验收同上；
5. `gpt2-medium` 只作尺寸压力测试，不替代不同架构验证。

步骤验收：1–2 完成后用 v159 父版本各生成一次 GPT-2 linear/attention parent JSON，与
Qwen parent 的机制签名（正/负 case 数、最差 layer/role）并列归档；3–4 全部通过后才允许
进入 §6 集成。所有跨模型结果标记 `cross-model-probe`，不与 Qwen proxy 或官方分数混排。

### 2.4 复杂度准入（2026-09-02 源码审计补充）

官方总时间是固定调用图的确定性函数（n 为序列长度，C 为通道数）：

```text
T_total = 168·t_w_calib + 24·t_a_calib + 250·t_linear_dyn + 200·t_attn_dyn
```

在线 API 的每个新机制在实现前先归入以下类别；类 2/类 3 直接禁止，不进入实现：

1. **类 0**：校准一次性 O(poly(C))，在线 O(n·C) 向量化——安全；
2. **类 1**：在线常数因子增大（常数次额外 O(n·C) pass）——可接受，须申报；
3. **类 2**：在线随 n 超线性（n² 前向、per-token 循环）——禁止，v128–v131 超时根因；
4. **类 3**：在线数据依赖搜索/迭代（无法静态定界）——禁止。

当前 v159 在线路径的两个架构不变量必须保持：动态 Q/K/V 的 refine 上限是绝对常数
（Q `16_384` / K `24_576` / V `24_576` 个 64-block），且候选评估为批量张量展开后一次
argmin，无 Python 逐块循环。任何新机制破坏任一条即落入 v128–v131 风险类别。

复杂度验收不靠计时：静态检查在线路径算子清单与上界；本地秒数只作同机 A/B，不预测官方。

## 3. 统一实验生命周期

每个 Linear 或 Attention 机制按以下顺序执行：

1. 声明 parent SHA、唯一变化、focus、control、预期复杂度变化和失败条件；
2. 单 state/API smoke，验证合法状态、CPU/CUDA device 和连续域不变量；
3. 目标侧 Qwen compact 与保存的 parent 精确配对；
4. compact 通过后运行一次目标侧 Qwen default；
5. default 通过后运行封存的 GPT-2 真实前向配对；最终候选再运行一次 Pythia/OPT；
6. 两线独立通过后，运行一次 Qwen 完整 168 Linear + 120 Attention 集成审计；
7. 记录 source SHA、JSON/report、六 API 时间和决定，更新同一 v159 归档，不为微优化创建新版本。

父版本结果只运行一次。已有 JSON 使用 replay，不重复消耗 API。`--full-cases` 只有发现 shape、
length 或显存边界问题时才运行，不能用作日常排名。

固定命令骨架：

```powershell
# 接口 smoke（第 2 步）：顺序前缀 case，只判六 API/状态/设备合法性，不判效果
.venv\Scripts\python.exe evaluator/official_eval.py --solution solution.py --linear-only --compact-panel --linear-cases 4 --cache-mode read --nvfp4-cache-mode auto --capture-device cuda --algorithm-device cuda
.venv\Scripts\python.exe evaluator/official_eval.py --solution solution.py --attention-only --compact-panel --attention-cases 4 --cache-mode read --nvfp4-cache-mode auto --capture-device cuda --algorithm-device cuda

# Qwen Linear / Attention 单侧开发门禁（第 3 步，compact 与保存的 parent 精确配对）
.venv\Scripts\python.exe evaluator/official_eval.py --solution solution.py --linear-only --compact-panel --cache-mode read --nvfp4-cache-mode auto --capture-device cuda --algorithm-device cuda
.venv\Scripts\python.exe evaluator/official_eval.py --solution solution.py --attention-only --compact-panel --cache-mode read --nvfp4-cache-mode auto --capture-device cuda --algorithm-device cuda

# Qwen default（第 4 步）：只保留当前目标侧，去掉 --compact-panel
# 跨模型 CLI 扩展完成后的封存门禁（第 5 步）
.venv\Scripts\python.exe evaluator/cross_model_eval.py --model gpt2 --solution solution.py --name parent --linear-only --cache-mode read --capture-device cuda --algorithm-device cuda
.venv\Scripts\python.exe evaluator/cross_model_eval.py --model gpt2 --solution solution.py --name parent --attention-only --cache-mode read --capture-device cuda --algorithm-device cuda
```

零 API replay（父 JSON 已存在、候选 JSON 已保存时的配对复核，不重新调用任何候选 API；
`--candidate-json` 必须配 `--baseline-json`，且不能与 `--solution` 组合）：

```powershell
.venv\Scripts\python.exe evaluator/official_eval.py --linear-only --compact-panel --baseline-json artifacts/official_eval/v159-<parent>.json --candidate-json artifacts/official_eval/v159-<candidate>.json
```

replay 判据：case identity 精确匹配 `(layer, role, test_window, split, length)`、
`mse_standard`、`reference_energy` 与 parent 一致；不满足即 `ERROR`，不放宽比较。

## 4. Linear 优化线

### L0. 固定父基线（DONE）

- 当前 v159 CUDA compact：`0.705508`，56/0/0，API `52.321s`；相对同设备 v158
  `+0.149185`，56/56 改善。
- CUDA Linear default：`0.633526`，median/q25/worst-quartile
  `0.626581/0.536043/0.434968`，167/1/0；唯一负例为 layer 22 `o`、length 10。
- default API `269.435s`：Weight calibration `208.971s`，Activation dynamic `60.464s`。
- transformed samples 与 Weight Gram 复用后 compact 输出 56/56 不变，API `51.055s`。

### L1. 校准热点分解与等价降复杂度

目标：不改变候选、接受条件和输出，先降低 Weight calibration（default API `208.971s`）。

**计时装置**：新建 `workbench/l1_calib_timing_probe.py`，import 根目录 `solution.py` 的
`hif4_calibration_and_quantize_weight` 并包裹计时（CUDA 侧 `torch.cuda.synchronize` +
`time.perf_counter`），不修改提交文件。五阶段切分与源码锚点：

| 阶段 | 激活路径锚点（solution.py） |
| --- | --- |
| ① joint 候选搜索 | 7955–8010 嵌套循环；内层 `_linear_output_candidate_metrics`（5031） |
| ② Weight GPTQ | `_gptq_quantize_weight`（8080 调用，定义 5323）+ `_transformed_covariance`（8056） |
| ③ Weight e2e refine | `_weight_e2e_refine`（5123） |
| ④ 激活侧 Gram/Hessian | 变换样本 8132–8145、`weight_output_gram` 8168–8170、单次 Cholesky 8249–8262 |
| ⑤ CPU state 构造与搬运 | `_cpu_state_tensor` 调用点 |

输出格式：每 `(layer, role)` 一行五列毫秒，末尾按阶段汇总 mean/total，写
`logs/l1_calib_timing_probe.md`。验收：探针只读不改输出；先在 layer `0/8/15/23` 全 role
跑通，再覆盖 28 个 compact Weight state。

按热点只做批处理、共享 Gram/Cholesky、避免重复 dequant/reconstruct、复用 candidate metric 等
数学等价优化。每项要求 Qwen compact 56 case `delta=0`；只降低局部计时但增加显存峰值或在线
复杂度的实现不保留。当前里程碑是 Linear default API 相对 `269.435s` 明显下降，官方时间仍只
认官方回传。

热点嫌疑排序（2026-09-02 二次审计修正；先计时区分占比再动手）：

1. **joint 嵌套搜索**：`smooth_candidates`（identity + `_WEIGHT_SMOOTH_ALPHAS` 变体）×
   permutation（`[best_perm, sorted_j]`）× `candidate_block_sizes=(4, 8)` ×
   `_BLOCK_SMOOTH_SEEDS=(0,1,2,3)`；每次调用 `_linear_output_candidate_metrics` 内部对
   **每个 activation sample** 重复 `_linear_pair_transform` + `_dense_to_hif4` +
   `_dequantize_hif4` + 两次 matmul（5047–5084）。样本变换/编码随候选数线性重复，是
   批处理/缓存的首要目标；
2. **`_weight_e2e_refine` + `_gptq_quantize_weight`** 的 per-block 候选评估（5123/5323）；
3. **`_transformed_covariance` 全协方差 O(m·C²)**（定义 4782，激活调用 8056）与
   `weight_output_gram`/Cholesky（每 weight 恰一次，O(C³) × 168）。

死代码声明（修正 2026-09-02 首版审计的错误结论）：`_ADAPTIVE_ACT_GPTQ_REG = False`
（行 78，reg 候选循环 8184–8248 不执行）且 `_ADAPTIVE_OFFSETS = False`（行 80，offset
候选循环 8264 起不执行）。激活路径不存在“reg 循环内重复 Cholesky”——首版排序第 2 条
基于死分支，作废；激活路径每 weight 仅单次 Cholesky。L2 也不得把这两条死分支列为消融
对象。

### L2. 有界复杂度消融

L1 完成后，一次只消融一项。消融对象只允许是**激活路径**的搜索维度（常量锚点 solution.py）：

1. joint block-smooth 搜索维度：`_BLOCK_SMOOTH_SEEDS = (0,1,2,3)`（行 54）→ 如 `(0,1)`；
   `_BLOCK_SMOOTH_SIZES = (4, 8)`（行 53）→ 如 `(8,)`；
2. RMS smooth 候选加倍：`_WEIGHT_SMOOTH_RMS = True`（行 317）→ `False`；
3. `_weight_e2e_refine`（5123）的候选评估次数/`max_refine_ratio`；
4. `_WEIGHT_SMOOTH_ALPHAS_WIDE`（行 45）宽层 alpha 数量。

方法：复制 `solution.py` 到 `workbench/ablate_<name>.py`，只改单个常量，评测命令
`--solution workbench/ablate_<name>.py` 其余不变；产物标记 `ablation`，不进版本序列。
关闭项必须在 Qwen compact/default 和 GPT-2 父子配对中均不产生系统性回归（判定同
§2.2 全部条目）。不得用减少测试 case、缩短 token 或只保留浅层来伪造加速。失败项立即
恢复，不组合多个负向消融。

显式排除：`_ADAPTIVE_ACT_GPTQ_REG`（行 78）与 `_ADAPTIVE_OFFSETS`（行 80）已是
`False`，对应候选循环本就不执行，不列为消融对象，也不得顺手翻回 `True`。

### L3. 精度机制

复杂度稳定后才增加一个精度机制：在现有 A/W 联合坐标中做单次、共享分解的 block residual /
GPTQ 更新。硬约束：

- 保持 `XWᵀ = (XR)(WR⁻ᵀ)ᵀ` 连续域不变量；
- Hessian/Gram 使用最终部署坐标；
- Weight 与 Activation 联合验收，不能分别优化后假定可相加；
- 对 shape class 使用统一规则，不针对 q/k/v/o/fc/proj 或 layer 写表；
- 不增加第二轮完整 oracle、per-token candidate loop 或新的 block/seed/rank 网格；
- 分解只允许在 ≤64 维 block 内进行（沿用 `_gptq_initialize64`/`_cholesky_inverse_factor`
  模式），禁止整矩阵 Schur/求逆：168 次校准 × O(C³) ≈ 10¹³ FLOPs 不可行，block 内
  168 × B × 64³ ≈ 3×10⁹ 可行。

算法草案（复用现有原语，全部已存在于 solution.py）：最终 `(best_d, best_perm, size, seed)`
选定后、`_gptq_quantize_weight`（8080）之前，对部署坐标 weight（8039–8046
`_linear_pair_transform` 输出）按 64 维 block 执行一次 `_gptq_initialize64`（1631）+ 单次
`_coordinate_descent64`（1681）sweep，block Gram 取自已在部署坐标的 `gram_full`（8056）；
输出与现行 GPTQ 基线在同一接受准则（`_weight_e2e_refine` 的 margin/threshold）下竞争，
恒等候选保留。每 block 64³ FLOPs，总量 168 × B × 64³ ≈ 3×10⁹——类 0（校准一次性），
在线路径零改动。

fc/proj/o 只作为误差定位重点，不作为硬编码路由依据。特别禁止为当前唯一 layer-22 `o` 负例
增加专属规则；只有跨层、跨模型同类 shape 都显示同一问题时才修改统一机制。

## 5. Attention 优化线

### A0. 固定父基线

父代码为当前 v159 Linear + v158 Attention。先运行并保存（产物命名沿用
`artifacts/official_eval/vNNN-<desc>.json` 惯例）：

1. CUDA Attention-only compact → `artifacts/official_eval/v159-attn-compact-parent.json`；
2. compact 通过后一次 Attention-only default 120 cases →
   `artifacts/official_eval/v159-attn-default-parent.json`；
3. GPT-2 Attention parent JSON → `artifacts/official_eval/v159-attn-gpt2-parent.json`，
   后续候选只做配对。

依赖关系：1、2 只用现有 `official_eval` CLI，不依赖 §2.3 跨模型扩展，可立即执行；3 依赖
§2.3 第 1–2 项（`cross_model_eval.py` 场景隔离 + 配对）完成。

报告必须按 Q/K/V、QK、QKV、layer、length、split 输出 logits MSE、softmax probability MSE 和
KL。静态 Linear q/k/v role 与动态 Attention Q/K/V 必须分开命名和解释。API 时间必须分开记录
Attention calibration 与 Q/K/V dynamic（当前 v159 六 API 分解里这两项尚无独立基线数字，
A0 首跑即建立）。

### A1. 等价复杂度清理

先审计 `hif4_calibration_attention` 与动态 Q/K/V 的重复 covariance、pair transform、编码和 state
搬运。只允许共享中间量、批处理和删除重复计算；六 API 调用数、候选集合与输出必须不变。

v158 官方只比 v86 增加约 `0.3s`，因此任何新 Attention 路径必须替换现有计算，不能叠加历史
v128/v129 的 Gram sweep、PAWV、多轮 dynamic refine 或 length-keyed state。

共享预计算的具体位置与改法（2026-09-02 审计）：

1. **K 居中共享**：`_center_attention_k`（2774）的输出只依赖
   `(k_sample, center_mode, center_value)`，与 alpha 候选 `candidate_d` 无关；但
   `_attention_candidate_metrics`（8434）在每个 alpha 候选内重复对全部 k_samples 居中。
   改法：签名增加可选参数 `precomputed_centered_k`；在 center_mode 循环（8932）内、alpha
   循环（8964）前按当前 center_mode（mode 4 时 `center_value=sac_center`，每层固定）算好
   centered K 样本传入。mode 0 恒等无需预计算；
2. **block_signs 共享**：`_attention_rotation_signs(kv_num_heads, head_dim, seed)`
   （8466–8470 调用）只依赖 `(kv_num_heads, head_dim, seed)`；第二阶段 block smooth 候选的
   (size, seed) 组合有限，按 `{(seed, size): signs}` 预计算一次传入。

注意 `sac_center` 本身已是每层只解一次（8683–8699，统计循环之前），无需改动。验收：
Attention compact 全部 case 输出与 parent **逐位一致**（浮点 delta=0，非容差比较）、
六 API 调用数不变、只允许 Attention calibration 计时下降；任何 case 不一致即回退。

### A2. Q/K 配对精度

机制已存在，无需新写迭代（2026-09-02 二次审计修正）：`_solve_k_center_scale_aware`
（2729，C41）已实现量化感知不动点迭代 `c = mean_tokens(K − dequant(Q(K − c)))`，从
`c = 0` 起步（恒等候选恒可采纳，不会差于不居中），轮数为 `_ATTN_CENTER_ALTERNATIONS`，
docstring 明确记载 softmax 精确不变性。当前它对 GQA 被双重关闭：

- `_ATTN_SCALE_AWARE_CENTER_GQA = False`（行 351）使 8686–8699 的每层一次求解跳过
  （`sac_center = None`），并使候选循环 8933–8940 跳过 mode 4；
- `_ATTN_CENTER_MODES = (0, 2)`（行 319）候选竞争本身不含 4。

**A2 = 两行开关变更**：`_ATTN_SCALE_AWARE_CENTER_GQA → True` 且
`_ATTN_CENTER_MODES → (0, 2, 4)`。校准成本为每层一次有界不动点
（`_ATTN_STATS_TOKENS` 采样行 × rounds × O(C)），类 0；在线路径零改动——state 保存固定
center，dynamic 仍为一次 center + encode。

数学前提已推导确认（2026-09-02）：per-head 公共平移 c_h 使
`logits'_ij = logits_ij + q_i·c_hᵀ/√d`，对固定行 i 是不依赖 j 的标量，即行常数，softmax
严格不变；GQA 下每个 Q head 各自成行常数，causal mask 为加性 -inf 不受影响，K 平移不触碰
V。连续域严格无损，收益全部来自量化域。实现硬条件：center 必须是 per-head 常向量（非
per-token），量化在平移后域进行，在线不引入迭代（类 0）——现有实现已全部满足。

验收重点：Q/K 单侧可能变差，但 QK、logits、probability 和 KL 必须在短长序列与跨模型上同向；
不得只看 Q/K operand MSE。若 length 10 改善而 512/1024 回归，直接拒绝，不建立长度路由。

### A3. 编码器与 V

只有 A2 稳定后，才分别评估：

- Q/K：用统一的低成本编码规则替换现有 refine，不与 Matrix-Smooth 叠加搜索；
- V：只允许一次固定 encode 改进，不使用 `PᵀP`、PAWV、token/length 路由或对侧在线 Gram。

Q/K 与 V 必须分成两个实验。V 改进不能用 QK 正向掩盖，QK 改进也不能用 V/output 正向掩盖。

"替换"必须是净替换：验收加静态断言——在线 per-tensor 算子数不高于现状；叠加式改法直接
拒绝（类别判定见 2.4）。

## 6. 集成与提交门禁

Linear 与 Attention 各自通过 Qwen default + GPT-2 后，按以下顺序集成：

1. 当前 v159 + 已通过的 Linear 变化，Attention 字段一致性检查；
2. 在该 Linear 上合入已通过的 Attention 变化，Linear case 必须与集成前一致；
3. 运行一次完整 Qwen default，保存六 API 时间与调用数；
4. 运行一次 GPT-2 集成验证和一次 Pythia/OPT 最终验证；
5. 只有本地机制、尾部、跨模型和复杂度均可解释时才提交官方。

官方未知时间写 `timeNA`，本地 `269s/291s` 不能换算为官方 `<300s`。官方结果优先于所有本地
proxy；若官方与本地再次反转，记录反转并收紧跨模型/数学门禁，不调整本地权重拟合官方分数。

## 7. 当前执行顺序

1. **评测基础设施**：为 `cross_model_eval.py` 增加场景隔离和父子配对；随后适配一个
   Pythia/OPT 真实前向模型。
   完成判据：新 CLI 生成 GPT-2 linear/attention parent JSON，禁用侧 API 调用数为 0；
   Pythia hook 捕获与参考前向 max abs diff < 1e-5。
2. **Linear L1**：完成阶段热点分解，继续数学等价降复杂度。
   完成判据：`logs/l1_calib_timing_probe.md` 覆盖 28 个 compact Weight state 的五阶段
   分解；至少一项等价优化在 compact 全 case delta=0 下使 default API 计时下降。
3. **Attention A0**：建立 Attention-only compact/default 与 GPT-2 parent 基线；A0 第
   1–2 步只运行现有评测器，与 Linear L1 并行推进，不串行等待（第 3 步依赖本节项 1）。
   完成判据：三个 A0 JSON 归档；Attention calibration 与 Q/K/V dynamic 的独立时间
   基线建立。
4. **Linear L2/L3 与 Attention A1/A2**：分别执行，禁止同时修改。
   完成判据：L2 每个消融有 compact + GPT-2 配对记录（通过或恢复二选一）；L3/A2 各自
   通过 §2.2 全部条目；A1 验收逐位一致（delta=0）。
5. **最终集成**：每条线独立通过跨模型门禁后只运行一次。
   完成判据：§6 五步全部走完，六 API 时间与调用数归档，复杂度静态检查通过。

任何 AI 接手时只从本节第一个未完成项继续；不得跳过跨模型门禁，也不得从归档计划恢复旧的
ROAB、无约束 sweep、PAWV 或多轮在线搜索。
