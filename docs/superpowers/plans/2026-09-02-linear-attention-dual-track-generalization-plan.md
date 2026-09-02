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

先扩展 `evaluator/cross_model_eval.py`：

1. 增加与主评测器一致的 `--linear-only`、`--attention-only`，禁用侧不得调用 API；
2. 父版本 cache/JSON 只生成一次，候选使用同 case 做配对；
3. `gpt2` 作为每个通过 Qwen default 的强制真实前向验证；
4. 在最终候选上增加一个不同架构验证，优先本地 `pythia-160m`（rotary/fused-QKV）；若适配成本
   过高则用 `opt-125m`，但必须记录真实 module mapping，禁止伪造 Qwen role；
5. `gpt2-medium` 只作尺寸压力测试，不替代不同架构验证。

所有跨模型结果标记 `cross-model-probe`，不与 Qwen proxy 或官方分数混排。

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
# Qwen Linear / Attention 单侧开发门禁
.venv\Scripts\python.exe evaluator/official_eval.py --solution solution.py --linear-only --compact-panel --cache-mode read --nvfp4-cache-mode auto --capture-device cuda --algorithm-device cuda
.venv\Scripts\python.exe evaluator/official_eval.py --solution solution.py --attention-only --compact-panel --cache-mode read --nvfp4-cache-mode auto --capture-device cuda --algorithm-device cuda

# Qwen default：只保留当前目标侧，去掉 --compact-panel
# 跨模型 CLI 扩展完成后的封存门禁
.venv\Scripts\python.exe evaluator/cross_model_eval.py --model gpt2 --solution solution.py --linear-only --cache-mode read --capture-device cuda --algorithm-device cuda
.venv\Scripts\python.exe evaluator/cross_model_eval.py --model gpt2 --solution solution.py --attention-only --cache-mode read --capture-device cuda --algorithm-device cuda
```

为 `cross_model_eval.py` 实现 replay 时直接复用 `official_eval._paired_effect_diagnostics` 的 case
identity 与标准臂一致性检查，不另写一套宽松比较逻辑。

## 4. Linear 优化线

### L0. 固定父基线（DONE）

- 当前 v159 CUDA compact：`0.705508`，56/0/0，API `52.321s`；相对同设备 v158
  `+0.149185`，56/56 改善。
- CUDA Linear default：`0.633526`，median/q25/worst-quartile
  `0.626581/0.536043/0.434968`，167/1/0；唯一负例为 layer 22 `o`、length 10。
- default API `269.435s`：Weight calibration `208.971s`，Activation dynamic `60.464s`。
- transformed samples 与 Weight Gram 复用后 compact 输出 56/56 不变，API `51.055s`。

### L1. 校准热点分解与等价降复杂度

目标：不改变候选、接受条件和输出，先降低 Weight calibration。

实现入口：`hif4_calibration_and_quantize_weight`，按以下阶段增加 workbench 计时，计时器不得进入
正式提交：

1. Smooth/permutation/block-Hadamard candidate metric；
2. Weight GPTQ + AdaRound；
3. Weight e2e refine；
4. Activation Gram/Hessian、adaptive regularization、adaptive offset；
5. CPU state 构造与 dynamic state transfer。

按热点只做批处理、共享 Gram/Cholesky、避免重复 dequant/reconstruct、复用 candidate metric 等
数学等价优化。每项要求 Qwen compact 56 case `delta=0`；只降低局部计时但增加显存峰值或在线
复杂度的实现不保留。当前里程碑是 Linear default API 相对 `269.435s` 明显下降，官方时间仍只
认官方回传。

### L2. 有界复杂度消融

L1 完成后，用未编号 workbench 一次只消融一项：

1. adaptive activation regularization 候选；
2. adaptive offset 候选；
3. 窄层 joint smooth 搜索；
4. Weight e2e refine 的重复候选评估。

关闭项必须在 Qwen compact/default 和 GPT-2 父子配对中均不产生系统性回归。不得用减少测试
case、缩短 token 或只保留浅层来伪造加速。失败项立即恢复，不组合多个负向消融。

### L3. 精度机制

复杂度稳定后才增加一个精度机制：在现有 A/W 联合坐标中做单次、共享分解的 block residual /
GPTQ 更新。硬约束：

- 保持 `XWᵀ = (XR)(WR⁻ᵀ)ᵀ` 连续域不变量；
- Hessian/Gram 使用最终部署坐标；
- Weight 与 Activation 联合验收，不能分别优化后假定可相加；
- 对 shape class 使用统一规则，不针对 q/k/v/o/fc/proj 或 layer 写表；
- 不增加第二轮完整 oracle、per-token candidate loop 或新的 block/seed/rank 网格。

fc/proj/o 只作为误差定位重点，不作为硬编码路由依据。特别禁止为当前唯一 layer-22 `o` 负例
增加专属规则；只有跨层、跨模型同类 shape 都显示同一问题时才修改统一机制。

## 5. Attention 优化线

### A0. 固定父基线

父代码为当前 v159 Linear + v158 Attention。先运行并保存：

1. CUDA Attention-only compact；
2. compact 通过后一次 Attention-only default 120 cases；
3. GPT-2 Attention parent JSON，后续候选只做配对。

报告必须按 Q/K/V、QK、QKV、layer、length、split 输出 logits MSE、softmax probability MSE 和
KL。静态 Linear q/k/v role 与动态 Attention Q/K/V 必须分开命名和解释。

### A1. 等价复杂度清理

先审计 `hif4_calibration_attention` 与动态 Q/K/V 的重复 covariance、pair transform、编码和 state
搬运。只允许共享中间量、批处理和删除重复计算；六 API 调用数、候选集合与输出必须不变。

v158 官方只比 v86 增加约 `0.3s`，因此任何新 Attention 路径必须替换现有计算，不能叠加历史
v128/v129 的 Gram sweep、PAWV、多轮 dynamic refine 或 length-keyed state。

### A2. Q/K 配对精度

第一候选是在 v158 解析式 Matrix-Smooth 后加入固定两次的 K 公共中心更新。K 的 head 内公共平移
保持 softmax logits 行偏置不变；state 只保存一个固定 center，dynamic 仍为一次 center + encode。

验收重点：Q/K 单侧可能变差，但 QK、logits、probability 和 KL 必须在短长序列与跨模型上同向；
不得只看 Q/K operand MSE。若 length 10 改善而 512/1024 回归，直接拒绝，不建立长度路由。

### A3. 编码器与 V

只有 A2 稳定后，才分别评估：

- Q/K：用统一的低成本编码规则替换现有 refine，不与 Matrix-Smooth 叠加搜索；
- V：只允许一次固定 encode 改进，不使用 `PᵀP`、PAWV、token/length 路由或对侧在线 Gram。

Q/K 与 V 必须分成两个实验。V 改进不能用 QK 正向掩盖，QK 改进也不能用 V/output 正向掩盖。

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
2. **Linear L1**：完成阶段热点分解，继续数学等价降复杂度。
3. **Attention A0**：建立 Attention-only compact/default 与 GPT-2 parent 基线。
4. **Linear L2/L3 与 Attention A1/A2**：分别执行，禁止同时修改。
5. **最终集成**：每条线独立通过跨模型门禁后只运行一次。

任何 AI 接手时只从本节第一个未完成项继续；不得跳过跨模型门禁，也不得从归档计划恢复旧的
ROAB、无约束 sweep、PAWV 或多轮在线搜索。
