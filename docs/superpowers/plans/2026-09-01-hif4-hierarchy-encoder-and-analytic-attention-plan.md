# HiF4 唯一活动计划：层级编码器与解析式等价变换

> 状态：**ACTIVE / SUPERSEDES 17816-ANCHOR PLAN**
>
> 更新：2026-09-01
>
> 目标一：Linear mean 向 `0.8` 做可达性验证并持续提升
>
> 目标二：官方六 API 端到端时间严格 `<300s`
>
> Linear 研究期间冻结 v86 Attention；Attention 实验与 Linear 分开执行

## E0. 先修复本地评测趋势代理（当前阻塞项）

**目的：** 让本地结果至少能回答“同一公开输入、同一调用图下候选是否更好”，并在已知官方
同 cohort 锚点上显式报告顺序反转；在 E0 通过前，不用本地 proxy 分数选择 Linear/Attention
算法，也不把本地时间当作官方 `<300s` 结论。

**已确认的失真来源：**

1. v1 的 hash 前缀选例集中在少数 WikiText 文档/offset，validation 只有单一 split；
2. 旧 NVFP4 E4M3 scale 把所有小于 `2^-6` 的值夹到正规数，忽略 `2^-9` subnormal；
3. 一次修改把校准放在每个 case 内，给 output-aware 候选额外的 per-case oracle；官方历史调用
   记录明确是 168 个 layer/role Weight state、24 个 Attention state，然后复用到动态 case；
4. 450 case 等权显示值被误读为官方总分，且跨官方权重 revision 的版本被混排。

**已实现的 proxy-v2 设计：**

- train 只做 calibration；validation/test 交替做 holdout，窗口来自不同文档并使用固定哈希
  offset，Attention 长度保持 `[10,128,512,1024,1024]` 的轮换扩展；
- 默认使用固定分层真实 W/A panel：Linear 覆盖 24 层×7 role、每个 layer/role 一个真实窗口
  （168 cases），Attention 覆盖 24 层×五个官方 holdout 长度（120 cases）；不施加
  Linear:Attention 权重。`--full-cases` 才展开全部窗口做 stress，case 数 CLI 限制仍只能用于
  smoke，不能改变校准状态生命周期；
- E4M3 scale 使用包含 subnormal 的 `e4m3-subnormal-ceil-v1`，标准 HiF4 分母仍由评测器
  独立冻结；
- Linear 每个 layer/role 只调用一次 calibration（前两折），Attention 每层只调用一次
  calibration（五折），动态阶段每 case 只调用对应 API；
- 主指标为真实 case 的 `linear_mean`、`attention_mean` 和未加权 `overall_mean`；
  `trend_diagnostics` 只对同一官方 cohort 做 pairwise 顺序审计，不把官方分数拟合进结果。

**E0 验收：**

1. `tests/test_official_eval.py`、codec/合法性检查和 cache 协议检查通过；
2. v86、v138/v140/v147 使用同一 proxy-v2 cache 重跑，报告分层真实 W/A panel、实际 API calls、逐
   role/layer 结果和趋势审计；必要时另行记录 `--full-cases` stress；
3. 若锚点顺序仍反转，记录为 `inversion_detected`，继续扩充数据/模型 stress panel；不得
   调权重或硬编码分数把反转“修正”为通过；
4. E0 完成前，L0–L4 只做代码审计和单层 smoke，不据此晋级新算法。

E0 产物是 `evaluator/official_eval.py`、`evaluator/nvfp4_sim.py`、测试和一份执行日志；旧
`official-shape-v1` JSON/cache 保持 immutable，只能标为历史诊断。

## E0.5. 把误差源直接写进评测结果（当前优化入口）

**目的：** 主分数只回答“候选整体是否变好”，不能指导下一步改哪一侧。评测器必须在同一份
真实 W/A、同一 calibration state 和同一动态输出上给出可复现的控制臂；这些控制臂只在
evaluator 内重算，不改变六个 API 的调用图、主分数或官方时间代理。

**Linear 四臂：**

```text
E00 = standard W + standard A
E10 = candidate W + standard A
E01 = standard W + candidate A
E11 = candidate W + candidate A
```

每个 case 记录四个输出 MSE、W/A operand relative-MSE、W-only/A-only/Both gain，以及
`interaction_gain=(E10+E01-E00-E11)/E00`。结果同时按 role、layer、shape、test length、split
聚合；`case_scores.linear` 保留逐 case 定位信息。正 interaction 表示超加性互补，负值表示
收益重叠或递减，不能把它误读为独立 W/A 增益；若两侧单独都恶化而 Both 改善，结果标为
`paired_coordinate_coupling_likely`，应优先优化等价变换与双侧配对。

**Attention 六臂与中间量：**

```text
E000 = standard Q/K/V
E100 = candidate Q only
E010 = candidate K only
E001 = candidate V only
E110 = candidate Q+K
E111 = candidate Q+K+V
```

每个 case 记录 Q/K/V/QK/Both 输出 MSE、QK 与 QKV interaction，并记录 logits MSE、softmax
probability MSE 和 `KL(reference || estimate)`。结果按 layer、长度和 split 聚合，重点对比
`10/128/512/1024` 长度桶，区分 logits、softmax 和 V/output 的误差传播；若 Q/K 单独恶化而
QK 改善，结果标为 `paired_qk_coupling_likely`，不把单侧控制臂当作独立可部署算法。

**当前实现与使用：** `evaluate_solution(..., decomposition=True)` 默认开启；CLI 的
`--no-decomposition` 仅用于快速 smoke。JSON 新增 `decomposition.linear`、
`decomposition.attention` 和 `diagnostic_config`，Markdown 报告新增 W/A/Q/K/V 控制臂表。
没有额外候选 API 调用，主 `score` 字段保持原始未加权真实 panel 定义；`--full-cases` 的分数
只能作为 stress 记录，不能与默认 panel 混排。

**决策顺序：** 先用固定 v86 Attention 做 Linear 四臂归因，再冻结 Linear 做 Attention 六臂
归因；只有定位到具体 role/layer/length 的误差源后，才进入 L1–L4 或 Attention 独立实验。
若 W-only、A-only 和 Both 的方向不一致，优先检查坐标变换/部署状态交互，不继续盲调参数。

## E0.6. 跨模型结构探针（已完成，持续作为诊断）

**问题：** `赛事说明书.txt` 只公开六个 API 和数据组织，没有公开模型名称、层数、hidden size、
GQA 或位置编码。Qwen2.5-0.5B 是当前 `proxy-v2` 的本地结构假设，不能被当作官方结构事实。

**已实现：** `evaluator/cross_model_eval.py` 支持本地 GPT-2 checkpoint 的真实前向捕获。它将
fused `attn.c_attn` 按真实权重切分为 Q/K/V，以 MHA (`q_heads=kv_heads`) 和绝对位置编码运行，
并把单一 GELU `mlp.c_fc` 作为一个 `ffn_in` role，不伪造 `fc_gate`/`fc_up`。它复用
`official_eval.py` 的 NVFP4/HiF4 codec、合法性检查、状态生命周期和 W/A、Q/K/V 误差源分解；
cache、JSON 和报告与 Qwen proxy 分离，协议标为 `cross-model-probe-v1`。

**已观测（同一 GPT-2 72 Linear + 60 Attention panel）：**

| 候选 | 官方分数 | GPT-2 Linear | GPT-2 Attention | GPT-2 overall |
|---|---:|---:|---:|---:|
| v86 | 16744 | 0.375010 | 0.411100 | 0.391414 |
| v147 | 16579 | 0.518011 | 0.411100 | 0.469415 |
| v140 | 15838 | 0.519794 | 0.497247 | 0.509545 |

官方 `v86 > v147 > v140` 与 GPT-2 `v140 > v147 > v86` 完全相反；Qwen proxy 也存在同向
反转。结论不是“GPT-2 更接近官方”，而是：Qwen 结构确实不能证明官方结构，但换模型后反转仍在，
故失真还来自隐藏权重/数据分布、case 构成或官方实现细节。跨模型结果只用于判断算法是否依赖
Qwen 特有的 GQA/RoPE/SwiGLU，并指导 role 级优化，不能进入官方分数拟合或晋级门槛。

**后续用法：** 每个新 Linear 机制先跑 Qwen 分层 panel，再用同一 GPT-2 cache 做小 panel
（默认可选 `--linear-cases/--attention-cases`），只检查合法性、误差源方向和明显结构回归；
若两种结构的改善方向冲突，回到数学不变量和逐 role 误差，不围绕模型名称继续调参。需要更接近
Qwen 层数的压力测试时，再单独生成 `gpt2-medium` cache，不与当前 panel 混排。

## E0.7. 外部 hif4 role 归因（已完成，指导下一轮）

**问题边界：** 直接运行上游 [youxilee/hif4](https://github.com/youxilee/hif4) 的
`real_data_eval.py`（commit `dd5ee6515323169dbd4133b3d4fd1ff1cb7be646`），在同一 GPT-2
12 层、`amax6 / seq128 / calib2 / test2 / config=current` 下复测 v84/v86/v140/v147。
这里的 Linear `q/k/v` 是 fused `c_attn` 拆出的三个**静态投影层**，不是 Attention 动态
Q/K/V；动态 Attention 另计一列。完整逐层和临时消融见
[`hif4 外部 role 归因日志`](../../../logs/execution/2026-09-01-hif4-external-role-attribution-v140-v86.md)。

**v140 相对 v86 的外部结果：**

| role | v86 | v140 | Δ | 12 层符号 | 当前判断 |
|---|---:|---:|---:|---:|---|
| q | .6221 | .6630 | +.0409 | 12 正 / 0 负 | 不作为第一目标 |
| k | .6341 | .7241 | +.0900 | 12 正 / 0 负 | 不作为第一目标 |
| v | .6133 | .6218 | +.0085 | 11 正 / 1 负 | 基本中性 |
| o | .5578 | .5560 | −.0018 | 5 正 / 7 负 | 近中性，先冻结 |
| fc | .5558 | .5107 | **−.0452** | **0 正 / 12 负** | **第一嫌疑** |
| proj | .5373 | .5221 | **−.0153** | 6 正 / 6 负 | **第二嫌疑，层级异常** |

这不是“Linear mean 变好所以 qkv 没问题”的推断，而是逐层符号和形状消融共同支持：
q/k 稳定改善、v 近中性，fc 全层回归，proj 虽均值较小但有严重单层回归；o 的均值不足以
支持立即重写。外部临时控制进一步显示：只关闭 `rows < channels` 的 proj-ROAB，proj 从
`.5221` 升到 `.5430`；关闭 `rows > channels` 的 fc-ROAB 无变化；关闭 fc-CAT 只恢复
`.0037`；关闭 fc-BOAT 反而降到 `.4599`。因此 fc 不是简单“删掉一个 refine”，而是要保留
BOAT 的结构收益并重做 expansive 编码/scale；proj 可以先做 ROAB-off 负向控制。

**与官方的关系和置信度：** 官方只给出总分，且 v147 的提交 SHA 未确认；外部脚本使用内置
synthetic text、候选私有标准 codec 和 causal GPT-2，不能复刻官方绝对分数。它与 Qwen
`proxy-v2`、独立 WikiText `cross_model_eval.py` 的整体排序冲突，所以不用于晋级或权重拟合。
但在“v140 代码在真实 GPT-2 causal 输入下哪个 Linear role 退化”这个问题上，证据置信度排序为：
`fc` 高、`proj` 中高、`o` 低；q/k/v 不再作为第一修复目标。官方 v147 相对 v86 低 165 分且
Attention 仍为 v86 组合，故下一轮应把官方风险归因优先投向 Linear 的 fc/proj，而不是动态
Attention 的 Q/K/V。

**后续硬约束：**

1. 先冻结静态 q/k/v 和 o；square shape 暂不通过调用顺序猜 role，避免把 qkv/o 一起误改。
2. `proj (rows < channels)` 先跑 ROAB-off 对照，再跑 Activation-only Decoupled Encoder；
   `fc_gate/up (rows > channels)` 保留 BOAT，单独设计 expansive scale/encoder。每一步都要有
   per-role/per-layer 误差，不接受只看 Linear mean 的提升。
3. O 只有在跨 fold、跨结构持续为负时才开专属路由；动态 Attention 继续采用 v86 的 Q/K 配对
   路径，V 不增加搜索。
4. 新评测报告必须同时输出静态 Linear role（q/k/v/o/fc/proj）和动态 Attention
   Q/K/V 的控制臂，明确标注二者不是同一层级，不能把 qkv 投影退化误报为 Attention Q/K/V
   问题。

## 0. 本计划替代什么

旧活动计划
[`2026-09-01-hif4-linear-0.8-under-300s-plan.md`](../archive/plans/2026-09-01-hif4-linear-0.8-under-300s-plan.md)
已归档。旧计划把 17816 视为新 Linear 框架锚点，并优先推进 block-Schur Weight GPTQ、
低秩 Activation GPTQ 和 A3 双侧残差。重新核对源码、JSON、计时和单侧 oracle 后，这个顺序
不再成立：

1. 17816 只有用户确认的官方总分，源码、SHA、Attention 配置和官方时间仍未同步，不能作为
   可复现代码父版本。
2. 当前根目录 `solution.py` 是 v140 Linear、v86 Attention 和一轮额外 A3 的单文件组合；它不是
   17816 源码。
3. 当前 A3 并非一次 block-Schur 增量，而是在已有 `_crossfold_weight_output` 后再次调用同一个
   完整离散求解器。它把本地 Linear 从 pre-A3 的 `0.5073546371` 提到 `0.5100503237`，同时把
   API 从 `222.2266s` 增到 `300.3507s`，新增约 `78.1s`。
4. 用户已确认 v147 官方结果为 `16579 / 211s`：时间通过，但分数低于 v86 的 `16744`，因此
   v147 已标记 `REJECTED`。由于 v147 目录源码曾被替换，官方提交 SHA 仍为 `unconfirmed`。
5. 历史五层×七 role 单侧 oracle 为：当前双侧 `0.523019`、Weight perfect `0.704170`、
   Activation perfect `0.820357`。它不是当前全量成绩，但足以说明只改善 Weight 不能承担
   `0.8` 目标，Activation 才是主要误差源。
6. 当前本地 API 分解约为：Weight calibration `219.694s`、Activation dynamic `18.716s`、
   Attention calibration `55.616s`、Attention Q/K/V dynamic `6.325s`。主矛盾是 Linear
   calibration 的重复离散求解，不是动态完整 Hessian 的存储。

因此新计划从“继续叠加求解器”改为“重写合法 HiF4 编码决策、解析构造等价坐标系、把校准
oracle 编译成低复杂度部署规则”。Block-Schur 和双侧残差只保留为后期一次性残差步骤，不再是
第一优先级。

## 1. 不变边界与证据纪律

### 1.1 活动源码与场景冻结

- 正式候选只修改根目录 `solution.py`，保持完整、单文件、自包含和六个公共 API。
- Linear 实验固定 v86 Attention，不同时修改 Attention。
- Attention 实验固定已选 Linear，不同时修改 Linear。
- 17816 源码到位前只记录它是官方精度事实，不重建、不猜测、不把文字描述当成源码。
- 当前根的最近完整同行为结果来自
  `artifacts/official_eval/active-v140-linear-v86-attention-a3-official-shape-v1.json`；源码清理后的
  SHA 与该 JSON 不同，因此新代码变化前先恢复一个可复现、不可变的 pre-A3 对照。

### 1.2 评测与时间

- 唯一本地主评测器是 `evaluator/official_eval.py`，当前协议为 `proxy-v2`；旧
  `official-shape-v1` 只作 immutable 历史诊断。
- 模型/数据固定为 Qwen2.5-0.5B、分层真实 W/A panel、Attention calibration lengths
  `[10,128,512,1024,1024]`、validation/test holdout 和同一只读 proxy-v2 cache；
  `--full-cases` 仅用于额外 stress。
- 校准调用保持官方状态图：168 个 layer/role Weight state、24 个 Attention state，动态
  case 复用 state；不能按 case 重新校准。默认全量 case 不使用任何 5:4 或其它人为权重。
- 本地只比较同机 A/B 的 `linear_mean`、`attention_mean`、逐 role/case 和六 API 分解。
- 官方 `<300s` 只由官方回传确认。本地秒数不得覆盖 v86 通过和 v128/v129/v130/v131 超时事实。
- 小样本 workbench 只用于回答机制问题；正式晋级必须跑完整 proxy-v2，并通过 E0 趋势审计。

### 1.3 版本纪律

- 每个正式版本只引入一个数学机制；alpha、seed、rank、offset、block size 等内部比较写入同一
  workbench，不逐项编号。
- 未晋级版本必须标记 `_rejected`，官方超时版本标记 `_timeout`。
- 不再修改已有 immutable archive；当前 v147 源码/JSON 混淆先通过独立审计记录修复，不覆盖
  原始 JSON。
- 产生正式版本或状态变更后执行 `git diff --check`、提交、推送并核验工作树。

## 2. 新的统一问题定义

对 Linear，最终目标仍是：

\[
\min_{R,Q_X,Q_W\in\mathcal H}
\mathbb E\|XW^T-Q_X(XR)Q_W(WR^{-T})^T\|_F^2,
\]

其中 `R` 必须可逆且部署复杂度受限，`\mathcal H` 是合法 HiF4 五字段码域。

现有框架的局限不是没有输出目标，而是：

1. 决定 mantissa code 的编码尺度与最终 E6M2 解码尺度基本绑定；
2. Smooth/Permutation/Hadamard 依赖有限候选搜索，未直接求解两侧二阶几何平衡；
3. 输出度量主要通过在线 Gram/coordinate refine 兑现，校准和动态都重复做离散搜索；
4. Weight 侧重复完整 output oracle，只取得千分位增益。

新框架分三层：

1. **表示层：** 解耦编码尺度与合法解码尺度，改变 code assignment，但不改变最终格式；
2. **坐标层：** 用解析式层级矩阵平衡替代候选式 Smooth/Permutation/Hadamard；
3. **编译层：** 用昂贵校准 teacher 生成合法最优决策，再编译成动态阶段的一次阈值/LUT 编码。

## 3. Linear 主线

### L0. 恢复可复现父版本与建立最小 attribution workbench

**目的：** 在写新算法前恢复可信基线，并确认当前 A3、Activation 编码器和 Weight 编码器各自的
真实贡献；同时把外部 role 结论落到本地可复现的最小控制臂。L0 不分配正式版本号。

**代码/证据入口：**

- `solution.py` 的 `_dense_to_hif4`、`_crossfold_weight_output`、`_refine_activation`；
- pre-A3 JSON：`v147-v86-attention-v140-linear-official-shape-v1.json`；
- 当前 A3 JSON：`active-v140-linear-v86-attention-a3-official-shape-v1.json`；
- v148 失败 JSON；
- 历史 L0 单侧 oracle，只作方向证据。

**工作：**

1. 从 Git 历史恢复 pre-A3 可复现单文件对照，不覆盖现有 v147 目录和原始 JSON。
2. 在固定层/role 的未编号 workbench 中记录：parent、只加当前第二次 output pass、合法
   Activation block oracle、合法 Weight block oracle。
3. 对每个 case 记录真实输出相对 MSE、Weight/Activation 各自误差、接受 block 数和 API 热点。
4. 按 `q/k/v/o/fc_gate/fc_up/proj` 汇总；对 `proj` 记录 ROAB-off，对 `fc_gate/up` 记录
   BOAT-on/off 和 CAT-on/off。q/k/v/o 只做冻结回归基线，不在同一 workbench 中混入新编码器。

**完成条件：** 得到可复现 source SHA，并能解释 `0.507355→0.510050` 的增益来自哪些 role，且
不再把当前 A3 称为 block-Schur。

**失败处理：** 若无法恢复 JSON 对应源码，保留 JSON 为 `non-reproducible evidence`，以当前根
建立新的、独立编号的基线，不伪造旧归档。

### L1. Role-gated Activation-only Decoupled HiF4 Encoder

**唯一机制：** 只改变动态 Activation 的 HiF4 code assignment；Weight、等价变换和 v86
Attention 全部冻结。第一实现只覆盖外部归因指向的 `proj` 与 `fc_gate/fc_up`，q/k/v/o
沿用 parent，避免把静态 qkv 投影和动态 Attention Q/K/V 混为一个问题。

对于一个 64-channel block，引入只存在于编码阶段的高精度尺度 `s_q`：

\[
z=\operatorname{round}\left(\frac{4x}{s_q d_{lv2}d_{lv3}}\right),\qquad
\hat x=s_d d_{lv2}d_{lv3}\frac{z}{4}.
\]

- `s_q` 只决定 mantissa/sign code，不保存、不参与解码；
- `s_d` 仍投影为合法 E6M2；
- lv2/lv3、mantissa、sign 和输出 shape 不变；
- 固定 code 后，在部署 Weight metric 下闭式更新 `s_d`，再投影到最近及相邻合法 E6M2；
- incumbent 始终保留，任何提前停止都返回合法五字段。

这不是继续扩大 offset 搜索。历史 E6M2 oracle只覆盖 stored-scale 邻域，没有覆盖编码尺度和解码
尺度解耦后的 code assignment。

**动态复杂度约束：** 用固定、向量化的编码尺度候选或闭式更新替换现有部分 coordinate refine；
不得新增 Python per-coordinate/per-candidate 循环，六 API 调用数不变。

**role 路由：** `rows < channels` 的 `proj` 先使用 ROAB-off 作为负向控制，再接入解耦 encoder；
`rows > channels` 的 `fc_gate/fc_up` 保留 BOAT，只替换其编码/scale teacher。square shape 的
q/k/v/o 暂不依据调用顺序区分，统一保持 parent；如果后续必须拆分，必须先在 state 中写入
可验证的 role 标记或输入统计签名。

**最小验证：**

1. 合法 round-trip 与 reference validation；
2. 同一 block 上比较 parent、解耦 encoder、合法 exhaustive teacher 的输出 metric；
3. 两个 calibration fold 均改善才进入完整评测；
4. 完整评测固定 v86 Attention，记录逐 role 和 Activation dynamic 时间。

**失败处理：** 若 teacher 有明显空间但快速 encoder 不兑现，转 L3 编译；若 teacher 本身无空间，
停止该表示族，不扩大 `s_q` 网格。

### L2. Hierarchical Matrix Balance

**唯一机制：** 用解析式 2×2 层级矩阵平衡替换当前 Smooth/Permutation/Hadamard 候选框架；
L1 编码器和 Attention 固定。

对一个 2-channel pair，计算最终坐标下的：

\[
A=\mathbb E[X^TX],\qquad B=W^TW.
\]

求 SPD `S` 满足：

\[
SAS=B,
\]

并取 `MM^T=S`。部署：

\[
X'=XM,\qquad W'=WM^{-T},\qquad X'W'^T=XW^T.
\]

按 HiF4 层级执行固定 butterfly wiring：

1. lv3 的 4-channel group 内配对；
2. 同一 lv2 的两个 lv3 group 间配对；
3. 64-channel block 内的 lv2 group 间配对。

每层只计算一次解析矩阵，不搜索 alpha、seed、angle 或 block size。正则化由样本数和矩阵 trace
确定，不做参数 sweep；变换条件数受数值合法范围约束。

**复杂度约束：** 新 pair transform 必须替换现有 multiplier/permutation/FWHT，不能叠加。动态
仍为固定 stage 的 `O(TC)`；同机算子数和 API 时间应不高于所替换路径。

**最小验证：** 连续输出不变量、逆变换稳定性、两折合法量化输出 MSE、逐 role 接受情况。

**失败处理：** 若解析平衡连续域正确但量化回退，记录是哪个层级 wiring 造成回退；不改成角度网格
或学习型 dense rotation。

### L3. Oracle-to-Encoder Compilation

**唯一机制：** 不改变合法码域和坐标变换，只把 calibration 上的合法输出 oracle 编译成低成本
动态规则。

Teacher 在 calibration 上允许使用最终部署 `Q_W` 的完整输出 metric。Student 只能读取动态阶段
便宜、尺度不变的局部特征：

- 64/8/4 group 的 `amax` 比值；
- `rms/amax`；
- subgroup 幅值排序；
- calibration 写入 state 的静态 Weight metric 分组权重。

Student 输出 `s_q/s_d` 选择、lv2/lv3 决策或少量合法候选索引。部署只允许阈值、查表和批量张量
运算，不运行 teacher、不保存 token/output tensor、不做候选循环。

**验收：** 在交换 fold 上改善真实 `Q_XQ_W^T` 输出误差；动态复杂度不高于 L1；完整评测才决定
是否晋级。

**失败处理：** 若 teacher-student gap 大，报告不可压缩的决策特征；不扩大成神经网络或复杂
meta-router。

### L4. Weight Decoupled Encoder

**唯一机制：** 把 L1 已验证的编码/解码尺度解耦应用到 Weight；Activation、坐标变换和 Attention
冻结。

Weight 固定 code 后使用变换后 `H_X` 的二次型闭式更新 stored scale；所有输出仍是完整合法
HiF4 block。实现必须向量化共享输出行的 Hessian 计算，不再重复完整 `_crossfold_weight_output`。

若 L4 有效，再评估一次性 block residual：每个 block 最多重新编码一次，分解每层复用一次，禁止
第二轮完整 candidate oracle。这个 residual 是后续独立版本，不与 L4 同版实现。

## 4. Linear `0.8` 可达性判断

在 L1–L4 过程中维护四条同口径曲线，不分配微版本号：

1. 当前合法 player；
2. 新坐标系 + 当前 encoder；
3. 新坐标系 + 快速 decoupled encoder；
4. 新坐标系 + 合法 teacher/oracle。

判断依据是剩余误差比例，而不是把本地 mean 换算成官方分数。若合法 teacher 在多个 fold/role 上
仍不能接近 `0.8`，下一步扩展表示/坐标系；若 teacher 可达而 student 不行，继续编译；若 student
已接近 teacher，不再增加 GPTQ 迭代。

## 5. Attention 独立主线：同复杂度或更低复杂度

Linear 稳定前不修改 Attention。Attention 实验启动后固定 Linear，父版本仍以 v86 Attention 的
官方 `16744 / 222.7s` 通过事实为时间/泛化锚点。

v128/v129 的本地 Attention `0.837789/0.836579` 只说明公开数据存在空间；它们的 Attention
calibration 为 `199.8/126.4s`、Q/K dynamic 合计约 `32s`，且官方均 timeout，不能整条迁移。

### A0. v129 增益归因 workbench

不建立版本，只在固定 Attention 小样本上拆分：

1. v129 静态 transform + v86 快速 encoder；
2. identity transform + v129 Gram dynamic refine；
3. v129 完整路径；
4. v86 parent。

目的只回答 `0.836` 的增益来自静态坐标还是动态三轮 Gram refine；不重新做全量排名。

### A1. Analytic Matrix-Smooth Q/K

**唯一机制：** 用解析式 GQA group-local 2×2 矩阵平衡替换 reciprocal diagonal、permutation、
Hadamard 和 GQRB candidate search。

对每个 GQA group 聚合 Q covariance `A`，K 使用共享 head covariance `B`，求 `SAS=B` 和
`MM^T=S`，部署：

\[
Q'=QM,\qquad K'=KM^{-T},\qquad Q'K'^T=QK^T.
\]

使用与 HiF4 4/8 层级对齐的固定 2×2 pair wiring。calibration 只计算一次解析状态，并只进行一次
parent/child deployed Attention 复评。

**复杂度契约：** Q/K dynamic 至多一次 `O(TC)` pair transform + 一次 HiF4 encode；不得保留
Gram64 sweep、angle/seed/alpha/block 网格。

### A2. Quantization-aware K Fixed Point Center

**唯一机制：** 保持 A1 transform 和 Q/V 不变，仅更新 K 公共平移。

K head 的公共平移是 softmax 精确不变量。固定当前合法量化 code 后，执行固定两次：

\[
c\leftarrow\operatorname{mean}(K-\hat K(K-c)).
\]

最终 state 只保存一个 center vector；动态复杂度与当前 K centering 相同。

### A3. Attention Decoupled Encoder

**唯一机制：** 将 L1 验证的解耦 HiF4 encoder 分别接入 Q、K、V，替换现有 offset/refine 路径；
A1/A2 state 固定。

- Q/K 在精确等价变换后编码；
- V 不引入 `P^TP`、length-keyed state 或 per-token coordinate search；
- 输出格式和 Attention 计算不变；
- dynamic 每个 operand 只执行一次固定复杂度 encode。

### A4. Static Softmax-Fisher Importance

**唯一机制：** 不改变 transform/encoder，只用一个固定 token view 的 softmax Jacobian构造 Q/K
静态 Fisher diagonal，替换当前对侧二阶矩 importance。state schema 和 dynamic 运算不增加。

V 不部署 PAWV；V 的提升只来自 A3 encoder。若 Fisher calibration 开销高于被替换的候选选择，
则拒绝该实现，不通过缩短隐藏 token 分布来伪造时间收益。

### Attention 复杂度硬约束

正式 Attention 候选必须同时满足：

1. API 调用数与 v86 相同；
2. calibration 为一次固定统计 pass 和最多一次 parent/child 输出复评；
3. Q/K dynamic 各一个固定 pair transform 和一个 encode；
4. V dynamic 一个 encode；
5. 无 per-sequence/per-token candidate loop；
6. 无 Gram coordinate sweep、PAWV、长度字典或随序列增长的搜索。

复杂度按算子结构判定，本地秒数只做同机 A/B；官方时间状态在回传前写 `unknown`。

## 6. 执行顺序

1. **L0 证据修复与 role attribution**：恢复可信 pre-A3 对照，复核外部 hif4 指向的 fc/proj
   回归，并明确当前 A3 的 role 收益和真实代价。
2. **L1 Role-gated Activation-only Decoupled Encoder**：先处理 proj，再处理保留 BOAT 的 fc；
   q/k/v/o 冻结作为回归对照。
3. **L2 Hierarchical Matrix Balance**：用解析变换替换候选式 D/P/H，不叠加部署算子。
4. **L3 Oracle compilation**：把输出最优合法决策蒸馏成动态一次编码。
5. **L4 Weight Decoupled Encoder**：复用已验证表示机制，禁止重复完整 output pass。
6. **0.8 可达性复判**：比较 player/student/teacher 曲线，决定继续表示、编译还是停止该框架。
7. **官方单变量提交**：固定 v86 Attention，只提交一个 Linear 数学机制；以官方分数和 `<300s`
   共同判定。
8. **Attention A0–A4**：Linear 稳定后独立推进，每版只替换一个机制并遵守复杂度契约。

## 7. 最小产物与失败记录

每个正式算法只要求能改变决策的产物：

- 完整单文件源码和 SHA256；
- parent 与唯一算法变化；
- `proxy-v2` 命令、模型/data revision、cache、device；
- Linear/Attention mean、必要的逐 role/case；
- 六 API 与 wall time；
- 官方分数/时间/status，未知写 `unregistered/NA`；
- 接受或拒绝原因。

workbench 只保留一个汇总 JSON/Markdown，不为内部参数分配版本。失败若能否定一个数学机制，保存
一个代表性 `_rejected`；no-op、重复 sweep 或无归因价值的实现不归档。

## 8. 当前决策

- 不继续扩大 alpha、offset、seed、rank、block 数、sweep 或 damping。
- 不把当前第二次 `_crossfold_weight_output` 继续包装成 block-Schur。
- 不优先实现低秩 Activation GPTQ；当前动态 Activation 不是主要时间热点。
- 不把 v128/v129 高 Attention 路径直接合并到 v86。
- Linear 第一正式机制是 role-gated Activation-only Decoupled HiF4 Encoder：先 proj，后 fc；
  静态 q/k/v/o 暂不改动。
- Attention 第一正式机制是 Analytic Matrix-Smooth Q/K，且只能在 Linear 稳定后启动。
- Block-Schur、双侧联合残差和学习型 rotation 均降为后续备选；只有新的表示/解析框架证明正向后
  才重新评估。
