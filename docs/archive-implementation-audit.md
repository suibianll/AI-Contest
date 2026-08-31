# 归档算法实现与可复现性审计

> 审计日期：2026-08-31
> 范围：`solutions/` 下 v000–v126 候选（排除工具目录 `.mimosa`）、`logs/execution/`、当前根 `solution.py` 以及所有历史计划。
> 结论性质：这是实现审计和实验可复现性审计，不是官方成绩承诺。

## 1. 审计口径

本次审计把每个负结果分成三类：

1. **算法负结果**：源代码完整、目标函数和部署路径一致，且在固定 Qwen 全层 panel 上确实不如父版本。
2. **实现或指标问题**：结果不能直接用于否定算法，因为写回路径、门禁目标或统计对象存在明显错位；需要修复后重跑。
3. **不可复现结果**：归档只有结果而没有被评估的源码，或者缺少源 SHA；可以保留为历史线索，但不能作为严格的算法结论。

本地主指标仍然是 Qwen2.5-0.5B 同口径 panel：

\[
P = 250 g_L + 200 g_A,
\]

其中 \(g_L\) 和 \(g_A\) 分别是 672 个 Linear、96 个 Attention 诊断 case 的平均 gain。五模型 raw sum 只作结构性回退诊断，不能和官方分数相加或换算。

## 2. 当前根与已确认效果

根目录 `solution.py` 是 v126：在 v125 全部 Linear/Attention 机制上修复 B2 PAWV 的变长 calibration，以长度字符串分组 diagonal，校准与动态 V 精确匹配当前行数，未命中回退；同时删除未使用 low-rank 时仍执行的 full `P^TP/eigh`。当前规范 LF SHA256：

`47e2e3ab76c6deaac8de47bbcbd8f689cf5989dc8ff9e9081a887ec89e819b08`

| 指标 | v125 已测 precision parent；v126 未重跑 full |
|---|---:|
| Qwen Linear mean | 0.509760 |
| Qwen Attention mean | 0.842039 |
| Qwen shaped panel | **295.847849** |
| Qwen native total | 423.394380 |
| Qwen API time | 2653.580314 s |
| 官方分数 | 尚无提交结果 |

当前正式路径的有效组件是：

- **BOAT**：RMS 对角平衡与 4/8/16/64 signed-Hadamard，属于两侧 operand-local 的等价变换。
- **cross-fold Weight-HSDQ**：用 `AᵀA` 二阶增量搜索离线权重参数，并用另一折校验。
- **Gram-hierarchy Activation-HSDQ**：用静态变换后权重的 `WᵀW` 选择激活层级/offset，并限制 block 与 sweep 数。
- **Expansive-FFN CAT balance**：仅对 `rows > channels` 的结构形状使用固定 α=0.25 RMS 对角 balance；v106 通过 Qwen full-layer。
- **B1 GQRB margin**：GQA group-local 正交 Q/K mixing；只在完整部署复评有 margin 时接纳。
- **B2 PAWV diag-only**：用 attention probability 的 token-row 对角 Hessian 做 V 的离散 refinement；跨 token 低秩项未进入根。
- **L4a final deployed-Gram row gate**：仅对 `rows > channels` 且 `channels <= 1024`
  的 expansive FFN，在 v107 parent 和最终部署 `G_q=W_q.T@W_q` 候选之间做完整 Gram
  二次型逐行选择；不变差的行才写回，避免 block surrogate 回退。
- **L5a block-local permutation**：在每个 64 维 hierarchy block 内，用独立
  `amax/rms` pressure 选择至多一个固定排列，与 `D`、signed-Hadamard 同步作用于
  W/A；两折 operand-local gate 不通过则回退 identity。该等价坐标变换 full-layer
  panel `295.482473`，Linear `0.508298`。

最新正向链为：v086 `267.307909` → v098 `293.793700` → v100 `293.797301` → v106 `294.272633` → v107 `295.157057` → v109 `295.239309` → v110 `295.242780` → v111 `295.482473` → v115 `295.680651` → v116 `295.734045` → v117 `295.785829` → v118 `295.808212` → v121 `295.811281` → v124 `295.820229` → v125 `295.847849`。C1a v119 不改变分数，只把 API 时间从 `2249.746s` 降至 `2040.505s`；C1b v121 两轮 refresh 再增加 `0.003069` panel，但 API 回升至 `2180.450s`；v125 `max_blocks=8` 再增加 `0.027620` panel，但 API 升至 `2653.580s`。其中最大跃迁来自 C86 实验集合重写为 clean 单一路径；v106 的增益只来自 expansive `fc_gate`，v107 的增益来自窄输入 q/k/v/o 的 Gram-gated Global-LRH，v109 的增益来自 expansive FFN final-Gram row gate，v110 再来自其上 GALS 小预算，v111 来自 block-local 等价排列，v115 来自窄输入 rank-16 off-block factor，v116 来自宽 `proj(d=4864)` rank-4 off-block factor，v117 来自 full `G_64` hierarchy coordinate sweep，v118/v121/v124/v125 来自结构化 block-circulant proposal、gradient refresh、rank 与 block budget 扫描及完整部署 Gram gate。

## 3. 归档源码审计结果

### 3.1 v092 / v105 full-hierarchy LRH 写回审计（原结论已更正）

文件：[`v092 solution.py`](../solutions/20260830_v092_a3-lrh-r8-rejected_score292.426982_time382s/solution.py)

早期审计曾认为 v092 的 `_polish_weight_lrh` 根据 `den_work` 重新搜索了
`scale_codes`、`lv2`、`lv3`，随后在最后调用：

```python
return _write_codes(parent, codes)
```

但逐行复核保存的 v092 源码后，没有找到上述 scale/lv2/lv3 搜索；该版本实际
沿用 parent denominator，再把 mantissa/sign 交给 `_write_codes`。因此“搜索结果
被写回函数丢弃”是对源码的过度描述，不能作为 v092 结果的既定事实。

为消除这个不确定性，v105 实现了真正的原子 full-hierarchy 候选：E6M2 scale
offset、lv2、lv3、sign、mantissa 一起生成并由新 denominator 解码；合成测试
`29 passed`。Qwen 五层×七 role 的 35-case screen 评估 70 个 fold 候选，仅 1 个
通过交换 fold，且最终 0/35 case 改变 stable parent，`both_player` 与 L0 逐条相同。

**当前影响**：v092 的 `292.426982` 仍只代表保存的旧实现；v105 的正确写回
复验则显示当前 LRH 在两折 calibration 上泛化不足。该算法族按 active plan
标记为“corrected LRH rejected”，不再扩大 rank、block 数或 sweep。证据见
[`v105 archive`](../solutions/20260830_v105_l1-full-hierarchy-lrh-rejected_screen523019_time266s/)
和 [`L1 execution log`](../logs/execution/2026-08-30-l1-full-hierarchy-lrh.md)。

### 3.2 P1：v095 的全局 LRH 用错接受门禁

文件：[`v095 solution.py`](../solutions/20260830_v095_a6-global-activation-lrh-rejected_score282.616646_time374s/solution.py)，约 715–717 行。

全局低秩候选优化的是 block Gram/global surrogate，但最后用未加权逐元素 MSE 作回退保护：

```python
initial_error = (q_initial - x).square().sum(...)
refined_error = (q - x).square().sum(...)
keep = refined_error <= initial_error
```

实际部署的激活影响是二次型 \(e^T(W^TW)e\)，不是 \(\|e\|_2^2\)。因此 surrogate 可能在 Gram 目标上变好，却被错误的 MSE 接纳；也可能反过来在关键方向变坏而 MSE 仍通过。

**影响**：v095 的 `282.616646` 是“全局 LRH + 错误门禁”的结果，不能单独作为全局 LRH 的最终否定。
**动作/结果**：v107 已用部署量化权重的完整 `G_q=W_q.T@W_q` 做逐行 Gram gate；
五层 screen `0.52894931`，24 层 Linear mean `0.5069966356`、panel `295.157057`，
相对 v106 分别 `+0.003538` 和 `+0.884423`。Gram/MSE 冲突率 `0.567475`，证明旧
gate 会改变候选裁决。4-block 版本因此成为当前精度 parent；时间 `481.036527s`
暂不作为探索期否决，L3 代码/结果归档在
[`v107 archive`](../solutions/20260830_v107_l3-global-lrh-precision-parent_score295.157057_time481s/)。

### 3.3 P1：v099 PAWV rank-8 的低秩度量在最终 Q/K 变换前计算

文件：[`v099 solution.py`](../solutions/20260830_v099_b2-pawv-lowrank-rejected_score334.101693_time16s/solution.py)，约 1128–1135 行与 1275–1297 行。

代码先用原始 `q_samples/k_samples` 构造 `row_diagonal,row_lowrank`，然后据此量化 V；之后才选择并应用 reciprocal RMS、K-center、Hadamard、GQRB mixing，最终 Q/K Gram 在变换后才重新计算。于是 PAWV 的 rank-8 \(P^TP\) 度量和最终 attention 几何不在同一坐标系。

**影响**：v099 layer-1 `334.101693` 的回退可能部分来自 stale metric，而不是 rank-8 低秩项本身。
**动作**：先确定最终 Q/K 变换，再用变换后的 attention probability 重建 diag/low-rank metric；采用交替 Q/K/V 更新，并用真实部署输出复评。diag-only 仍是当前安全基线。

### 3.4 P1：v102/v103 GALS 的统计对象和角色识别仍不严谨

v102/v103 的 GALS 使用校准时浮点 `weight_t` 计算的 `gram64`，而实际权重已经经过 HSDQ 成为 `weight_params`。v104 的 A7 直接把 Gram 换成量化后近似 `WqᵀWq`，单层变好但全层回退且超时，这说明统计对象错位是一个合理怀疑点，但不是唯一原因。

另外，v103 所谓 role-aware 并没有读取显式 role/模块身份，而是把 `weight_shape` 通过 `weight_channels <= 2048 and weight_rows <= weight_channels` 当作角色代理。形状代理会把不同模块混到同一门控，不能证明“真正 role-aware GALS”无效。

**影响**：v102/v103 的负结果只能否定“当前静态浮点 Gram + 形状代理 + 稀疏预算”组合。
**动作**：重新生成最终 `weight_params` 后的 Gram；在 API 内传入显式、只读 role id；用三折/异构模型筛选后再决定是否部署。不要直接恢复全局 GALS。

### 3.5 P1：v108 首次 L4a screen 是路由 no-op，v109 已完成有效复验

文件：[`v108 archive`](../solutions/20260831_v108_l4a-final-weight-gram-screen-rejected_scoreNA_timeNA/)。

v108 首次实现把 `dense.shape[0] > dense.shape[1]` 当作权重的 expansive shape 判断；
动态 API 中 `dense` 的第一维是 token 数（128），不是离线权重的 output-row 数，因而
所有真实调用都没有进入 final-Gram 路由。其 screen `0.5289493081` 与 v107 完全相同，
不能作为 L4a 算法失败证据。v108 归档 README 已追加该更正，并保留源快照以便复核。

随后 v109 改为 calibration 阶段写入静态 `final_gram_route`，并使用双候选：v107
parent 与最终部署 Gram 候选；对完整 `G_q` 做逐行精确比较，只接纳不增大
`e^T G_q e` 的行。五层 screen `0.5292690913`，24 层 full-layer Linear mean
`0.5073256468`、panel `295.239309`，相对 v107 分别 `+0.0003290112`、`+0.0822528`。
v109 已归档为当前精度 parent；API `517.285773s` 只作为探索期时间记录。

### 3.5 P2：v087–v091 源码缺失，负结果不可完全复现

`solutions/20260830_v087...` 至 `v091...` 目录目前只有 `result.md`，没有被评估的 `solution.py`，结果文件也没有 source SHA；早期的 v028 scale-code probe 也只有诊断结果、没有提交源。v087 的 E1 源可以从 Git 提交 `7e1f818` 的父版本恢复；A2–A5 的精确源快照无法从普通 Git 历史唯一恢复。

**影响**：这些版本的分数和时间可以作为日志事实，但无法排除临时 patch、环境差异或未归档参数造成的影响。
**动作**：将它们标记为 `non-reproducible`；不覆盖已有结果，不伪造源码；未来任何候选必须把完整源、规范化 SHA、参数 JSON 和结果一起归档。

### 3.6 P2：历史目录名与修订官方结果不一致

- v031 目录名仍为 `score14613_time159.2s`，但修订官方口径为 `21864 / 161.3s`。
- v066 目录名为 `scoreNA_timeNA`，但修订官方口径为 `22557 / 217.2s`。
- v072 目录名仍为 `scoreNA_timeNA`，但用户最新确认官方为 `22662 / 226s`。

这不是算法实现 bug，但容易让脚本或读者误把旧目录名当成最终成绩。后续不改写不可变目录，统一在 `solutions/README.md` 和审计表中以 canonical result 字段为准。

## 4. 没有发现明确实现错误的近期候选

以下源码结构和结果记录没有发现足以推翻其结论的硬错误；它们仍可能是算法泛化失败，但负结果在当前证据下可保留：

| 版本 | 审计结论 |
|---|---|
| v093 CAT-inspired BOAT-2 | 变换顺序和正交不变量基本自洽；代价高、单层不迁移，暂按算法/泛化失败处理。诊断 helper 忽略 permutation，但不在部署路径。 |
| v094 frozen-Q(A) ridge/Qronos | Woodbury/ridge 公式与离线 Q(W) 目标数值自洽；主要问题是过拟合和 455.73s 超时。 |
| v098 B1 GQRB margin | group-local 矩阵保持正交，候选经完整部署复评；结果可作为稳定父版本。 |
| v100 B2 PAWV diag-only | 对角 token-row 目标与写回路径一致，当前根已通过 Qwen 与五模型确认。 |
| v104 A7 quantized-weight Gram | 统计替换本身已正确落地；layer-1 正向不能迁移到全层，且 470.58s 超时，按组合算法失败处理。 |
| v106 expansive-FFN CAT balance | 固定 α=0.25 的结构路由，full-layer `+0.475332` panel，API `412.65s`；仅 fc_gate 改善，作为当前 parent。 |
| v109 L4a final deployed-Gram row gate | v107 parent 与最终 `G_q` 候选做完整二次型逐行门控；full-layer panel `295.239309`、Linear `0.507326`，较 v107 `+0.082253` panel；API `517.29s`，当前精度 parent。 |
| v110 L4b final-Gram GALS | 解析 critical-scale offsets，最多 4 个高损失 block；两折完整 Gram 正向才启用，在线完整 `G_q` 行级 gate；full-layer panel `295.242780`、Linear `0.507340`，较 v109 `+0.003470` panel；API `701.90s`，前一精度 parent。 |
| v111 L5a block-local permutation | 每个 64 个通道块的压力排序/交错候选，两折 operand-local gate；full-layer panel `295.482473`、Linear `0.508298`，较 v110 `+0.239693` panel；API `726.094s`，前一精度 parent。 |
| v115 L6a rank-16 global LRH | 窄输入 off-block factor rank 8→16；full-layer panel `295.680651`、Linear `0.509091`，较 v111 `+0.198179` panel；API `716.483s`，前一精度 parent。 |
| v116 L6b wide rank-4 factor | 宽输入 `d>1024,d<=8192` 的 off-block factor；full-layer panel `295.734045`、Linear `0.509305`，较 v115 `+0.053394` panel；API `739.425s`，前一精度 parent。 |
| v117 L6c full `G_64` hierarchy | 固定 scale、最多 4 个高损 block 的 `lv2/lv3` 坐标 sweep；full-layer panel `295.785829`、Linear `0.509512`，较 v116 `+0.051784` panel；API `2019.475s`，前一精度 parent。 |
| v118 L6d structured block-circulant factor | 宽输入最多 4 个 `64×64` kernel 生成跨 block proposal，full-layer panel `295.808212`、Linear `0.509601`，较 v117 `+0.022382` panel；API `2249.746s`，前一精度 parent。 |
| v119 C1a structured proposal vectorization | 与 v118 全部 score 字段逐位相同；reference/vectorized `atol=1e-6`，API `2040.505s`（`−9.30%`），当前 precision/time parent。 |
| v120 C1b block refresh | 一次 refresh screen `0.5333730058`，低于 v118 screen，已拒绝；完整源码已归档。 |
| v121 C1b structured gradient refresh×2 | 两轮 refresh；screen `0.5333964596`，full panel `295.811281`、Linear `0.509614`，较 v119 `+0.003069`；38 项测试/compliance 通过，API `2180.450s`；用户确认官方 runtime timeout，仅按 accuracy-first 精度证据保留。 |
| v122 C1c rank-2 | screen `0.53336284`，低于 v118/v121；完整源码已归档，未跑 full-layer。 |
| v123 C1c max-blocks-2 | screen `0.53335171`，低于 v118/v121；完整源码已归档，未跑 full-layer。 |
| v124 C1c structured rank-8 | screen `0.53343639`；full panel `295.820229`、Linear `0.509649`，较 v121 `+0.008948`；26 项核心测试/compliance 通过，API `2323.911s` 超时但 accuracy-first 接受。 |
| v125 C1c rank-8 / max-blocks-8 | screen `0.53358298`；full panel `295.847849`、Linear `0.509760`，较 v124 `+0.027620`；Attention `0.842039` 逐位不变；API `2653.580s`，仅作 precision-only 证据，runtime invalid。 |

最近候选的静态/运行时 Linear 合规扫描均为 `violations=0, static=0`；本次没有发现把 `A@W` 输出监督写入在线 `Q(A)` 的新违规。v107 Attention 合约专项审计（见 [`2026-08-31-v107-attention-contract-audit.md`](../logs/execution/2026-08-31-v107-attention-contract-audit.md)）曾在固定 `seq=128` 下通过，但官方变长 mini sample 已确认 v100/v107 的 `_build_pawv_metric` 会发生 shape mismatch。v72 随后以 `22662 / 226s` 官方通过，确认 v66/v72 Attention 闭包是安全边界。v126 已按长度分组修复 PAWV，并通过 `[10,128,512,1024,1024]` 完整公开 calibration API；合规、本地 smoke 和官方通过仍分开记录。

## 5. 已实现、已验证、未验证的方向矩阵

| 算法族 | 已实现并保留 | 已实现但回退 | 当前仍未验证/需要修复 |
|---|---|---|---|
| Linear 基础 | BOAT、**v111 L5a block-local permutation**、cross-fold Weight-HSDQ、Gram-hierarchy Activation-HSDQ、**v106 expansive CAT balance**、**v107 Gram-gated Global-LRH**、**v109 final deployed-Gram row gate**、**v110 final-Gram GALS**、**v115 L6a rank-16**、**v116 L6b wide rank-4**、**v117 L6c full `G_64` hierarchy**、**v118 L6d structured factor**、**v119 C1a vectorization**、**v121 C1b refresh×2**、**v124 C1c rank-8**、**v125 C1c max-blocks-8**、512-row weight sampling、历史稳定 A@W/JDRQ 组件 | blockwise BOAT-2、全宽/逐块 A@W、大步长 headroom、full-H、**v105 corrected full-hierarchy LRH** 等 | 稀疏 Schur、统计元路由、外部差异审计、C2/C3 跨模型与 state/time 压缩 |
| Attention | GQA head-local rotation、MHA K-center、B1 GQRB margin、B2 PAWV diag-only | causal CVaR、全模型 K-center、PAWV rank-8 当前实现 | 最终 Q/K 变换后的 PAWV rank/position bucket、交替 Q/K/V、真正 role-aware 的结构门控 |
| 变换/CAT | 固定低自由度 CAT/BOAT 子集、共享 Hadamard | R64、CAT β 网格、full-H selector、BOAT-2 | 低自由度新坐标系或外部实现差异对照，尚无可部署候选 |
| 诊断/工程 | E0-G/D0 scale oracle、C0 五模型确认、clean 单路径重写 | 全局 scale 扩张部署 | 三折合成宽度矩阵、元策略路由、计算预算重分配 |

## 6. 归档与计划治理结论

旧计划中确实存在“写成已执行、实际只跑了子集”的内容，主要包括：

- E1 计划承诺的合成宽度覆盖、三折验证和完整 progressive beam 尚未完成；当前 v087 只完成了一个高成本实现。
- grid/consolidated/accuracy-first 计划中的 CAT/Householder、frozen-Q(A)、global-LRH 曾被列为下一步，但对应实验已经有结果，且 v092/v095 还带实现缺陷，不能继续按旧文字推进。
- 计划中把 v 的 GALS 正向 oracle 当作部署机会，但 v102/v103 已证明当前 sparse/shape-proxy 组合不稳定；v109/v110 已完成无 role-id 的 final-weight-Gram 与 GALS 复验，下一阶段转向 L5 结构路线。
- 官方提交/兑换率校准从未完成，不能用本地 panel 推断官方 36000 距离。

治理动作：L6、C1 计划主体已完成；当前唯一 active 计划仍是 [`2026-08-31-hif4-active-c1-structured-linear-plan.md`](superpowers/plans/2026-08-31-hif4-active-c1-structured-linear-plan.md)，下一步为 C2/C3；其余实施计划和流程文档均位于 [`docs/superpowers/archive/plans/`](superpowers/archive/plans/)。`plans/README.md` 与归档索引只描述导航及历史性质，不产生下一步指令。

## 7. 审计后的优先级

1. 正式提交线从已通过的 v72 `22662 / 226s` 出发，只移植 Linear 变化并冻结其 Attention 闭包；v125/v126 必须先通过 C3 压缩与完整复测，不能直接提交。
2. L1 的 v105 corrected full-hierarchy LRH 已完成并拒绝；不再扩大其自由度。
3. L3 v107、L4a v109、L4b v110、L5a v111、L6a v115、L6b v116、L6c v117、L6d v118、C1a v119、C1b v121、C1c v124/v125 均已完成并产生精度/time parent；v120、v122、v123、L5b/v112、
   L5c/v113、L5d/v114 已按 screen 归档拒绝，L5e 已完成可达性 checkpoint。
4. 当前只执行唯一活跃计划的 C2/C3 structured Linear 路线；Linear 不回到已否决的 sampler、
   joint residual、H32/H64 或 group-only solver，先完成跨模型审计，再做最终 state/time
   checkpoint。Attention PAWV rank/position
   metric 继续独立延后。
5. 每个实验必须保存完整源和 SHA；只要没有完整源，就标为不可复现，不把结果当作硬上限。

## 8. 本次可复核检查

- 当前根静态/运行时 Linear 合规扫描：`violations=0`、`static_violations=0`，8 个 review items 均是允许的离线 `A@W` 人工复核提示。
- 当前根代表性测试：L6a/L6b/L6c/L6d 定向集合（含 `tests/test_global_activation_lrh.py`、`tests/test_l5a_joint_transform.py`、`tests/test_linear_compliance_guard.py`、`tests/test_linear_error_decomposition.py`、`tests/test_expansive_cat.py`）共 **36 passed**。
- 全量 pytest 本次为 `51 passed, 38 failed, 1 skipped, 5 errors`。失败主要来自仍针对已删除 C39/FULL64/JDRQ 内部接口的历史测试，以及 Windows 临时目录权限错误；本次只改文档、归档和导航，没有把这些历史测试伪装成当前根通过。
- 计划目录检查：`docs/superpowers/plans/` 仅有一份非 README 计划；归档计划的相对链接检查通过。
