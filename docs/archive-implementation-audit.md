# 归档算法实现与可复现性审计

> 审计日期：2026-08-30
> 范围：`solutions/` 下 98 个 v000–v104 候选（排除工具目录 `.mimosa`）、`logs/execution/`、当前根 `solution.py` 以及所有历史计划。
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

根目录 `solution.py` 是 v100 的 B2 PAWV **diag-only** + B1 GQRB 路径；v101 是同一源码的五模型确认，不是另一套算法。当前规范 LF SHA256：

`617482cee04ff9514a8d41226b651336e4b8b86692673308e835de1091693eba`

| 指标 | 当前根 v100/v101 |
|---|---:|
| Qwen Linear mean | 0.501558 |
| Qwen Attention mean | 0.842039 |
| Qwen shaped panel | **293.797301** |
| Qwen native total | 417.882506 |
| Qwen API time | 392.423565 s（C0 复测 401.130873 s） |
| 官方分数 | 尚无提交结果 |

当前正式路径的有效组件是：

- **BOAT**：RMS 对角平衡与 4/8/16/64 signed-Hadamard，属于两侧 operand-local 的等价变换。
- **cross-fold Weight-HSDQ**：用 `AᵀA` 二阶增量搜索离线权重参数，并用另一折校验。
- **Gram-hierarchy Activation-HSDQ**：用静态变换后权重的 `WᵀW` 选择激活层级/offset，并限制 block 与 sweep 数。
- **B1 GQRB margin**：GQA group-local 正交 Q/K mixing；只在完整部署复评有 margin 时接纳。
- **B2 PAWV diag-only**：用 attention probability 的 token-row 对角 Hessian 做 V 的离散 refinement；跨 token 低秩项未进入根。

最新正向链为：v086 `267.307909` → v098 `293.793700` → v100 `293.797301`。其中最大跃迁来自 C86 实验集合重写为 clean 单一路径，不能简单归因于单个新公式。

## 3. 归档源码审计结果

### 3.1 P0：v092 的 LRH 层级结果被写回函数丢弃

文件：[`v092 solution.py`](../solutions/20260830_v092_a3-lrh-r8-rejected_score292.426982_time382s/solution.py)

`_polish_weight_lrh` 内部根据 `den_work` 重新搜索 `scale_codes`、`lv2`、`lv3`，并以新的层级构造 `q_work`。但是函数最后调用：

```python
return _write_codes(parent, codes)
```

`_write_codes` 只把 mantissa/sign 写回 `parent` 的参数副本，不会写回刚刚搜索出的 scale/lv2/lv3。也就是说，候选的 mantissa 是按新 denominator 计算，最终却和旧 denominator 配对，优化目标和部署表示不一致。

**影响**：v092 的 `292.426982` 只能说明“当前这份写回错误的 LRH 实现失败”，不能证明正确的 full-hierarchy LRH 必然失败。
**动作**：修复为原子写回完整层级参数（或在候选对象中同时携带全部 code），先做 layer-1，再做一次全层验证；在修复结果前，该方向标记为“未验证”，不能放入停止清单。

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
**动作**：用与部署一致的 block-Gram objective 做逐候选 gate，并加入跨 fold/跨模型确认；若修复后仍回退，再把结论升级为算法负结果。

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

### 3.5 P2：v087–v091 源码缺失，负结果不可完全复现

`solutions/20260830_v087...` 至 `v091...` 目录目前只有 `result.md`，没有被评估的 `solution.py`，结果文件也没有 source SHA；早期的 v028 scale-code probe 也只有诊断结果、没有提交源。v087 的 E1 源可以从 Git 提交 `7e1f818` 的父版本恢复；A2–A5 的精确源快照无法从普通 Git 历史唯一恢复。

**影响**：这些版本的分数和时间可以作为日志事实，但无法排除临时 patch、环境差异或未归档参数造成的影响。
**动作**：将它们标记为 `non-reproducible`；不覆盖已有结果，不伪造源码；未来任何候选必须把完整源、规范化 SHA、参数 JSON 和结果一起归档。

### 3.6 P2：历史目录名与修订官方结果不一致

- v031 目录名仍为 `score14613_time159.2s`，但修订官方口径为 `21864 / 161.3s`。
- v066 目录名为 `scoreNA_timeNA`，但修订官方口径为 `22557 / 217.2s`。

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

最近候选的静态/运行时 Linear 合规扫描均为 `violations=0, static=0`；本次没有发现把 `A@W` 输出监督写入在线 `Q(A)` 的新违规。合规通过不等于精度通过，二者分开记录。

## 5. 已实现、已验证、未验证的方向矩阵

| 算法族 | 已实现并保留 | 已实现但回退 | 当前仍未验证/需要修复 |
|---|---|---|---|
| Linear 基础 | BOAT、cross-fold Weight-HSDQ、Gram-hierarchy Activation-HSDQ、512-row weight sampling、历史稳定 A@W/JDRQ 组件 | blockwise BOAT-2、全宽/逐块 A@W、大步长 headroom、full-H 等 | 修复后的 full-hierarchy LRH、Gram-objective gate 的 Global-LRH、最终权重 Gram + 显式 role 的 GALS、E1 三折/beam 版本 |
| Attention | GQA head-local rotation、MHA K-center、B1 GQRB margin、B2 PAWV diag-only | causal CVaR、全模型 K-center、PAWV rank-8 当前实现 | 最终 Q/K 变换后的 PAWV rank/position bucket、交替 Q/K/V、真正 role-aware 的结构门控 |
| 变换/CAT | 固定低自由度 CAT/BOAT 子集、共享 Hadamard | R64、CAT β 网格、full-H selector、BOAT-2 | 低自由度新坐标系或外部实现差异对照，尚无可部署候选 |
| 诊断/工程 | E0-G/D0 scale oracle、C0 五模型确认、clean 单路径重写 | 全局 scale 扩张部署 | 三折合成宽度矩阵、元策略路由、计算预算重分配 |

## 6. 归档与计划治理结论

旧计划中确实存在“写成已执行、实际只跑了子集”的内容，主要包括：

- E1 计划承诺的合成宽度覆盖、三折验证和完整 progressive beam 尚未完成；当前 v087 只完成了一个高成本实现。
- grid/consolidated/accuracy-first 计划中的 CAT/Householder、frozen-Q(A)、global-LRH 曾被列为下一步，但对应实验已经有结果，且 v092/v095 还带实现缺陷，不能继续按旧文字推进。
- 计划中把 v 的 GALS 正向 oracle 当作部署机会，但 v102/v103 已证明当前 sparse/shape-proxy 组合不稳定；真正 final-weight-Gram + role-id 版本仍未验证。
- 官方提交/兑换率校准从未完成，不能用本地 panel 推断官方 36000 距离。

治理动作：保留一份新的唯一活跃计划 [`2026-08-30-hif4-active-optimization-plan.md`](superpowers/plans/2026-08-30-hif4-active-optimization-plan.md)，其余实施计划和流程文档全部移到 [`docs/superpowers/archive/plans/`](superpowers/archive/plans/)。`plans/README.md` 与 `archive/plans/README.md` 只描述导航和历史性质，不再各自提出下一步。

## 7. 审计后的优先级

1. 官方接口恢复后，先提交当前根 v100，获得第一个真实兑换率锚点。
2. 若继续本地优化，第一实验是**修复 v092 的完整层级写回**，先 layer-1 再全层；不是直接再发明一个 LRH 变体。
3. 第二实验是 v095 的 **Gram-objective gate** 修复；若仍失败，才把 Global-LRH 归入已证伪路线。
4. 第三实验才是最终 Q/K 变换后重建 PAWV rank/position metric；Linear 方向并行做 final-weight-Gram + 显式 role 的 GALS 小预算复筛。
5. 每个实验必须保存完整源和 SHA；只要没有完整源，就标为不可复现，不把结果当作硬上限。

## 8. 本次可复核检查

- 当前根静态/运行时 Linear 合规扫描：`violations=0`、`static_violations=0`，8 个 review items 均是允许的离线 `A@W` 人工复核提示。
- 当前根代表性测试：`tests/test_gals_hierarchy.py`、`tests/test_reference_hif4.py`、`tests/test_linear_compliance_guard.py` 共 **23 passed**。
- 全量 pytest 本次为 `51 passed, 38 failed, 1 skipped, 5 errors`。失败主要来自仍针对已删除 C39/FULL64/JDRQ 内部接口的历史测试，以及 Windows 临时目录权限错误；本次只改文档、归档和导航，没有把这些历史测试伪装成当前根通过。
- 计划目录检查：`docs/superpowers/plans/` 仅有一份非 README 计划；归档计划的相对链接检查通过。
