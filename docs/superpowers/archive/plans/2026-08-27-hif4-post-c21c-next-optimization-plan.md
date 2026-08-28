# HiF4：C21-C 之后的下一阶段优化实施计划

> 日期：2026-08-27  
> 当前基线：C21-C / v025  
> 官方成绩：14437  
> 官方评测耗时：166.6 秒  
> 最终目标：在完全遵守赛事规则、官方评测耗时严格小于 300 秒的前提下，持续提高 Linear 与 Attention 精度，并判断 26000 分是否在当前框架内可达。

## 1. 文档定位

本文档是 C21-C 合规基线完成后的独立优化计划，不覆盖、不修改已经归档并关闭的 `2026-08-27-hif4-26000-algorithm-implementation-plan.md`。

本文档要解决四个问题：

1. 当前实现还剩下哪些真实误差来源；
2. 下一种值得验证的算法是什么，以及为什么它没有被既有失败实验否定；
3. 其他 AI 应如何逐步实现、验证、归档和决定是否提交；
4. 如果主方案失败，后续应按什么顺序切换方向，而不是继续无效搜索。

本阶段研发时间充裕，因此不为离线分析、可行性探针和开发实验设置人为的短时限。可以使用更完整的层、组件、折叠和参数扫描来获得可信结论。唯一不可放宽的时间要求是：**最终提交版本的官方评测总时间必须严格小于 300 秒**。

## 2. 第一优先级：赛事合规红线

所有设计、实现、测试和实验必须先满足本节，再讨论精度和速度。

### 2.1 绝对禁止项

在任何校准、搜索、打分、排序、早停、消融或测试代码中，都不得：

- 以任何形式计算 `A@W`；
- 通过转置、分块、逐列、逐行、批处理、低秩展开或等价收缩间接计算上述结果；
- 使用全精度输出、输出误差、教师输出或评测输出作为拟合目标；
- 利用输出误差反推、筛选或调整 `Q(A)`；
- 构造与输出拟合目标代数等价的交叉项或迹形式；
- 使用隐藏评测信号、排行榜反馈或保留集结果进行参数调优；
- 让权重信息参与激活量化器 `Q(A)` 的生成或候选打分。

这条规则也适用于调试脚本、一次性探针、单元测试和“只为验证公式”的代码。不存在研发阶段例外。

### 2.2 合法的信息流

激活与权重必须保持两条独立的信息流：

- 激活侧：只允许使用激活样本自身，优化激活硬重构误差、饱和率、尾部误差和分组统计；
- 权重侧：只允许使用权重自身，以及由激活单独计算出的合法二阶统计 `H_A = A^T A / N + damping * I`；
- 坐标变换侧：可以使用激活统计和权重统计分别评价候选，但最终接受条件必须显式保证两侧都不恶化；
- Attention：继续使用独立的合成输入验证动态路径，不得引入 Linear 输出监督。

### 2.3 合规实现原则

1. `Q(A)` 的搜索函数只接收激活张量、量化配置和纯激活统计。
2. `Q(W)` 的搜索函数只接收权重张量、合法 Hessian、量化配置和纯权重统计。
3. 坐标候选返回两个独立指标：`activation_ratio` 与 `weight_ratio`，不得合并成输出误差。
4. 所有候选必须先通过静态禁用模式、运行时钩子和人工代码审查。
5. 新代码中一旦出现可疑的跨操作数矩阵收缩，立即停止实验并删除该实现路径。

## 3. 当前工程与实验结论

### 3.1 当前 Champion

当前可提交基线为 C21-C / v025：

- 官方分数：14437；
- 官方耗时：166.6 秒；
- 本地 Linear 均值：0.5311；
- 本地 Attention 均值：0.4497；
- 合规检查：通过；
- 当前 `solution.py` 虽包含后续实验的禁用代码，但有效行为与 C21-C 等价。

官方分数与本地 Linear 指标之间只能做工程近似。当前有两种独立估计：

- v024→v025 单变量对照（Attention 逐位相同）实测斜率约 **259 分/pp**；
- v001/v002/v013/v024 四锚点联合拟合约 **300 分/pp**（v025 代入旧拟合的残差约 190 分）。

两种方法相差约 ±15%。所有换算必须使用区间 **259～300 分/pp**，阈值类战略判断一律以保守端（259 分/pp，即所需 Linear 更高的一端）为准：

- 22000 分需要 Linear 约 0.78（斜率 300）～0.8226（斜率 259）；
- 25000 分需要 Linear 约 0.86～0.9382；
- 26000 分需要 Linear 约 0.90～0.9768。

注意 22000 分所需 Linear 的区间下端 0.78 恰好落在 C24 绝对门附近：**战略判断对斜率高度敏感**，在获得新的官方锚点收窄区间之前，第 10.2 节的阶段决策不得只引用单一斜率。该外推并非赛事保证，但足以说明：从 14437 到 26000 需要约 37～45 个 Linear 百分点，不可能仅靠若干 1～2 个百分点的局部补丁实现。

### 3.2 已确认的误差结构

操作数消融结果约为：

- 仅量化激活：0.7655；
- 仅量化权重：0.7647。

说明剩余误差并非单侧主导，激活和权重贡献近似对半。任何只改善一侧、明显伤害另一侧的方法都难以成为稳定 Champion。

### 3.3 已归档实验的约束

#### C22：随机全宽 R64 变换失败

- 72 个组件全部回退；
- 分散种子仍普遍恶化激活或权重的局部指标；
- 校准成本高，CPU 比例约 1.52。

结论：随机稠密全宽混合不是下一步方向。该实验只否定随机全宽混合，不否定具有层级结构、可解释且可折叠的等价变换。

#### C23：full-64 权重 GPTQ 有精度信号但实现成本过高

- Linear 提高约 1.93 个百分点；
- full-H 局部损失下降约 20.95%；
- 固定六项 6/6 为正；
- CPU 比例约 1.55，完整方案投入产出比不足。

分组件提升约为：q +2.12、k +3.61、v +0.92、o +2.11、fc +1.87、proj +0.95 个百分点。

结论：更强的权重量化搜索确实有空间，但 full-64 全覆盖不是最终实现。它应作为后备方向拆分，寻找贡献最大的阶段和覆盖率。

#### C28：固定坐标中的激活 scale-code 搜索接近穷尽

- 合法固定坐标精确码本 oracle 的总能量比例约 0.919；
- 当前坐标内只剩约 8.1% 的激活能量改进空间。

结论：继续在相同坐标系中细化激活 scale code，理论收益有限。下一步应先改变坐标分布，再使用现有硬量化器。

### 3.4 当前最重要的判断

当前框架的主要瓶颈不是“还没把同一个量化器搜得足够细”，而是现有层级量化器看到的坐标分布仍不理想。下一步应寻找：

- 不改变模型数学语义；
- 不增加动态推理算子；
- 能同时降低激活和权重局部量化难度；
- 可以折叠进现有状态；
- 完全不依赖输出拟合；
- 最终仍能稳定控制在 300 秒内的结构化坐标变换。

## 4. 主方案 C29：Hierarchy-Aligned Equivalent Scaling

中文名称：**层级对齐等价缩放**。下文简称 HAES。

### 4.1 核心思想

当前 Linear 路径已经包含对角平滑、排列和层级正交旋转。C28 说明固定坐标中的码本已经接近局部上限，C22 又说明随机全宽旋转会破坏层级量化结构。

HAES 不再增加随机混合，而是在排列后、层级旋转前加入一个与 4/8/16/64 量化层级对齐的结构化对角缩放 `S`：

```text
activation coordinates: A_s = A · D^-1 · P · S^-1 · R
weight coordinates:     W_s = W · D    · P · S    · R
```

其中：

- `D` 是当前对角平滑；
- `P` 是当前通道排列；
- `R` 是当前层级正交旋转；
- `S` 是待搜索的层级对齐正对角矩阵。

`S` 在两侧使用互逆因子，且 `R` 为正交结构，因此它只重新分配坐标尺度，不改变原始 Linear 算子的精确语义。实际实现时不增加新的动态操作，而是把 `S` 逆散射并折叠回 `D`。

### 4.2 为什么该方案没有被 C22/C28 否定

- C22 搜索的是随机、稠密、全宽的 R64 混合；HAES 是对角、稀疏参数化、层级对齐的尺度重分配。
- C28 在固定坐标系中优化激活码本；HAES 先改变坐标分布，使原有码本看到新的输入。
- C23 的结果说明权重侧仍有可观空间；HAES 同时评价激活与权重，而不是只追求激活误差。
- HAES 可折叠进现有 `smooth_inv` 或等价状态，不增加推理阶段矩阵操作。

### 4.3 参数化

**S 粒度与组件实际 block size 绑定（强制）**：现有实现中各组件的
`block_smooth_size` 由按组件搜索在 `(4, 8, 16)`（含关闭 0）中选择，
而 S 插在 P 之后、R 之前，量化层级（lv3 per-4）却定义在 R 之后的
坐标系：

- `R=0` 或 `H4`：S 与 R 精确交换，per-4 对齐完好；
- `R=H8`：每个 H8 块混合两个相邻 4 组，S 的有效分辨率退化到约 per-8；
- `R=H16`：退化到约 per-16，且 Hadamard 的能量平均效应进一步钝化组间尺度差异。

因此 S 的组大小必须与该组件实际生效的 `block_smooth_size` 一致
（组件关闭 block smooth 即 size=0 时，S 组大小取 4）。S 组内为常量时
与任意块对角 R 精确交换，折叠进 D 的性质不变。禁止全组件统一 S 粒度。
探针必须按「S 粒度 × 组件 block size」分桶报告；任何把不同 block
size 组件混入同一 S 粒度的聚合结论无效。

以每个 64 通道块为基本单元，在每个 4 通道子组上共享一个缩放参数。每个 64 块共有 16 个离散参数：

```text
s_g = 2 ^ (z_g / 8)
z_g ∈ {-4, -3, ..., 3, 4}
```

因此单组缩放范围约为 `[2^-0.5, 2^0.5]`。初始值全部为 0，即 `S = I`。

每次更新一个 64 块后，对该块 16 个 `z_g` 做中心化：

```text
z_g <- clip(round(z_g - mean(z)), -4, 4)
```

中心化用于去掉与全局平滑重复的自由度，并防止整个块无意义地整体放大或缩小。

第一轮只实现 4 通道共享参数（受绑定规则约束：即仅在组件 block size
≥4 时可用，实际等价于允许的最细粒度）。只有第一轮产生稳定正收益后，
才允许追加以下消融：

- 更粗 S 粒度（8/16 通道共享，仅对 block size ≥ 该粒度的组件合法）；
- 4 通道参数 + 每 16 通道一个低频修正（低频修正项不受绑定约束，
  但必须单独消融）；
- 步长分母 4、8、16；
- 最大码范围 3、4、6。

不得在主机制尚未成立前一次性引入多层参数，以免无法归因。

### 4.4 折叠到现有 `D`

`S` 定义在排列后的坐标顺序中。现有代码使用：

```python
transformed = (dense * scale.unsqueeze(0)).index_select(-1, permutation)
```

因此折叠时应执行：

```python
s_original = torch.ones_like(d_parent)
s_original[permutation] = s_permuted
d_new = d_parent * s_original
```

最终状态只保存或使用 `d_new`，不新增动态 `S` 算子，不修改 Attention 动态路径，不扩展提交接口。

实现前必须用纯坐标 round-trip 和对角映射单元测试确认散射方向。测试不得通过构造 Linear 输出进行确认。

### 4.5 合法目标函数

对于每个候选缩放，分别计算：

```text
L_A = mean((A_s - dequantize(quantize(A_s)))^2)
```

权重侧先由变换后的激活单独计算：

```text
H_A = A_s^T · A_s / N + damping · I
```

再使用当前合法的权重硬重构或 Hessian 加权局部损失得到 `L_W`。`Q(A)` 的搜索过程绝不能读取权重或 `L_W`。

相对父版本定义：

```text
ratio_A = L_A(candidate) / L_A(parent)
ratio_W = L_W(candidate) / L_W(parent)
```

激活尾部定义为高分位或最大饱和相关误差的稳定统计，并计算：

```text
tail_ratio = tail_A(candidate) / tail_A(parent)
```

候选排序分数：

```text
score = max(ratio_A, ratio_W) + 0.1 * max(0, tail_ratio - 1)
```

候选必须先满足 Pareto 安全条件才允许参与排序：

```text
ratio_A <= 1 + eps
ratio_W <= 1 + eps
tail_ratio <= 1.01
```

建议 `eps = 1e-4`，只用于浮点噪声容忍，不允许以平均改善掩盖单侧明显回退。

### 4.6 搜索范围

研发时间充裕，因此分级执行，而不是只检查少数有利样本。

#### 第 0 级：S 网格 oracle 微探针（先于任何搜索代码的实现）

在实现坐标下降搜索之前，先用零成本 oracle 回答机制上限问题（沿用
C28 的否决模式，避免 C23"机制成立但成本失控"教训的对称风险——
"成本可控但机制空转"）：

- 取代表性子集（建议 3 层 × 全 6 组件），在 pre-R 放置下直接枚举
  S 的离散网格（z ∈ 9 档，粒度按 §4.3 绑定规则与组件 block size
  一致），每组独立取激活硬重构误差最小的 z——这是单侧 oracle
  上界，无需 Pareto 联动与中心化；
- 码字用现有量化器重适配；
- 报告该 oracle 相对父版本的激活硬重构误差降幅，及其占 per-4
  自由 scale 余量（26%～38% relRMSE）的份额，并按
  「S 粒度 × block size」分桶（H16 组件预期显著低于 H4 组件，
  分桶即对齐质量证据）。

**否决门**：若 oracle 激活能量降幅 < 5%（相对父版本），C29 机制
上限不足，直接判定主机制失败并转 C30，不实现坐标下降搜索。该探针
只需半天量级成本。

#### 第一级：全工程可行性探针

覆盖：

- 全部 12 层；
- q、k、v、o、fc、proj 六类组件；
- 至少两个独立校准折叠；
- 多个步长配置；
- 初始搜索使用 64 行激活和 128 行权重，随后用完整校准重算。

输出每个层/组件的：

- 父版本 `L_A`、`L_W`、tail；
- 候选 `L_A`、`L_W`、tail；
- 两侧 ratio；
- 被选择的 64 块数；
- 非零 `z` 数量和分布；
- 两折的一致性；
- 搜索耗时和最终重算耗时。

#### 第二级：完整候选

只有探针证明机制成立后，才修改 `solution.py` 并跑完整 Linear、Attention、固定六项、合规和计时矩阵。

### 4.7 难块选择

对每个 64 通道块，计算父版本归一化难度：

```text
difficulty_b = max(
    L_A_block / median(L_A_blocks),
    L_W_block / median(L_W_blocks)
)
```

默认选择最难的 25% 块进行搜索，其余块保持 `z = 0`。同时在探针中扫描覆盖率：

```text
coverage ∈ {0.10, 0.25, 0.50, 1.00}
```

目的不是强行降低计算，而是判断收益是否集中。如果全覆盖更稳定，可在最终方案中使用全覆盖；如果收益只来自少量难块，则采用稀疏覆盖并固定规则。

### 4.8 坐标搜索

每个被选中的 64 块执行离散坐标搜索：

1. 从全零 `z` 开始；
2. 对 16 个 4 通道组依次尝试 `delta ∈ {-1, 0, +1}`；
3. 对候选做范围裁剪和块内中心化；
4. 只保留 Pareto 安全候选；
5. 选择 `score` 最低的候选；
6. 最多执行两轮 sweep；
7. 一轮没有任何严格改善时立即停止该块；
8. 最后用完整校准样本重算，而不是直接采用搜索子样本分数。

实现应批量化：

- 行维度批处理；
- 候选 `delta` 批处理；
- 64 块批处理；
- 允许在 sweep 和 16 个组上保留小循环；
- 禁止为每个元素启动 Python 循环。

### 4.9 双折交叉验证

候选不能只在生成它的样本上验收。

执行顺序：

1. fold-0 搜索，fold-1 验证；
2. fold-1 搜索，fold-0 验证；
3. 两个方向都必须满足激活、权重非退化；
4. 两个方向的综合改善符号必须一致；
5. 通过后，使用全部校准数据重建一次最终 `z`；
6. 最终 `z` 只能由开发校准集确定，不得读取 holdout 或官方反馈。

### 4.10 可行性探针通过门槛

HAES 进入正式实现必须同时满足：

- 全矩阵汇总的 `max(ratio_A, ratio_W) <= 0.92`，即最差一侧至少降低约 8%；
- 至少 75% 的层/组件点同时满足两侧不恶化；
- q、k、v、o、fc、proj 六类组件中至少五类的中位数改善为正；
- 双折改善方向一致；
- 激活 tail 总体回退不超过 1%；
- 非有限值为 0；
- 所有对角尺度在预注册边界内；
- 合规静态扫描无违规。

如果总体改善在 5%～8% 之间，但跨层、跨组件和双折高度稳定，可以保留为“弱正机制”，继续做一次参数化消融；低于 5% 或依赖极少数点时直接判定主机制失败。

## 5. C29 实现任务清单

### 5.1 阶段 A：建立独立探针

新增：

- `evaluator/hierarchy_scale_probe.py`
- `tests/test_hierarchy_scale_probe.py`

探针必须复用正式量化路径，不得复制出一个与提交逻辑不同的近似量化器。建议接口：

```python
def hierarchy_scale_from_codes(z_codes, group_size=4, denominator=8):
    ...

def scatter_permuted_scale(s_permuted, permutation):
    ...

def apply_hierarchy_aligned_d(d_parent, s_permuted, permutation):
    ...

def hierarchy_scale_operand_metrics(
    activation,
    weight,
    d_parent,
    permutation,
    config,
):
    ...

def search_hierarchy_aligned_scale(
    activation_search,
    activation_validation,
    weight_search,
    d_parent,
    permutation,
    config,
):
    ...
```

接口设计要求：

- 激活量化函数不接收 `weight`；
- 权重指标函数可以接收激活产生的 Hessian，但不得返回或使用输出拟合量；
- 搜索结果显式包含两侧独立指标；
- 随机抽样使用固定种子；
- 输出 JSON/Markdown 双份结构化结果；
- 所有配置写入结果，确保可复现。

### 5.2 阶段 B：探针单元测试

至少实现以下测试：

1. `z = 0` 时与父版本逐元素等价；
2. `s_original[permutation] = s_permuted` 的映射方向正确；
3. 折叠后的 `D` 与显式坐标缩放的中间坐标一致；
4. 互逆对角缩放 round-trip 恢复原坐标；
5. 中心化后码值合法且确定；
6. 候选单侧恶化时被 Pareto gate 拒绝；
7. tail 超过阈值时被拒绝；
8. 固定种子重复运行结果一致；
9. 禁用功能时结果与 C21-C 完全一致；
10. 非有限输入或非法尺度明确报错；
11. 测试源码不构造被赛事禁止的跨操作数结果；
12. Attention 相关状态与路径不发生变化。

### 5.3 阶段 C：运行完整探针矩阵

运行顺序：

0. 先运行 §4.6 第 0 级 S 网格 oracle 微探针；未过 5% 否决门则终止 C29，不进入后续步骤；
1. 单层单组件 smoke test，确认数值和内存；
2. 全 12 层 × 6 组件 × 2 folds；
3. 扫描 `coverage = 0.10/0.25/0.50/1.00`；
4. 扫描步长分母 `4/8/16`；
5. 对排名靠前的配置用完整校准行重算；
6. 输出层、组件、配置三种聚合表；
7. 根据预注册门槛作出 pass/fail，不凭主观印象挑配置。

研发耗时可以较长；允许离线探针超过 300 秒。探针不是提交代码。应优先获得覆盖完整、可复现的机制结论。

### 5.4 阶段 D：接入 `solution.py`

只有阶段 C 通过后执行。

建议新增内部函数：

```python
_hierarchy_scale_from_codes
_scatter_permuted_scale
_apply_hierarchy_aligned_d
_hierarchy_scale_operand_metrics
_search_hierarchy_aligned_scale
```

新增功能开关：

```python
_HIERARCHY_ALIGNED_SCALE = False
```

接入期间保持默认关闭，完成全套验证并准备候选归档时才打开。状态格式原则上不变，最终只让已有 `smooth_inv` 或对应对角状态反映 `d_new`。

接入检查：

- 禁用时 `solution.py` 哈希行为与 C21-C 一致；
- 启用后没有新动态矩阵运算；
- 没有新增超出接口的持久状态；
- Attention 路径不读取 HAES 状态；
- 所有尺度裁剪、dtype 和 device 与父实现一致；
- CPU 与 CUDA 路径结果在容差内一致；
- 所有搜索只发生在允许的预处理/校准阶段。

### 5.5 阶段 E：正式开发集验证

完整验证矩阵：

- Linear：全部层、全部组件，offset 0，`amax=6`；
- 固定六项：offset 0/97/193/389，`amax=6`；offset 0，`amax=4`；pow2，offset 0；
- Attention：MHA 与 GQA、causal 与 non-causal；
- 576 个合成 Attention 样例全部保持父版本结果；
- 静态合规扫描；
- 运行时合规钩子；
- 确定性复跑；
- CPU 串行最终计时；
- 必要时补充 CUDA 一致性测试，但官方门槛以官方环境和可比 CPU 计时为准。

不得仅凭平均值提升接受候选。必须同时报告：

- 平均值；
- 六组件分别变化；
- 各层最差回退；
- 固定六项 6/6 结果；
- 两折结果；
- tail 和饱和率；
- 校准耗时、动态耗时、总耗时；
- 合规结果。

### 5.6 阶段 F：正式候选验收

分为三个层级，避免把“机制有效”和“值得消耗官方机会”混为一谈。

#### 机制有效

- 激活与权重局部损失均稳定下降；
- 双折一致；
- 没有明显层/组件集中退化；
- 全部合规检查通过。

#### 可进入本地 Champion 比较

- Linear 均值至少提高 1.5 个百分点；
- 固定六项至少 5/6 为正，且第六项不得显著为负；
- 任一组件平均回退不超过 0.2 个百分点；
- Attention 576 样例与父版本一致；
- 确定性通过；
- 预计最终官方总耗时低于 300 秒。

#### 可消耗 holdout / 官方提交机会

- 参数、覆盖率、步长和随机种子已经冻结；
- 不再根据 holdout 修改实现；
- 本地完整矩阵通过；
- 合规复核通过；
- 时间门槛以显式公式表达：`推算官方耗时 = 166.6s × (候选本地 CPU 串行 algorithm-stage 秒数 / C21-C 本地 CPU 串行 algorithm-stage 秒数)`，C21-C 本地串行 stage 约 61.3s，故推算官方耗时不得超过 285 秒（等价：本地 stage ratio ≤ 约 1.71，本地 stage ≤ 约 105 秒），为官方 300 秒硬限制留出环境波动余量；
- 候选归档必须同时报告三项计时：本地串行 stage 秒数、相对 C21-C 的 ratio、推算官方秒数；
- 如果推算官方耗时在 285～300 秒之间，必须先做纯性能优化并重新验证数值等价；
- holdout 只运行一次，并按预注册规则决定提交或归档。

说明一：285 秒指**推算官方耗时**（按上述公式换算），不是本地计时本身——本地到官方的放大系数约 2.7 倍（61.3s → 166.6s），本地 105 秒 ≈ 官方推算 285 秒。

说明二：该口径相对上一计划对晋级候选的 ratio ≤ 1.15 纪律是**有意的放宽**（166.6s 基线下 270s 硬上限对应 ratio ≈ 1.71，基于 C21-C 官方实测时间得出）。两套口径不得混用：上一计划的归档沿用旧口径，本计划候选一律使用本节公式。285 秒是最终提交的工程安全缓冲，不是研发实验的硬限制；不能因为探针或校准搜索耗时较长就过早否定有希望的机制，最终版本应通过折叠、缓存、向量化或离线固定参数把官方路径压到 300 秒以内。

## 6. 性能优化要求

HAES 理论上可以完全折叠进 `D`，因此动态推理时间应与 C21-C 近似。额外成本主要来自校准搜索。

性能工作按以下顺序执行：

1. 先验证算法正确和精度信号；
2. 用 profiler 区分校准搜索、权重量化、Attention 和 Python 调度成本；
3. 批量化候选、块和样本；
4. 缓存父版本量化结果、块统计和 Hessian；
5. 对完全相同 shape/config 的常量结构复用索引与 Hadamard 元数据；
6. 避免在内层循环重复 device/dtype 转换；
7. 最终只保留胜出配置所需路径，探针扫描代码不进入提交热路径；
8. 在精度完全一致的前提下，才允许做近似或剪枝；
9. 使用官方可比的串行命令进行至少三次计时，报告中位数和最大值，并按 §5.6 公式换算推算官方耗时（本地 stage 秒数、ratio、推算官方秒数三项齐全）；
10. 最终任何一次可信测量的推算官方耗时达到或超过 300 秒，都不得提交。

不得通过降低校准覆盖、减少必要验证或改变算法语义来伪造低耗时。性能优化必须附带数值等价测试。

## 7. Attention 专项优化轨

Attention 不能只作为 Linear 候选的回归保护项。本节建立一条与 C29 Linear 主线相互独立的 Attention 优化轨，用于继续提高 MHA/GQA 精度、修复现有尾部债务，并验证官方隐藏 Attention 场景是否仍存在高价值弱项。

Attention 专项候选必须从 C21-C 的当前 Attention 行为出发，先独立归档，再与已晋级的 Linear 候选组合。禁止在同一个首次实验中同时改变 Linear 和 Attention，以免无法归因。

### 7.1 为什么此前优先 Linear

当前 Attention 并非未经优化。A1 相对 B0 已获得稳定的大幅本地提升：

- MHA 六组固定配置平均 causal `+5.71pp`、non-causal `+7.74pp`；
- GQA 两组固定配置平均 causal `+8.44pp`、non-causal `+10.28pp`；
- 当前 MHA offset-0 causal 为 `0.4497`；
- 当前 C21-C 保留 A1 Attention 路径，后续 C11～C28 没有改变其有效行为。

已有深化实验也提供了明确约束：

- C2 独立 Segment-CVaR 因重置 causal 历史导致 MHA causal `-3.42pp`，已拒绝；
- C2a 保留完整上下文，但以 CVaR 作为排序目标仍使 MHA causal/non-causal 约 `-0.53/-0.52pp`，已拒绝；
- A2 H64 聚合均值有正信号，但出现 MHA 单层约 `-1.97pp` 和 GQA non-causal 尾部退化，已关闭；
- A3 更换 head 级 V importance 仅带来约 `-0.06pp`，且有单层退化；
- L1 数据驱动 scale 降低逐块损失，却使 MHA causal/non-causal 约 `-0.60/-0.85pp`，证明元素重构 proxy 与 softmax 目标可能错位；
- E1 合成矩阵中 heavy-tail 场景仍为负，是已登记的 Attention 尾部债务。

Linear 被作为 26000 主线的原因是官方兑换证据，而不是认为 Attention 已经达到上限：

- C21→C21-C 的 Attention 逐位不变单变量对照显示，Linear 每提高 `1pp` 约对应官方 `259` 分；
- 历史多锚点拟合中，MHA Attention 每提高 `1pp` 只对应不超过约 `8.6` 分，该估计置信度较低，但量级明显弱于 Linear；
- 即使把本地 MHA Attention 从 `0.4497` 提高到 1，按现有映射也不足以填补 14437→26000 的缺口。

因此资源优先级是：Linear 负责主要分数突破，Attention 负责额外增益、隐藏场景保险和尾部鲁棒性。研发时间充裕，所以 Attention 仍应完成以下独立机制验证。

### 7.2 Attention 合规边界

赛事规则零仍然完全适用：Linear 校准不得以任何形式计算 `A@W`，也不得利用其输出拟合或反推 `Q(A)`。

当前工程对规则的保守分流如下：

1. Linear 路径绝不能调用任何 Attention scorer；
2. Attention 专用的 `QK^T`、mask、softmax probability 属于 Attention 算子自身的必要统计；
3. 新 Attention 搜索优先使用 logit/probability 误差，不使用 Linear Weight，不产生 Linear 输出；
4. `PV` 形式的最终 Attention 输出只用于独立评测与终验，不作为 A4 主搜索目标；
5. 如果官方进一步书面明确禁止使用 Attention 输出拟合 Attention 量化器，则必须关闭现有 A1 output selector，先建立 probability-only 合规基线，再执行本节；
6. 在规则未明确扩大前，本节仍不得把 Attention 的允许操作外推到 Linear。

必须新增静态调用图检查，证明：

- `hif4_calibration_linear` 及其可达函数不能到达 `_attention_forward`；
- Linear state 不包含任何 Attention probability/output 派生字段；
- Attention state 不读取 Linear Weight 或 Linear evaluator 结果；
- 所有 Attention 候选只影响 q/k/v state 和对应动态 API。

### 7.3 A4：Headwise Pareto QK Search

中文名称：**逐 Head Pareto QK 搜索**。这是 Attention 专项的第一候选，也是优先级最高的 Attention 机制。

#### 7.3.1 问题定义

当前实现虽然生成 per-head 的缩放向量和排列，但 center mode、缩放候选类型、alpha 和 permutation basis 的接受决策仍以整层候选为单位。容易出现以下问题：

- 大多数容易 head 主导整层平均值；
- 少数困难 head 在 GQA/non-causal 或特定 offset 上退化；
- 一个 KV head 在 GQA 中服务多个 Q heads，其最优尺度可能与 MHA 不同；
- H64 整层选择的正均值不能阻止个别 head 的分布漂移。

Attention 各 head 的 probability 在 softmax 前后彼此独立，GQA 中也可以按“一个 KV head + 它对应的一组 Q heads”作为独立搜索单元。因此 A4 改为每个 KV head group 单独选择候选，再组合为整层 state。

#### 7.3.2 严格等价候选

对于 KV head `h` 及其对应 Q head group，搜索以下候选。

K 的 token-invariant centering：

```text
center_h ∈ {
    none,
    mean,
    midrange,
    median,
    trimmed_midrange(q05, q95)
}
```

每种 center 都必须对当前 K Tensor 的 token 维计算一个 `[head_dim]` 常量并从所有 token 中减去。对同一个 query，该操作只给所有 key logits 加上相同常量，因此量化前 softmax probability 严格不变。

Q/K 成对缩放：

```text
d_h = stat(K_h)^alpha / stat(Q_group_h)^(1-alpha)

stat  ∈ {peak, RMS}
alpha ∈ {0.125, 0.250, 0.375, 0.500, 0.625, 0.750, 0.875}
```

同时保留 identity。Q 侧乘 `d_h`，K 侧乘 `d_h^-1`，并继续使用当前 `[1/16, 16]` 安全范围。

共享排列 basis：

```text
permutation_h ∈ {
    identity,
    hierarchy_balanced(Q, K),
    Q_range_only,
    K_range_only
}
```

同一 KV group 内的所有 Q heads 与 K head 使用相同的 head 内排列，量化前点积严格不变。

#### 7.3.3 Attention 内在目标

对每个 head/group 单独计算：

```text
L_ref = Q_ref K_ref^T / sqrt(head_dim)
L_hat = Q_hat K_hat^T / sqrt(head_dim)
P_ref = softmax(mask(L_ref))
P_hat = softmax(mask(L_hat))
```

由于 softmax 对每行常量平移不敏感，先对有效 key 区间的 logit 误差做行中心化：

```text
delta = L_hat - L_ref
delta_centered = delta - mean_valid_keys(delta)

loss_logit = mean(delta_centered^2)
loss_prob  = mean((P_hat - P_ref)^2)
```

同时计算有限、对极小 probability 稳定的 Jensen-Shannon divergence 作为诊断，不直接作为首轮硬目标。

主排序使用父版本归一化指标：

```text
r_prob  = loss_prob(candidate)  / loss_prob(parent)
r_logit = loss_logit(candidate) / loss_logit(parent)

score = r_prob + 0.10 * max(0, r_logit - 1)
```

不得用普通 Q/K 元素 MSE 取代该终验。元素 MSE只允许作为便宜的候选预筛指标。

#### 7.3.4 分桶鲁棒性

对每个完整 Attention 序列只计算一次 logits/probability，然后从结果中切分 query 行。不得像已失败的 C2 那样分段重算并重置 causal 历史。

固定分桶：

- query position 四分位；
- reference attention entropy 四分位；
- causal / non-causal；
- calibration sample；
- MHA/GQA 在外部完整回归中分别统计。

A4 不重复 C2a 的 `mean + CVaR` 加权排序，而采用：

- mean probability loss 为主目标；
- 最差 bucket、non-causal 和高/低 entropy 为硬约束；
- 尾部只决定是否接受，不通过扩大权重压倒均值。

#### 7.3.5 两阶段搜索

为避免一次构造过大的笛卡尔积：

第一阶段只搜索 center × scale，排列保持 parent：

1. 每个 KV head group 构造全部 center/scale 候选；
2. 用 fold-0 排序；
3. 保留 Pareto 前四名；
4. 用 fold-1 验证；
5. 反向执行 fold-1 搜索、fold-0 验证；
6. 两个方向都非退化的候选才进入第二阶段。

第二阶段对前四名追加四种 permutation basis：

1. 每个 head 最多 16 个组合；
2. 使用完整部署 Q/K 量化路径重算 probability 指标；
3. 两折均改善才接受；
4. 没有合法候选时该 head 保留完整 parent state；
5. 将各 head winner 组合成整层 state 后，再做一次整层重算，防止实现拼接错误。

#### 7.3.6 A4 Pareto gate

每个被替换的 head 必须同时满足：

```text
search fold:     r_prob < 0.99
validation fold: r_prob <= 1.00
causal mean:     <= parent
non-causal mean: <= parent * 1.002
worst bucket:    <= parent * 1.005
```

整层组合还必须满足：

- 两折 probability mean 均改善；
- 任一 position/entropy bucket 回退不超过 0.5%；
- 非有限值为 0；
- MHA 与 GQA 的候选选择率、回退率完整记录；
- parent head 保留必须逐元素复现 A1；
- 动态路径不增加新的稠密矩阵操作。

### 7.4 A4 实现任务

#### 7.4.1 独立探针

先新增：

- `evaluator/attention_headwise_probe.py`
- `tests/test_attention_headwise_probe.py`

建议函数：

```python
def _attention_center_candidates(k_head, modes):
    ...

def _attention_head_probability_metrics(
    q_group,
    k_head,
    q_state,
    k_state,
    causal,
    buckets,
):
    ...

def _search_headwise_qk_candidate(
    search_samples,
    validation_samples,
    parent_head_state,
    config,
):
    ...

def _compose_headwise_attention_state(head_results, parent_state):
    ...
```

探针覆盖当前可用真实 GPT-2 全 12 层，并分别运行：

- MHA `q_heads=12, kv_heads=12, head_dim=64`；
- GQA `q_heads=12, kv_heads=6, head_dim=64`；
- head_dim=128 合成 smoke；
- causal/non-causal；
- 至少两个 calibration folds。

研发时间允许完整扫描，不得只挑选历史上容易提升的层或 head。

#### 7.4.2 `solution.py` 接入

探针通过后再新增：

```python
_ATTN_HEADWISE_PARETO = False
_ATTN_HEADWISE_CENTER_MODES = (0, 1, 2, 3, 4)
_ATTN_HEADWISE_ALPHAS = (0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875)
```

建议内部函数：

```python
_center_attention_k_headwise
_attention_head_probability_metrics
_search_attention_headwise_pareto
_compose_attention_headwise_states
```

state 设计：

- `multiplier` 继续使用扁平 per-channel Tensor；
- `permutation` 继续使用扁平 headwise permutation；
- `center_mode` 扩展为 scalar 或 `[kv_heads]` 的小 Tensor，保持旧 state 可读取；
- 每个 head 的 identity/parent 选择不增加额外字段；
- 禁用开关时返回字段和数值必须与 C21-C 逐元素一致。

动态实现要求：

- 所有 head center 一次 reshape 后向量化计算；
- 不允许逐 token Python 循环；
- mean/midrange/median/trimmed-midrange 只沿 token 维归约；
- 同一 center Tensor 对该 head 的全部 token 使用；
- Q/K paired scale 和 permutation 保持严格一致；
- calibration 搜索可以较慢，最终动态路径只能增加必要的归约和逐元素操作。

#### 7.4.3 A4 单元测试

至少覆盖：

1. 每种 K center 在量化前只造成逐 query 的 logit 常量平移；
2. softmax probability 在容差内不变；
3. Q/K 互逆 scale 的点积不变量；
4. MHA 共享排列不变量；
5. GQA 中一个 KV head 与对应多个 Q heads 的共享排列不变量；
6. scalar center mode 向后兼容；
7. headwise center mode shape/state 合法；
8. 一个 head 回退不会改变其他 head；
9. 两折符号不一致时拒绝；
10. 最差 bucket 超门时拒绝；
11. 禁用时与 C21-C 的 576 case 逐 case 一致；
12. Linear 调用图不能到达新增 Attention scorer；
13. 固定输入重复运行完全确定；
14. CPU/CUDA 在容差内一致。

### 7.5 A5：Hierarchy-Aligned Small-Block QK Rotation

中文名称：**层级对齐小块 QK 旋转**。

A2 只证明固定 H64 的跨分布尾部不安全，不足以否定与 HiF4 层级对齐的 R4/R8/R16。A5 必须作为 A4 之后的独立候选，不能与 A4 首次实现捆绑。

#### 7.5.1 变换

对每个 KV head group 使用相同的块对角正交矩阵：

```text
Q'_group = Q_group R_h
K'_h     = K_h R_h

R_h ∈ {I, blockdiag(H4), blockdiag(H8), blockdiag(H16)}
```

每个 Hadamard block 使用确定性 sign pattern。因为 `R_h R_h^T = I`，量化前 Q/K 点积不变。

候选：

```text
block_size ∈ {4, 8, 16}
seed       ∈ {0, 1, 2, 3, 4, 5, 6, 7}
```

不得重新测试固定全层 H64，也不得通过增加 H64 seed 重复 A2。

#### 7.5.2 搜索

1. 父版本固定为已通过的 A4；如果 A4 失败，则父版本为 C21-C/A1；
2. 每个 KV head group 先用元素局部误差预筛 24 个小块候选；
3. 每个 block size 最多保留两个 seed；
4. 使用 A4 的完整 probability/bucket 双折指标终验；
5. 每个 head 可独立选择 identity/R4/R8/R16；
6. 组合后运行整层部署路径；
7. 任一验证 fold、mask 或尾部 bucket 不安全则局部回退 identity。

#### 7.5.3 动态实现

扩展现有 `_apply_attention_rotation`：

- state 保存 `[kv_heads]` block size 和每个 head 的 sign code，或保存等价紧凑 Tensor；
- GQA 中对应 Q heads复用 KV head 的 rotation；
- 按相同 block size 的 heads 分桶后批量 FWHT；
- 禁止为每个 token/head 启动 Python 循环；
- R4/R8/R16 使用 butterfly，不构造 dense `[head_dim, head_dim]` 矩阵；
- identity heads 不进入 FWHT；
- 计时必须分别报告 calibration、q dynamic、k dynamic 和总路径增量。

#### 7.5.4 A5 晋级门

- offset-0 Attention 综合均值相对父版本至少 `+0.5pp`，正式 Champion 比较建议至少 `+1.0pp`；
- MHA/GQA causal 均不得下降超过 `0.2pp`；
- non-causal 均不得下降超过 `0.2pp`；
- 任一层回退不超过 `0.5pp`；
- E1 576 case 全矩阵均值不下降；
- `saturated_logits` 与 `heavy_tail` 场景均值不下降；
- 任一合成 case 相对父版本下降不得超过 `2pp`；
- Linear 所有分项逐位不变；
- 最终官方总耗时严格小于 300 秒。

### 7.6 A6：Position-Aware V Refinement

中文名称：**位置感知 V 精修**。

当前 V importance 是每个 head 一个常量。当 `head_dim=64` 时，一个 HiF4 64 块恰好对应一个 head 的一个 token，head 内常量只会整体缩放该块损失，通常不能改变块内离散解。这解释了 A3 更换 `E[P]`、`E[P^2]` 后收益极小。

A6 不再调整 head 常量，而是利用 causal Attention 中不同 key position 被复用程度不同的事实，改变有限精修预算在 token/head blocks 之间的分配。

#### 7.6.1 合法位置统计

由 calibration Q/K 计算 probability `P`，对每个 KV head 和 key position：

```text
g_causal[h, s]    = sum_valid_queries_t P_causal[h, t, s]^2
g_noncausal[h, s] = sum_t P_noncausal[h, t, s]^2
```

按归一化位置 `s / seq_len` 压缩成 16 或 32 个 bins：

```text
g[h, bin] = max(
    normalize(g_causal[h, bin]),
    normalize(g_noncausal[h, bin])
)
```

该统计只来自 Attention probability，不读取 Linear Weight，不产生 Linear 输出。

#### 7.6.2 动态使用

state 新增小型 `position_importance[kv_heads, bins]`。动态量化 V 时：

1. 根据当前 `seq_len` 把每个 token 映射到归一化 position bin；
2. 对 head_dim=64，每个 token/head block 使用对应位置权重；
3. 对 head_dim=128，同一 head 的两个 64 块共享位置权重；
4. 位置权重只影响困难块排名和精修预算，不改变标准 HiF4 fallback；
5. 未进入精修集合的块必须与 parent 逐元素一致；
6. 任何无法映射的 shape/seq_len 回退均匀权重。

#### 7.6.3 实现前 oracle

新增 `evaluator/v_position_budget_probe.py`，先回答：

- 当前 V 精修预算是否真的未覆盖高 `g` 位置；
- 16/32 bins 相比均匀排名能否提高 attention-weighted residual 捕获率；
- 收益是否在 MHA/GQA、causal/non-causal 和不同 offset 上一致；
- 当前接近全量 refine 时，位置重排是否仍有实际选择差异。

只有满足以下条件才接入 `solution.py`：

- 固定相同 refine block 数，position-aware 排名捕获的加权残差至少提高 15%；
- 至少 75% 层为正；
- MHA/GQA 方向一致；
- 16/32 bins 结果不依赖单一 offset；
- 估计动态额外成本小于父 Attention 路径的 5%。

如果 oracle 显示当前 refine 已接近全量、排名改变不能影响量化结果，则立即拒绝 A6，不实现更复杂的 token-token Hessian 或跨 token coordinate descent。

#### 7.6.4 A6 晋级门

- Attention 综合均值至少 `+0.5pp`；
- causal 和 non-causal 均不下降；
- MHA/GQA 均不下降；
- 最差层回退不超过 `0.5pp`；
- E1 overall、heavy-tail、saturated logits 全部非退化；
- q/k state 与输出逐元素保持父版本；
- Linear 全部分项不变；
- 最终官方总耗时严格小于 300 秒。

### 7.7 Attention 完整评测矩阵

任何 A4/A5/A6 候选必须运行：

真实模型固定矩阵：

- MHA amax6 offsets `0/97/193/389`；
- MHA amax4 offset `0`；
- MHA pow2 offset `0`；
- GQA amax6 offsets `0/193`；
- causal/non-causal 双轨；
- 全 12 层逐层结果；
- mean、median、worst layer、win rate；
- position/entropy bucket 结果。

合成矩阵：

- 8 场景 × MHA/GQA × causal/non-causal × 3 mode × 3 seed，共 576 case；
- overall、各场景、最差 case；
- 特别单列 `heavy_tail`、`saturated_logits`、`qk_dynamic_imbalance`；
- 与 C21-C 逐 case delta。

工程验证：

- feature-off 与 C21-C 逐元素等价；
- state shape、dtype、元素数合法；
- CPU/CUDA 一致；
- 至少两次确定性复跑；
- q/k/v dynamic 分项计时；
- 完整官方可比串行计时至少三次；
- 静态合规和运行时调用图审计。

### 7.8 Attention 候选的总体晋级规则

#### 机制有效

- 目标局部指标跨 folds、layers 和 heads 一致改善；
- 完整部署 Attention 指标方向一致；
- 没有通过少数极端 head 掩盖多数退化；
- 合规、状态、确定性全部通过。

#### 可成为 Attention Champion

- 固定真实矩阵综合至少 `+1.0pp`；
- MHA/GQA、causal/non-causal 四个主分区均不出现超过 `0.2pp` 的均值退化；
- 最差层不低于父版本超过 `0.5pp`；
- E1 overall 和关键尾部场景不下降；
- 官方预计时间严格低于 300 秒。

#### 可与 Linear Champion 合并

- Attention 候选已经独立归档并冻结；
- Linear 候选已经独立归档并冻结；
- 合并版本只做交互验证，不重新调两侧参数；
- Linear 固定六项保持 Linear 父版本；
- Attention 完整矩阵保持 Attention 父版本；
- 合并后重新执行合规、确定性和三次计时；
- 合并后官方预计时间严格低于 300 秒。

Attention 候选不因研发时间较长被拒绝，但由于历史官方兑换率较弱，弱于 `+1pp` 的 Attention 改动原则上不单独消耗剩余 holdout。优先把 holdout 留给最强 Linear 候选和最终组合版本。

### 7.9 Attention 执行顺序

其他 AI 必须按以下顺序执行：

1. 复现 C21-C/A1 的八组真实 Attention 固定结果与 576 case；
2. 只增加 head/bucket telemetry，确认不改变结果；
3. 实现 A4 独立 probe，不修改默认 solution 行为；
4. 完成 A4 全层 MHA/GQA 双折探针；
5. A4 通过后接入并运行完整矩阵；
6. A4 归档后才开始 A5 R4/R8/R16；
7. A5 归档后运行 A6 position-budget oracle；
8. A6 oracle 通过才接入动态路径；
9. 每个候选失败时恢复上一个 Attention Champion；
10. 只将最强 Attention Champion 与最强 Linear Champion 组合；
11. 组合通过完整矩阵和 300 秒硬门后才考虑 holdout/官方提交；
12. 所有失败和弱正结果均归档，不重复 H64/CVaR/head-constant V 搜索。

## 8. 后备方案 C30：Hessian-Aware Hierarchical Permutation

如果 C29 失败，下一方向是**Hessian 感知层级排列**，而不是继续随机旋转。

### 8.1 动机

当前排列未必让强相关、相似量化难度的通道落入同一个 4/8/16/64 层级块。随机 R64 已失败，但结构化重排仍可能改善层级量化器的局部性，并且最终只复用现有 `permutation` 状态，动态成本为零。

### 8.2 合法图构造

使用变换后激活协方差或 Hessian `H`，以及由权重自身硬重构得到的逐通道残差能量：

```text
r_i = sum_rows(E_weight[row, i]^2)
edge(i, j) = abs(H[i, j]) * sqrt(r_i * r_j)
```

再加入幅值不兼容惩罚，避免把幅值分布相差过大的通道强行放进同一小组：

```text
penalty(i, j) = abs(log(scale_i) - log(scale_j))
utility(i, j) = edge(i, j) - lambda * penalty(i, j)
```

这里只使用合法的激活二阶统计和权重自身残差，不使用输出拟合或跨操作数结果。

**合规预裁定（强制前置步骤）**：`edge(i, j) = |H_A[i,j]| * sqrt(r_i * r_j)` 把激活二阶统计与权重残差统计以逐元素乘积形式组合，是迄今所有候选中最接近灰线的模式——两个操作数各自合法、该乘积也不是 `A@W` 收缩，按白名单"分别计算 operand-local 指标、再以预注册规则组合排名"可以辩护，但现有 `evaluator/linear_compliance_guard.py` 的运行时污点跟踪未必认识"分别派生的逐通道统计量的逐元素乘积"这一形式。因此在实现任何 C30 代码之前，必须：

1. 先把该 edge 模式作为最小用例提交 `linear_compliance_guard` 裁定；
2. 若 guard 判违规：检查是污点跟踪的保守误报还是实质问题，必要时扩展 guard 的白名单定义并配套回归测试，裁定结论与理由写入预注册条目；
3. 若最终裁定不通过：C30 直接否决，不进入图构造阶段，不得带着"可能合规"的实现继续。

### 8.3 分层分组算法

1. 在每个 64 通道候选池内建立图；
2. 用贪心匹配或谱排序形成 4 通道组；
3. 将两个 4 组聚合成 8 组；
4. 将两个 8 组聚合成 16 组；
5. 将四个 16 组组成 64 块；
6. 每一级都最大化组内 utility，同时约束幅值范围；
7. 生成确定性 permutation；
8. 用独立激活/权重局部损失进行 Pareto 验收；
9. 折叠到现有排列状态，不增加动态操作。

### 8.4 C30 可行性门槛

- 相比父排列，16 通道以内捕获的合法图边权至少提高 20%；
- 激活幅值不兼容惩罚增加不超过 5%；
- 激活和权重硬重构指标均不恶化；
- 双折一致；
- 完整 Linear 至少提高 1.5 个百分点后才进入 holdout；
- 最终官方路径严格小于 300 秒。

若图指标改善但硬量化指标无改善，立即停止，不得继续通过改权重系数追逐开发集。

## 9. 后备方案 C31：C23-lite 贡献拆分

如果 C29、C30 均失败，回到 C23 的已知正信号，但必须拆出低成本子集。

### 9.1 要回答的问题

分别测量 C23 各阶段的边际贡献：

- 仅 scale beam；
- GPTQ 初始化；
- full-H coordinate descent；
- hierarchy refine；
- 不同覆盖率；
- 不同 beam 宽度；
- 不同 shape class。

### 9.2 预注册扫描矩阵

```text
coverage ∈ {0.05, 0.10, 0.15, 0.25}
beam     ∈ {1, 2, 4}
stage    ∈ {scale-only, init-only, coord-only, hierarchy-only, combinations}
```

先运行分阶段探针，输出每个阶段的：

- 局部权重损失下降；
- Linear 增益；
- CPU 增量耗时；
- 每 0.1 个百分点增益的秒数；
- 层/组件稳定性；
- 内存峰值。

### 9.3 shape 约束

当前接口不能仅凭 shape 区分 q/k/v/o，因为这些窄方阵形状可能相同。因此：

- 不得声称只对其中某一个名字启用，除非接口确实提供合法、稳定的组件标识；
- shape gating 必须按实际可观察 shape class 应用于同类组件；
- fc/proj 可按不同形状单独分析；
- 不得使用调用顺序等脆弱隐式信号猜测组件名。

### 9.4 C31 验收

C31 的定位是低成本增量，不是假定它能单独达到 26000。只有满足以下条件才保留：

- 至少提高 1.0～1.5 个 Linear 百分点；
- 固定六项稳定；
- 相比 C23 大幅降低校准耗时；
- 总官方预估时间低于 300 秒；
- 完全合规；
- 与 C29 或 C30 组合时需要重新做完整消融，不能直接相加推算。

## 10. 26000 分可达性判断方法

### 10.1 当前估计

按 §3.1 的区间映射，26000 分需要 Linear 约 0.90（斜率 300）～0.9768（斜率 259）；本节及后续阶段决策一律取保守端 **0.9768**，相对当前 0.5311 差距约 0.4457，即 44.57 个百分点。

已知单项实验的量级：

- C23 full-64：约 +1.93 个百分点；
- 常规局部微调：通常更低；
- 固定坐标激活 oracle 剩余能量有限。

因此在不出现新的结构性突破时，当前框架通过微调直接达到 26000 的概率很低。

### 10.2 阶段性决策阈值

用以下结果判断是否仍应冲击 26000：

1. 如果 C29 单独带来至少 +5 个百分点，说明坐标重构是高杠杆方向，应继续做多层级 S 与 C30 联合设计；
2. 如果 C29/C30 各自只有 +1～2 个百分点，说明它们是增量优化，不能支撑 26000；
3. 如果三条方向组合后仍低于 Linear 0.65，应停止宣称当前框架可达 26000；
4. 若要达到 26000，下一代框架必须一次性解决更大比例的激活和权重误差，而非继续细化现有码本；
5. 任何高分方案都必须先证明合规，不能以输出拟合换取分数。

### 10.3 可能需要的框架级突破

若 C29～C31 无法显著提升，后续研究应转向：

- 对赛事允许的数据格式、量化格式和状态语义重新做严格解释；
- 设计与 HiF4 层级码本联合优化的新型可逆结构，而不是单独优化 scale；
- 研究能同时改善两侧分布、且可完全折叠的更大类结构化变换；
- 在官方规则允许范围内增加模型内先验，但不得引入输出监督；
- 向官方确认边界模糊的允许操作，再决定是否实现。

在没有框架级证据前，不应把 26000 当作通过参数扫描必然可达的目标。

## 11. Holdout 与官方提交策略

当前 holdout 预算已使用 1/3，剩余 2/3。

规则：

1. C29/C30/C31 和 A4/A5/A6 的探针、参数扫描及开发验证都不得使用 holdout；
2. 原则上保留一次机会给最强且完全冻结的独立机制候选，优先 Linear 高杠杆候选；
3. 另一次机会优先留给“最强 Linear Champion + 最强 Attention Champion”的冻结组合；
4. 弱于 `+1pp` 的 Attention 候选不单独消耗 holdout，除非公开证据证明官方 Attention 权重显著高于当前估计；
5. 如果 Linear 主线全部失败而 Attention 出现跨矩阵的大幅提升，可把第一机会转给该 Attention 候选，但必须预先记录原因；
6. holdout 结果只用于接受或拒绝，不用于修改参数；
7. 一旦根据 holdout 修改参数，修改后的版本视为新候选，不能继续复用同一 holdout 结论；
8. 官方提交前必须记录代码哈希、配置、环境、计时和合规报告；
9. 官方结果无论好坏都必须归档，不得只保存成功结果。

## 12. 归档与日志要求

每个候选使用新的编号和独立目录。建议：

```text
solutions/20260827_v029_c29-haes-<status>/
solutions/20260827_vNNN_a4-headwise-pareto-qk-<status>/
solutions/20260827_vNNN_a5-small-block-qk-rotation-<status>/
solutions/20260827_vNNN_a6-position-aware-v-<status>/
solutions/20260827_vNNN_c30-hessian-permutation-<status>/
solutions/20260827_vNNN_c31-c23-lite-<status>/
```

`vNNN` 必须在实际归档时取下一个可用版本号，不得让 Linear 与 Attention 并行实验占用相同编号。

每个目录至少包含：

- `solution.py` 快照；
- `result.md`；
- 完整配置；
- 原始指标 JSON；
- 计时记录；
- 合规报告；
- 与父版本的差异摘要；
- 接受/拒绝原因；
- 可复现命令。

同步更新：

- 优化执行日志；
- solution archive 索引；
- 当前 Champion 指针；
- holdout 使用计数；
- README 中的当前成绩与状态（若 Champion 发生变化）。

拒绝实验也必须归档。结论必须说明它排除了什么、没有排除什么，防止未来重复错误解释。

## 13. 其他 AI 的逐步执行指令

接手本计划的 AI 必须严格按以下顺序工作。Linear 与 Attention 是两条独立候选轨，可以在各自独立工作区中推进，但在分别归档前不得合并：

1. 阅读赛事规则、C21-C、C22、C23、C28、A1、A2、A3、C2/C2a 的结果和最新执行日志；
2. 记录当前 `solution.py` 哈希和 Git 状态，不覆盖用户已有修改；
3. 运行 C21-C 基线，确认 Linear 固定矩阵、Attention 八组真实矩阵、E1 576 case 和确定性；
4. 新建 C29 独立探针与单元测试，不修改默认提交行为；
5. 另建 A4 独立 Attention probe，只增加 telemetry 和搜索代码，不修改默认行为；
6. 两条轨分别完成合规测试和单层/head smoke test；
7. C29 跑全 12 层 × 6 组件 × 双折探针；A4 跑全 12 层 MHA/GQA × 双折探针；
8. 两条轨分别按预注册门槛自动生成 pass/fail；
9. 只有 pass 的机制才接入各自候选 `solution.py`；
10. 接入后先验证 feature-off 等价，再验证 feature-on 数值；
11. Linear 候选跑固定六项和 Attention 不变量；Attention 候选跑八组真实矩阵、576 case 和 Linear 不变量；
12. 各自通过后再做 profiler 和性能优化；
13. 每个候选用至少三次可比串行测量确认最终时间，按 §5.6 公式换算的推算官方耗时必须严格低于 300 秒；
14. 分别冻结参数并归档；
15. C29 失败则转入 C30，C30 失败后转入 C31；
16. A4 失败则转入 A5，A5 完成后才运行 A6 oracle；
17. 只组合已经独立晋级并冻结的 Linear/Attention Champion；
18. 组合版本重新运行全部精度、合规、确定性和计时矩阵；
19. 再按第 11 节决定是否使用剩余 holdout；
20. 三个 Linear 方向均失败则停止同类 Linear 微调；三个 Attention 方向均失败则停止同类 Attention 微调；分别提交上限报告，不得无依据扩大随机搜索。

任何步骤发现合规疑点时，优先停止并审查；不得带着“可能合规”的实现继续跑分。

## 14. 明确不再重复的方向

除非出现新的理论证据或官方规则变化，不再执行：

- 更多随机 R64、双 Hadamard 或仅更换随机种子；
- 固定坐标系中的 C28 scale-code 细化；
- 原样恢复 C23 full-64 全覆盖实现；
- 固定全层 H64、只增加 H64 seed 或再次提交已拒绝的 A2；
- 把 Segment-CVaR/C2a 加权目标原样恢复为 Attention 排序器；
- 继续更换每个 head 一个常量的 V importance 公式；
- 只看单一 offset 的候选筛选；
- 使用 holdout 或官方结果调参数；
- 任何形式的 Linear 输出拟合；Attention 的 `PV` 仅可用于本节定义的独立终验，不得进入 Linear，也不得取代 A4 的 probability 主目标；
- 通过放宽 300 秒限制接受不可提交方案；
- 为赶开发时间而省略全层、全组件和双折验证。

## 15. 完成定义

本计划只有在以下任一状态成立时才算完成：

### 状态 A：产生新 Champion

- 新候选通过全部精度、稳定性、Attention、合规和计时门槛；
- 最终官方耗时严格小于 300 秒；
- holdout/官方结果已归档；
- README、日志和 Champion 指针已同步。

### 状态 B：主方向被可信否定

- C29、C30、C31 均完成预注册的 Linear 全矩阵探针；
- A4、A5、A6 均完成预注册的 Attention 探针或 oracle；
- 失败原因有数据支持并已归档；
- 剩余误差和框架上限重新估计；
- 明确说明 26000 需要何种框架级突破；
- 不再重复已否定的搜索。

## 16. 当前建议结论

下一步应建立两条彼此独立的机制轨：Linear 首先执行 C29 HAES，Attention 首先执行 A4 Headwise Pareto QK Search。C29 是冲击总分的高优先级方向，因为它既绕开了 C22 的随机混合失败，也突破了 C28 的固定坐标上限；A4 则利用当前尚未开发的 per-head 组合自由度，在不重复 H64/CVaR 失败路径的前提下改善 MHA/GQA 和尾部场景。

研发阶段应充分运行完整探针，不用人为压缩实验时间；但最终候选和最终组合都必须通过严格的 300 秒官方硬门槛。若 C29 不能产生至少中等规模、跨层稳定的提升，则按 C30、C31 顺序验证；若 A4 失败，则按 A5、A6 顺序验证。两条轨分别得出结论后再更新框架上限，不得用 Attention 的弱官方兑换率掩盖 Linear 缺口，也不得因为 Linear 是主分来源而停止验证 Attention 的隐藏场景和尾部收益。
