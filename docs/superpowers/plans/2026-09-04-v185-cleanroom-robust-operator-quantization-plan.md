# v185 计划：clean-room 稳健算子量化重写（2026-09-04）

> 状态：ACTIVE / IMPLEMENTATION
>
> 本计划不继承或复制现有 `solution.py` 的实现，只使用公开接口、
> `evaluator/reference_hif4.py` 的合法格式定义和已经确认的机制经验。
> v182 保持官方分数父，v180 保持时间预算父；v184 工作区不属于本计划。

## 1. 目标

从空白文件重新实现六个正式 API，建立一个可解释、低有效自由度、可独立验证的候选：

1. 输出始终是合法 HiF4 五字段；
2. Linear 直接优化校准折上的部署 MatMul 输出误差；
3. Attention 直接优化校准折上的部署 Attention 输出误差；
4. 高维统计只能产生少数解析候选，不能直接生成高维自由参数；
5. 每个机制都有 identity/parent 回退；
6. 在线路径不计算 Hessian、Gram、矩阵逆或数据相关候选搜索；
7. 单文件、自包含、脱离仓库可导入。

本轮首先验证“从零构建的强正则算法能否在跨模型和隐藏分布上比复杂历史堆栈更稳健”，
不预设它必须立即超过 v182。

## 2. 数学模型

### 2.1 HiF4 可行域

每个 64 元素 block 输出：

```text
x_hat[i] = scale_factor
           * scale_lv2[group8(i)]
           * scale_lv3[group4(i)]
           * sign[i] * mant[i]
```

其中 `scale_factor` 是有限 E6M2，`scale_lv2/lv3 ∈ {1,2}`，
`sign ∈ {-1,0,1}`，`mant ∈ {0,0.25,...,1.75}`。

给定顶层 scale 后，逐 8 元素组穷举其 8 个合法 lv2/lv3 组合，mantissa 使用固定网格最近点；
这是本实现唯一的基础离散求解器。

### 2.2 Linear

允许一个严格保持连续乘积不变的对角变换：

```text
X' = X D
W' = W D^-1
```

`D` 由校准激活 RMS 与权重 RMS 的解析平衡得到，并向 identity 收缩、限制条件数。
校准折分别比较 identity 与该单一候选的真实部署输出误差；只有 median 正向且 worst-fold
不过界时才启用。最终 Weight 只编码一次。

Weight 的 HiF4 层级选择使用校准激活二阶矩的对角权重；在线 Activation 的层级选择使用
最终量化权重列能量。二者仍以真实 MatMul fold loss 作总门控，不部署完整 Hessian。

### 2.3 Attention

按顺序只允许四个低自由度动作：

1. **K token-mean centering**：每个 KV head 从当前 K 序列减去同一向量；该操作只给每个
   query 的 logits 加常数，softmax 连续域严格不变。
2. **解析 Q/K 对角平衡**：每个 KV head、每个 head-dim 使用校准 RMS 构造 `D`，部署
   `Q'=QD, K'=KD^-1`，连续 logits 不变；log-space 向 identity 固定收缩并 clamp。
3. **收缩 logits gain**：按 KV head 拟合量化 logits 到参考 logits 的回归斜率，log 域
   固定收缩；以固定非对称指数分配到 Q/K，最终乘积仍为单一 `gamma`。
4. **门控 +4 scale code**：比较基础 offsets 与只增加 `+4` 的候选；只有跨校准折真实
   Attention 输出损失稳定改善时，state 才允许在线使用 +4。

每一步只与上一步结果二选一，不做 alpha、rank、block size、seed 或 threshold 网格搜索。

## 3. 泛化约束

### 3.1 低有效自由度

- Linear：每个 state 只有 identity/analytic-D 一个二选一决策；
- Attention：每一步只有 off/on；Q/K 参数在 KV group 内共享；
- 所有连续参数向 identity 收缩并有固定范围；
- 不允许 per-row、per-token、per-case 保存自由参数。

### 3.2 Cross-fit 门

候选在每个校准样本上独立计算相对 parent 的算子损失改善：

```text
delta_f = (loss_parent_f - loss_candidate_f) / (loss_parent_f + eps)
```

只在以下条件同时满足时接受：

```text
median(delta_f) > fixed_min_gain
min(delta_f) > -fixed_worst_tolerance
```

Attention 同时检查 causal 与 non-causal 两个固定视图。门只负责 off/on，不调参数。

### 3.3 Trust region

- 对角 Linear transform 限制在 `[0.5, 2.0]` 并按 64-block 去除几何均值漂移；
- Attention Q/K balance 限制在 `[0.5, 2.0]`；
- logits gain 限制在 `[0.75, 4/3]`；
- E6M2 只在标准 code 邻域 `{0,-1,+1,+2,+3}`，通过门后才加入 `+4`；
- 动态只精化最高基础损失的一小部分 block，其余保持标准编码。

## 4. 复杂度设计

- 基础 HiF4 编码一次完成；offset 邻域只处理 top-loss block；
- Weight gate 只使用固定行数的权重 probe 和固定 token 数校准激活；
- Attention gate 只使用每个校准样本的固定前缀；
- 在线无 Python 逐元素循环；scale 候选数量固定；
- state 只保存 CPU tensor、bool、int、float；
- 不保留校准 token、A@W、QK、Attention output 或 Hessian。

## 5. 文件与版本边界

- 新源码：`solutions/20260904_v185_cleanroom-robust-operator_scoreNA_timeNA/solution.py`
- 结果：同目录 `result.md`
- 根 `solution.py` 保持 v182，除非未来官方 RETAINED；
- 不读取或修改 v184 工作区；
- 不从任何旧 solution 导入实现。

## 6. 实施步骤

1. 从零实现 NVFP4 dequant、E6M2 encode/decode、标准 HiF4 hierarchy solver；
2. 实现稀疏 top-loss offset refinement；
3. 实现 Linear analytic-D、probe gate、最终 Weight 编码与动态 Activation；
4. 实现 Attention reshape/GQA、K centering、Q/K balance、logit gain、+4 gate；
5. 实现六个公开 API 和合法 CPU state；
6. 运行 `py_compile`、脱离仓库六 API smoke、reference validator；
7. 运行目标侧最小 evaluator smoke；
8. 依次运行 Qwen compact、default、GPT-2、OPT；
9. 记录接口、状态、finite、case 配对、耗时与跨模型风险；
10. 本地结果只决定 ERROR/REJECTED/可提交候选，不换算官方分数。

## 7. 裁决规则

- 任一 API、HiF4 合法性、finite、state 或调用图失败：`ERROR`，修实现；
- Qwen/GPT-2/OPT 出现同构大幅回归：`REJECTED`；
- 本地 mixed 但机制可达、control 正确、复杂度有界：保留为候选，由用户决定是否消耗配额；
- 若未来官方 `>17598` 且 `<300s`：RETAINED；否则 REJECTED/TIMEOUT；
- 不围绕 fold、阈值、收缩、offset 集或 refine ratio 做邻域扫描。
