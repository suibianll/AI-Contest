# HiF4 量化算法泛化评估环境

## 1. 目标

该环境用于评估 NVFP4→HiF4 算法修改是否具有稳定泛化收益，而不是只在少量固定合成样本上取得更高分。它复现赛事评分公式和主要判题流程：

\[
Score=\frac{MSE_{STD}-MSE_{PLAYER}}{MSE_{STD}}
\]

其中 Linear 比较最终矩阵乘输出，Attention 比较完整 softmax Attention 输出。评估器不会用 \(A@W\) 拟合激活量化参数；它只在算法运行完成后计算输出误差，因此不违反赛题限制。

schema v2 的正式代理协议固定为 FP32 输出计算和 non-causal Attention。标准 HiF4 的 scale factor 先按 E6M2 量化，再对每个 8-value group 穷举全部 8 种合法 `(lv2, lv3_left, lv3_right)` 组合，以最小重构 MSE 选择层级指数。候选输出和 calibration state 按任务书严格校验，每个在线样本使用独立 state 深拷贝。

E6M2 有 64×4−1=255 个有限正数编码：只排除保留的 `(63,3)` NaN 编码，最小 scale 是 `2^-48`，最大是 49152；与最小非零 S1P2 mantissa 组合后的最小最终值是 `2^-50`。对比分析中“254 个合法 scale、最小 scale 为 `2^-50`”的推断与详细接口不一致，未采纳。

本地数据仍是合成泛化代理，不是官方数据集，因此绝对分不能映射为排行榜分数；其用途是筛选稳定的相对改进。

## 2. 防止过拟合的机制

| 机制 | 实现 | 解决的问题 |
|---|---|---|
| 开发集与隐藏集分离 | `--split dev/holdout` | 避免直接对最终评估样本调参 |
| 动态隐藏种子 | HMAC 根据 campaign、attempt 派生，不写入报告 | 避免长期复用固定随机种子 |
| 代码冻结 | 评估前记录 candidate/incumbent 的 SHA256 | 防止结果与代码版本不对应 |
| Holdout 预算 | 默认最多 3 次，每次写入 campaign | 避免把隐藏集逐渐变成开发集 |
| 配对比较 | 新旧实现使用完全相同的数据 | 降低随机波动，提高版本判断能力 |
| 分布漂移 | 校准/测试幅值漂移、稀疏、相关、均值偏移、离群值 | 评估校准状态的迁移能力 |
| 结构漂移 | 多 head/GQA/head_dim/seq_len；正式分数 non-causal，causal 可作为独立鲁棒性 campaign | 避免只适配单一 Attention 形状或污染正式权重 |
| 尾部约束 | 负分率、灾难性负分率、P05、最差 10% 均值 | 对齐赛事中负分会直接扣分的风险 |
| 性能约束 | candidate/incumbent 量化耗时比 | 避免精度提高但五分钟超时 |

隐藏种子并不是安全沙箱：代码拥有者仍可主动读取本地密钥。它提供的是可审计的实验纪律。最可靠的流程是只让 CI 或另一位队员持有 campaign 目录，并只返回聚合结果。

## 3. 数据覆盖

Linear 包含 balanced、hierarchy、outlier、heavy-tail、校准—测试幅值漂移、64-block 内相关、稀疏和均值偏移；Attention 包含 balanced、Q/K 动态范围失衡、K 均值偏移、heavy-tail、logits 饱和、V 离群值、Q/K 相关性，并轮换 MHA/GQA、不同 head 数和 head_dim。默认只把 non-causal 纳入正式分数；需要 causal 压力测试时使用单独配置，配置指纹会自动选择独立 campaign。

评估规模分三档：

| 档位 | 用途 | 种子/样本规模 |
|---|---|---|
| `smoke` | 接口、合法性和基本回归 | 1 seed，少量场景 |
| `standard` | 日常版本选择 | dev 3 seeds、holdout 5 seeds，完整主要分布 |
| `soak` | 提交前压力测试 | dev 8 seeds、holdout 12 seeds，更大矩阵与更多结构 |

## 4. 推荐运行流程

### 4.1 当前 CPU/Torch 环境：先验证评估协议

```powershell
.\.venv\Scripts\python cli.py evaluate solution.py `
  --tier smoke --device cpu --split dev --root .
```

当前活动评测只执行 Torch CPU/CUDA 路径，不使用 NumPy 模拟。仓库中的旧
NumPy 模拟器文件保留为下载资料，不会被 CLI、worker 或评分器导入。

### 4.2 PyTorch 环境：开发集精确评估

```powershell
.\.venv\Scripts\python cli.py evaluate solution.py `
  --tier standard --device cpu --split dev --root .
```

先根据开发集完成参数和算法选择。建议一次只改变一个机制，并保留消融结果。不要因为单个 seed 或单个场景下降就立即针对性补丁；只有能解释的系统性失败才应修改算法。

### 4.3 候选冻结后的隐藏集晋级

```powershell
.\.venv\Scripts\python cli.py evaluate solution.py `
  --tier standard --device cpu --split holdout --root .
```

每次 holdout 调用都会消耗一次预算并生成一批新种子。若代码变化，应先回到 dev；不要连续调用 holdout 搜索阈值。

### 4.4 提交前压力测试

```powershell
.\.venv\Scripts\python cli.py evaluate solution.py `
  --tier soak --device cpu --split dev --root .
```

随后还必须执行官方 `self_check.py`，并在鲲鹏 920B 或相近 CPU 上确认总时间低于 5 分钟。schema v2 的 `standard` 轨共享 300 秒总超时，而不是每个 seed 各 300 秒；`soak` 是额外压力测试，不代表官方时限。当前仓库没有官方数据集，所以本地评测不能替代真实比赛测试。

## 5. 晋级标准

默认配置要求同时满足：平均比赛分数非负；负分率不超过 10%；小于 -10% 的灾难性负分不超过 2%；最差 10% 平均分不低于 -5%；相对旧版本平均分至少提升 0.2 个百分点；负分率不得上升；候选量化时间不超过当前 Champion 的 2.0 倍，同时整条 standard 轨不得超过 300 秒。

建议以业务风险调整阈值，但不能在看到 holdout 结果后修改同一 campaign 的阈值。schema、样本规模、阈值、时限、计算精度或掩码模式变化后，CLI 会按策略指纹建立新 campaign，旧报告不得与新协议绝对分混算。赛事最终分数是 case 分数之和，因此报告同时保留平均分、总分和分组结果；版本选择优先顺序建议为：合法性与无异常 → 负分率与尾部 → 配对平均增益 → 总运行时间 → 绝对平均分。

## 6. 输出解释

输出 JSON 主要字段如下：

- `candidate.summary`：平均分、中位数、P05、最小值、最差 10% 均值、负分率、灾难性负分率、bootstrap 95% 区间和分场景统计。
- `comparison`：新旧实现逐 case 配对分数差、胜/平/负、平均差及 bootstrap 区间。
- `decision.checks`：每个晋级条件是否通过；只有 PyTorch 精确后端可能最终 `promote=true`。
- `metadata.seed_commitment`：本轮隐藏种子的承诺值，报告不包含真实隐藏 seed。
- `campaign.json`：所有评估历史、代码哈希和 holdout 使用次数，便于审计。

## 7. 当前实现合规审计

本地审计已覆盖实际候选 `solution.py` 与独立评测器：

- 六个竞赛接口均可加载；输出 shape、HiF4 五字段、255 值 E6M2、0.25 mantissa 步进、有限实数、dense strided/no-grad 均被校验。
- state 校验覆盖 CPU、dtype、dense strided、finite、深度/节点、4096 UTF-8 字节字符串与键；每个在线测试收到独立深拷贝。
- Linear 校准路径不计算 `A @ W`，只使用激活二阶统计、Weight 统计和量化残差；Attention 校准使用完整 softmax 输出代理，符合两类算子的不同规则边界。
- GQA head 映射、non-causal softmax、逐组精确标准 HiF4 基线和逐 case 配对评分均由独立 Torch 评测器执行；causal 实现保留作可选鲁棒性测试。
- 活动路径不导入 NumPy；CUDA 只用于精度筛选，CPU 计时是权威轨。当前机器没有 CUDA，因此 GPU 轨只能在安装 CUDA 版 PyTorch 后验收。
- 仓库未提供真实比赛数据；本系统是可审计的合成泛化代理，不能据此宣称官方排行榜分数。提交前仍需运行官方检查和鲲鹏/相近 CPU 实测。

## 8. 版本开发纪律

1. 每个版本只修改一类机制，例如 scale 搜索、排列、Attention 门控或精修比例。
2. 使用相同 dev campaign 做配对消融，至少跨 3 个 seed。
3. 若平均分提高但负分率、P05 或最差场景恶化，不晋级。
4. 只有开发集通过全部门控才运行 holdout；单版本最多一次正式 holdout。
5. holdout 未通过时分析失败类别，但下一版必须使用新 attempt 的隐藏种子。
6. 最终版本运行 `soak + self_check + 官方环境计时`，并保存代码 SHA256 与结果 JSON。
