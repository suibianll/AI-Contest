# HiF4 量化算法泛化评估环境

## 1. 目标

该环境用于评估 NVFP4→HiF4 算法修改是否具有稳定泛化收益，而不是只在少量固定合成样本上取得更高分。它严格复现赛事分数：

\[
Score=\frac{MSE_{STD}-MSE_{PLAYER}}{MSE_{STD}}
\]

其中 Linear 比较最终矩阵乘输出，Attention 比较完整 softmax Attention 输出。评估器不会用 \(A@W\) 拟合激活量化参数；它只在算法运行完成后计算输出误差，因此不违反赛题限制。

## 2. 防止过拟合的机制

| 机制 | 实现 | 解决的问题 |
|---|---|---|
| 开发集与隐藏集分离 | `--split dev/holdout` | 避免直接对最终评估样本调参 |
| 动态隐藏种子 | HMAC 根据 campaign、attempt 派生，不写入报告 | 避免长期复用固定随机种子 |
| 代码冻结 | 评估前记录 candidate/incumbent 的 SHA256 | 防止结果与代码版本不对应 |
| Holdout 预算 | 默认最多 3 次，每次写入 campaign | 避免把隐藏集逐渐变成开发集 |
| 配对比较 | 新旧实现使用完全相同的数据 | 降低随机波动，提高版本判断能力 |
| 分布漂移 | 校准/测试幅值漂移、稀疏、相关、均值偏移、离群值 | 评估校准状态的迁移能力 |
| 结构漂移 | 多 head/GQA/head_dim/seq_len、causal 与非 causal | 避免只适配单一 Attention 形状 |
| 尾部约束 | 负分率、灾难性负分率、P05、最差 10% 均值 | 对齐赛事中负分会直接扣分的风险 |
| 性能约束 | candidate/incumbent 量化耗时比 | 避免精度提高但五分钟超时 |

隐藏种子并不是安全沙箱：代码拥有者仍可主动读取本地密钥。它提供的是可审计的实验纪律。最可靠的流程是只让 CI 或另一位队员持有 campaign 目录，并只返回聚合结果。

## 3. 数据覆盖

Linear 包含 balanced、hierarchy、outlier、heavy-tail、校准—测试幅值漂移、64-block 内相关、稀疏和均值偏移；Attention 包含 balanced、Q/K 动态范围失衡、K 均值偏移、heavy-tail、logits 饱和、V 离群值、Q/K 相关性，并轮换 MHA/GQA、不同 head 数和 head_dim。默认同时评估 causal/non-causal；如果判题规则确定 mask，可固定为对应模式。

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

随后还必须执行官方 `self_check.py`，并在鲲鹏 920B 或相近 CPU 上确认总时间低于 5 分钟。当前仓库没有提供官方 `self_check.py`，所以本地评测是独立代理，不能替代真实比赛测试集。

## 5. 晋级标准

默认配置要求同时满足：平均比赛分数非负；负分率不超过 10%；小于 -10% 的灾难性负分不超过 2%；最差 10% 平均分不低于 -5%；相对旧版本平均分至少提升 0.2 个百分点；负分率不得上升；量化时间不超过旧版本的 1.2 倍。

建议以业务风险调整阈值，但不能在看到 holdout 结果后修改同一 campaign 的阈值。赛事最终分数是 case 分数之和，因此报告同时保留平均分、总分和分组结果；版本选择优先顺序建议为：合法性与无异常 → 负分率与尾部 → 配对平均增益 → 总运行时间 → 绝对平均分。

## 6. 输出解释

输出 JSON 主要字段如下：

- `candidate.summary`：平均分、中位数、P05、最小值、最差 10% 均值、负分率、灾难性负分率、bootstrap 95% 区间和分场景统计。
- `comparison`：新旧实现逐 case 配对分数差、胜/平/负、平均差及 bootstrap 区间。
- `decision.checks`：每个晋级条件是否通过；只有 PyTorch 精确后端可能最终 `promote=true`。
- `metadata.seed_commitment`：本轮隐藏种子的承诺值，报告不包含真实隐藏 seed。
- `campaign.json`：所有评估历史、代码哈希和 holdout 使用次数，便于审计。

## 7. 版本开发纪律

1. 每个版本只修改一类机制，例如 scale 搜索、排列、Attention 门控或精修比例。
2. 使用相同 dev campaign 做配对消融，至少跨 3 个 seed。
3. 若平均分提高但负分率、P05 或最差场景恶化，不晋级。
4. 只有开发集通过全部门控才运行 holdout；单版本最多一次正式 holdout。
5. holdout 未通过时分析失败类别，但下一版必须使用新 attempt 的隐藏种子。
6. 最终版本运行 `soak + self_check + 官方环境计时`，并保存代码 SHA256 与结果 JSON。
