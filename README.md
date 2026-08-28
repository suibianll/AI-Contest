# HiF4 量化竞赛工程

华为 2026 算法竞赛 NVFP4 → HiF4 赛道的开发工作区。根目录
`solution.py` 是唯一活跃、可提交的算法文件；历史候选保存在
`solutions/`，不会被运行时引用。

英文版：[README_EN.md](README_EN.md)

## 当前状态

- 最新合规官方锚点：v025 / C21-C，`14437 / 166.6s`。
- 历史 v024 得分为 `16043 / 173.8s`，但包含官方后来明确禁止的
  Linear 输出监督路径，不作为后续合规父版本。
- 当前根 `solution.py`：C38，本地 Linear `0.5695`、Attention causal
  `0.4497`、CPU algorithm-stage 约 `99s`；官方已提交
  `14092 / 170.57s`（与本地倒挂，归档 v030，待 A/B 对照 v025 二分
  定位；官方时间余量 43%，时间非瓶颈）。
- 当前源码 SHA256：
  `648A27B3560EF7F5D939CD409301E445E5065047CBD5438C1A73A013730E467F`。
- C38 固定矩阵：offset 0/97/193 的 Linear 分别为
  `0.5695 / 0.5629 / 0.5766`。

本地时间和本地分数仅用于候选比较，不冒充官方结果。任何官方结果都应与
实际提交 SHA、分数和时间一起归档。

## 官方硬约束

1. 不得显式或隐式计算 `A@W`，并利用其输出拟合、选择或反推 `Q(A)`。
2. 输出必须是合法 HiF4 五字段，API、state、shape、dtype 和设备必须符合要求。
3. 最终官方评测总时间必须严格小于 `300s`。
4. 不使用 holdout 或官方分数反向调参。

除上述规则外，不设置固定的增益、coverage、beam、单组件非退化或中间时间门槛。
开发阶段允许完整扫描和超过 300 秒的诊断实验；发现精度信号后，再通过算法和实现
优化压入最终时间限制。

## 当前算法

### Linear

1. 按 NVFP4 scale 和 E2M1 载荷重建浮点参考。
2. 搜索 SmoothQuant 对角缩放、通道排列和 4/8/16 小块正交变换。
3. 生成标准 HiF4 参数，并执行 4/8/16 组二阶精修。
4. Weight FULL64 使用合法激活 Hessian：
   - scale beam 保留 2 路；
   - 输入宽度 `<=1024` 的矩阵覆盖全部 64-block；
   - 宽层覆盖率为 `0.25`；
   - 执行 GPTQ 初始化、一次 64 维坐标下降和层级 toggle；
   - 已删除无收益的第二轮坐标下降。
5. Activation 动态路径使用 sample-local HiF4 编码和当前已验证的 4/8 组精修。

### Attention

当前保留 A1 路径：Smooth-QK、K midrange 居中、headwise permutation、
MHA/GQA 对齐和真实 Attention 双 mask 安全选择。固定 H64、Segment-CVaR 和
无收益的 V importance 候选保持关闭。

## 开发原则

- 真实部署路径的配对分数是候选裁判；oracle 和局部损失只用于诊断、排序和解释。
- 不用任意百分比阈值在实现前否决候选，除非存在严格数学不可能证明。
- 允许同时删除冗余计算并重新分配预算；完成后再做消融归因。
- 保留完整的精度—时间 Pareto 曲线，不因单次负结果宣称整个赛事空间不可达。
- 小而稳定的正增量可以累计，不要求每个候选达到固定百分点。
- 失败实验照常归档，但失败结论只约束被实际测试的实现和配置。

## 工程结构

```text
solution.py                         唯一活跃提交文件
evaluator/real_data_eval.py         真实 GPT-2 配对评测
evaluator/synthetic_attention_eval.py
                                    576-case Attention 安全矩阵
evaluator/cap_oracle.py             固定坐标误差空间诊断
evaluator/linear_compliance_guard.py
                                    Linear 合规静态/运行时检查
evaluator/holdout_eval.py           受预算保护的 holdout 评测
tests/                              发布、格式、合规和算法测试
solutions/                          不可变候选归档
docs/superpowers/logs/              执行日志和校准记录
docs/superpowers/plans/             当前通用流程
docs/superpowers/archive/plans/     已失效优化计划，仅供历史查阅
```

## 运行评测

使用工程虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r evaluator\requirements.txt
```

真实模型评测：

```powershell
.\.venv\Scripts\python evaluator\real_data_eval.py `
  --solution solution.py --model models/gpt2 --device cuda
```

GQA 示例：

```powershell
.\.venv\Scripts\python evaluator\real_data_eval.py `
  --solution solution.py --model models/gpt2 --device cuda --kv-heads 6
```

Attention 合成矩阵：

```powershell
.\.venv\Scripts\python evaluator\synthetic_attention_eval.py `
  --solution solution.py
```

完整测试：

```powershell
.\.venv\Scripts\python -m pytest -q
```

## 记录位置

- 当前优化事实以根 `solution.py`、最新执行日志和可复现评测输出为准。
- 历史版本及其结论见 [solutions/README.md](solutions/README.md)。
- 最新执行记录见
  [2026-08-26-optimization-execution-log.md](docs/superpowers/logs/2026-08-26-optimization-execution-log.md)。
- 候选归档流程见
  [2026-08-26-solution-archive-workflow.md](docs/superpowers/plans/2026-08-26-solution-archive-workflow.md)。
- 旧优化计划已移入 `docs/superpowers/archive/plans/`，不再作为后续执行依据。
