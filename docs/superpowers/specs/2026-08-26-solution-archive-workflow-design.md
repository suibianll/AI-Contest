# HiF4 方案归档与评测工作流设计

日期：2026-08-26
状态：已确认，等待实施计划

## 1. 目标

将工程调整为一个轻量、可追溯的算法优化工作区：根目录始终只有一个正在优化和准备提交的 `solution.py`，本地真实 GPT 评测代码独立放置，所有获得官方结果的算法版本按分数和耗时归档。通过本地分项结果、官方总分和实验结论的连续记录，判断本地评测与官方评测是否一致，并据此选择下一轮优化方向。

本设计不恢复旧评测系统，不引入数据库、归档脚本、自动晋级、复杂兜底或单元测试框架。

## 2. 目录结构

```text
AI竞赛/
├── solution.py
├── evaluator/
│   ├── real_data_eval.py
│   ├── nvfp4_sim.py
│   └── requirements.txt
├── solutions/
│   ├── README.md
│   ├── 20260825_v000_v9-baseline_score9000plus_timeNA/
│   │   ├── solution.py
│   │   └── result.md
│   └── 20260826_v001_current-baseline_score10250_time127s/
│       ├── solution.py
│       └── result.md
└── docs/
    └── superpowers/
        ├── specs/
        └── plans/
```

职责边界如下：

- 根目录 `solution.py` 是唯一活跃算法，允许持续修改，也是每次上传官方评测的文件。
- `evaluator/` 只保存真实 GPT 本地评测的最小运行代码和依赖声明。
- `solutions/` 只保存已经获得官方结果或具有明确历史价值的不可变算法版本。
- `docs/superpowers/` 保留现有设计方案和实施计划，不参与评测运行。
- `.git/` 和 `.venv/` 作为版本管理和本地运行环境继续保留。

## 3. 归档命名

版本目录统一使用：

```text
YYYYMMDD_vNNN_topic_scoreSCORE_timeTIMEs
```

约束：

- 日期使用官方结果确认日期；历史版本使用其进入仓库或成为基线的日期。
- 版本号从 `v000` 开始，三位十进制、严格递增且不复用。
- `topic` 使用简短 ASCII kebab-case，描述该版本的唯一主要机制。
- `score` 和 `time` 固定放在目录名末尾。
- 得分和耗时均使用官方结果，不用本地代理分数替代。
- 历史得分只有近似值时使用 `score9000plus` 这类显式标签。
- 耗时未知时使用 `timeNA`；官方超时时使用 `time300plus`。
- 已归档目录不因后续成为或失去 Champion 而改名。

当前两个初始归档为：

- `20260825_v000_v9-baseline_score9000plus_timeNA`：从 Git 恢复已删除的 `solution_v9_champion.py`，历史只确认官方约 9000+，耗时未知。
- `20260826_v001_current-baseline_score10250_time127s`：归档当前根目录 `solution.py`，官方得分 10250，耗时 127 秒。

## 4. 版本内容

每个版本目录固定包含：

```text
solution.py
result.md
```

其中 `solution.py` 必须是当次官方提交文件的逐字节副本。`result.md` 固定记录：

- 版本号、日期和源码 SHA256；
- 相对上一版本的唯一主要改动；
- 优化假设；
- 本地评测模型、层数、序列长度、校准样本数、测试样本数和 NVFP4 模式；
- 本地 Linear q/k/v/o/fc/proj 分项；
- 本地 Attention 分数和评测耗时；
- 官方得分和官方耗时；
- 相对上一版本的官方分数与耗时变化；
- 状态：`champion`、`accepted` 或 `rejected`；
- 结论、失败原因和下一步建议。

历史资料缺失时填写 `NA` 并说明来源限制，不推测精确值。

## 5. 总表

`solutions/README.md` 是人工维护的版本总表，按版本号升序记录：

| Version | Date | Topic | Local Linear | Local Attention | Local Time | Official Score | Official Time | Delta | Status | Directory |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |

其中 `Local Linear` 记录六类 Linear 分项的简单平均值，仅用于快速浏览；完整分项保存在对应 `result.md`。`Delta` 是相对前一官方版本的得分变化。目录链接使用相对路径。

## 6. 优化与归档流程

每轮严格遵循以下顺序：

1. 从根目录活跃 `solution.py` 开始，只引入一个主要机制变化。
2. 使用 `evaluator/real_data_eval.py` 运行真实 GPT 本地评测。
3. 记录七类本地分数、评测参数和耗时；本地评测失败则停止该候选。
4. 手动上传同一份根目录 `solution.py` 到官方评测。
5. 官方返回得分和耗时后，无论提升还是退化，都创建下一个归档版本。
6. 复制当次官方提交文件为归档目录内的 `solution.py`。
7. 校验活跃提交文件和归档副本 SHA256 完全一致。
8. 填写 `result.md`，并在 `solutions/README.md` 追加一行。
9. 若官方结果提升且值得作为新基线，状态标为 `champion`；否则标为 `accepted` 或 `rejected`。
10. 根目录 `solution.py` 继续作为下一轮唯一活跃代码，不从归档目录直接编辑。

本地评测命令为：

```powershell
.\.venv\Scripts\python evaluator\real_data_eval.py --solution solution.py --model <GPT-2模型目录>
```

## 7. 如何发现优化方向

每次只改变一个主要机制，使官方分数变化可归因。归档后按以下顺序分析：

1. 比较本地六类 Linear 分项，定位收益或退化集中在哪类算子。
2. 比较本地 Attention 分数，判断 Q/K/V 改动是否影响真实 softmax 输出。
3. 比较官方总分变化，判断本地分项趋势能否预测官方结果。
4. 比较官方耗时变化，计算每单位耗时带来的得分收益。
5. 本地与官方同向时，沿该机制做下一次单变量优化。
6. 本地改善但官方退化时，在 `result.md` 标记“本地代理失配”，优先调整评测数据或停止该机制。
7. 负结果必须归档，防止重复测试已经证伪的方向。

## 8. 最小验证与异常记录

不增加单元测试代码。每次归档只要求：

- `solution.py` 通过 Python 语法检查；
- 本地真实 GPT 评测完成并产生七类分数；
- 归档代码与官方提交文件 SHA256 一致；
- `result.md` 和总表中的官方得分、耗时、状态一致。

异常按下列方式记录：

- 本地评测失败：不提交官方，不建立正式版本目录。
- 官方退化或负分：正常归档，状态为 `rejected`。
- 官方超时：使用 `scoreNA_time300plus`，正文记录超时。
- 历史信息不完整：使用 `NA` 或显式近似标签，不制造精确值。
- 已归档算法代码不修改、不覆盖；记录补充仅允许修改 `result.md` 和总表。

## 9. 实施范围

本轮实施只完成：

1. 创建 `evaluator/` 并移动现有真实 GPT 评测代码和依赖声明。
2. 创建 `solutions/README.md`。
3. 从 Git 恢复并归档 v9 历史算法。
4. 归档当前 10250/127 秒版本。
5. 保持根目录 `solution.py` 为活跃副本。
6. 保留全部 `docs/superpowers/` 文档。

不恢复旧 `hif4_system/`、测试、报告、模拟实验脚本或其他已删除评测代码。
