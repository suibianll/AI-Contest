# HiF4 量化评测项目

本项目用于华为 2026 算法赛题的 HiF4 量化方案开发。输入是 NVFP4
（E2M1 carrier + block scale）数据，输出是 HiF4 表示；目标是在保持
计算格式合法的前提下，使反量化结果尽可能接近 NVFP4 参考结果。评测覆盖
Linear 层和 Attention 投影两条路径，得分是相对标准 HiF4 转换的 MSE 改善。

当前根目录的 `solution.py` 已切换为已知最高分的 `youxilee/hif4` v2.0
方案（官方分数 15000+）。它来自提交
`6abbf36e1208ac7afffd2ba3e2e4a8aa9a1f3757`，并与归档版本保持相同 SHA256。

## 项目结构

```text
solution.py                         当前唯一活跃、可直接提交的算法文件
evaluator/
  nvfp4_sim.py                      权威 NVFP4 编码/解码模拟器
  real_data_eval.py                 真实 GPT-2 评测器
  requirements.txt                  评测依赖
solutions/
  README.md                         版本、分数和耗时总表
  YYYYMMDD_vNNN_.../solution.py     不可变的历史算法源码
  YYYYMMDD_vNNN_.../result.md       该版本的来源、结果和结论
docs/superpowers/                   设计方案、实施计划和归档流程文档
```

`solution.py` 是活跃文件；`solutions/` 只保存已经提交或明确记录过的版本，
不会作为运行时依赖。`docs/superpowers/` 保留完整设计过程，不参与比赛提交。

## 算法概要

当前 v2.0 方案包含以下核心阶段：

1. 使用官方 NVFP4 scale 规则和 E2M1 carrier 还原浮点参考值。
2. 对每个 Linear 层收集校准激活，搜索 SmoothQuant 缩放、通道置换和
   权重/激活误差重要性；宽层使用更细的 alpha 网格。
3. 在校准通过 safety gate 时尝试 block-diagonal Hadamard 变换，枚举
   block size 4/8/16 和确定性 sign seed；不满足改善阈值时回退到对角路径。
4. 对权重、激活及 Q/K/V 的高误差 HiF4 block 做有预算的层级 scale
   refinement，并用绝对误差排序、二次统计量和边界扩展提高收益。
5. Attention 路径按真实 Q/K/V 张量校准，支持 MHA/GQA 的 head 分组，
   最终通过相同的动态量化接口生成可提交的 HiF4 参数。

所有状态都是 CPU 上的普通张量/标量，动态量化阶段只依赖校准得到的状态，
不依赖评测器内部实现。

## 评测方法

评测器加载真实 GPT-2 权重和文本前向激活，抓取每层的 Q/K/V、Attention
投影和 FFN 输入。每个样本同时生成：

- NVFP4 反量化结果：参考值；
- 标准 HiF4 结果：基线；
- 当前 `solution.py` 的校准/动态量化结果：候选值。

每个 Linear 或 Attention 样本使用同一公式：

```text
score = (MSE(standard, reference) - MSE(candidate, reference))
        / MSE(standard, reference)
```

然后对层和测试批次取平均。当前评测器与远程 hif4 项目的核心口径一致，
但支持通过 `--solution` 和 `--model` 指定候选文件和模型路径。

## 环境与运行

建议使用项目内虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r evaluator\requirements.txt
```

运行默认的 GPT-2 12 层、2 个校准批次和 2 个测试批次：

```powershell
.\.venv\Scripts\python evaluator\real_data_eval.py `
  --solution solution.py --model gpt2
```

首次使用 `gpt2` 时需要准备模型；也可以传入已下载的本地模型目录：

```powershell
.\.venv\Scripts\python evaluator\real_data_eval.py `
  --solution solution.py --model D:\models\gpt2
```

开发阶段可降低规模以快速比较算法方向：

```powershell
.\.venv\Scripts\python evaluator\real_data_eval.py `
  --layers 1 --seq 16 --calib 1 --test 1
```

参数含义：`--layers` 为层数，`--seq` 为序列长度，`--calib`/`--test` 为
校准/测试批次数，`--mode` 可选 `amax6`、`amax4`、`pow2`，`--kv-heads`
用于 GQA 烟测。评测输出包含 q/k/v/o/fc/proj 六项 Linear 分数及 Attention
分数；运行时间请在命令外层记录。

## 版本归档流程

每次优化只修改根目录 `solution.py`，完成本地评测后再提交官方评测。收到
官方分数和耗时后：

1. 新建 `solutions/YYYYMMDD_vNNN_topic_scoreSCORE_timeTIME/`。
2. 将实际提交的根 `solution.py` 原样复制进去。
3. 在同目录 `result.md` 记录源码哈希、单一改动、局部分项分数、官方分数、
   耗时、结论和下一步方向。
4. 在 `solutions/README.md` 追加一行；未知值使用 `NA`，近似值使用明确的
   `plus` 或 `time300plus` 标记。
5. 核对根文件和归档文件 SHA256 相同后再提交 Git。

当前版本记录：

- v000：旧 v9 基线，约 9000+；
- v001：原活跃基线，10250 分、127 秒；
- v002：`youxilee/hif4` v2.0，15000+，已设为当前活跃 solution。

虚拟环境 `.venv/`、Python 缓存和其他本地产物已由 `.gitignore` 排除，
不会进入算法归档或比赛提交。
