# HiF4 可持续评测与优化系统

这是一个 Torch-only 的本地评测系统。竞赛主代码始终是实际的
`solution.py`：系统在独立 Python 进程中加载它，使用独立实现的 NVFP4/标准
HiF4 基准计算分数，再把结果写成可审计的 JSON 报告。现阶段先使用 CPU
完成全链路验证；安装 CUDA 版 PyTorch 后，CUDA 只用于精度筛选，CPU 仍是
最终计时和晋级的权威轨。

## 当前基线

下载的 v9 代码已按原始字节固化到 `solution_v9_champion.py`，作为初始 Champion；当前待评估、后续可自动修改的竞赛主代码仍是 `solution.py`。
初始 Champion SHA256 为：

```text
a6b8b858156164333d1d3ca25c6233b4845061f40a16d4cf74695ecdbb9041f7
```

CPU smoke 套件包含 9 个配对 case。最近一次本地运行的平均分约为
`0.205917`；该数值仅用于记录当前环境的基线，候选晋级仍必须通过配置中
全部统计门槛。

## 环境安装

在中国大陆网络环境下，可以使用项目虚拟环境和镜像源安装 CPU 依赖：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
# CPU wheel（当前验证环境）
.\.venv\Scripts\python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch==2.13.0
```

当前验证版本为 Python 3.12.13、PyTorch `2.13.0+cpu`、pytest 9.1.1，未安
装 NumPy。PyTorch 可能打印一次“Failed to initialize NumPy”警告，这是
Torch 的可选互操作提示，不代表系统使用 NumPy 模拟。

## 常用命令

首次建立注册表时固化 v9 Champion（只执行一次）：

```powershell
.\.venv\Scripts\python cli.py init --champion solution_v9_champion.py --root .
```

评估当前候选 `solution.py` 的 CPU 权威 smoke：

```powershell
.\.venv\Scripts\python cli.py evaluate solution.py --tier smoke --device cpu --split dev --root .
```

报告位于 `reports/`，包含候选哈希、配置哈希、环境、设备权威性、计时、
case 明细、聚合分数和 seed commitment。holdout 报告只保存 commitment，
不保存原始 holdout seed。

安装兼容 CUDA 的 PyTorch 后可执行精度筛选：

```powershell
.\.venv\Scripts\python cli.py evaluate solution.py --tier standard --device cuda --split dev --root .
```

CUDA 不可用时命令会明确失败（退出码 4），不会自动退回其它后端。完整
候选验证的顺序是 GPU dev → CPU dev → holdout：

```powershell
.\.venv\Scripts\python cli.py validate --candidate candidate.py --tier standard --root .
```

验证成功后候选会进入 `registry/versions/`，再显式晋级：

```powershell
.\.venv\Scripts\python cli.py promote --candidate-id <id> --root .
.\.venv\Scripts\python cli.py history --root .
.\.venv\Scripts\python cli.py rollback --root .
```

晋级会重新计算源码哈希，并要求 GPU dev、CPU dev、holdout 三份报告均为
`passed` 且绑定同一候选哈希；任何源码改动、超时、崩溃或门槛失败都会被拒
绝。旧版本目录不会被删除，rollback 只改变 Champion 指针并追加历史事件。

## 评测与优化边界

- `hif4_system/formats.py` 和 `scoring.py` 是独立的标准基准，不调用候选的
  私有量化函数。
- `hif4_system/compliance.py` 检查六个竞赛接口、NumPy 导入、文件 I/O、
  非法状态和疑似用 `A @ W` 拟合校准的路径。
- 活跃评测路径只使用 Torch；仓库中保留的旧 NumPy 模拟器仅作历史参考，
  不会被当前 CLI 或 worker 导入。
- `hif4_system/runner.py` 使用隔离 worker，父进程负责超时、崩溃和结果协
  议；Torch tensor 不跨进程传递。
- `hif4_system/statistics.py` 的 bootstrap 只使用 Torch，不依赖 NumPy。
- `hif4_system/optimizer.py` 目前只提供安全的 `CandidateGenerator` 协议和
  不含 holdout 字段的 `DevFeedback`。后续自动修改算法时，生成器只能看到
  Champion 快照和 dev 反馈，候选仍必须通过完整评测、预算和晋级门禁。

退出码为：0 成功，2 参数/配置错误，3 合规拒绝，4 评测失败或超时，5 晋级
门禁失败。

运行全部测试：

```powershell
.\.venv\Scripts\python -m pytest -q
```
