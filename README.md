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

评测协议已升级到 schema v2。修正后的 CPU standard/dev 共 96 个配对 case：
当前候选总分 `56.840941`、均分 `0.592093`，v9 Champion 总分
`52.661592`、均分 `0.548558`；配对平均增益为 `+0.043535`，95%
聚类 bootstrap 区间 `[0.036403, 0.051993]`，胜/平/负为 `92/0/4`。
候选负分率为 `1.04%`，v9 为 `3.13%`。旧 schema v1 报告包含近似标准
HiF4 基线和双掩码加权，不能与 v2 绝对分直接比较。以上仍是合成代理，
不等同于官方榜单；已知官方结果是 v9 约 9000+、当前候选 10250、耗时 127 秒。

schema v2 固定使用 FP32 输出计算和 non-causal Attention；标准 HiF4 在每个
8-value group 上穷举 8 种合法 lv2/lv3 组合，E6M2 使用 255 个有限合法值。
causal 只作为显式配置的鲁棒性实验；配置变化会创建独立 campaign，避免
旧 holdout 与新协议混用。

## 环境安装

在中国大陆网络环境下，可以使用项目虚拟环境和镜像源安装依赖：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
# CPU wheel（无 CUDA 时）
.\.venv\Scripts\python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch==2.13.0
```

当前验证版本为 Python 3.12.13、PyTorch `2.6.0+cu124`、pytest 9.1.1，
GPU 为 NVIDIA GeForce RTX 3060 Ti。环境中可能因 Torch 依赖存在 NumPy，
但活跃评测、统计和数据生成路径均不导入 NumPy，也不使用 NumPy 模拟；
竞赛算法与评测后端仍是 Torch-only。

## 常用命令

首次建立注册表时固化 v9 Champion（只执行一次）：

```powershell
.\.venv\Scripts\python cli.py init --champion solution_v9_champion.py --root .
```

评估当前候选 `solution.py` 的 CPU 权威 smoke：

```powershell
.\.venv\Scripts\python cli.py evaluate solution.py --tier smoke --device cpu --split dev --root .
```


没有 CUDA 或在日常迭代阶段，可让候选与 Champion 在同一批数据上执行
CPU 配对审计：

```powershell
.\.venv\Scripts\python cli.py audit --candidate solution.py --incumbent solution_v9_champion.py --tier standard --split dev --root .
```

报告位于 `reports/`，包含候选哈希、配置哈希、环境、设备权威性、计时、
聚合分数和 seed commitment。dev 保存 case 明细；holdout 不保存原始 seed
或逐 case 结果，只返回候选/Champion 聚合统计和配对门禁。

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
  私有量化函数；标准层级指数使用逐组精确枚举。
- `hif4_system/compliance.py` 检查六个竞赛接口、NumPy 导入、文件 I/O、
  E6M2/mantissa 合法性、非法状态和疑似用 `A @ W` 拟合校准的路径。状态
  字符串/键受 4096 UTF-8 字节限制，每个在线样本获得独立深拷贝。
- standard 轨共享 300 秒总超时；候选量化计时不包含评测器的 state 深拷贝
  与输出校验开销。
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
