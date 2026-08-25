# HiF4 本地评估优化系统设计

日期：2026-08-25  
状态：已由用户确认  
初始算法：`solution.py` v9，SHA256 `a6b8b858156164333d1d3ca25c6233b4845061f40a16d4cf74695ecdbb9041f7`

## 1. 目标

构建一个单机、文件式、可审计的 HiF4 评估与持续优化系统。第一阶段只评估人工提供的候选，并以现有 v9 为初始 Champion；第二阶段通过稳定接口接入参数搜索和受约束的代码候选生成，不改写评估核心。

系统只使用真实 PyTorch 路径，不提供 NumPy 仿真或无 Torch 回退。GPU 用于大规模精度筛选，CPU 用于最终晋级计时和比赛环境近似验证。

## 2. 设计原则

1. 真实算子优先：Linear 比较最终矩阵乘输出，Attention 比较完整 softmax Attention 输出。
2. 标准 HiF4 是独立基线，不复用候选代码中的内部实现。
3. 候选与 Champion 使用完全相同的数据进行逐 case 配对。
4. 精度与性能分轨：CUDA 结果不参与 CPU 运行时间门控。
5. 防止隐藏集过拟合：holdout 种子保密、调用限额、结果只暴露聚合信息。
6. 所有结果绑定源码 SHA256、配置、环境和随机种子承诺。
7. Champion 更新可回滚；失败或中断不能破坏当前 Champion。

## 3. 范围

### 3.1 第一阶段

- 实现真实 Torch NVFP4/HiF4 编解码和标准 HiF4 基线。
- 实现 Linear、Attention、causal/non-causal、MHA/GQA 的真实评分。
- 实现候选静态合规检查、运行时合法性检查和超时隔离。
- 实现 GPU 精度轨、CPU 精度/性能轨和 smoke/standard/soak 分层评估。
- 实现实验 campaign、报告、Champion 注册表、晋级和回滚。
- 将当前 v9 固化为初始 Champion，并生成首份 CPU/GPU smoke 报告。

### 3.2 第二阶段预留

- 参数网格与 Optuna 风格搜索适配器。
- 受约束的源码模板变体生成器。
- 候选去重、早停、资源预算和失败反馈。
- 优化器只能读取 dev 聚合指标；不能读取 holdout 原始 case 或密钥。

### 3.3 非目标

- 不构建多机任务队列、Web 服务或数据库平台。
- 不用 GPU 时间推断官方鲲鹏 CPU 时间。
- 不声称合成数据结果等同于官方排行榜结果。
- 不自动提交比赛代码。

## 4. 总体架构

```text
候选源码
   |
   v
静态合规检查 ----失败----> 拒绝并保存诊断
   |
   v
GPU smoke/standard 精度筛选
   |
   v
均值、P05、负分率、灾难性负分和置信区间门控
   |
   v
CPU 配对复评与运行时间门控
   |
   v
限额 holdout 验证
   |----------------------|
   v                      v
晋级 Champion          保留 Champion
归档旧版本             记录失败维度
```

模块边界：

- `hif4_system/evaluator/`：标准格式、合成数据、真实算子和逐 case 评分。
- `hif4_system/compliance/`：AST、接口、文件 I/O、非法状态和输出检查。
- `hif4_system/campaign/`：dev/holdout 种子、预算、清单和报告。
- `hif4_system/runner/`：独立子进程、设备选择、超时和分层执行。
- `hif4_system/gates/`：统计汇总、配对比较和晋级规则。
- `hif4_system/registry/`：Champion、候选快照、原子切换和回滚。
- `hif4_system/optimizer/`：第一阶段只定义候选生成协议。
- `tests/`：单元、集成和端到端验证。
- `cli.py`：稳定的用户入口。

## 5. 数据与执行流程

### 5.1 初始化

`init --champion solution_v9_champion.py` 校验 v9 哈希和接口，复制不可变快照到注册表，写入 Champion 指针。初始固化只声明合法性和可执行性，不声明相对提分。

### 5.2 精度轨

GPU 轨运行 smoke/standard/soak 精度评估。每个候选和 Champion 在相同 seed、case、mask 和 compute dtype 上运行。报告保留逐 case 内部数据，但用户默认看到聚合结果；holdout 报告不暴露真实 seed。

### 5.3 性能轨

CPU 轨在独立进程中预热后计时，记录候选量化时间、总墙钟时间、峰值内存和相对 Champion 比例。线程数和设备固定写入环境快照。CPU 结果是唯一参与性能晋级门控的数据。

### 5.4 晋级

候选必须依次通过：

1. 静态合规与六接口检查；
2. 输出格式、有限值、shape、dtype、state 深度和节点限制；
3. GPU dev 精度门控；
4. CPU 配对精度与运行时间门控；
5. 限额 holdout 门控。

通过后先写入完整候选快照、配置和报告，再以原子文件替换 Champion 指针。旧 Champion 保留在历史中，可直接回滚。

## 6. 评分与门控

单 case 分数：

```text
score = (mse_std - mse_player) / max(mse_std, epsilon)
```

默认聚合指标：总分、均值、中位数、P05、最小值、最差 10% 均值、负分率、低于 -10% 的灾难性负分率、胜/平/负、配对均值差和 95% bootstrap 区间。

Bootstrap 使用 Torch 随机采样实现，不依赖 NumPy。默认晋级门槛继承现有评估协议：候选平均分非负、负分率不高于 10%、灾难性负分率不高于 2%、最差 10% 均值不低于 -5%、相对 Champion 平均提升至少 0.2 个百分点、负分率不增加、CPU 量化时间不超过 Champion 的 1.2 倍。门槛可在 campaign 创建时配置，holdout 首次运行后锁定。

## 7. 数据集与设备策略

合成套件覆盖 Linear 的 balanced、hierarchy、outlier、heavy-tail、校准/测试漂移、相关、稀疏和均值偏移；Attention 覆盖 balanced、Q/K 动态范围失衡、K 均值偏移、heavy-tail、logits 饱和、V 离群、Q/K 相关、MHA/GQA、多 head、不同 head_dim/seq_len 和 causal/non-causal。

- CUDA：快速筛选和大规模精度实验，可配置 `fp32`、`bf16`。
- CPU：正确性复评、计时和最终晋级；默认 `fp32`、`bf16`。
- 设备差异超过容忍阈值时，候选不得自动晋级，报告标记为需要人工复核。

## 8. 隔离与错误处理

每次候选评估在独立子进程加载源码。父进程负责超时、退出码、stdout/stderr、结果协议和临时目录清理。以下情况立即拒绝候选并归档原因：导入失败、接口缺失、违规文件 I/O、Linear 校准路径疑似计算 `A @ W` 后拟合、非法 HiF4 参数、NaN/Inf、shape/dtype 不符、state 超限、设备不一致、超时或子进程崩溃。

评估报告采用临时文件完整写入后原子替换。campaign 清单只在报告落盘后追加记录。Holdout 调用预算在启动前预留，异常也记为已消耗，防止通过反复中断探测隐藏集。

## 9. 文件布局

```text
config/
  default.json
hif4_system/
  evaluator/
  compliance/
  campaign/
  runner/
  gates/
  registry/
  optimizer/
registry/
  champion.json
  versions/<version-id>/solution.py
campaigns/<campaign-id>/
  campaign.json
  .holdout_secret
  reports/
tests/
cli.py
requirements.txt
```

运行产物目录加入 `.gitignore`；算法源码、配置、测试和文档纳入版本控制。

## 10. CLI

```powershell
python cli.py init --champion solution_v9_champion.py
python cli.py evaluate solution.py --tier smoke --device cuda
python cli.py validate --candidate candidate.py --tier standard
python cli.py promote --candidate-id <id>
python cli.py history
python cli.py rollback
```

`validate` 负责完整 GPU→CPU→holdout 流程；`promote` 只接受已经通过完整门控且源码哈希未变化的候选。

## 11. 依赖与安装

建立项目本地虚拟环境，安装 CPU/CUDA 兼容的 PyTorch 和 pytest。pip 使用可配置的中国大陆镜像；PyTorch wheel 来源与 CUDA 版本单独配置，安装后必须记录 `torch.__version__`、CUDA runtime、GPU 型号和 CPU 信息。运行时不自动下载任何数据。

## 12. 测试策略

- 格式单测：NVFP4/HiF4 编解码、标准基线和边界编码。
- 评分单测：已知张量的 Linear/Attention MSE 和比赛公式。
- 接口单测：六个 API、状态限制和禁止文件 I/O。
- 统计单测：聚合、Torch bootstrap、配对 key 和门控边界。
- campaign 单测：种子承诺、holdout 限额、门槛锁定和异常消耗。
- registry 单测：初始化、原子晋级、哈希变化拒绝和回滚。
- 设备测试：CPU/CUDA 结果一致性及 CUDA 不可用时的明确失败。
- 端到端测试：v9 smoke 的 GPU 评估、CPU 评估和初始 Champion 固化。

## 13. 第一阶段验收标准

1. 全部单元与集成测试通过。
2. v9 通过静态合规、接口和输出合法性检查。
3. v9 完成 GPU smoke 精度评估和 CPU smoke 精度/性能评估。
4. Champion 注册表指向与原始 v9 相同 SHA256 的不可变快照。
5. 首份报告包含环境、配置、源码哈希、聚合指标和所有门控结果。
6. 系统不导入任何 NumPy 仿真模块，且没有无 Torch 自动回退。
7. CUDA 不可用时 GPU 命令明确失败，但 CPU 流程仍可独立运行。

## 14. 后续自动优化接口

`CandidateGenerator` 接收只读 Champion 快照、允许修改的参数空间和 dev 聚合反馈，输出新的候选源码及生成元数据。优化器不得调用 holdout；完整评估器负责去重、预算、门控和最终晋级。结构性代码生成与数值参数搜索使用同一候选协议，从而保持实验记录和安全边界一致。
