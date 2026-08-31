# v107 与 v31、v51、外部实现的 Attention 输出对照

日期：2026-08-31  
目的：针对官方 v107 `Attention / wrong answer` 回传，使用同一输入、同一 NVFP4 codec、同一 Attention API，对比 v31、v51、外部实现和 v107 的 state 与 Q/K/V 输出，排除输出契约或数值污染。

## 1. 对照范围与协议

- 对照源码：
  - v31：`solutions/20260828_v031_c39-fw-official14613_time159.2s/solution.py`
  - v51：`solutions/20260829_v051_c47b-grouping-threshold005_scoreNA_timeNA/solution.py`
  - 外部：本地归档的 `solutions/20260826_v002_youxilee-hif4_score15000plus_timeNA/solution.py`
  - v107：`solutions/20260830_v107_l3-global-lrh-precision-parent_score295.157057_time481s/solution.py`
- 所有候选均按赛事规定的公开 API 调用；使用评测器同一份 `nvfp4_sim.nvfp4_encode`、同一份 `reference_hif4.validate_state` 和 `validate_hif4_params`。
- Qwen 真实缓存：`artifacts/real_model_suite/cache/qwen2.5-0.5b__seq128__calib2__test4__layersall__schema1.pt`，24 层，2 个 calibration window、4 个 test window，`q_heads=14, kv_heads=2, head_dim=64`，CPU。
- 另外运行固定的 synthetic GQA/MHA contract case；下文主表使用 24 层真实 Qwen cache，attention MSE 是同一 NVFP4 Q/K/V 的非 causal GQA 前向结果与参考前向的均方误差。

注意：本地可复核的“外部实现”是归档 v002 源码，不应误称为最新外部 v2.7 源码。最新 v2.7 的提交/结果已在独立审计中记录，但当前仓库没有对应的完整源文件；因此本报告的外部逐输出数字只代表归档 v002。

## 2. 24 层 × 4 test window 结果

每个候选共 96 个 test batch、288 个 Q/K/V 输出；每个候选应有 72 个 Q/K/V state 对象。`contract` 包含 state 校验、HiF4 参数逻辑 shape、输出 shape、CPU/finite 检查。

| 候选 | state failures | contract failures | 非有限输出 | Q/K/V shape failures | attention MSE mean | median | max case | max abs output diff |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v31 | 0/72 | 0/96 | 0/288 | 0/288 | 0.003825193693 | 0.003321634489 | 0.01451323181 | 3.216447592 |
| v51 | 0/72 | 0/96 | 0/288 | 0/288 | 0.003825193693 | 0.003321634489 | 0.01451323181 | 3.216447592 |
| 外部 v002（归档） | 0/72 | 0/96 | 0/288 | 0/288 | 0.005298729964 | 0.003531764378 | 0.02867037989 | 3.378484726 |
| v107 | 0/72 | 0/96 | 0/288 | 0/288 | **0.001692483761** | **0.001360747672** | **0.005817570724** | **2.42027235** |

结论很明确：在同输入同 codec 的公开 API 对照中，没有发现 v107 的 Attention 输出字段缺失、shape 错误、dtype/device 错误、NaN/Inf 或 state 不合规；而且 v107 的 Attention 数值误差均值、median、最坏 case 和最大绝对输出差都优于这三个基准。

## 3. 逐版本输出差异

完整 Q/K/V 输出 hash 的逐层相等数：

| 比较 | 完全相等的层数 | 解释 |
|---|---:|---|
| v31 vs v51 | 24/24 | 两个版本的 Attention 部署输出逐层逐位相同；v51 的 Linear 改动没有污染 Attention。 |
| v31 vs 外部 v002 | 12/24 | 外部实现有 12 层输出不同；不是统一的 codec/API shape 错误，而是实现选择差异。 |
| v31 vs v107 | 0/24 | v107 有意新增/改变了 Linear 侧 Global Activation-LRH/Gram 状态；Attention 输出不同是预期现象，不等于错误。 |

抽查真实 Qwen 层的 Attention MSE：

| 层 | v31 | v51 | 外部 v002 | v107 |
|---:|---:|---:|---:|---:|
| 0 | 0.000011989658 | 0.000011989658 | 0.00006170484 | **0.000010855354** |
| 11 | 0.0061549381 | 0.0061549381 | 0.0061549381 | **0.0017654899** |
| 23 | 0.0064214191 | 0.0064214191 | 0.0096449436 | **0.0040288536** |

典型 Qwen state 结构也符合预期：v31/v51/外部的 q、k、v state 通常分别为 3、3、2 个 tensor；v107 为 q、k、v 分别 4、4、1 个 tensor，其中新增的 `gram64`、multiplier、rotation sign 和 importance 只服务于 Linear/部署状态。所有这些 state 均通过评测器校验并保持 CPU、finite、无梯度。

## 4. 对官方 `wrong answer` 的判断

这组实验不能访问官方隐藏输入，因此不能证明隐藏数据一定正确；但它排除了“v107 Attention 五字段输出本身必然坏掉”这一假设。当前更可信的待验证原因是：

1. 官方打包/提交文件与本地归档源不一致，或隐藏 wrapper 的输入 shape/role 路由不同；
2. v107 Linear 新增的 `deployment_gram` 在 Qwen 形状上约占 2.6 GiB，导致提交端内存/进程状态异常，错误被官方归类为 `wrong answer`；
3. v107 API 约 481 s，超过 300 s 门槛，官方（2026-08-31 确认）在最新 300s 限制下判为 timeout；
4. 仍需使用官方同包环境，对 v106（已通过 Attention 的时间 parent）和 v107 做逐 API、逐 case 的状态/输入 shape/首个错误阶段对照，才能定位隐藏契约问题。

因此，在没有官方隐藏样本之前，不应回退 Attention 算法或把 v107 的本地 Attention 输出判为错误；应优先做 v106/v107 同包复测，并隔离 `deployment_gram` 的内存影响。

## 5. 可复现命令与证据

本次脚本使用评测器的真实 Qwen cache，逐版本加载历史 `solution.py`，调用公开 Attention API，随后运行 `validate_state`、`validate_hif4_params` 和同一非 causal GQA reference forward。结果摘要见本文第 2 节；v107 更早的随机 MHA/GQA 与 5 场景合规矩阵见 [`2026-08-31-v107-attention-contract-audit.md`](2026-08-31-v107-attention-contract-audit.md)。

