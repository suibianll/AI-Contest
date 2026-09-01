# 当前状态：官方最高分更新为 17816，Linear 新框架成立

更新：2026-09-01。

## 0. 最新官方进展

用户确认：一套新的 Linear 算法已经在官方评测上产生明确提升，当前最高总分达到
**17816**，比旧官方最高 v86 的 16744 高 **1072 分**。

新算法不是 v138–v145 的局部调参延续，而是完整的：

- SmoothQuant + Permutation + block Hadamard 等价变换搜索；
- 变换后完整协方差驱动的 Weight GPTQ；
- 部署权重输出 Gram 驱动的 Activation GPTQ；
- quadratic AdaRound、E6M2 offset、data-driven refinement 和 edge extension；
- proxy 与 e2e 混合选择，窄层执行联合变换搜索。

新提交的版本号、源码 SHA、官方运行时间以及对应 Attention 配置尚未提供，因此当前只把
**17816 记为用户确认的官方精度锚点**，不伪造时间结论，也暂不建立源码归档。完整理论分析和
后续执行顺序见
[`活动计划`](superpowers/plans/2026-09-01-hif4-linear-0.8-under-300s-plan.md)。

## 1. 版本结论

- **旧仓库内官方基线：v86，16744 分 / 222.7s。** 新的用户确认最高分为 17816，但源码与
  官方时间尚未同步到仓库。
- 根目录 [`solution.py`](../solution.py) 当前仍是 v140 实验代码，SHA256
  `52521F1B996BF67641C22A90132ED7A7BCA477976D8A05BEC411CC9E04AA7C90`；它只比 v138
  本地 Linear 高 `0.00003513`，没有官方结果，不再称为“当前最优”。
- v138/v139 虽在官方 `<300s` 内通过，但只有 `15715/15716`，比 v86 低约 1029 分；
  v138–v145 这条“压缩 Attention 后继续叠 Linear 局部模块”的路线已经失败并关闭。
- 下一阶段不从 v140 继续调参；拿到 17816 源码后，以它作为新 Linear 父版本，并冻结该提交
  对应的 Attention 做单变量归因。

## 2. 评测口径

本地统一使用 [`evaluator/official_eval.py`](../evaluator/official_eval.py) 的
`official-shape-v1`：Qwen2.5-0.5B、24 层、250 Linear + 200 Attention case、Attention
calibration 长度 `[10,128,512,1024,1024]`。主字段是 `linear_mean`、`attention_mean`、
六 API `api_total_seconds` 和 `wall_seconds`。

该协议使用固定公开模型、WikiText 窗口和固定 case 选择，而官方使用隐藏数据与未公开的新权重。
因此它只用于算法诊断和同机耗时记录，不能继续作为官方排序器，也不能把本地时间换算为鲲鹏时间。

## 3. 当前结果表

| 版本 | Linear mean | Attention mean | API(s) | Wall(s) | 官方结果 | 状态 |
|---|---:|---:|---:|---:|---:|---|
| v84 | 0.406668 | 0.718107 | 279.191 | 300.848 | 16517 / 252.563s | 官方通过 |
| **v86** | **0.406668** | **0.719696** | **299.302** | 321.996 | **16744 / 222.7s** | **官方基线** |
| v128 | 0.465655 | 0.837789 | 310.732 | 332.557 | timeout | 失败 |
| v129 | 0.465655 | 0.836579 | 248.363 | 270.606 | timeout | 失败 |
| v130 | 0.471837 | 0.836579 | 295.437 | 317.607 | timeout | 失败 |
| v131 | 0.473131 | 0.836579 | 294.835 | 317.708 | timeout | 失败 |
| v134 | 0.507320 | 0.834256 | 289.042/289.832 | 312.315/313.181 | 未提交 | 本地研究父版本 |
| v135 | 0.500812 | 0.834256 | 290.823 | 313.365 | 未提交 | REJECTED |
| v136 | 0.500132 | 0.834256 | 287.816 | 310.472 | 未提交 | REJECTED |
| v137 | 0.507163 | 0.834256 | 296.755 | 319.306 | 未提交 | REJECTED |
| v138 | 0.507320 | 0.715942 | 187.935–192.996 | 210.855–216.324 | **15715 / 208s** | 官方通过但路线失败 |
| v139 | 0.507278 | 0.715942 | 193.389 | 217.196 | **15716 / 202s** | 官方通过但路线失败 |
| v140 | 0.507355 | 0.715942 | 205.365 | 229.337 | 未提交 | 本地微增益，不再作为最优 |
| v141–v145 | 0.281760–0.506256 | 0.715942 | 204.681–211.460 | 228.127–234.842 | 未提交 | REJECTED；源码已清理 |

完整原始数据见 [`artifacts/official_eval/`](../artifacts/official_eval/)，官方回传记录见
[`logs/execution/`](../logs/execution/)。

## 4. v86 与 v138 的关键反转

当前本地协议下：

| 对比 | Linear 差值 | Attention 差值 | 本地 API 差值 | 官方时间差值 | 官方分数差值 |
|---|---:|---:|---:|---:|---:|
| v138 − v86 | +0.100651 | **−0.003754** | −111.37s | −14.7s | **−1029** |

v138 的本地等权显示为 `27001.827`，明显高于 v86 的 `24560.627`，但官方排序完全相反。
这证明当前 Linear 本地增益没有在隐藏评测上转化，而 v86 的 Attention 表示对官方分数非常重要。

v138 也不是原样 v86 Attention：它缩小统计 token、候选 block/seed 和输出终选范围，并删除了
v86 的部分 scale-aware/output-aware 机制。此前把它描述为“v86 级静态 Attention”是不准确的。

## 5. 已关闭的算法路线

以下方向不再通过调参数继续：

- v138 的缩减 Attention shortlist；
- v139 连续 output-aware gain；
- v140 局部 reciprocal pair/ROAB-P2；
- v141–v145 非对称选列 BDLR、锚点冻结和阻尼变体；
- v128–v131 动态 Q/K Gram、PAWV 和随序列放大的 Attention 搜索；
- 增加 alpha、offset、sweep、block 数、阻尼、角度或候选槽位的局部扫描。

这些路线要么官方超时，要么官方分数低于 v86，要么只有固定本地 panel 上的 `10^-5–10^-4`
级差值，不能支撑继续投入。

## 6. 新的理论算法主线

新的唯一活动计划是
[`2026-09-01-hif4-linear-0.8-under-300s-plan.md`](superpowers/plans/2026-09-01-hif4-linear-0.8-under-300s-plan.md)。核心顺序为：

1. 先对 17816 框架做结构消融，拆分等价变换、Weight GPTQ、Activation GPTQ 和 HiF4
   层级 refine 的真实贡献。
2. 将 GPTQ 改成以完整 64-channel HiF4 block 为原子的 block-Schur 求解。
3. 用低秩+块对角 Hessian/Woodbury 形式保留跨块 Activation GPTQ，同时降低动态复杂度。
4. 固定两轮联合 Weight–Activation 优化，直接控制
   `E_AW^T + AE_W^T - E_AE_W^T`。
5. 用相同部署 stage 数的学习型 butterfly 扩展固定 Hadamard，而不叠加第二套变换。
6. 让 scale/lv2/lv3/mantissa 作为完整层级状态参与 block oracle。
7. 在这套已被官方验证的新框架上重新测 Joint oracle，判断 `linear_mean=0.8` 可达性。

## 7. 归档现状与待整理项

已完成：

- v128–v131 的 `result.md` 和目录名均标记 `TIMEOUT`；
- v135–v137 的 `result.md` 和目录名均标记 `REJECTED`；
- v132/v133 已补齐 `RETAINED / LOCAL HISTORICAL PARENT` 结果文件；
- v134 标记为 `RETAINED / LOCAL RESEARCH PARENT`，不代表官方可提交；
- v140 ROAB-P2 改为 `REJECTED / LOCAL-ONLY`，归档目录标记 `_rejected`；
- 空的重复 v140 curvature 目录已删除；
- v141–v145 失败源码目录删除，逐次 JSON/日志保留。

2026-09-01 归档整理（区分新旧评测分数体系）：

- 旧协议产物移入 `*/archive/legacy-*-20260901/`：`artifacts/oracle_dashboard/` →
  `artifacts/archive/legacy-oracle-dashboard-20260901/`；`artifacts/jdrq/` →
  `artifacts/archive/legacy-jdrq-diagnostics-20260901/`；`logs/candidates/` →
  `logs/archive/legacy-candidates-20260901/`；根目录 `solution_b0_tmp.py` →
  `logs/archive/legacy-root-files-20260901/`；
- v031 目录名由旧面板 `official14613` 更正为官方 `21864`（同步更新
  `evaluator/official_eval.py` ARCHIVE_MANIFEST 与 `solutions/README.md`）；
- v125 screen 记录目录更名为 `20260831_v125b_c1c-block8-screen-positive`，版本号恢复全局唯一；
- 清理 12 个 pytest 临时目录残留（`.pytest-tmp-*`、`.tmp-pytest`、`.tmp_pytest`、
  `artifacts/pytest_tmp*`，均为 gitignore 覆盖的本地临时目录）；
- 根 README 新增「分数体系与归档对照」章节，三套分数体系（官方旧权重 / 官方新权重 /
  本地协议分）明确隔离，`solutions/` 目录名中的数字字段口径已在索引中说明。

以后微参数实验不分配版本号；只有新数学算法、官方提交或一个代表性失败实现进入
`solutions/`，目录名和 `result.md` 必须同时标注 `retained/rejected/timeout`。
