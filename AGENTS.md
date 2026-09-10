# HiF4 竞赛执行规则

> 整理：2026-09-10。这里只维护稳定执行约束；计划进度、逐版本结果和历史解释不在此追加。

## 1. 范围与唯一入口

- 只完成用户明确提出的任务；只改必要文件，不扩展评测、重构或清理。普通分析、文档修改只做静态核验。
- 用户最新明确指令优先。仓库规则顺序：本文件 → [4B 测试指引](docs/4b-panel-testing-guide.md) → [计划入口](docs/superpowers/plans/README.md)指定的唯一活动总计划及执行附录 → workbench 状态。低优先级文件不能新增门禁。
- 优化开始时读计划入口和指定计划，再按需读[当前状态](docs/current-solution-status.md)、[版本索引](solutions/README.md)、目标源码与评测器。只在计划入口登记当前计划，其他文档链接到入口，不复制活动卡片快照。
- 事实依据：绑定源码 SHA 的官方回传 → 当前确认记录 → 原始 result/log/JSON。推测、未提交、本地失败、官方失败必须区分；历史日志和 workpackages 不提供下一步指令。
- 历史判读：旧权重分数不可与当前分数混排；提交配额、0.5B/OOD/跨模型门、本地时间预测、L1/负向损失门均已退役。按需查[09-07](docs/stale-information-inventory-2026-09-07.md)、[09-05](docs/stale-information-inventory-2026-09-05.md)、[09-04](docs/stale-information-inventory-2026-09-04.md)、[09-02](docs/stale-information-inventory-2026-09-02.md)，后续修订与当前规则优先；不要求每次重读全部历史。
- 冲突按上述优先级处理，只修正本次相关入口；不得把整库历史同步当作继续执行的前置任务。保护其他会话未提交文件。

## 2. 完整父与官方裁决

| 用途 | 版本 | 官方分数 / 时间 | SHA256 |
|---|---|---|---|
| 当前完整父 | v202 Linear + v195 Attention | 18053 / 281s | `56DC805D6E5A3AEF896DB8021045740292735725D688B48E3D4393E55EFCB2BD` |
| 回退根 | current Linear + v195 Attention | 18053 / 289s | `839ADB1E617C3115C6B55071A34B281C5DB0FF2AA070ADBBC71FD1549E761D7F` |

- 每个正式候选从当前最高分完整根构建，保留六 API；一个机制、一个固定配置、固定候选数量。侧隔离只作诊断，不建立侧父或侧晋级线，不机械组合历史侧结果。
- 官方分数提高且时间 `<300s` 才晋级，先绑定计分源码 SHA；本地正向不替换根。组合从已确认的较高分完整父重建，并由完整官方结果裁决。
- 官方提交次数无限制，不重复相同 SHA 或逐位等价 A/B。不同算法 ±1～4 分不能当作随机噪声。
- 本地分数、Δmean、L1、holdout、拟合改善幅度与 api_seconds 只作机制诊断，不设提交或晋级阈值，不换算官方分数或时间。不得以本地时间预测决定必然超时或设置 280/296s 等回退门。
- 已有本地 REJECTED 保留原记录，但不写成官方证伪；不自动重跑历史候选。无可达变化或最终逐位恢复父的候选不提交。

## 3. 算法与实现边界

- 正式提交为根 `solution.py`，单文件自包含、脱离仓库可导入六 API：`hif4_calibration_and_quantize_weight`、`hif4_dynamic_quantize_activation`、`hif4_calibration_attention`、`hif4_dynamic_quantize_q`、`hif4_dynamic_quantize_k`、`hif4_dynamic_quantize_v`。
- 不从其他 Python 文件、仓库/归档路径或 importlib 加载提交实现；编码器、解码器、E6M2、层级与状态逻辑留在模块内，通过 `evaluator/reference_hif4.py` 合法状态检查。独立导入验证脚本不受此提交实现限制。
- 动态 API 只执行校准编译的规则，不带入校准搜索、完整矩阵求逆或未限制的 Python 候选循环。Attention 动态 API 另禁止 Gram contraction 和候选循环；复杂计算放校准阶段。
- Linear 优化实际输出误差 `XW^T - Q(XR)Q(WR^{-T})^T`；变换保持连续乘积不变，Gram/Hessian 与最终部署坐标一致。operand MSE、连续代理下降不能替代真实硬编码输出证据。
- Linear 使用全部 Qwen3.5-4B 校准数据做 A@W 低维拟合，不拆 fit/select，不以泛化、独立窗口负向否决探索。Attention 参数学习使用 calibration folds，按固定多折规则聚合，学习/选择/独立验证职责明确；holdout 不参与选参或提交否决。
- 记录 attempted/accepted、硬码变化及最终输出变化，验证真实分支可达；死分支、no-op、单个近似求解器失败不证明整族饱和。
- 新机制先按需查[已关闭机制与证据边界](docs/closed-mechanism-evidence.md)，保留具体实现及其参数邻域关闭约束。不扫 threshold/seed/alpha/offset/fold/coverage/候选数量，不增加模型/layer/role 专属路由。修复已证实错误或改变目标/求解算法须说明实质差异，不能只换名称。
- 保留 C76.4；不重开 V 码分配或改变官方解码格式。用户成功机制锚点只支持研究方向，源码未绑定前不能替换父或证明本地实现有效。

## 4. 测试与归档

1. 固定完整父与 SHA；已有同 evaluator/协议/cache/panel/device/SHA 结果用 `--reuse-existing`，不重跑。
2. 检查六 API、合法 state、finite、reachability、关闭机制 control 与未改子系统 control；训练类候选做随机形状 contract smoke。
3. 官方前跑目标侧 4B shard0 排除接口错误。活动算法开发卡要求的目标侧六 shard 完成一次，用于真实输出诊断；数值不成为提交门。官方正向后只补必要 interaction audit，不重复已有目标侧全量。
4. 合法、可达、非等价的单一代表候选归档并交官方裁决。正式归档前完成脱离仓库单文件导入检查。
5. 保存源码、配置、父子 SHA、JSON/report、local proxy、API total、wall time 与官方状态。官方未知记 `unregistered/NA`，不填本地秒数；ERROR、NO_EFFECT、本地 REJECTED、官方 REJECTED、官方 TIMEOUT 明确区分。拒绝候选目录含 `rejected`，无实质算法/复杂度变化不分配版本号。
6. 官方回传及时绑定计分 SHA 和归档 SHA，再更新状态、版本索引、计划进度及执行记录；未复测源码不能继承结果。

日常入口 `evaluator/eval.py`（eval-v3），使用 `.venv\Scripts\python.exe` CUDA 环境。目标侧参数、缓存口径和命令见 [4B 指引](docs/4b-panel-testing-guide.md)。不新增 0.5B、OOD、GPT-2/opt 或 fresh-default 计时；兼容后端 `official_eval.py` 的输出不混排。

本地全六 shard 为 336 Linear + 72 Attention，`gain=1-MSE_PLAYER/MSE_STD` 为同 NVFP4 输入下实际 case 等权指标；不能由面板、官方样例数或侧分推断隐藏权重、榜首差距归属或官方分数。

## 5. 缓存、并发与 Git

- 校准缓存 identity 包含 solution SHA/pack/device；命中时校准 API 秒数和调用数为 0，不跨缓存状态比较 api_seconds。dense cache 与校准产物缓存必须区分，cohort 不匹配不静默合并。
- 定期清理 `artifacts/official_eval/cache/proxy-v3-calibration/` 的死候选缓存，保留当前根、回退根和正在运行/待复用候选。用 `workbench/cache_cleanup/prune_calibration_cache.py`，先 dry run、核对绝对目录后 `--apply`；多会话必须加 `--min-age-hours 2`。
- **绝不删除 `artifacts/official_eval/cache/qwen3.5-4b-proxy-v2.pt` dense 主缓存。** 本次任务无关的缓存不顺手清理。
- 多会话 GPU 评测串行，启动前检查占用；独立 workbench/artifact，版本号登记和根切换串行，不能覆盖其他执行者的源码或状态。
- 不覆盖原始 JSON/report/执行日志，修正另写日志。实质代码或状态更新后 `git diff --check`、只提交本次文件、push 并核验 status；不提交 ignored cache、`.codegraph/`、临时目录或无关改动。
- 新实验细节写计划/日志；本文件只更新稳定约束和官方父表。分析与本次规则整理见[推进瓶颈审计](docs/optimization-stall-analysis-2026-09-10.md)。
