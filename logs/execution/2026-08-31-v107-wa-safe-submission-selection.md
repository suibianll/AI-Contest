# v107 官方 Attention WA 与安全提交候选选择

日期：2026-08-31  
> **已被后续官方结果取代：** 用户于 2026-08-31 确认 v100 同样出现 Attention
> `wrong answer`，且不是 timeout。本文件保留当时的选择依据，不再产生提交建议。
> 新裁决见 [`v100 官方 WA 边界审计`](2026-08-31-v100-official-wa-boundary-audit.md)。
> 2026-08-31 结果纠错：**v107 并非 timeout**，其官方结果仍为 Attention `wrong answer`
> （与 v100 同类）；真正在新限制下官方判为 timeout 的是 v98（B1 GQRB margin 版），
> 见 [`v98 官方 timeout`](2026-08-31-v98-official-timeout.md)。

原结论：官方复测首选 `v100`，保留 `v066` 作为已确认通过的控制组；不提交 v107。

## 1. 与已有官方成绩版本的对照

已有同口径官方锚点：v31 `21864 / 161.3s`、v51 `22451 / 234s`、v66 `22557 / 217.2s`。这些版本均完成过官方评分；v107 的官方 Attention 场景返回 `wrong answer`。

本地专项审计已经确认：

- v100、v106、v107 的 Attention 校准及 Q/K/V 动态函数没有差异；v106/v107 的 Attention state 与输出逐 tensor 相等。
- v107 与 v31、v51、归档外部实现使用同一 Qwen cache、同一 NVFP4 codec 时，state、五字段参数、shape、CPU/finite 校验均为 0 failures；v107 Attention MSE mean `0.00169248`，低于 v31/v51 的 `0.00382519`。
- 因此没有证据支持“v107 Attention 数值输出本身损坏”。

v107 相对 v106 唯一关键新增是 Linear Global Activation-LRH 及完整 `deployment_gram=W_q^T W_q`。如果官方 runner 同时保留多个 Linear state，Qwen 形状累计额外状态约 2.6 GiB；v107 本地 API `481.04s` 也超过最新官方 `300s` 限制（2026-08-31 修订），但官方实际判定仍为 Attention `wrong answer`（用户确认非 timeout）；本地超时风险仅作历史解释保留。

隐藏输入 shape、提交包与归档源不一致仍不能完全排除，但优先级低于资源原因。

## 2. 候选比较

| 候选 | Qwen full panel | API | 对 300s 余量 | `deployment_gram` | 官方证据 | 裁决 |
|---|---:|---:|---:|---|---|---|
| v66 | 旧代理 Qwen `350.152420` | 旧代理 `151.91s` | 大 | 无 | `22557 / 217.2s` 已通过 | 官方控制组 |
| clean stable parent | `293.755106` | `382.15s` | `-82.15s` | 无 | 未提交 | 更保守备用 |
| **v100** | **`293.797301`** | **`392.42s`** | **`-92.42s`** | **无** | 未提交；合约通过 | **首选** |
| v106 | `294.272633` | `412.65s` | `-112.65s` | 无 | 未提交；合约通过 | 分数更高但时间风险偏大 |
| v107 | `295.157057` | `481.04s` | `-181.04s` | 约 2.6 GiB 累计风险 | 官方 Attention WA（非 timeout） | 不提交 |

v100 相比 v106 只损失 `0.475332` panel（约 `0.16%`），换取 `20.23s` API 余量；相比 clean stable parent 只多 `0.042195` panel、增加 `10.27s`。按项目当前以六 API 累计时间为门禁的口径，v100 是精度与通过概率的最佳折中。

## 3. v66/v100 同场复核

命令使用 Qwen layer 0 固定缓存、`seq=128 / calib=2 / test=4 / amax6 / CPU`，在一次 evaluator 运行中同时加载注册的官方锚点 c66 与归档 v100：

| 候选 | layer-0 panel | Linear panel | Attention panel | API |
|---|---:|---:|---:|---:|
| c66 | `314.731294` | `131.580914` | `183.150380` | `25.196s` |
| v100 | **`336.037091`** | **`150.767630`** | **`185.269461`** | **`18.559s`** |

两者均通过当前 evaluator 的状态、HiF4 参数和时间有效性预筛。该单层结果只用于接口/方向复核，不替代 v100 已归档的 24 层结果，也不预测官方绝对分数。

## 4. 提交文件

- 首选源码：`solutions/20260830_v100_b2-pawv-diagonly-active_score293.797301_time392s/solution.py`
- 规范 LF SHA256：`617482cee04ff9514a8d41226b651336e4b8b86692673308e835de1091693eba`
- 根 `solution.py` 仍保持 v125 precision-only 研究版本；本次没有改动 active 算法。

若 v100 官方仍返回同类 Attention WA，应立即提交 v66 原文件作为控制；若 v66 通过而 v100 失败，再把问题收敛到 clean Attention 隐藏 shape/状态契约，而不是 Linear `deployment_gram`。
