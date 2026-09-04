# eval-v3 官方成绩重评审（2026-09-04）

## 目的

将用户入口从旧的一次性 proxy-v2 评测切换到 `evaluator/eval.py`，并用统一的
proxy-v3 分片协议重跑仓库中有官方分数的源码。官方分数只作为独立观测值做同 cohort
pairwise 审计，未用于拟合、校正或换算本地 gain。

## 执行配置

- 入口：`evaluator/eval.py`；底层 `proxy_v3_eval.py`；诊断 `proxy_v3_analyze.py`。
- dense cache：`artifacts/official_eval/cache/qwen2.5-0.5b-proxy-v2.pt`，一次进程加载。
- 场景：`both`；shard `0,1,2,3,4,5`；每 shard 4 个分散层。
- 每个版本总覆盖：Linear `336`（24 层 × 7 role × validation/test），Attention `48`
  （24 层 × 两个配对窗口），共 `384` case；这与官方隐藏 panel 不等价。
- 校准缓存按 solution SHA、state keys、输入 hash、设备和 PyTorch 版本隔离。

## 结果

### new-weight

成绩清单中的 `29` 个 new-weight 版本全部完成 `6/6` shard：

- `29/29` 源码评测成功；`29/29` case 覆盖完整；有限值检查通过；跨 shard identity 无重复。
- 同 cohort pairwise：`324` concordant、`80` inverted、`2` tied，非平局一致率
  `324/(324+80)=0.801980`。该数值仅说明新代理的排序诊断情况，不能作为官方分数预测精度。
- 其中有 `15` 个近零本地差异反转（`|Δlocal| < 0.002`，报告保留前 12 个样例）；
  v188 相对 v180/v182/v183/v186 的本地微小正向与官方 −2/−3/−3/−4 不一致，
  正好验证近零信号不能作为晋级依据。
- 唯一合理性标记：`v159` 的官方分数绑定 SHA 无法从当前归档确认（源码仍可运行）。
- `v162` 标准 codec 在新协议下 local gain 为 `0.0`，作为零点/codec 端到端 sanity check 通过。

完整报告：[`audit.md`](../../artifacts/proxy_v3/official-audit/audit.md)；机器结果：
[`audit.json`](../../artifacts/proxy_v3/official-audit/audit.json)。

### old-weight

成绩清单中的 `12` 个 old-weight 版本均尝试完成 `6/6` shard：

- `11/12` 完整成功；`v002` 六个 shard 均因源码遗留的 CUDA/CPU 混用报错，未伪造结果。
- 所有 old-weight 输出都标记 `official_cache_cohort_mismatch`：当前 dense cache 是
  new-weight，故这些数值只能证明接口/稳定性，不能用于跨 cohort 算法排序。
- 在可成功的 11 个版本上，pairwise `31` concordant、`23` inverted、`1` tied，非平局一致率
  `0.574074`；该结果进一步确认历史 cohort 不应与当前分数混排。

完整报告：[`audit.md`](../../artifacts/proxy_v3/official-audit-oldweight/audit.md)；机器结果：
[`audit.json`](../../artifacts/proxy_v3/official-audit-oldweight/audit.json)。

## 修复与判读

首轮复用旧校准产物时发现 PyTorch `weights_only` 无法反序列化 `TorchVersion`。v3 现将
`torch.__version__` 规范化为字符串，并在 `auto` 模式把损坏/旧 allow-list 产物按 stale cache
重校准、原子覆盖；`read` 模式仍会显式失败。修复后 v162 复跑通过。

本次审计没有把 v3 API 秒数与官方时间相减或换算：v3 的 case 数、校准窗口和调用图是为快速
诊断设计的，官方时间仍只能用 fresh default 的既有分解模型单独预测。所有报告均写有
`official_score_equivalent=false`。
