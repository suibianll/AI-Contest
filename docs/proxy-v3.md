# eval-v3 分片评测与诊断工具

`evaluator/eval.py` 现在是本地评测系统的稳定入口。它默认使用 `proxy-v3` 分片协议：一次进程
加载 dense cache，按层均衡分片，校准状态按 solution SHA 和输入身份持久化，并把每个分片的
热点/失败原因写入 JSON 和 Markdown。原 `evaluator/official_eval.py` 保留为 proxy-v2
兼容/reference 后端，不再作为用户日常评测入口，也不会被新系统静默改写。

## 运行方式

先按现有流程生成 proxy-v2 dense cache，然后使用替换后的入口运行全部 shard：

```powershell
.venv\Scripts\python.exe evaluator\eval.py `
  --solution solution.py --name candidate --scenario both `
  --cache artifacts\official_eval\cache\qwen2.5-0.5b-proxy-v2.pt `
  --calibration-cache-mode auto `
  --output-dir artifacts\proxy_v3\system
```

输入 dense cache 由 `--cache` 指定，默认读取
`artifacts/official_eval/cache/qwen2.5-0.5b-proxy-v2.pt`（OOD 使用 `--ood` 读取 OOD cache）。每个
shard 覆盖 4 个分散层：

- Linear：全部 7 个 role，validation/test 两个配对窗口，共 28 个 state、56 个 case；
- Attention：同 4 层、两个配对窗口，共 8 个 case；
- 六个 shard 的 state 并集覆盖 24 层，单个 state 只出现一次。

校准产物位于 `artifacts/official_eval/cache/proxy-v3-calibration/`，按 solution SHA、场景、state keys、输入
hash、设备和 PyTorch 版本校验。`auto` 命中后会跳过 calibration；结果会明确标记
`calibration_cache_hit=true`、`calibration_timing_measured=false`，因此不能用于官方时间预测。

父子批量筛选可使用稳定入口（底层顺序 runner 仍保留）：

```powershell
.venv\Scripts\python.exe evaluator\eval.py `
  --baseline-solution solutions\parent\solution.py `
  --solution solution.py --name candidate --scenario linear `
  --output-dir artifacts\proxy_v3\candidate-run
```

入口会在一次进程中保持 dense cache，避免每个 shard 重读 11GB 快照。`--reuse-existing` 只接受
协议、场景、分片、源路径和 SHA 全部匹配的 JSON；否则自动重跑，避免“猜测式”复用旧结果。

## 官方成绩重评审

对成绩清单中所有有官方分数的版本运行同一 proxy-v3 协议：

```powershell
.venv\Scripts\python.exe evaluator\eval.py `
  --official-audit --cohort new-weight --scenario both `
  --shards 0,1,2,3,4,5 `
  --output-dir artifacts\proxy_v3\official-audit
```

清单由 `evaluator/official_results_v3.py` 维护，显式区分 `old-weight`/`new-weight`，并记录源码是否
可复现。审计输出 `audit.json` / `audit.md` 包含每个版本的完整 case 覆盖、有限值、重复 identity、
源 SHA、官方时间是否越过 300 秒，以及同 cohort 的官方/本地 pairwise concordance。后者只用于
检查代理排序是否合理；本地 gain 不参与官方绝对分数换算，报告会明确写出
`official_score_equivalent=false`。

若要复核历史 old-weight 成绩，必须显式指定 `--cohort old-weight`。当前 dense cache 是
new-weight，系统会把每条结果标为 `official_cache_cohort_mismatch`；这类输出只能用于接口/稳定性
回归，不能用于算法排序或官方分数推断。若已有分片文件，追加 `--reuse-existing` 可只重建汇总报告。

## 诊断方式

单独分析已有 JSON：

```powershell
.venv\Scripts\python.exe evaluator\proxy_v3_analyze.py `
  --baseline artifacts\proxy_v3\parent.json `
  --candidate artifacts\proxy_v3\candidate.json `
  --mechanism-type analytic `
  --focus-linear-roles q,k
```

输出包含：

- 同 case 的 `delta_mean`、median、L1、最差 20% tail、正/负/零计数；
- role、role family、layer、shape、split、length 热点和最差 case；
- 若输入结果有 decomposition 字段，则显示 W-only/A-only、Q/K/V、interaction、logit/probability
  的分量方向；
- API 时间排序。只有“新鲜 default panel、未命中校准缓存”才套用已有官方时间模型；shard/缓存秒数
  永远不换算官方分数或官方时间；
- OOD 配对时输出 `delta(in-ood)`，并按现行 `0.01` 门禁给出原因。

`--focus-linear-roles` 只做目标 role 与未修改 control 的配对检查；它不会新增候选路由或改变
evaluator 的调用图。v3 的任何正向结果仍只是本地筛选证据，官方分数必须通过正式评测确认。
