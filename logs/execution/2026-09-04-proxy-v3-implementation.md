# proxy-v3 旁路评测与诊断工具实施记录

日期：2026-09-04
范围：新增并切换稳定评测入口；未修改 `evaluator/official_eval.py`、根 `solution.py` 或官方候选。

## 产物

- `evaluator/proxy_v3_eval.py`：六个确定性平衡 shard。每个 shard 选 4 个纵深层；Linear 覆盖
  全部 7 个 role、validation/test 配对窗口（28 state、56 case），Attention 覆盖同层的双窗口（8 case）。
- `evaluator/proxy_v3_analyze.py`：父子 exact-case 配对、mean/median/L1/tail、role/layer/shape/
  split/length 热点、组件方向、focus/control、OOD gap、API 分段耗时和官方时间资格判断。
- `evaluator/proxy_v3_runner.py`：顺序跑 shard；两侧分别累计统计，连续两个非正向 shard 提前停止，
  `--reuse-existing` 只重放已有 JSON；同一 runner 进程只加载一次 multi-GB dense cache，并保存每 shard
  的详细分析 JSON/Markdown。
- `evaluator/eval_system.py` / `evaluator/eval.py`：稳定用户入口、官方成绩 manifest 和 `--official-audit`，
  将官方成绩作为独立观测做同 cohort 顺序与完整性审计。
- `docs/proxy-v3.md`：命令、缓存语义和判读边界。

## 关键边界

v3 读取现有 `proxy-v2` dense cache，不捕获新数据、不改变六个 API 的调用协议。校准 artifact
按 solution SHA、训练数据 hash、校准窗口、state keys、场景、设备和 PyTorch 版本校验；命中缓存
会跳过校准，并把 `calibration_timing_measured=false`，因此 shard/cache 秒数不进入官方时间模型。
分析器只允许当前本地趋势门禁 `delta_mean > 0 && L1 < 0.02`、OOD `|delta gap| <= 0.01`；
不拟合或换算官方绝对分。

## 验证

```text
51 passed in 7.36s
py_compile: passed
evaluator/official_eval.py: no diff
```

`git diff --check` 对本次新增/修改文件通过；工作区中其他 session 的既有文件未处理。
