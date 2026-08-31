# HiF4 solutions archive（轻量索引）

更新时间：2026-08-31。归档目录中的 `solution.py` 是只读证据；当前可提交代码只在仓库根目录
[`solution.py`](../solution.py)。本文件只保留入口、官方锚点和最近一次统一复评结果，逐次实验细节放在
[`logs/execution/`](../logs/execution/)。

## 当前口径

- 活动本地 profile：`sampled-means-v2`。
- 同一批样例同时计算均值和时间，Qwen2.5-0.5B cache 固定为 `seq=128 / calib=2 / test=4`。
- 官方面板是 250 Linear + 200 Attention；当前 cache 不复制 case，能实现的最近整数构成为
  **112 Linear + 96 Attention**（比例 `1.1667`，官方比例 `1.25`）。Linear 取 4 个分层层，Attention
  覆盖 24 层；因此均值和耗时不会再使用两套样例。
- 主指标只有 `Linear mean`、`Attention mean`；`v2 panel = 250·Linear mean + 200·Attention mean`
  仅是固定面板下的本地排序量，不是官方绝对分数。
- 本地 API/Wall 时间只可在相同硬件、cache、profile 下做 A/B；不能把本地秒数直接等同官方秒数。
- 官方分数采用过不同权重：v001–v074 主要是旧权重，v84 是目前确认的新权重结果，绝对值不能混排。

v84 曾有一条旧的 sampled-means-v1 CPU 记录（224L/32A，API 422.615s）；当前表格中的
v84 使用与 v98/v100 完全相同的 sampled-means-v2 CUDA 配置。两条记录均保留，但旧记录
只用于历史复现，不能用于当前时间排名。详见
[v84/v98/v100 运行时差异分析](../logs/execution/2026-08-31-v84-v98-v100-runtime-analysis.md)。

## 当前根版本

根文件 SHA256 与 v127 归档完全一致：
`f15e112c7e832d019ee83d707acd9d72fef121a306e4cc3b50dbbc2cbb574924`。

| source | Linear mean | Attention mean | v2 panel（诊断） | Local API | Local Wall | 官方状态 |
|---|---:|---:|---:|---:|---:|---|
| root / v127 | **0.522453** | **0.842024** | **299.018** | 177.039s | 180.430s | 尚未提交 |

完整 JSON：[root-v127-sampled-means-v2.json](../artifacts/real_model_suite/root-v127-sampled-means-v2.json)，
报告：[root-v127-sampled-means-v2.md](../logs/execution/2026-08-31-root-v127-sampled-means-v2.md)。

## 有官方记录的归档版本：统一 v2 复评

以下所有本地列均来自同一 Qwen cache 和 `112/96` 样例。官方列是平台/用户确认的历史事实；
`v2 panel` 不是官方分数换算。

| 版本 | 官方分数 / 时间 | v2 Linear | v2 Attention | v2 panel | Local API | 备注 |
|---|---:|---:|---:|---:|---:|---|
| v001 | 10250 / 127s | 0.374042 | 0.504234 | 194.357 | 28.797s | 历史基线 |
| v002 | 15313 / 137s | 0.504169 | 0.558166 | 237.675 | 153.483s | CUDA 设备混用；CPU 算法设备复测通过 |
| v013 | 15799 / 144s | 0.506761 | 0.654816 | 257.654 | 49.873s | 历史结果 |
| v024 / C21 | 16043 / 173.8s | 0.521622 | 0.654816 | 261.369 | 57.147s | 后续合规审查不采用其监督路径 |
| c21 | 14437 / 166.6s | 0.456105 | 0.654816 | 244.989 | 55.250s | 合规控制组 |
| c38 | 14092 / 170.57s | 0.415997 | 0.654816 | 234.962 | 72.146s | 官方回退案例 |
| c39 / v031 | 21864 / 161.3s | 0.451339 | 0.654816 | 243.798 | 69.755s | 旧权重锚点 |
| c40 | 14432 / 216.667s | 0.443476 | 0.654816 | 241.832 | 128.556s | 官方回退案例 |
| c41b / v034 | 21864 / 159.4s | 0.451339 | 0.654816 | 243.798 | 69.537s | 旧权重锚点 |
| c47b / v051 | 22451 / 234s | 0.449899 | 0.654816 | 243.438 | 145.918s | 旧权重锚点 |
| c66 / v066 | 22557 / 217.2s | 0.444620 | 0.657497 | 242.654 | 144.738s | 旧权重锚点 |
| v72 / C74 | 22662 / 226s | 0.444847 | 0.657497 | 242.711 | 154.793s | 旧权重；官方通过 |
| v74 / C75 | **22750 / 239.387s** | 0.452721 | 0.657497 | 244.680 | 165.299s | **旧权重官方通过基线** |
| v84 / C84 | 16517 / 252.563s | 0.489389 | **0.739172** | **270.182** | 239.910s（独立复测 234.361s） | **新权重官方通过；<300s** |

### 有官方失败/超时裁决的归档（不参与通过版本排名）

这些版本没有可用的官方分数，但官方已给出 WA 或 timeout；以下是同一 v2 口径下的本地复评，
用于定位精度/时间原因，不能把本地 panel 当成官方成绩。

| 版本 | 官方裁决 | v2 Linear | v2 Attention | v2 panel | Local API |
|---|---|---:|---:|---:|---:|
| v98 | timeout（>300s） | 0.516969 | 0.842022 | 297.647 | 169.000s |
| v100 PAWV fixed | 原始 WA；修复线仍 timeout | 0.516969 | 0.842024 | 297.647 | 176.158s |
| v107 PAWV fixed | Attention WA（原始） | 0.526490 | 0.842024 | 300.027 | 187.127s |
| v121 PAWV fixed | timeout（>300s） | **0.531834** | 0.842024 | **301.363** | **1571.187s** |

复评文件：

- [统一复评摘要](../logs/execution/2026-08-31-official-archive-recheck-v2.md)
- [v001–c66 v2 JSON](../artifacts/real_model_suite/official-anchors-sampled-means-v2.json) / [报告](../logs/execution/2026-08-31-official-anchors-sampled-means-v2.md)
- [v002 CPU 复评 JSON](../artifacts/real_model_suite/official-anchor-v002-sampled-means-v2-cpu.json) / [报告](../logs/execution/2026-08-31-official-anchor-v002-sampled-means-v2-cpu.md)
- [v72/v74 v2 JSON](../artifacts/real_model_suite/official-anchors-v72-v74-sampled-means-v2.json) / [报告](../logs/execution/2026-08-31-official-anchors-v72-v74-sampled-means-v2.md)
- [v84 v2 独立复测 JSON](../artifacts/real_model_suite/v84-sampled-means-v2-cuda.json) / [报告](../logs/execution/2026-08-31-v84-sampled-means-v2-cuda.md)
- [v98/v100/v107/v121 v2 JSON 与报告](../logs/execution/2026-08-31-official-archive-recheck-v2.md)（各自 JSON/报告文件名按候选前缀保存）

外部 `youxilee/hif4` 的 `24153 / 239s` 只作为不可导入的参考，未伪装成本地候选。

## 归档与复评规则

1. 新实验先修改根 `solution.py`，通过本地合规检查后再复制到新的 `solutions/YYYYMMDD_vNNN_.../`；归档源码不覆盖。
2. 未有官方结果的目录使用 `scoreNA_timeNA`，本地均值/时间不得写进 Official Score/Time。
3. 官方返回后只追加提交 SHA、官方分数、官方时间和日期；若评分权重变化，必须标注 revision，不能与旧权重混排。
4. 所有候选必须用活动 profile v2 复评；旧 `sampled-means-v1 (224/32)` JSON 仅供历史复现，不参与当前排名。
5. 详细实验记录放 `logs/execution/`；本索引只更新当前根、官方锚点和最新统一复评入口。

## 常用命令

```powershell
# 统一复评官方锚点（Qwen、固定 cache、v2）
.venv\Scripts\python.exe evaluator\real_model_suite.py `
  --models qwen2.5-0.5b --cache-mode read `
  --evaluation-profile sampled-means-v2 `
  --candidates v001 v013 v024 c21 c38 c39 c40 c41b c47b c66 v72 v74 v84 `
  --output artifacts\real_model_suite\official-archive-recheck-v2.json `
  --report logs\execution\official-archive-recheck-v2.md
```

v002 需单独使用 `--algorithm-device cpu`（其历史实现混用了 CPU/CUDA tensor）；不要为了让
表格完整而吞掉异常或把失败结果当成零分。
