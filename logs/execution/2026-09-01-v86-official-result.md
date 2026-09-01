# v86 官方结果（新评分权重，16744 / 222.7s）

> 日期：2026-09-01
> 来源：用户回传官方评测结果。
> 归档：`solutions/20260830_v086_c86-attn-block-final_scoreNA_timeNA/`
> 源码 SHA256：**raw `E7A16D6991DBB70A593FBE87D0C5D1D8FD38F801665354A01FFAF2F0A96F03CD`**
> （与归档 `result.md` 记录一致；该文件含 8696 处 CRLF，LF 归一值为
> `CDE19F98ED3639659AF563B5AD0A9954CE4BA8C04FDF89568D1966CB3735A21E`，不要混用两种口径）。

## 1. 官方结果

| 候选 | 官方分数 | 官方时间 | 裁决 | 相对 v84 |
|---|---:|---:|---|---|
| v84 / C84 | 16517 | 252.563 s | pass（新权重） | 基准 |
| **v86 / C86** | **16744** | **222.7 s** | **pass（新权重）** | **分数 `+227`，时间 `−29.863 s`** |

v86 是目前**新评分权重下分数最高、且运行最快的官方通过点**，距 300 s 限制还有
`77.3 s` 余量（v84 为 `47.437 s`）。

v86 = v084 + C86。C86 的唯一改动是 Attention 侧：Q/K 共享 head-local
block-Hadamard 候选（block 4/8/16，seed 0），用最终 offset/refinement 点阵
打分器排序，只把胜出的整数对与静态符号写入 Q/K state；Linear 侧完全不动。

## 2. 本地 official-shape-v1 复测（250 Linear + 200 Attention）

| 候选 | Linear mean | Attention mean | API total(s) | Wall(s) | 官方时间 |
|---|---:|---:|---:|---:|---:|
| v084 | 0.406668 | 0.718107 | 279.191 | 300.848 | 252.563 s |
| v086（旧受干扰记录） | 0.406668 | 0.719696 | 462.239 | 501.257 | 222.7 s |
| **v086（2026-09-01 空闲重测）** | **0.406668** | **0.719696** | **299.302** | **321.996** | **222.7 s** |

精度符合预期：Linear 逐位相同（C86 不触碰 Linear），Attention `+0.001589`。

## 3. 时间悖论：本地慢 65%，官方反而快 12%

本地 API `279.191 → 462.239`（**+65.6%**），官方却 `252.563 → 222.7`
（**−11.8%**）。官方/本地比值：

- v84：`252.563 / 279.191 = 0.905`（API）、`0.839`（Wall）
- v86：`222.7 / 462.239 = 0.482`（API）、`0.444`（Wall）

**同一算法族内比值跨度约 1.9 倍**。这说明即便纠掉了形状错配（详见
[v100 超时根因分析](2026-08-31-v100-official-timeout-analysis.md) 的
250/200 vs 224/32 错配表），本地时间仍不能可靠预测官方时间。

## 4. 逐 API 分解：Linear 侧异常膨胀指向测量漂移

| API | v084 | v086 | 差值 | C86 是否应影响 |
|---|---:|---:|---:|---|
| `hif4_calibration_and_quantize_weight` | 194.236 | 325.770 | **+131.53 (+67.7%)** | **否（纯 Linear）** |
| `hif4_dynamic_quantize_activation` | 35.854 | 45.720 | +9.87 (+27.5%) | **否（纯 Linear）** |
| `hif4_calibration_attention` | 43.817 | 83.334 | +39.52 (+90.2%) | 是 |
| `hif4_dynamic_quantize_q` | 2.246 | 2.823 | +0.58 | 是 |
| `hif4_dynamic_quantize_k` | 1.595 | 2.417 | +0.82 | 是 |
| `hif4_dynamic_quantize_v` | 1.442 | 2.175 | +0.73 | 是 |

C86 只改 Attention，但**不受其影响的 Linear 两项却涨了 67.7% 与 27.5%**。
在算法上无法解释，最可能是本机测量漂移（GPU/CPU 争用、批处理位置、热降频）。
因此 `462.239 s` 应视为**含漂移的上界**，不是 v86 的真实本地成本。

## 5. 2026-09-01 空闲重测

在无其他 Python/评测进程占用的机器状态下，使用只读
`qwen2.5-0.5b-official-shape-v1.pt` cache 完整执行 168 次 Weight、250 次 Dynamic
Activation、24 次 Attention calibration 和各 200 次 Q/K/V。新结果为：

```powershell
.venv\Scripts\python.exe -u evaluator\official_eval.py `
  --solution solutions\20260830_v086_c86-attn-block-final_scoreNA_timeNA\solution.py `
  --name v086-idle-rerun-20260901 `
  --cache artifacts\official_eval\cache\qwen2.5-0.5b-official-shape-v1.pt `
  --cache-mode read --algorithm-device cuda `
  --output artifacts\official_eval\v086-idle-rerun-20260901-official-shape-v1.json `
  --report logs\official_eval\v086-idle-rerun-20260901-official-shape-v1.md
```

- Linear mean：`0.40666821449674884`
- Attention mean：`0.7196960689329899`
- API total：`299.30157260096166 s`（本地 API `<300` indicator=True）
- Wall：`321.9955865999218 s`（仅诊断字段）

旧 `462.239/501.257s` 是并发/漂移污染的上界，保留在原始报告中以保证审计链完整，
但本次空闲重测才是当前 v86 的本地结果。完整原始输出见
[`v086 idle rerun JSON`](../official_eval/v086-idle-rerun-20260901-official-shape-v1.json)
和 [`v086 idle rerun report`](../official_eval/v086-idle-rerun-20260901-official-shape-v1.md)。

## 6. 结论与纪律影响

1. **v86 取代 v84 成为主锚点**：官方分数更高、时间更短、余量更大
   （`77.3 s` vs `47.437 s`）。后续提交线应以 v86 为父版本参考。
2. **Attention 改动本身不是超时根因**：C86 是 Attention 侧改动，官方时间反而
   下降。真正触发超时的是 PAWV/GQRB 的 per-seq_len 动态分组 + Python 循环。
3. **本地时间不能作为官方时间的预测器**，只能作为同机同批次的 A/B 相对比较。
   任何"本地 ≤150 s 安全"类预算规则只能用于**同一批次内**的相互比较，
   不能外推到官方 300 s。
4. 需要官方时间判断时，唯一可靠手段是提交官方评测。
