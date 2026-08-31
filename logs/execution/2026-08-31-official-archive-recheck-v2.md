# 官方结果归档算法统一复评（sampled-means-v2）

日期：2026-08-31  
模型：Qwen2.5-0.5B；cache：`seq=128 / calib=2 / test=4 / layers=all`；设备：CUDA。  
活动样例：`112 Linear + 96 Attention`，同一批样例同时计算两个均值和 API 时间。

## 结果

`Linear mean` 和 `Attention mean` 的单 case gain 为
\(g=(\mathrm{MSE}_{std}-\mathrm{MSE}_{player})/\mathrm{MSE}_{std}\)。`v2 panel` 只是
\(250\bar g_L+200\bar g_A\) 的本地形状量，不是官方分数。

| 版本 | 官方分数 / 时间 | Linear mean | Attention mean | v2 panel | API |
|---|---:|---:|---:|---:|---:|
| v001 | 10250 / 127s | 0.374042 | 0.504234 | 194.357 | 28.797s |
| v002 | 15313 / 137s | 0.504169 | 0.558166 | 237.675 | 153.483s（CPU alg） |
| v013 | 15799 / 144s | 0.506761 | 0.654816 | 257.654 | 49.873s |
| v024 / C21 | 16043 / 173.8s | 0.521622 | 0.654816 | 261.369 | 57.147s |
| c21 | 14437 / 166.6s | 0.456105 | 0.654816 | 244.989 | 55.250s |
| c38 | 14092 / 170.57s | 0.415997 | 0.654816 | 234.962 | 72.146s |
| c39 / v031 | 21864 / 161.3s | 0.451339 | 0.654816 | 243.798 | 69.755s |
| c40 | 14432 / 216.667s | 0.443476 | 0.654816 | 241.832 | 128.556s |
| c41b / v034 | 21864 / 159.4s | 0.451339 | 0.654816 | 243.798 | 69.537s |
| c47b / v051 | 22451 / 234s | 0.449899 | 0.654816 | 243.438 | 145.918s |
| c66 / v066 | 22557 / 217.2s | 0.444620 | 0.657497 | 242.654 | 144.738s |
| v72 / C74 | 22662 / 226s | 0.444847 | 0.657497 | 242.711 | 154.793s |
| v74 / C75 | 22750 / 239.387s | 0.452721 | 0.657497 | 244.680 | 165.299s |
| v84 / C84 | 16517 / 252.563s | 0.489389 | 0.739172 | 270.182 | 239.910s |

### 官方失败/超时归档（同口径复评）

| 版本 | 官方裁决 | Linear mean | Attention mean | v2 panel | API |
|---|---|---:|---:|---:|---:|
| v98 | timeout（>300s） | 0.516969 | 0.842022 | 297.647 | 169.000s |
| v100 PAWV fixed | 原始 WA；修复线仍 timeout | 0.516969 | 0.842024 | 297.647 | 176.158s |
| v107 PAWV fixed | Attention WA（原始） | 0.526490 | 0.842024 | 300.027 | 187.127s |
| v121 PAWV fixed | timeout（>300s） | 0.531834 | 0.842024 | 301.363 | 1571.187s |

## 复评证据

- v001–c66：[JSON](../../artifacts/real_model_suite/official-anchors-sampled-means-v2.json)，[详细报告](2026-08-31-official-anchors-sampled-means-v2.md)。
- v002 CPU 设备复测：[JSON](../../artifacts/real_model_suite/official-anchor-v002-sampled-means-v2-cpu.json)，[报告](2026-08-31-official-anchor-v002-sampled-means-v2-cpu.md)。CUDA 复测失败原因为归档实现把 CPU tensor 与 CUDA tensor 混用；不是评测器把失败当成零分。
- v72/v74：[JSON](../../artifacts/real_model_suite/official-anchors-v72-v74-sampled-means-v2.json)，[报告](2026-08-31-official-anchors-v72-v74-sampled-means-v2.md)。
- v98：[JSON](../../artifacts/real_model_suite/v98-official-timeout-sampled-means-v2.json)，[报告](2026-08-31-v98-official-timeout-sampled-means-v2.md)。
- v100：[JSON](../../artifacts/real_model_suite/v100-pawv-fixed-sampled-means-v2.json)，[报告](2026-08-31-v100-pawv-fixed-sampled-means-v2.md)。
- v107：[JSON](../../artifacts/real_model_suite/v107-pawv-fixed-sampled-means-v2.json)，[报告](2026-08-31-v107-pawv-fixed-sampled-means-v2.md)。
- v121：[JSON](../../artifacts/real_model_suite/v121-pawv-fixed-sampled-means-v2.json)，[报告](2026-08-31-v121-pawv-fixed-sampled-means-v2.md)。

## 解释边界

1. v001–v074 的官方总分主要属于旧权重，v84 是当前确认的新权重；不能用本表的本地 panel 反推两套官方绝对分数。
2. v2 对官方比例做“最近整数、禁止复制”的实现：`112/96=1.1667`，官方 `250/200=1.25`；在 24 层、4 窗口、7 Linear role 的 cache 上，4 个 Linear 层（112 case）比 5 个层（140 case）更接近目标。
3. v84 在当前 v2 下 Attention 和总 panel 最高，但这只说明本地排序；是否满足官方格式和 300s 端到端限制仍需官方测试。
4. v84 的旧 v1 CPU 记录（422.615s）不参与本表；v84/v98/v100 的源码级长度预算、
   O(T^2d) 放大和官方通过/超时差异见
   [v84/v98/v100 运行时分析](2026-08-31-v84-v98-v100-runtime-analysis.md)。
