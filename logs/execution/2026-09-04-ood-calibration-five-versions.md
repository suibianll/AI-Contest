# 2026-09-04 OOD 标定重跑（v158/v160/v168/v171/v176 + v182 汇总）

## 目的

落实 `docs/official-local-fitting-analysis-2026-09-04.md` §6 的标定建议：在已知官方裁决的
版本上验证 `gain_in − gain_ood` 的判别力。不消耗官方配额。

## 运行

- OOD 五跑：`--ood`，逐版本显式 `--output/--report`（`ood-v<NNN>.json/.md`），
  OOD NVFP4 缓存全命中，单跑 1m50s~5m35s，总 16m38s。
- in-dist 补跑两跑：v158 both-default（此前只有旧 compact）、v176 attention-only-default
  （此前只有 compact）；in-dist NVFP4 缓存命中。
- 归档 SHA 全部与 solutions/*/result.md 台账核对一致：
  v158 `18F9DE03...8277`、v160 `33B1D061...680D`、v168 `5988AE47...AC79`、
  v171 `4469B85B...A844`、v176 `DFA69838...5CC2`。

## 标定表（default 口径，同机；OOD = 3 域 15 窗口面板）

| 版本 | 官方裁决 | 侧 | gain_in | gain_ood | gap |
|---|---|---|---:|---:|---:|
| v158 | 16861 RETAINED（+） | linear | 0.448180 | 0.419648 | **+0.028532** |
| v158 | 〃 | attention | 0.735752 | 0.719135 | +0.016617 |
| v160 | 17532（+，历史父） | linear | 0.633526 | 0.617210 | +0.016317 |
| v160 | 〃 | attention | 0.742354 | 0.720900 | +0.021453 |
| v168 | 14005（+，A1 step +60） | attention | 0.741474 | 0.719470 | **+0.022004** |
| v171 | 13657（A4 **−348** REJECTED） | attention | 0.740808 | 0.719199 | +0.021610 |
| v176 | 13964（C1 **−41** REJECTED） | attention | 0.737024 | 0.721814 | +0.015210 |
| v182 | 17598（+，完整官方父） | linear | 0.636609 | 0.620706 | +0.015903 |
| v182 | 〃 | attention | 0.741829 | 0.721239 | +0.020591 |

standard Linear 侧（v168/v171/v176）在两个面板上均精确 **0.000000** → 评测链路确定性
再次得到验证（PLAYER 与 STD 逐位一致）。

## 结论（诚实判读）

1. **gap 家族带稳定**：Attention 侧全部版本落在 +0.015~+0.022；当前 Linear 谱系
   （v160/v182）稳定在 +0.016 左右。父版本 v182 基线位于家族带内，无异常。
2. **小幅官方负在 OOD 上无特征**：v171（−348）Δgap = −0.0004（与父不可区分）；
   v176（−41）Δgap = −0.0068，但其负向来自 in-dist 增益更低（0.7370 vs 0.7415），
   OOD 增益反而略高于父（0.7218 vs 0.7195）——方向与"过拟合暴露"相反。
   → **OOD 判据不能预测/解释机制质量类的小幅官方负**。
3. **OOD 判据的定位收窄**：它只针对 §6 设计目标中的**分布拟合型大失败**
   （v140/v155/v156 的 −1163 级："本地 +0.06、官方 −1163"）。该级别失败不在本标定集内
   （早于 proxy-v2 可比口径）。可执行阈值：**|Δgap| 超出家族带（±0.007）即异常**，
   带内 OOD 不携带超出 in-dist 的额外信息。
4. **正向一致性证据**：v158 旧 Linear（pre-L1batch）in-dist 0.448 / gap +0.0285，
   v160/v182 新 Linear in-dist 0.634~0.637 / gap +0.016——Linear 谱系演进同时改善了
   分布内增益与跨分布稳健性，官方分（16861→17532→17598）与"非纯分布拟合"互证。
5. 逐域结构：zh 域 Linear 系统性最低（语言迁移），Attention 三域均匀；
   v176 的 news 域（0.7341）高于父、zh 域（0.7164）低于父，域间分化大于均值差异。

## 对后续候选的使用规则（登记）

- 候选判读仍以 in-dist paired + `L1 < 0.02` 为主门禁；OOD 为**旁路警报**：
  `|Δ(gain_in − gain_ood)| > 0.01`（家族带外）→ 拟合型机制嫌疑，禁止提交官方；
- 带内 Δgap 不作为否决或晋级依据；
- OOD 面板继续不参与 proxy 排名、不与 in-dist 混排。

## 产物清单

- OOD：`artifacts/official_eval/ood-v{158,160,168,171,176}.json` + `logs/official_eval/ood-v*.md`
- in-dist 补跑：`artifacts/official_eval/v158-default-both.json`、`v176-attn-default.json`
- 原始 stdout：`logs/official_eval/*.{run.log}`（7 份）
- v182 基线见 `logs/execution/2026-09-04-ood-suite-baseline-v182.md`
