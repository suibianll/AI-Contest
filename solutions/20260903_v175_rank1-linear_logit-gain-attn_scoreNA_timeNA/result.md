# v175 候选：组合（v166 rank-1 Linear + v168 A1 logits 增益 Attention）

> 状态：**CANDIDATE — 两侧逐位保留各自官方父侧机制；计划 §13 最后一步，提交官方**
>
> 构造：v166 基底（v160 Linear + rank-1，官方 `4590 / 226s`）删除 standard Attention
> tail，注入 v168 的 A1 logits gain（官方 `14005 / 210s`）；Linear 与 Attention 均为
> 各自官方正向父侧
>
> 候选 SHA256：`33B25EBEDA87DC1D09F08531A68E056C6D6587A43A336A5DD93B2B642F8A8F88`
>
> 官方结果：`unregistered / NA`

## 1. 唯一构造（预注册，计划 §6/§13 组合规则）

两个独立官方父侧组合为单文件：
- **Linear** = v166（fold-median rank-1 残差重分布，官方 `4590/226s`，+3 over v163）；
- **Attention** = v168（per-KV-head 解析 logits 增益 folded into multiplier，官方
  `14005/210s`，+60 over v164）；
- 非目标侧零污染：组合中 Linear 与 v166 逐位一致、Attention 与 v168 逐位一致
  （case 级 max |Δgain| = 0.0 已实测）。

## 2. 本地验证（描述性；官方裁决）

| 项目 | 结果 |
| --- | --- |
| 隔离导入 + 六 API | OK（六 API 各唯一） |
| linear compact 56（配对 v166） | candidate_mean `0.705628`，**mean_delta 0.0**（rank-1 逐位保留） |
| attention compact 4（配对 v168） | attention_mean `0.797753`，**max |Δgain| = 0.0**（A1 逐位保留） |
| gpt2 linear 72（配对 v159-lin 父） | **mean Δgain +0.000786**、median −0.000383、34+/38−/0=（win 0.472）；跨模型中性 |
| gpt2 attn compact 4（配对 v160 attn 父） | **mean Δgain +0.002447**、median +0.001125、3+/1−/0=（与 v168 同值，逐位保留） |
| API 时间 | linear compact 53.8s / attention compact 11.5s（同父侧） |

## 3. 判读（§3.3 / §13）

```text
interaction = S(v175) − S(v166) − S(v168) + 1001
S_pred      = 4590 + 14005 − 1001 = 17594（相对 v160 的 17532 +62，距榜首 4171）
closure     = (S(v175) − 17532) / 4233
```

`S(v175) > 17532` → 组合成为新完整官方父版本；`≤` → 保留两侧独立父侧，不伪造可加收益。
时间：分量和 `290s`，v160 式共享折扣后约 `262s`，余量充裕。

## 4. 复现

```powershell
.venv\Scripts\python.exe -u evaluator\official_eval.py --solution solutions\20260903_v175_rank1-linear_logit-gain-attn_scoreNA_timeNA\solution.py --linear-only --compact-panel --cache-mode read --nvfp4-cache-mode auto --capture-device cuda --algorithm-device cuda --baseline-json artifacts\official_eval\v166-compact-linear-smoke.json --output artifacts\official_eval\v175-compact-linear.json --report logs\official_eval\v175-compact-linear.md

.venv\Scripts\python.exe -u evaluator\official_eval.py --solution solutions\20260903_v175_rank1-linear_logit-gain-attn_scoreNA_timeNA\solution.py --attention-only --compact-panel --cache-mode read --nvfp4-cache-mode auto --capture-device cuda --algorithm-device cuda --baseline-json artifacts\official_eval\v168-compact-attn.json --output artifacts\official_eval\v175-compact-attn.json --report logs\official_eval\v175-compact-attn.md
```
