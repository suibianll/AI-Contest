# v183 候选：attention block-smooth 搜索 refine 全覆盖（Attention）

> 状态：**REJECTED（官方 2026-09-04）— v183 `17598/279.7s`，与父 v182 分数持平
> （step_gain 0），按预注册规则 `S≤17598` → REJECTED，覆盖率族关闭；
> 时间 279.7s < 300s（+6.7s 校准 refine 覆盖成本，非超时）。官方父保持 v182。**
>
> 来源诊断：[`2026-09-04-gate-audit-and-coverage-diag.md`](../../logs/execution/2026-09-04-gate-audit-and-coverage-diag.md)
> 方向 1（时间预算→覆盖率）唯一有正余量的 cap。
>
> 构造：v182 完整官方父（`17598/273s`）+ 仅提高校准期 attention block-smooth
> 搜索的 final-quantizer refine 覆盖：`_ATTN_BLOCK_SMOOTH_REFINE_RATIO 0.50→1.00`、
> `_ATTN_BLOCK_SMOOTH_REFINE_BLOCKS 24_576→131_072`。diff 仅 2 行常量；
> Linear 与 v182 逐位一致；在线 Q/K/V refine 与激活路径未动。

## 1. 机制（方向 1：覆盖率转化）

- block-smooth 搜索在校准期对每个候选 block 配置用 final quantizer 重建
  Q/K 并按 deployed MSE 选优；refine ratio 0.50 截断了候选比较时的精化覆盖。
- 提高到 1.00 使搜索期的候选评估与部署路径一致（全 refine），只改变
  校准期选出的 block signs/size，不增加任何在线算子。
- 单预注册配置（0.50→1.00 一步，非网格扫描）。

## 2. 硬检查（全部通过）

| 检查项 | 结果 |
| --- | --- |
| 单文件脱离仓库导入六 API | OK |
| diff 仅 2 常量（Linear/在线路径未触碰） | OK（git diff 核验） |
| 合法 state / 有限输出 / 无 NaN | OK（全部评测通过） |
| 机制 reachability | Qwen default 25/120 case 输出改变（11+/14−/95=），OPT 60 case 全不变（该路径模型相关但零输出破坏） |
| 校准时间 | GPT-2 实测 calibration_attention +0.397s/12 states（1.024×），官方时间风险可忽略 |
| Attention control（Linear） | 与 v182 逐位一致（代码未改） |

## 3. 本地描述性评测（配对 v180 attention baseline = v182 attention，逐位一致）

| 场景 | cases | mean Δgain | median | 改善/回归/不变 |
| --- | ---: | ---: | ---: | ---: |
| Qwen compact | 4 | 0.000000 | 0 | 0/0/4（哨兵未触发） |
| **Qwen default** | **120** | **+0.000511** | 0 | 11/14/95 |
| GPT-2 compact | 4 | −0.036537 | 0 | 0/3/1 |
| GPT-2 default | 60 | −0.005042 | 0 | 4/6/50 |
| OPT-125m default | 60 | 0.000000 | 0 | 0/0/60 |

- Qwen default +0.000511 与 D1 的 local +0.000356（官方 +3）同阶。
- GPT-2 default 轻微负（6/60 回归、median 0）；控制臂 Q-only/K-only 为正
  （+4.7/+7.5）但联合 QK 略负，呈轻微搜索过拟合迹象；标记
  **model-specific-risk**，在 D1 先例带内（D1 GPT-2 compact −0.009 → 官方 +3）。
- OPT 全不变：机制 reachability 模型相关，但无输出破坏。

## 4. 官方裁决

配额账本：v183 为第 4 个（4/10）。裁决规则：
`S(v183) > 17598` 且 `<300s` → RETAINED；`≤ 17598` 且 `<300s` → REJECTED
（覆盖率族关闭，不扫 ratio 邻域）；`>300s` → TIMEOUT。

官方回传 `17598/279.7s`：相对 v182 `step_gain=0`、时间 `+6.7s`。因此按上述
预注册规则判为 **REJECTED**，覆盖率族关闭；v182 保持完整官方父。配额不退还，账本
为 `4/10`，剩余 6。

## 5. 复现

```powershell
.venv\Scripts\python.exe -u evaluator\official_eval.py --solution solutions\20260904_v183_attn-bsm-full-refine_rejected\solution.py --attention-only --cache-mode read --nvfp4-cache-mode auto --capture-device cuda --algorithm-device cuda --baseline-json artifacts\official_eval\v180-attn-default.json --output artifacts\official_eval\v183-attn-default.json --report logs\official_eval\v183-attn-default.md
```

源码 SHA256：`d94f37cc7b5370b1c2bc070157166d060936371c4e65e354dad3746090771f24`
（v183 官方拒绝归档）。根 `solution.py` 保持 v182 官方父 `F3E39E99...A438`。
