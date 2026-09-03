# v182 候选：L-R2 融合 rank-2 残差重分布（Linear）

> 状态：**候选待官方评测（已提交请求）；硬检查全部通过**
>
> 计划：[`2026-09-04-post-v180-linear-rank2-plan.md`](../../docs/superpowers/plans/2026-09-04-post-v180-linear-rank2-plan.md)
>
> 构造：v180 完整官方父 + Linear 侧把 v166 的 rank-1 残差重分布推广为一次融合
> rank-2 正交更新（U=[u1,u2]、V=[v1,v2]、V^T U≈0、连续域乘积严格不变）；
> Attention 与 v180 逐位一致（未触碰 Attention API）。

## 1. 机制（L-R2）

- 父 rank-1：`R1 = I + u1 v1^T`，`v1^T u1 = 0`，`u1 = 0.25 d2`（v166 原逻辑不变）。
- L-R2 在 span(v1,u1) 正交补上用部署坐标基础 HiF4 残差算子
  `C(x) = Ea^T(Ea x)/||Ea||^2 + Ew^T(Ew x)/||Ew||^2` 做固定 128 次 power
  iteration，提取 d3/d4，sign-align + median 聚合 + Gram-Schmidt，固定
  `v2 = d3`（单位）、`u2 = 0.25 d4`（范数 0.25）。
- 融合 rank-2：`U=[u1,u2]`、`V=[v1,v2]`、`V^T U ≈ 0` →
  `R = I + U V^T`、`R^-1 = I - U V^T`（Woodbury），
  `A' = A + (A U) V^T`、`W' = W - (W V) U^T`，`A'W'^T = AW^T` 严格保持。
- Gram/Hessian 用最终坐标 rank-2 公式精确更新，之后沿用父 GPTQ/hierarchy/encode。
- 动态激活：一次融合 `dense + (dense @ U) @ V^T`（单 GEMM 对，无循环/无候选/Gram）。

## 2. 固定参数（不允许调参）

```text
rank = 2
coefficient = 0.25        # 继承 v166
power iterations = 128   # 继承 v166
folds = even/odd (4 folds)
aggregation = component-wise median
```

## 3. 硬检查（全部通过）

| 检查项 | 结果 |
| --- | --- |
| 单文件脱离仓库导入六 API | OK（compact/default/GPT-2/OPT 全跑通） |
| reference_hif4 state/五字段合法 | OK（无 state 错误） |
| 无 NaN/Inf、有限输出 | OK |
| rank-2 reachability | compact 28/28、default 168/168、GPT-2 72/72、OPT 72/72 全部 = 1 |
| `||V^T U||F` 接近 0 | vtu_cross_max 实测 6.99e-10 ~ 1.86e-08 |
| 连续域乘积不变 | 数学成立（V^T U=0 → Woodbury R^-1=I-UV^T） |
| Attention control | 与 v180 逐位一致（diff 仅 Linear 权重校准 + 激活路径 6 个 hunk） |
| 动态 API 无 Gram contraction/候选循环 | OK（融合 [D,2] GEMM） |

## 4. 本地描述性评测（配对 v180；官方不参与决策）

| 场景 | cases | mean Δgain | median Δgain | 改善/回归 | median MSE ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen compact linear | 56 | −0.000093 | +0.000073 | 30/26 | 0.999585 |
| Qwen default linear | 168 | +0.000020 | +0.000048 | 85/83 | 0.999897 |
| GPT-2 linear | 72 | +0.001171 | +0.000307 | 37/35 | 0.998737 |
| OPT-125m linear | 72 | +0.025632 | +0.000787 | 40/32 | 0.998331 |

- Qwen compact cross-holdout：28/28 对 validation/test 同号；gain gap median 0.0165。
- Qwen default 其余 role/family/层分布与父同构（W/A 深度负、interaction 巨大正）。
- 跨模型整体非负/微弱正，无 `model-specific-risk` 标记。
- 本地 proxy 只作描述；官方结果决定提升与否。

## 5. 时间记录（本地，非官方）

- Qwen default linear API total：v182 332.4s vs v180 282.4s（+50s，主要来自 rank-2
  校准 power iteration 增量；在线仅增加一列融合 GEMM）。
- 官方时间由官方评测回传后登记；不提交等价时间 A/B。

## 6. 官方裁决

配额账本：v182 为第 3 个（3/10，本地无 model-specific-risk）。等待用户回传官方分数：
`S(v182) > 17597` 且 `<300s` → RETAINED 新父；`<= 17597` 且 `<300s` → REJECTED（rank
扩展族关闭）；`>300s` → TIMEOUT（rank 扩展族关闭，不降 rank 重试）。

## 7. 复现

```powershell
# compact
.venv\Scripts\python.exe -u evaluator\official_eval.py --solution solutions\20260904_v182_rank2-linear_v180-attn_scoreNA_timeNA\solution.py --linear-only --compact-panel --cache-mode read --nvfp4-cache-mode auto --capture-device cuda --algorithm-device cuda --baseline-json artifacts\official_eval\v180-compact-linear.json --output artifacts\official_eval\v182-compact-linear.json --report logs\official_eval\v182-compact-linear.md

# default
.venv\Scripts\python.exe -u evaluator\official_eval.py --solution solutions\20260904_v182_rank2-linear_v180-attn_scoreNA_timeNA\solution.py --linear-only --cache-mode read --nvfp4-cache-mode auto --capture-device cuda --algorithm-device cuda --baseline-json artifacts\official_eval\v180-linear-default.json --output artifacts\official_eval\v182-linear-default.json --report logs\official_eval\v182-linear-default.md
```

源码 SHA256：`f3e39e993a436e217cb4811525c81239f82a6ec58845a0646e183a824c33a438`
（v182 候选归档）。根 `solution.py` 保持 v180 官方父 SHA
`2BA40122...8AA3`；正式提交入口在 v182 官方 RETAINED 前不改为候选。
