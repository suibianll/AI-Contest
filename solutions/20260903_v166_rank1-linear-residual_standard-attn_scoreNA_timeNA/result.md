# v166 候选：rank-1 残差重分布 Linear + 标准 Attention（侧向隔离计划首个 Linear 机制）

> 状态：**RETAINED — 官方** **`4590 / 226s`，成为新 Linear 父侧**
>
> 官方共同基线：v162 `1001 / 146s`；Linear 官方父侧：v163（v160 Linear + standard Attention）
> `4587 / 202s`；完整方案锚点：v160 `17532 / 232s`
>
> 父版本：v160 归档（Linear 机制基底），SHA
> `33B1D061CE6BFCD92659C597BE4830BB9B910E646FF518433DA67B925AE8680D`；
> Attention 侧为 v163 追加段（standard codec，逐位复用）
>
> 候选 SHA256：`9C0EAC6A7CA883A1F8962C11735744271259460F5EBBF23D530A5BBCF12B4646`
>
> 官方结果：**`4590 / 226s`（2026-09-03 回传）**

## 1. 唯一算法机制（预注册，侧向隔离计划 §6）

秩 1 可逆残差重分布，在 v160 Linear 的**权重编码调用之前**以重绑定方式插入
（部署坐标系）：

```text
R = I + u v^T,   v = d1,  u = 0.25 * d2,  d2 ⊥ d1
W' = W_dep R^{-T} = W_dep - (W_dep v) u^T     （u⊥v 时精确，无除法）
A' = A_dep R      = A_dep + (A_dep u) v^T     （动态 O(TD)）
```

- **残差探针**：部署坐标下的基础 HiF4 codec（`_dense_to_hif4`）编码残差——权重残差
  `W_dep - dequant(encode(W_dep))`，激活残差为两个校准 window 部署样本的
  even/odd 行折（4 折，每折 64 行）各自编码残差；

- **方向构造**：每折残差二阶矩（激活项与权重项各自 trace 归一后相加，不引入权重
  系数）的 top-2 特征向量（幂迭代 128 步、确定性起始、deflation 取第二方向）；
  符号固定（最大分量正值）后对齐 fold 0，跨折逐分量中位聚合；d2 对 d1 正交化；

- **常数**：c = 1/4（预注册）；稳定界 `|v^T u| ≤ 1/2`（正交构造下恒满足，保留
  预注册投影行）；d2 退化（正交化后范数 < 1e-6）时 u = 0，候选对该 state 精确
  回退父版本行为；

- **下游自洽**：Hessian `gram' = R^T G R`（rank-1 精确更新，O(D²)）、block grams 与
  importance 同步变换后，GPTQ 权重编码、activation importance/gram/h\_inv 全部由
  父流程自身在变换后坐标系中一次性计算（单次编码设计，无二次编码/无搜索循环）；
  宽输入层（无 gram\_full，如 proj）的 importance 用变换后样本估计；

- **动态路径**：父版本变换链之后加一行 `A + (A u) v^T`，其余不变；

- state 新增 `rank1_u/rank1_v`（CPU float32，通过 `validate_state` 通用校验），
  version 3→4。

## 2. 本地验证（描述性，官方裁决）

| 项目                                               | 结果                                                                                                                                                               |
| ------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 隔离导入 + 六 API（脱离仓库）                               | OK                                                                                                                                                               |
| linear compact 56（smoke）                         | `0.705628`（父 0.705508），API 49.8s（父 46.1s，1.08×）                                                                                                                  |
| linear default 168（配对 v163-linear = v160 Linear） | **0.636590 vs 0.633526**；paired mean **+0.003064**、median `-0.000364`、`78+/90-/0`、worst `-0.047075`（layer22 o）、best `+0.602086`（proj）；API 282.8s（父 227.4s，1.24×） |
| 按 role 配对                                        | **proj +0.025142**（收益全部来源，宽输入层，父版本无 GPTQ/Hessian 保护）；q +0.0003 / fc\_gate +0.0000 / k -0.0014 / v -0.0007 / o -0.0015 / fc\_up -0.0005                           |
| attention default 120（非目标控制）                     | mean **0.0**，配对 **0/0/120** 与 standard 逐位一致，API 0.821s                                                                                                           |
| W/A/Both/interaction 四臂                          | 已记录于 JSON `decomposition`（overall both 0.6366，interaction 520.4；proj W-only rel-MSE 7.12 印证该侧保护最弱）                                                               |

证据：`sidecal-v166-linear-default.json`、`sidecal-v166-attn-default.json`、
`v166-compact-linear-smoke.json`（`artifacts/official_eval/`，对应 `logs/official_eval/`）。

## 3. 风险记录（不阻止提交）

- median 微负（-0.0004）、90/168 case 回归：收益集中于 proj 一个 role；官方是否
  转移由 §3.1 指标裁决，本地不设准确率门槛；

- API 282.8s 本地（1.24×）：全部增量在 calibration（探针 + Hessian 变换 + 单次
  重绑定编码），动态路径仅加一次 O(TD) 外积；官方时间预计 \~250s，但本地秒数不
  换算官方，timeout 时按计划 §5 允许一次保持数学目标不变的纯复杂度重构；

- 官方判读（预注册）：`S_L = 4590 > 4587` → **官方正向** **`+3`（G\_L = +3）**，成为新 Linear
  父侧 `P_L = v166（4590 / 226s）`；按 §3.1 登记 `C_L = S_L - 1001 = 3589`、
  `G_L = S_L - 4587 = +3`、`P_L = G_L / 3586 ≈ 0.0008`、`R_L = C_L / 3586 ≈ 1.0008`；
  时间 `226s` 通过 `<300s`。

## 5. 官方回传结论（2026-09-03）

官方 `4590 / 226s`，相对 v163（`4587 / 202s`）step\_gain `+3`，官方正向并成为
**新 Linear 父侧**。收益方向与本地 default 的 proj role（`+0.0251`）归因一致；本地
90/168 回归 case 与 median 微负未在官方反转。按活动计划 §12 第 11 步父侧更新：
`P_L：v163 → v166`，下一个 Linear 候选从 v166 构造（保留 v160 固定口径比例）。

## 4. 复现

```powershell
.venv\Scripts\python.exe -u evaluator\official_eval.py --solution solutions\20260903_v166_rank1-linear-residual_standard-attn_scoreNA_timeNA\solution.py --linear-only --cache-mode read --nvfp4-cache-mode auto --capture-device cuda --algorithm-device cuda --baseline-json artifacts\official_eval\sidecal-v163-linear-default.json --output artifacts\official_eval\sidecal-v166-linear-default.json --report logs\official_eval\sidecal-v166-linear-default.md

.venv\Scripts\python.exe -u evaluator\official_eval.py --solution solutions\20260903_v166_rank1-linear-residual_standard-attn_scoreNA_timeNA\solution.py --attention-only --cache-mode read --nvfp4-cache-mode auto --capture-device cuda --algorithm-device cuda --baseline-json artifacts\official_eval\sidecal-v163-attn-default.json --output artifacts\official_eval\sidecal-v166-attn-default.json --report logs\official_eval\sidecal-v166-attn-default.md
```
