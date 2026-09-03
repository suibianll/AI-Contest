# v172 候选：L2 HiF4 层级约束 Babai 解码（低复杂度扩展计划第五包）

> 状态：**REJECTED（明确负优化；2026-09-04 按用户最终指示 '明确负优化的需要拒绝提交'）**
>
> 父版本：v166（官方 `4590 / 226s`，Linear 父侧）；Attention = v162 standard tail 逐位一致
>
> 候选 SHA256：`67283937D8E8767FDF760AFE70D4EFC778228F4ED4E52AF4A6F6F21769DF4F65`
>
> 官方结果：`unregistered / NA`

## 1. 唯一算法机制（预注册，计划 §9）

在最终部署变换坐标系的每个 64-block：`H_b = X_cal^T X_cal / N`（gram_full 对角块），
damping `0.01 * mean(diag(H_b))`，处理顺序按 damped 对角降序；
`Bmat = chol(H_b)^T @ diag(step)`（step_j = 0.25 * scale * lv2_j * lv3_j，父合法层级），
batched QR + 每 Weight row 自最后一维向前 nearest-plane rounding，code clip `[-7,7]`；
发生 clipping 时一次 no-clipping rescale（E6M2 scale 提升到容纳未裁剪 code 的最小合法
code，`_solve_exact_hierarchy` 重算 lv2/lv3 后重跑一次）；逐 (row, block) 比较
parent/Babai 的 block-Hessian loss，保留较小者，五字段原子写回。不跑第二 sweep、不搜
damping/order/scale。校准期调用一次，动态路径与 activation_state 逐位一致（零新增）。

## 2. 本地验证（描述性；官方裁决）

| 项目 | 结果 |
| --- | --- |
| 隔离导入 + 六 API | OK（子代理实现；重跑 compact status ok） |
| compact 56（配对 v166） | candidate_mean `0.657287` vs 父 `0.705628`（**−0.0483**、median −0.0443、`0+/48−`、worst −0.172）——**明确负：0 个正 case，拒绝提交** |
| 机制可达 | 是（attempted=parent blocks，Babai 输出与父不同，五字段写回） |
| API 时间 | 62.2s（父 compact 46.1s，1.35×）——描述性，官方时间不可预测 |

## 3. 判读（§3.3）

```text
step_gain = S(v172) − 4590
```

`>0` → RETAINED 成为新 Linear 父侧；`≤0` → REJECTED 转 L3（Trellis，独立解码算法，
不受 L2 结果取消）。官方回传裁决。

## 4. 复现

```powershell
.venv\Scripts\python.exe -u evaluator\official_eval.py --solution solutions\20260903_v172_babai-weight-decode_scoreNA_timeNA\solution.py --linear-only --compact-panel --cache-mode read --nvfp4-cache-mode auto --capture-device cuda --algorithm-device cuda --baseline-json artifacts\official_eval\v166-compact-linear-smoke.json --output artifacts\official_eval\v172-compact-linear.json --report logs\official_eval\v172-compact-linear.md
```
