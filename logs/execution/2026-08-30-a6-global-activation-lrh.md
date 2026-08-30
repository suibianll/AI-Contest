# A6 Global Activation-LRH — rejected

日期：2026-08-30  
状态：`archived-rejected`；主代码已恢复 stable parent。官方评测不可用，使用
固定 Qwen2.5-0.5B cache / qwen-official panel。

## 实现

对静态 transformed weight 构造合法 block Gram
\(B=\operatorname{blockdiag}(W^TW)\)，不分配完整 channels² 矩阵；随机 range
iteration 只通过

\[
 (W^TW-B)V = W^T(WV)-BV
\]

求出 off-block 的正特征子空间。保存的 `global_lrh` 列为
\(U=V_r\Lambda_r^{1/2}\)，动态 activation 目标为

\[
 L(q)=(q-x)^TB(q-x)+\|U^T(q-x)\|_2^2.
\]

逐 block HSDQ 的每个合法坐标更新同时纳入 block gradient、低秩 gradient
\(U(U^T(q-x))\) 与对应对角曲率。为防止近似误差扩大，另加逐行 L2 parent guard；
v3 将低秩能量缩放为 10%。state 仅含静态 CPU Gram/低秩张量，不含 calibration
output 或 residual。

## 结果

| 版本 | layer-1 panel | full panel | full Linear mean | Attention mean | API time |
| --- | ---: | ---: | ---: | ---: | ---: |
| v1 rank-8 full energy | `209.851980` | 未跑（单层门禁失败） | — | — | `16.19s` layer-1 |
| v2 + row L2 guard | `323.075547` | 未跑（单层门禁失败） | — | — | `16.57s` layer-1 |
| v3 10% low-rank energy（最高） | `323.076334` | `282.616646` | `0.457010` | `0.841829` | `373.97s` |
| stable parent | `336.035344` | `293.755106` | `0.501558` | `0.841829` | `382.15s` |

v3 全层 panel 比 parent 低 `11.138460`，因此即使运行时间合格也不进入主线。
跨层回退表明当前两折静态 \(W^TW\) off-block 结构不能稳定指导 validation 的
Q(A) 离散码点；不继续扩大 rank 或 power iteration。

## 裁决与证据

- v3 完整源码及三次本地结果保存在
  `solutions/20260830_v095_a6-global-activation-lrh-rejected_score282.616646_time374s/`。
- 主 `solution.py` 恢复 stable parent；A6 不改变当前最高分版本。
- 后续转 Attention 路线；Linear 的 global-LRH 仅作为理论上限诊断，不再作为部署
  候选。
