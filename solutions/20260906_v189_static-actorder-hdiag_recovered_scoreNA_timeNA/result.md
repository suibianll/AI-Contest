# v189 候选：静态部署 Hessian activation-GPTQ 块序重排

> 状态：**RETAINED（官方 2026-09-06）**，官方 **`17616/275s`**。本地门禁通过；
> 官方结果为独立观测，未把本地 proxy 换算成官方分数。

## 1. 版本与父版本

- 父版本：根 `solution.py` v186，官方 `17599/272s`。
- 父 SHA256：`f8495dca20334acbdad16fc18ee41a4970f31e1837fdeedcee9c70aee54e7eb8`。
- 候选 SHA256：`261202248a0146a2ee45f3df60bd1979bb8171b7c162921013b0024c848617af`。
- 归档源码：本目录 `solution.py`；上传包：本目录 `solution.zip`。
- 官方结果相对父版本 `+17/+3s`，通过 `<300s`；根目录已切换为 v189。

## 2. 唯一机制

v186 完成权重量化校准后，使用部署坐标中的 `W_hat` 列能量
`diag(W_hat.T @ W_hat)`，为完整 64-channel block 生成一次固定降序顺序。
在线 activation GPTQ 将 activation、`h_inv`、importance 和已有 block Gram 按同一
顺序重排，复用原有有限 GPTQ，再恢复输出布局。Attention、HiF4 编码器、连续域
变换和在线候选数量均不变；state 只增加合法的 CPU Tensor 排列和布尔标记。

## 3. R0/R1/R2 门禁

- 单文件导入六个正式 API：通过；`py_compile`：通过；专用测试：`2 passed`。
- 当前 `eval-v3` 六 shard、336 Linear cases：候选 mean
  `0.636799627541`，v186 baseline `0.632821506420`，signed delta
  `+0.003978121121`；median delta `+0.002777959610`，L1
  `0.005497954666 < 0.02`，正/负/零 `288/38/10`，所有输出有限且 case 唯一。
- 六 shard 最小逐 case delta `-0.064834778839`，收益分布跨 role/layer；该值仅作
  本地风险记录，不作官方分数换算。
- 当前 default-panel 新鲜组合：

  | 侧 | 候选 | v186 父基线 | 变化 |
  |---|---:|---:|---:|
  | Linear | `0.640258324430` | `0.636609486834` | `+0.003648837595` |
  | Attention | `0.752173407020` | `0.752173407020` | `0` |
  | Overall (288 cases) | `0.686889608842` | `0.684761120245` | `+0.002128488597` |

  v186 Linear 基线由 bit-identical 的 v182 Linear default 记录提供，Attention 基线由
  v186 Attention default 记录提供；两侧均使用同一 proxy-v2 cache 口径。

- OOD 六 shard：候选 in-dist mean `0.636799627541`、OOD mean `0.648734473521`；
  相对父版本的 `Δ(gain_in - gain_ood)` 为 `+0.000294570038`，满足
  `|Δgap| <= 0.01`。
- 新鲜 default 时间审计：`W_calib=268.196743s`、`A_calib=57.182195s`、
  `dyn_act=59.737831s`、`dyn_qkv=3.058645s`；六 API 分解模型预测
  `T_pred=279.956116s < 280s`，通过提交时间门。该预测不是官方实测时间。

## 4. 证据

- eval-v3 Linear：`artifacts/proxy_v3/static-actorder-hdiag-recovered-20260906/full-r5/static-actorder-hdiag-recovered-full/`
- eval-v3 OOD：`artifacts/proxy_v3/static-actorder-hdiag-recovered-20260906/ood-r6/static-actorder-hdiag-recovered-ood/`
- default-panel：`artifacts/official_eval/static-actorder-hdiag-recovered-fresh-default-r9.json`
- default-panel 报告：`logs/official_eval/static-actorder-hdiag-recovered-fresh-default-r9.md`
- 固定计划：`docs/superpowers/plans/2026-09-06-static-activation-gptq-order-plan.md`

## 5. 官方提交状态

赛事说明要求上传仅含 `solution.py` 的 `solution.zip`。本目录已生成该包并收到官方
回传 `17616/275s`；候选已 RETAINED，根目录同步为 v189。官方源码 SHA 与归档源码
一致：`261202248a0146a2ee45f3df60bd1979bb8171b7c162921013b0024c848617af`。
