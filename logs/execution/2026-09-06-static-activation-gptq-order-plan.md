# 静态部署 Hessian activation-GPTQ 块序计划执行记录

日期：2026-09-06  
计划：[`归档计划`](../../docs/superpowers/archive/plans/2026-09-06-static-activation-gptq-order-plan-candidate-archived.md)  
父：根 `solution.py` v186，SHA `F8495DCA...7EB8`

## 裁决

R0、R1、R2 全部通过，候选按 R3 归档为 v189。根 `solution.py` 未修改。官方结果尚未
返回，状态为 `UNREGISTERED / NA`；已生成赛事要求的仅含 `solution.py` 的
`solutions/20260906_v189_static-actorder-hdiag_recovered_scoreNA_timeNA/solution.zip`。

## R0：恢复与合法性

- 恢复源码是单文件副本，导入不依赖 `workbench` 或其他仓库实现。
- 六个正式 API 可导入；`py_compile` 通过；专用测试 `2 passed`。
- `gptq_block_order` 为 CPU `int16` 完整排列，block 固定 64；state 可序列化，输出有限。
- 保留字节码的非恒等顺序行为与重建源码逐 state 对齐，输出最大差为 `0.0`。

## R1：当前 eval-v3 配对

固定 proxy-v2 dense cache、CUDA 和六 shard 的 336 Linear cases：

| 指标 | 候选 | v186 baseline / 变化 |
|---|---:|---:|
| mean | `0.636799627541` | `0.632821506420` / `+0.003978121121` |
| median delta | `+0.002777959610` | — |
| L1 delta | `0.005497954666` | `<0.02` |
| 正/负/零 | `288/38/10` | — |
| 最小逐 case delta | `-0.064834778839` | 风险记录 |

收益跨 role/layer 分布；Linear-only shard scope 仅用于侧向配对，未被当作官方总分。
新鲜 default-panel 的 168 Linear + 120 Attention 结果为：

- 候选：Linear `0.640258324430`、Attention `0.752173407020`、Overall `0.686889608842`；
- v186 组合父基线：Linear `0.636609486834`、Attention `0.752173407020`、Overall
  `0.684761120245`；
- Overall delta `+0.002128488597`，Attention bit-identical。

v186 Linear 侧使用 bit-identical 的 v182 Linear default 记录，Attention 侧使用 v186
Attention default 记录；两侧同属 proxy-v2 default-panel 口径。

## R2：OOD 与时间

- OOD 六 shard：candidate in mean `0.636799627541`、OOD mean `0.648734473521`；
  相对父的 `Δ(gain_in-gain_ood)=+0.000294570038`，通过 `|Δgap|<=0.01`。
- 新鲜 default API timing：`W_calib=268.196743s`、`A_calib=57.182195s`、
  `dyn_act=59.737831s`、`dyn_qkv=3.058645s`。
- 已校准时间模型预测 `279.956116s < 280s`，通过提交资格门；API total
  `388.175412s` 和 wall `412.292138s` 是本地参考测量，不是官方时间。

## 证据与官方提交

- Linear evidence：`artifacts/proxy_v3/static-actorder-hdiag-recovered-20260906/full-r5/static-actorder-hdiag-recovered-full/`
- OOD evidence：`artifacts/proxy_v3/static-actorder-hdiag-recovered-20260906/ood-r6/static-actorder-hdiag-recovered-ood/`
- Default JSON：`artifacts/official_eval/static-actorder-hdiag-recovered-fresh-default-r9.json`
- Default report：`logs/official_eval/static-actorder-hdiag-recovered-fresh-default-r9.md`
- v189 source SHA：`261202248a0146a2ee45f3df60bd1979bb8171b7c162921013b0024c848617af`

当前仓库没有官方竞赛网页的自动上传接口，因此只能确认提交包已准备完成，不能把
网页上传写成已完成。收到官方回传后再追加裁决，不重复提交相同 SHA。
