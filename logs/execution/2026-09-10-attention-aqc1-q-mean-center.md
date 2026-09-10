# A-QC1 Q 侧 per-call 数据中心化执行记录

> 计划：[Attention Q 侧 per-call 数据中心化计划（A-QC1）](../docs/superpowers/plans/parallel/2026-09-10-attention-q-mean-center-plan.md)。
> 根 R0：v202 Linear + v195 Attention，官方 `18053/281s`，SHA256 `56DC805D...EFCB2BD`。

## 1. 实现（2026-09-10）

- 候选：`workbench/full_solution/attention-aqc1-q-mean-center/candidate/solution.py`
  （根逐位副本 + 纯插入，SHA `8d3d3cfb...`）。
- 与 A-MC1 同构、移到 Q 侧：`_nvfp4_to_hif4`（4184、4290–4298）新增 `q_mean_center`，
  在 rotation/center 之后、`_dense_to_hif4` 之前对 Q 做 `Q -= mean_tokens(Q)`（按 head 分组）；
  `hif4_dynamic_quantize_q`（10770）仅在 arm 标志开启时执行；A-QC1 段（11610–11724）
  先原样执行根校准，再逐层全 folds 真实 MSE gate，严格改善才写 `q_state["q_mean_center"]=1`。
  K/V/Linear 完全不动。

## 2. Control 结果（全部 PASS）

脚本：`workbench/full_solution/attention-aqc1-q-mean-center/control.py`，结果 `control_results.txt`。

1. arm 关闭：Q/K/V 五字段与输出与根逐位一致（flag 缺失与显式 0 均验证）。
2. arm 开启 + 非零均值 Q：Q 五字段显著变化（mant 21227、sign 16403、lv3 2033、lv2 568、
   scale 345 处）；dense softmax 输出 max|Δ|=1.74e-5（rel 1.98e-3）——Q 中心化无不变性，
   输出有意改变，机制可达；K/V 逐位不变。
3. 六 API 脱离仓库 importlib 独立导入通过。
4. `validate_state`/`validate_hif4_params` 通过；audit 含 `aqc1_arm/aqc1_gate_loss_parent/
   aqc1_gate_loss_candidate/aqc1_windows`；`root_rotation_frozen=True`。
5. V/Linear control 逐位一致；gate 接受/回退双路验证（8 种子 4 center / 4 parent +
   强制不改善注入正确回退）。

## 3. 评测

产物：`artifacts/proxy_v3/attention-aqc1-sixshard-full-20260910/`（六 shard 一次跑满，
`--stop-after-nonpositive 6`，`reasonableness_issues: 0`，API total 诊断 40.094s，无缓存命中）。

| shard(层) | delta_mean | +/-/0 |
|---|---:|---|
| 0（层0） | +0.000000 | 0/0/12 |
| 1（层1） | +0.000000 | 0/0/12 |
| 2（层8） | +0.000000 | 0/0/12 |
| 3（层15） | +0.000000 | 0/0/12 |
| 4（层22） | +0.000000 | 0/0/12 |
| 5（层5） | +0.000000 | 0/0/12 |

六 shard 72 case 全部与根逐位相同（overall `+0.533998` = baseline）：**6/6 层 gate 全部回退**——
Q 中心化在全部校准 folds 的真实部署 MSE 上没有任何一层严格改善。

## 4. 结果与裁决

- 裁决：`NO_EFFECT`。机制可达（control 2 证明五字段与输出可变）但六层 gate 全拒，
  输出与根逐位相同。按计划 §6 不占版本号、不提交官方、不以 midrange/中位数/trimmed 变体重试。
- Q 侧数据中心化关闭。至此 Attention 规则级空间全部裁决完毕：Q·K 不变量四类结构变换
  （对角 scale/置换/正交/K 平移）+ 两侧 per-call 定心规则均有定论；后续只由官方回传
  （v229 待官方）或 21071 锚点源码绑定驱动。
