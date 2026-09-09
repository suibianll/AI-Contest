# A-G1 Q/K 联合仿射 Gauge 执行记录

> 计划：[Attention Q/K 联合仿射 Gauge 优化计划](../docs/superpowers/plans/parallel/2026-09-10-attention-joint-affine-gauge-plan.md)。
> 总协调：[Linear 完整输出交叉残差纠码与双线协调计划](../docs/superpowers/plans/2026-09-10-linear-cross-residual-correction-plan.md)。
> 根 R0：v202 Linear + v195 Attention，官方 `18053/281s`，SHA256 `56DC805D6E5A3AEF896DB8021045740292735725D688B48E3D4393E55EFCB2BD`。

## 1. 实现（2026-09-10）

- 候选：`workbench/full_solution/attention-ag1-joint-affine-gauge/candidate/solution.py`（R0 完整副本 + A-G1 增量）。
- 改动点（候选文件行号）：
  - `_a2_train_rotation`（~11343–11525）：训练循环内新增每 KV group 零均值 log-scale `s[groups, head_dim]`，
    `d=exp(s)`，`q_t=(QR)*d`、`k_t=(KR+c)/d`；解析梯度 `grad_s = sum(q_t·dQ_t) − sum(k_t·dK_t)`；
    s 与 theta/center 共用同一 Adam（β=0.9/0.999、lr=0.01、同 32 步，无新增循环/步数/配置），
    更新后组内零均值投影并截断 `±log(2)`。s 与 center 一样不做梯度裁剪（与父实现一致）。
  - `_a2_true_path_gate_loss`（~11528）：新增 `log_scale` 参数，注入 player q/k state 走完整部署路径。
  - `hif4_calibration_attention`（~11608）：最后 fold gate 严格改善才写入 (R,c,s)，否则三者恢复父 state。
  - `_nvfp4_to_hif4`（~4184, ~4291）：`learned_rotation`/`learned_center` 之后、`_dense_to_hif4` 之前，
    Q 乘 `exp(+s)`、K 乘 `exp(−s)`；V 不变。
  - `hif4_dynamic_quantize_q/k`（~10779/~10816）：仅传参，无候选循环/搜索/求逆。
- state：`q_state`/`k_state` 各新增 CPU float32 `learned_scale`；`validate_state` 通过。

## 2. Control 结果（全部 PASS，GPU 空闲时运行）

脚本：`workbench/full_solution/attention-ag1-joint-affine-gauge/control.py`，原始结果 `control_results.txt`。

1. **s=0 逐位恢复父**：Q/K 五字段与最终输出 `torch.equal`。
2. **合成非零零均值 s**：dense softmax 输出 max|Δ|=3.7e-9（数值精度内不变）；hard 五字段显著变化
   （Q mant 12247 / lv3 1338 / lv2 1141 / scale 333 / sign 481 处），机制可达且非 no-op。
3. **六 API 独立导入**：脱离仓库 importlib 加载通过，calibration→dynamic_q 跑通、输出 finite。
4. **记录与合法 state**：audit 含 `a2_arm/a2_gate_loss_identity/a2_gate_loss_rotation/a2_scale_l2/
   a2_scale_max_abs/a2_scale_nonzero`；12 seed gate 探针 6 次接受，接受时 `learned_scale` 正确写入
   （CPU f32、组内零均值 ~1e-9、max|s|≤0.234），拒绝时三者完整回退父 state。
5. **V / Linear control**：固定输入下候选 V 五字段、Linear 两 API 与根逐位一致。

## 3. 评测

- shard0：`artifacts/proxy_v3/attention-ag1-shard0-20260910/`。接口正常（`reasonableness_issues: 0`），
  输出非等价（12 case：`+/-/0 = 6/6/0`），机制真实改变了 hard output。
  本地诊断（不换算官方分）：delta_mean `-0.002287`，worst-20% tail `-0.020354`；
  误差集中在 length=10（mean `-0.016953`，min `-0.041174`），length=1024 仅 `-0.001743`。
  API 时间：calibration 8.028s、dynamic q/k/v 合计 0.569s（shard0 口径，无额外外层循环）。
- 六 shard（禁用早停完整跑完）：`artifacts/proxy_v3/attention-ag1-sixshard-full-20260910/candidate/`
  （`attention-ag1-sixshard-20260910` 是被默认早停截断的中间产物，只作过程记录）。72 case 结果：

  | shard | delta_mean | +/-/0 |
  |---:|---:|---|
  | 0 | -0.002287 | 6/6/0 |
  | 1 | -0.021423 | 6/6/0 |
  | 2 | +0.000000 | 0/0/12 |
  | 3 | -0.003990 | 7/5/0 |
  | 4 | +0.000814 | 4/8/0 |
  | 5 | -0.004880 | 5/7/0 |

  等权均值 `-0.005294`；manifest candidate overall `+0.528703` vs baseline `+0.533998`；合计
  28/32/12（72 case）。`all_outputs_finite=true`、`expected_case_coverage=true`；API total
  （诊断，2 次校准缓存命中）29.136s。shard2 与根逐位不变；shard1 负向集中在 test split 与中长
  序列（length 128/512/1024 均负）。

  结论：机制可达（60/72 case 硬输出改变），但本地六 shard 净效应为负。

## 4. 结果与裁决

- 裁决：`REJECTED`（本地六 shard 负向，等权均值 `-0.005294`）。
- 归档：`solutions/20260910_v227_attention-ag1-joint-affine-gauge_rejected_scoreNA_timeNA/`
  （solution.py / result.md / config.json / official-result.json / verification.json）。
- 候选 SHA256：`165e1a6bc50abd417a2945778741a5839a5636e7c93008bc34a2af47b3de674f`
  （与 workbench 候选逐位一致，归档副本已复核）。
- 未提交官方，`official_status: unregistered/NA`（用户将统一做官方评测）。
- 按计划 §7：不减少 A2 步数、不缩小尺度范围、不拆分 head/block 重试，不追加 reciprocal 参数邻域。
  根保持 R0（v202 Linear + v195 Attention，官方 `18053/281s`）。
