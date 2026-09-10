# A-QB1 Q 侧加性 logit 偏置补偿执行记录

> 计划：[Attention Q 侧加性 logit 偏置补偿计划（A-QB1）](../docs/superpowers/plans/parallel/2026-09-10-attention-qk-logit-bias-plan.md)。
> 根 R0：v202 Linear + v195 Attention，官方 `18053/281s`，SHA256 `56DC805D...EFCB2BD`。
> 前卡 A-G1（v227）归因：gauge 窗口特异收益被单窗口 gate 反定价、联合训练拖垮 rotation；
> 本卡按其约束设计（冻结根全部已有状态、全 folds 训练与 gate）。

## 1. 实现（2026-09-10）

- 候选：`workbench/full_solution/attention-aqb1-q-bias/candidate/solution.py`（根逐位副本 + 255 行纯新增）。
- 改动点（候选行号）：
  - `_nvfp4_to_hif4`（4184、4290–4303）：新增 `learned_q_bias`，在 `learned_rotation`/`learned_center`
    之后、`_dense_to_hif4` 之前对 Q 逐元素加；异常静默退化为无偏置合法路径。
  - `hif4_dynamic_quantize_q`（10774）：仅传参，动态 API 只多一次加法。K/V/Linear 路径不动。
  - A-QB1 段（11613–11851）：`_aqb1_deployed_window_mse` / `_aqb1_train_q_bias` /
    `_aqb1_true_path_gate_losses` + 新 `hif4_calibration_attention`——先原样执行根校准
    （含根自己的 A2 rotation 与单窗 gate），再在其最终 state 上训 `b_q`（全部 5 folds 窗口等权、
    Adam 32 步、lr/β/clip 复用根 `_A2_TRAIN_*` 常量）并逐层全 folds 真实 MSE gate，
    严格改善才写 `q_state["learned_q_bias"]`。

## 2. Control 结果（全部 PASS，GPU 空闲时运行）

脚本：`workbench/full_solution/attention-aqb1-q-bias/control.py`，原始结果 `control_results.txt`。

1. **b_q=0 逐位恢复父**：Q/K 五字段与最终输出 `torch.equal`。
2. **合成非零 b_q**：Q 五字段显著变化（mant 21109、sign 12366、lv3 2880、lv2 1367、scale 382 处），
   最终输出 max|Δ|=5.9e-5（有意改变真实函数，符合机制定义）；K/V 逐位不变。
3. **六 API 独立导入**：脱离仓库 importlib 加载通过，calibration→dynamic_q 跑通、finite。
4. **合法 state + 记录**：`validate_state`/`validate_hif4_params` 通过；audit 含
   `aqb1_arm/aqb1_gate_loss_parent/aqb1_gate_loss_candidate/aqb1_steps/aqb1_windows/aqb1_bias_l2/
   aqb1_bias_max_abs/aqb1_bias_nonzero`；`root_rotation_frozen=True`（根 rotation 逐位不动）。
5. **V / Linear control**：与根全部逐位一致。
6. **Gate 双路验证**：合成数据 8/8 种子真实接受（改善 1.9%–2.7%）；注入有害 bias（全 50.0）正确
   回退为 parent 且不写字段。接受/回退两条路径均验证。

## 3. 评测

- shard0：`artifacts/proxy_v3/attention-aqb1-shard0-20260910/`，接口正常，`reasonableness_issues: 0`。
- 六 shard（72 case，`--stop-after-nonpositive 6` 跑满，禁用默认早停）：
  `artifacts/proxy_v3/attention-aqb1-sixshard-full-20260910/candidate/`。

| shard | delta_mean | +/-/0 |
|---:|---:|---|
| 0 | -0.041709 | 1/11/0 |
| 1 | -0.063230 | 0/12/0 |
| 2 | -0.063832 | 0/12/0 |
| 3 | -0.105430 | 0/12/0 |
| 4 | -0.013893 | 1/11/0 |
| 5 | -0.030970 | 1/11/0 |

等权均值 `-0.053177`，合计 3/69/0；manifest candidate overall `+0.480820` vs baseline `+0.533998`；
API total（诊断，1 次校准缓存命中）51.815s；shard3（层15）最差。

## 4. 结果与裁决

- 裁决：`REJECTED`（本地六 shard 强负向，六层全负），未提交官方，官方状态 `unregistered/NA`。
- 归档：`solutions/20260910_v228_attention-aqb1-q-bias_rejected_scoreNA_timeNA/`，
  候选 SHA256 `886a8b17736aa51a56d9cb595208975949fbfb815c597adc1ea1c7a7706c824a`。
- 归因：机制可达且 gate 真实接受（control 8/8 种子接受、改善 1.9%–2.7%），但全 folds 训练 +
  全 folds gate 仍全部六层负向——Q 偏置拟合到的"系统性 logit 偏差"是校准窗口特异而非量化器
  固有属性。结合 v227（gauge）与 A-RB1（舍入边界），逐通道/逐元素级校准拟合自由度在 Attention
  侧第三次被否决。按计划 §6 关闭本实现，不以步数/lr/fold/head 粒度重试。
- 根不变：v202 Linear + v195 Attention，官方 `18053/281s`。
