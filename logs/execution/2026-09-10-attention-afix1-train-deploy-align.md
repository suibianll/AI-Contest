# A-FIX1 A2 训练/部署前向对齐执行记录

> 计划：[Attention A2 训练/部署前向对齐计划（A-FIX1）](../docs/superpowers/plans/parallel/2026-09-10-attention-train-deploy-align-plan.md)。
> 根 R0：v202 Linear + v195 Attention，官方 `18053/281s`，SHA256 `56DC805D...EFCB2BD`。
> 依据：[推进瓶颈审计](../docs/optimization-stall-analysis-2026-09-10.md) §1/§5.3——训练前向裸
> `_dense_to_hif4` 与 gate 完整部署路径不一致。
> 前卡 A-MC1（v229）官方 TIMEOUT，A-QC1 关闭 `NO_EFFECT`；本卡是瓶颈审计后的第一张训练
> 目标保真度卡。

## 1. 实现（2026-09-10）

- 候选：`workbench/full_solution/attention-afix1-train-deploy-align/candidate/solution.py`
  （根逐位副本，只改 `_a2_train_rotation` 训练前向）。
- 机制：训练循环内 Q/K 的量化从裸 `_dense_to_hif4` 换成完整部署编码（与 gate 相同的
  `hif4_dynamic_quantize_q/k` 路径，player state 携带当前 rotation/center），反向 STE
  （`_m_attention_backward`）不变。参数化（仍是 rotation+center）、步数（Adam 32 步）、
  lr/β/clip、窗口、gate 全部与根相同；不新增变换族、不新增参数。
- 逐层记录训练前后 train loss、gate 父/候选 loss、`a2_arm`、相对根的 rotation/center 差异范数。

## 2. Control 结果（全部 PASS，GPU 空闲时运行）

脚本：`workbench/full_solution/attention-afix1-train-deploy-align/control.py`，原始结果
`control_results.txt`；种子探针 `seed_probe.py` / `seed_probe_results.txt`。

1. **0 步逐位恢复父**：Q/K/V 五字段与最终输出逐位一致。
2. **前向一致性**：训练前向与直接调用部署 API 逐位一致（8 次 Q + 8 次 K 录制核对，
   调用次数、player state 形状、子采样 token、rotation 步间更新均验证）。
3. **六 API 独立导入**：脱离仓库 importlib 加载通过，端到端输出 finite。
4. **合法 state**：`validate_state` / deployed params 检查通过。
5. **V / Linear control**：与根全部逐位一致。
6. **非等价性（种子探针）**：对齐前向把训练推向 (R, c) 流形上不同的点，候选与根非等价。

## 3. 评测

- shard0：`artifacts/proxy_v3/attention-afix1-shard0-20260910/`，接口正常；shard0 校准 API
  `11.017s`（根约 8s，对齐前向带来约 1.4× 校准开销——计划 §2 已声明的固定代价）。
- 六 shard（72 case，`--stop-after-nonpositive 6` 跑满）：
  `artifacts/proxy_v3/attention-afix1-sixshard-full-20260910/candidate/`。
- shard↔层对应：shard0→层0、s1→层1、s2→层8、s3→层15、s4→层22、s5→层5。

| shard（层） | delta_mean | +/-/0 |
|---:|---:|---|
| 0（层0） | -0.000133 | 7/5/0 |
| 1（层1） | -0.004687 | 8/4/0 |
| 2（层8） | +0.000000 | 0/0/12 |
| 3（层15） | -0.013820 | 4/8/0 |
| 4（层22） | +0.001307 | 7/5/0 |
| 5（层5） | -0.011968 | 3/9/0 |

等权均值 `-0.004884`，合计 29/31/12；manifest candidate overall `+0.529114` vs baseline
`+0.533998`；API total（诊断，1 次校准缓存命中）34.475s。

## 4. 结果与裁决

- 裁决：**官方 TIMEOUT (>300s)，REJECTED**（2026-09-10 用户回传"v230-attention也超时了"），
  精确秒数与分数未知；本地 `−0.004884` 保留为诊断，未获官方精度定价。见
  [官方回传记录](2026-09-10-v230-attention-afix1-official-timeout.md)。
- 归档：`solutions/20260910_v230_attention-afix1-train-deploy-align_rejected_scoreNA_timeNA/`，
  计分/归档 SHA256 `c2ff4ea0d6a3823e29351b616c330fa9358b588934e73019c382183130dfcd6f`。
- 时间归因：校准约 1.4× 的对齐前向成本与 v229 的校准期 gate 前向同成本类，在官方机上
  不可行；只关闭该实现，训练/部署对齐路线重试前必须先消除该校准成本，不缩步/缩窗重试。
- 解读：机制可达、非等价；但本地净负。层15 在 v227 和本卡两次重训 rotation 都明显变差
  （本次 `-0.013820`），而层15 在根中接受 rotation（gate +2.43%）——「冻结父 rotation 叠加
  增量」（A-MC1 式）与「重训 rotation」（本卡）的对照证据再次确认：根的 rotation 臂不宜重训。
- 版本号注意：并行 Linear 线 L-EM2 已登记 v230
  （`solutions/20260910_v230_linear-em2-groupstep-schedule_scoreNA_timeNA/`），构成第二次
  编号冲突（继 v204/v205）；按既有先例不重命名目录，引用时写全目录名。
- 根不变：v230 Linear（L-EM2）+ v195 Attention，官方 `18428/292s`（余量 8s）。
