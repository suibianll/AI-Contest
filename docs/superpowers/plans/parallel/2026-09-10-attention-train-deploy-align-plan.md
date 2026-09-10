# Attention A2 训练/部署前向对齐计划（A-FIX1）

> 状态：ACTIVE 执行附录，2026-09-10。
> 从属于当前活动总计划（Linear 线）。当前完整根为 v202 Linear + v195 Attention，官方 `18053/281s`。
> 本文件只负责 Attention A-FIX1；版本登记、组合与根切换由总协调线处理。
> 依据：[推进瓶颈审计](../../optimization-stall-analysis-2026-09-10.md) §1 与 §5.3——
> 「重点处理训练与部署量化路径不一致，冻结已证实有效的父 rotation」。

## 1. 问题定义（代码已核实）

根 `_a2_train_rotation`（`solution.py:11343`）的训练前向用裸 `_dense_to_hif4(q_rot/k_shift)`
（:11409-11410）量化 Q/K，而最终 gate `_a2_true_path_gate_loss`（:11470）用完整部署路径
`hif4_dynamic_quantize_q/k/v`（含 `_nvfp4_to_hif4` 的 offset 搜索、refine、margin 接受规则）。
训练优化的离散映射与部署执行的离散映射不同：STE 反向给的梯度是针对简化量化器的，
rotation/center 被优化到"简化路径下好"的点，部署路径下不一定好。gate 只在训练结束后检查一次，
无法修正 32 步轨迹积累的偏差。

## 2. 机制（固定，无新自由度）

把训练前向中的 Q/K 量化从 `_dense_to_hif4` 换成**完整部署编码**（与 gate 相同的
`hif4_dynamic_quantize_q/k` 路径，携带 player state 的 rotation/center 当前值），
反向仍用现有 `_m_attention_backward`（STE）从部署输出回传。即：只对齐前向离散映射，
不动参数化（仍是 rotation+center）、不动步数/lr/窗口、不动 gate、不动 V。

- 这**不新增变换族**：搜索的仍是同一 (R,c) 流形；关闭的 rotation/event/group/seed/block 邻域
  指"在同一训练目标上加密搜索"，本卡改的是训练目标的保真度。
- 不是 v222：v222 改梯度聚合与异常传播，本卡不改反向规则，只改前向量化器。
- 每步成本上升（部署编码含 offset 搜索），校准总时长会增加；这是已知的固定代价，
  不为压时间缩步数/窗口。

## 3. 固定算法

1. 从 R0 复制完整单文件候选；只改 `_a2_train_rotation` 内 Q/K 的前向量化调用，
   其余逐行保持（同一 Adam、同一 32 步、同一子采样、同一单窗口 gate）。
2. control：训练步数=0（或等价开关）时与根逐位一致；新前向产出的 Q/K 五字段与
   `hif4_dynamic_quantize_q/k` 直接调用逐位一致；六 API 独立导入；`validate_state` 通过；
   V/Linear 不变。
3. 记录每层训练前后 train loss、gate 父/候选 loss、`a2_arm`、相对根的 rotation/center 差异范数。
4. GPU 串行；先 shard0，再 `--shards 0,1,2,3,4,5 --stop-after-nonpositive 6` 跑满。
5. 本地正负只记录；合法、可达、非等价候选归档并交官方裁决（新规则：本地负向不再是提交门，
   官方状态 `unregistered/NA`，由用户统一评测）。

## 4. 完成条件

- 逐位相同或无层接受：`NO_EFFECT`；否则归档一个版本（本地正负只标注，官方待用户评测）；
- 官方 TIMEOUT：只关闭该实现，不以缩步/缩窗重试；
- 完成后本文件归档；不追加"对齐 V 侧""对齐 gate 窗口数"等第二批改动（那是新卡，需新证据）。

工作目录：`workbench/full_solution/attention-afix1-train-deploy-align/`。
日志：`logs/execution/2026-09-10-attention-afix1-train-deploy-align.md`。
输出：`artifacts/proxy_v3/attention-afix1-<run-id>/`。
