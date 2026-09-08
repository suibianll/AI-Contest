# continuous-attention 状态迁移与官方结果登记（2026-09-07）

> 历史契约：[已归档持续优化总计划](../../docs/superpowers/archive/plans/2026-09-07-continuous-linear-attention-plan-superseded.md)。
> 原因：A 侧此前按 v162-independent 结构写状态，协调者新结构（workbench/continuous_attention/）
> 看不到 R3/R2c/A2c 官方结果——本文件与 state.json/queue.md 完成迁移登记。

## 官方结果矩阵（全部官方实测，回传日期 2026-09-07）

| 卡 | 官方分/时间 | C_A | 归档路径 | solution SHA256 |
|---|---|---|---|---|
| **R3** 旋转+学习K-center（全门通过） | **14405 / 238s** | **13404** | `solutions/v162_attention_r3-rotation-center_allgates/` | `A5C679D7A2B349A879B2019B4A613244F5FEA7050E407B6475B9E29BD1C146DC` |
| R2c 栈内旋转（手工梯度） | 14387.8 / 240s | 13386.8 | `solutions/v162_attention_r2c-rotation-in-stack-manual_oodblocked/` | `CFDDCED7886A0CABC7251C57387536B86169185E7A6D8BE7A8647FEFAE677CD6` |
| A2c 独立旋转（手工梯度） | 9538 / 175s | 8537 | `solutions/v162_attention_a2c-rotation-manual-trainer_oodblocked/` | `FD902933ADF9AB70C816169BE744FAFE8D5DCDC51E1E1ED7B871076C973B406D` |
| R1 v189 栈 + 标准 Linear | 14009 / 211s | 13008 | `solutions/v162_attention_r1-v189-attnstack-recovery_officialNA_timeNA/` | `3619BFEB0E017555BD8FE31410888F80950A0128E3A2B94F643BD3F20B30BFC8` |

## 关键事实

1. **官方最佳 = R3（14405/238s）**，OOD Δgap +0.0065（按 2026-09-07 门禁修订为风险提示级）。
2. **官方环境 autograd 完全不可用**：A2b/R2b 官方回退（=父逐分）+ 本地 inference_mode 复现；
   A2c/R2c/R3 全部使用解析梯度训练器（parity: attention 1.1e-07 / Cayley 4.5e-06）。
3. **OOD 门排序价值被官方验证**：过门 R3 > 未过 R2c（+17.2）；绝对阈值校准待协调者
   （R2c 弃提交损失 +378.8 教训并存）。
4. **可加性**：A2c 独立 8537 vs R2c 栈内增量 378.8 ⇒ 交互 −8158（同覆盖 Q/K 坐标 DOF）。
5. **组合前景**：R3 + v189 Linear 侧 ⇒ 1001+3606+13404 ≈ **18011**（v189=17616，+395）。
6. **机制族结论**：Q/K 坐标（旋转）+平移（center）联合自由度在栈内已吸收
   （R3 default 0.7650 vs R2c 0.7678）；通往 0.9 需正交新族（A1/A2/A3 队列）。

## 结构迁移说明

- 历史 eval 结果保留在 `artifacts/proxy_v3/v162-independent/attention/`（r2c-id/r2c-ood/
  r3-id/r3-ood/r3-default 等，manifest 身份已核验）；新 run 按计划写
  `artifacts/proxy_v3/continuous/attention/`。
- 执行日志沿用 `logs/execution/v162-independent-attention.md`（完整历史）+ 本目录新
  `continuous-attention-<run_id>.md`（每轮独立）。
- 工作副本：`workbench/v162_attention/`（候选 candidate/candidate_b/candidate_c/
  candidate_v2/v3/v3b/v3c/v3d + fuzz 工具）；新机制候选将复制到
  `workbench/continuous_attention/<run_id>/`。
- 下一动作：A0 部署目标可验证化（evidence-repair §6 A-R1 + continuous-attention A0）。
