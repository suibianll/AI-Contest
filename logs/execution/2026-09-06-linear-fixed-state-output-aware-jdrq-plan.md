# Linear 冻结激活状态输出感知 JDRQ 执行记录

日期：2026-09-06  
计划：`2026-09-06-linear-fixed-state-output-aware-jdrq-plan`  
状态：**CLOSED / J1_REJECTED**  
根版本：v186（`F8495DCA20334ACBDAD16FC18EE41A4970F31E1837FDEEDCEE9C70AEE54E7EB8`）  
配对父：v189（`261202248a0146a2ee45f3df60bd1979bb8171b7c162921013b0024c848617af`）

## J0

候选文件：`workbench/linear_fixed_state_output_aware_jdrq_solution.py`  
候选 SHA-256：`aca047987fdffcf1c126a6ee37bca327ba0149e4ba5c65c47ae7b5a6900a9631`。

`py_compile`、单文件导入、六个 API 可见性和有限输出检查通过。运行输出确认 JDRQ
在全部 `896x4864` 宽下投影可达；root `solution.py` 未修改。

## J1

命令使用 eval-v3、CUDA、固定 proxy-v2 cache、`--linear-only --shards 0,1,2,3,4,5`，
baseline 为 v189。前两个 shard 后出现明确负向，按计划提前停止：

- 112 个配对 case：候选 mean `0.6343498783403304`，baseline mean
  `0.634575771204753`，整体 delta `-0.0002258928644226`；
- shard 0：delta mean `-0.0003604818362111497`，median `0`，L1
  `0.0003604818362111497`，`0+/8-/48=`；
- shard 1：delta mean `-0.00009130389263410246`，median `0`，L1
  `0.0002452943856971499`，`3+/5-/48=`；
- 最坏逐 case delta：shard 0 `-0.0040957015612991254`，shard 1
  `-0.004480031453440447`；主要负向来源为 `proj`，并见 `o`/`fc_gate`；
- 候选 API total `133.8395642 s`：weight calibration `99.9741984 s`，dynamic
  activation `33.8653658 s`。

## 决策

JDRQ 已满足 reachability，但在当前 eval-v3 的实际 `Q(A) @ Q(W)^T` 目标上为负，
关闭 JDRQ 接入及其邻域；J2 default/OOD/time、J3 官方提交均不执行，不分配 v190。
候选源码与摘要已归档至
`solutions/20260906_linear-fixed-state-output-aware-jdrq_rejected/`。

逐 case 原始证据：
[`manifest.md`](../../artifacts/proxy_v3/linear-fixed-state-output-aware-jdrq-20260906/j1/candidate/manifest.md)
