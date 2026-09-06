# 静态部署 Hessian 块序重排复核计划

> 创建：2026-09-06
> 状态：CLOSED / CANDIDATE-ARCHIVED-READY
> 父版本：根 `solution.py` v186（官方 `17599/272s`，SHA
> `F8495DCA20334ACBDAD16FC18EE41A4970F31E1837FDEEDCEE9C70AEE54E7EB8`）

## 1. 目的与边界

本计划只复核一个静态机制：在 `hif4_calibration_and_quantize_weight` 完成现有
v186 校准后，依据部署量化权重 `W_hat` 的列能量
`diag(W_hat.T @ W_hat)`，对完整 64-channel block 生成一次固定降序顺序；
恢复实现按原字节码的规则把完整 block 连续重排，连同激活、`h_inv`、importance
和已有 4x4 Gram 一起变换，执行同一个 activation GPTQ，再恢复输出布局。

它不改变权重编码、scale/lv/codebook、连续域变换、attention API 或在线候选数量。
块序是 calibration state 中的静态元数据，在线只做固定索引与既有 GPTQ；不能引入
校准搜索、逐样本选择、完整矩阵求逆或模型/层/role 路由。

历史产物 `artifacts/proxy_v3/linear_static_actorder_hdiag_full/` 曾记录旧评测入口下
336 个 Linear case 的 `0.6367996`，相对同次 v186 基线 `0.6328215`；原始 `.py`
已被清理，不能把该数字当作当前 eval-v3 结果。本轮使用保留 `.pyc` 行为重建为
独立单文件，旧数字只作恢复线索，最终以当前 `evaluator/eval.py` 为准。

## 2. 固定执行顺序

### R0：恢复与合法性

- 保留候选源为 `workbench/linear_static_actorder_hdiag_recovered_solution.py`；
  它必须能脱离 `workbench` 目录单文件导入六个正式 API。
- 仅允许 `gptq_block_order` 和标记字段进入 `activation_state`；顺序必须是
  `0..blocks-1` 的排列，block 大小固定为 64，所有 state tensor 可序列化且输出有限。
- 小形状 smoke、`py_compile`、现有 HiF4 合法状态检查通过后才进入评测。

### R1：当前 eval-v3 Linear 配对复核

使用固定 `proxy-v2` dense cache、CUDA、六个 shard，并以根 v186 作为 baseline。
候选必须在 default 336 Linear cases 上同时满足：

- `evaluation_scope` 为当前 `eval-v3`，case identity、`mse_standard`、energy 完整匹配；
- Linear mean 高于同一运行的 v186 baseline，且 focus median、q25/q75、worst
  quartile 不显示由单层/单 role 驱动的收益；
- 逐 case `L1 < 0.02`，负 case、validation/test 同号率和未修改 control 可解释；
- activation order reachability 在实际校准 state 中为 1，所有输出有限，无重复 case；
- 只记录 Attention 委托根版本的结果，不把 Linear 单侧 proxy 当官方分数。

若 R1 失败，标记 `REJECTED`，不扫 order、block score、threshold、seed 或 role
邻域；随后归档计划并继续寻找下一条未关闭的独立机制。

### R2：OOD 与时间审计

若 R1 通过，再对同一候选运行 `--ood` 六 shard，计算相对 v186 的
`Δ(gain_in - gain_ood)`。只有 `|Δgap| <= 0.01` 才允许进入提交准备；跨模型仅作
记录，不作晋级或否决依据。

用新鲜 default 调用统计估计官方时间，套用已校准的分解模型；预测时间必须 `<280s`
才具备官方提交资格。旧 shard wall time 不替代该时间审计。

### R3：归档与提交条件

仅在 R1/R2 通过且候选本地 default 分数高于当前本地最高有效父版本时：

1. 保存候选单文件、SHA256、父 SHA、完整 eval-v3 JSON/Markdown、OOD 和时间审计；
2. 在 `solutions/` 建立不可变的未裁决候选归档，官方字段写 `unregistered/NA`，
   不把本地 proxy 换算成官方绝对分；
3. 将该单文件作为一次新的官方候选提交。提交后根 `solution.py` 仍保持 v186，
   直到官方回传明确验证；不重复提交相同 SHA 或逐位等价版本。

若官方回传正向，才将它登记为新的正式父并更新根；若为负向或超时，保留完整证据、
关闭该机制族，不扫描其邻域。

## 3. 通过标准与停止条件

本计划不是对本地均值正向晋级的授权。合法性、有限输出、实际 reachability、
control 稳定性和当前 eval-v3 配对完整性是硬门；本地收益只决定是否进入 OOD/时间
审计，官方分数仍是唯一的最终裁决。Linear 目标 `0.9` 与 Attention 目标 `0.9`
继续作为长期目标，本候选不改变 Attention。

任何 R0/R1/R2 硬门失败都只产生一个固定配置的结论；不围绕该机制调参，不重跑已
确认的父版本，不修改根 v186。计划完成后移入 `docs/superpowers/archive/plans/`，
并在执行日志登记 `RETAINED`、`REJECTED`、`TIMEOUT` 或 `ERROR`。
