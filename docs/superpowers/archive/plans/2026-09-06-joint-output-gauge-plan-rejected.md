# 与网格共存的联合输出坐标计划

> 日期：2026-09-06。状态：**CLOSED / J0_REJECTED**。
> 这是 D-A/D-B 关闭后的新独立计划。根 `solution.py` 保持 v186；不重开已关闭的单侧
> scale、mantissa、rank、permutation、动态 per-call 或 Jacobian importance 邻域。

## 1. 依据与假设

D-A 显示 clip 只占最近网格 SSE 的 `8.89%`，D-B 的固定 64-block 子组重分区在重点层
没有 output oracle 余量，因此下一未直接测量的自由度是：**同一产品中的 X/W（或 Q/K）
是否需要共同选择进入 HiF4 网格的坐标**。当前 v186 虽在若干校准步骤使用 `A@W`，但
部署状态仍由分离的 operand codec 和固定候选门组成；本计划只测一个可折叠的、双侧同时
变化的静态 gauge，不把单侧 operand MSE 当成输出收益。

这里的“联合”指保持连续乘积不变的 `A_g = A G^{-1}`、`W_g = W G`（Attention 为
`Q_g = Q G`、`K_g = K G^{-T}`），再分别用合法五字段编码；不是单独改 sf/lv 值，
也不是把输出残差存入在线 state。若 oracle 没有 material gain，联合坐标路线关闭。

## 2. J0：固定 group-gauge 输出 oracle

使用 v186 已验证 calibration artifact 和 proxy-v2 cache，不改正式代码：

- Linear 固定层 `[0,8,15,23]`，重点 role `fc_gate/fc_up/proj`，并以其余四 role 作
  control；每个 state 使用全部五个 calibration window，每窗均匀取 32 token、权重取
  32 行、均匀取 4 个 64-block。
- 每个 64-block 固定按现有 `8×2×4` 位置切成 8 个连续 8-channel group。仅对每组使用
  一个共享的 paired diagonal gauge，候选集合预注册为
  `{2^-1/4, 1, 2^1/4}`；W 乘 gauge，X 乘逆 gauge。选择只在 calibration fold 内进行，
  用 leave-one-window-out 的实际 `X_hat W_hat^T` MSE，不能读取 holdout/test。
- 每个 fold 同时保留当前固定坐标 exact legal block arm，和 group-gauge 后的 exact
  legal block arm；两臂都调用 `evaluator/reference_hif4.py`，并报告 operand SSE、
  output MSE、fold gain、最坏 fold。该 arm 是离线 upper-bound/oracle，不是候选。
- Attention 使用 Q head 0 与最后 KV head 的对应 64-block，按 32 token 取样；同一组
  8-channel gauge 应用于 Q/K 的 paired diagonal（Q 乘 G，K 乘 G^-T），报告 Q-only、
  K-only、joint GQA output/logits/probability，并保留父动态量化作参照。

## 3. J0 固定门

Linear 重点层的 leave-one-window-out median output gain 必须 `> 0.01` 至少 3/4 层，
且至少一浅层（`<15`）和一深层（`>=15`）通过；control 的中位 gain 不得低于 `-0.01`。
Attention joint output gain 必须在至少 3/4 固定层的 fold-median 为正，且 logits 与
probability 不同时恶化。任一侧未过门只关闭该侧，另一侧仍按自身证据裁决。

通过 J0 也只允许进入 J1 机制卡：J1 才能把同一个固定 group-gauge 规则折叠到
`smooth_inv`/Attention `pair_transform`，做单一完整 eval-v3 候选。J0 失败不得扫
候选集合、fold、group size 或 gauge 指数邻域。

## 4. J1（条件）与官方边界

若 J0 通过，另立 J1 计划，固定一个非 output-aware 的统计实现（只用 calibration
fold 的 group RMS/输出无关量），加入跨 holdout、`L1 < 0.02`、OOD `|Δgap|<=0.01`
和接口/有限值检查；时间使用分解模型预测 `<280s`。不通过则根保持 v186。

完整候选必须先用 `evaluator/eval.py` 六 shard 做单侧 default audit；只有完整本地
Linear/Attention 均超过各自本地当前最高且时间安全，才归档版本并提交官方。官方分数
仍是唯一晋级裁决，本地数值不能换算官方绝对分。

## 5. 产物与停止

实现 `workbench/joint_output_gauge_probe.py` 和最小测试；产物根为
`artifacts/proxy_v3/joint-output-gauge-20260906/j0/`，manifest 必须绑定父源码、配置、
cache、参考解码器、工具 SHA 与 GPU 环境。结果无论正负都写入 `logs/execution/`；J0
完成后立即归档本计划或另立 J1，不在本计划内做邻域扫描。

## 6. 执行结果（2026-09-06）

J0 已在固定 v186、proxy-v2 cache、CUDA GPU 上完成。共执行 28 个 Linear state、4 个
Attention 层、每项 5 个 leave-one-window-out fold；manifest 绑定了父源码、配置、cache、
参考解码器和工具 SHA。结果见
[`J0 result.json`](../../../artifacts/proxy_v3/joint-output-gauge-20260906/j0/result.json)、
[`J0 report`](../../../artifacts/proxy_v3/joint-output-gauge-20260906/j0/report.md) 和
[`J0 manifest`](../../../artifacts/proxy_v3/joint-output-gauge-20260906/j0/run_manifest.json)。

- Linear：重点层中位 output gain 为 L0 `0`、L8 `0`、L15 `0`、L23 `0`；
  通过 `>1%` 的重点层为 `0/4`，control 中位 gain `0`，未通过固定门。
- Attention：joint fold-median output gain 为 L0 `-0.269746`、L8 `0`、L15 `0`、
  L23 `-0.079147`，正层 `0/4`；未通过固定门，且 L0/L23 的 probability gain 为负。
- 结论：固定 paired group-gauge oracle 未显示可部署材料收益，按预注册规则关闭，不扫描
  gauge、fold、group-size 或指数邻域；根 `solution.py` 保持 v186，未创建候选、未提交官方。
