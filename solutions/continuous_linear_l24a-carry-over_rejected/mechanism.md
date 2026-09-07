# L24-A：跨块残差 carry-over（REJECTED）

研究循环卡 L24-A，靶点 E2（合法投影损失 0.0320）。2026-09-08 实现并本地验证，**关闭**。

## 机制

L23b 逐块独立求解/接受/拒绝，拒绝块与合法投影的残差被丢弃。L24-A 采用
OBC/GPTQ 式输出坐标误差补偿：把每块连续解与合法解的差异残差
`C += Xb @ (Wd − W_legal)ᵀ` 携带到下一块的求解目标（`_l23_block_solve(Xb, R+C)`），
接受判定仍用真实输出残差保证 L_all 单调下降。

## 证据（4B shard0 paired，父 L4）

| 指标 | L23b | L24-A |
|---|---|---|
| shard0 fit_gain（校准折叠） | 0.9478 | **0.8243** |
| 宽层 accepted | 144/144 | **3/144** |
| shard0 delta_mean（vs L4 面板） | -0.2182 | -0.1125 |

**关闭原因**：carry 残差改变了后续块的求解方向，接受几乎全部崩溃（3/144），
校准拟合精度从 0.948 跌到 0.824——E2 未降（反而 E1/E2 合计大幅上升）。
按卡证伪判据（"同校准数据最终合法部署误差未降"）**关闭该实现**。

## 关闭粒度

只关闭"跨块 carry-over 这一实现"。E2 格仍 OPEN。E4（0.6143）仍是最大 OPEN 格，
但 L24-C 分桶实现已关闭，需新的 E4 机制卡。L23b（13639FB2）保持 READY_FOR_OFFICIAL。

## 产物

- 源码：`workbench/continuous_linear/l24-a-carry-over/candidate/solution.py`
  （SHA `96B5EAFB166DC4BC…`）
- 归档：`solutions/continuous_linear_l24a-carry-over_rejected/`
- 评测：`artifacts/proxy_v3/continuous/linear/l24-a-carry-over/shard0/`