# v185 官方结果：8446 / 165s（2026-09-04）

- 版本：`20260904_v185_cleanroom-robust-operator_rejected`
- 源码 SHA256：`3EA046594FB18DD86FD8CCFD2364A391039B0112E29986C8F949F9AF526C136C`
- 官方分数：`8446`
- 官方时间：`165s`
- 裁决：REJECTED

v185 是完全从六 API 与 HiF4 合法域重写的 clean-room 低自由度实现。官方时间远低于 300s，
因此失败不是超时；相对当前父 v186 `17599/272s` 少 `9153` 分，确认简单 Linear 对角平衡和
Attention K-center/QK-balance/logit-gain/稀疏 scale refine 无法恢复成熟块级算法的表达能力。

处置：关闭 v185 的 balance/gamma/refine 参数邻域；保留源码作研究基线。该次提交按用户账本
计为第 7 个，配额更新为 `7/10`、剩余 3。根 `solution.py` 保持 v186。
