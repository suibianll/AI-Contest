# v124 C1c structured rank-8 accepted

- 状态：accepted precision parent；仅将结构化 kernel rank `_ACT_STRUCTURED_LRH_COMPONENTS` 从 4 提升到 8，`max_blocks=4`、`refresh_mode=sweep2` 保持不变。
- source `solution.py` SHA256（规范 LF）：`4ad7b1219cf73f6570690e3c919a2cdb1777402f2e99ae4a21f1162ea838b690`。
- screen：Linear mean `0.53343639`，较 v121 `+0.00003993`；proj `0.41386025`。
- full Qwen：Linear mean `0.5096493233`，Attention mean `0.8420394885`，panel `295.8202285103`，native total `423.3201361314`。
- 相对 v121：Linear `+0.0000357905`，panel `+0.0089476344`，proj `0.4222010863→0.4225375610`；其余 role 与 Attention 不变。
- API `2323.9111777s`，wall `2356.2005468s`，超过 420s；accuracy-first 接受，C3 再压缩 state/计算。
- 26 项核心回归、screen/full 评测和静态/运行时 compliance 通过；`official_flow_valid=false` 仅因本地 CPU API 超时，不能直接当官方分数。

rank-2 v122 与 max_blocks-2 v123 已分别归档拒绝。
