# Linear 静态 activation-GPTQ 条件曲率块序候选（REJECTED）

状态：**REJECTED / C1 负向**。该目录是可复现实验快照，不是正式提交版本。

- 父：v189；本地 default Overall `0.686889608842`，根正式版本仍为 v186。
- 候选 SHA256：`54638e1146a07558b019c098f1496b3fa5d33f6805026e08a5ff855338e1d5ec`。
- C1 使用 eval-v3 Linear 六 shard 请求，但按负向门禁在 shard 1 后停止；实际 112 个配对 case。
- shard 0 / 1 的 Linear delta mean 为 `-0.000491` / `-0.000125`，L1 为
  `0.001316` / `0.001128`；shard 0 正/负/零为 `18/38/0`，shard 1 为 `27/29/0`。
- 主要回归角色为 `o` 与 `fc_gate`；shard 0 的 `o` mean delta 为 `-0.001068`，
  shard 1 的 `fc_gate` mean delta 为 `-0.001287`。
- 条件曲率排序真实 reachability 已确认；没有运行 default/OOD，也没有官方提交。

证据：`artifacts/proxy_v3/linear-static-gptq-conditional-curvature-20260906/c1/`。

该单一块序规则已关闭，不扫描 `diag(H)`/`diag(H^{-1})` 混合权重、反向排序或其他邻域。
