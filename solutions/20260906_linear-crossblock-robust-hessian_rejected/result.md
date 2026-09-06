# Linear 多折 cross-block Hessian 候选（REJECTED）

状态：**REJECTED / B1 负向**。这是研究快照，不是官方版本，不生成提交包。

- 父：v189，本地 Overall `0.686889608842`；根保持 v186。
- 候选 SHA256：`96a6bacbdc9057b0dfaa282d969347fd732b62c92e0f543db501a4789d22e62e`。
- B1 前两 shard 共 112 个配对 Linear case：shard 0/1 delta mean
  `-0.202975/-0.185789`，L1 `0.202975/0.185789`，均为 `0/48/8` 正/负/零。
- 证据：`artifacts/proxy_v3/linear-crossblock-robust-hessian-20260906/default-b1/`。

多折 covariance conditional 更新在真实部署输出上严重回退，已关闭且不扫邻域。
