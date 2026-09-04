# v187 官方结果：9167 / 169s（2026-09-04）

- 版本：`20260904_v187_attn-jacobian-sensitivity_research-retained`
- 源码 SHA256：`086535FB4205703524C5DF2378CF2557B7F4652DF03E6FA201C074F2094F8F65`
- 直接父：v185 `8446/165s`
- 官方结果：`9167/169s`
- 增量：`+721/+4s`
- 裁决：Jacobian sensitivity 机制官方正向，RETAINED 为研究机制；完整官方父仍为 v186

v187 唯一增加最终 Attention 输出的一阶 Jacobian Q/K 坐标 importance，压缩为 KV-group
共享的 `KV-head×64` 参数并由 leave-one-fold-out gate 部署。官方 +721 证明本地正向不是
同分布假象；169s 说明校准 Jacobian 与固定在线 importance 的成本可控。

总分仍比 v186 `17599/272s` 少 `8432`，差距来自 v185 clean-room 基座缺少成熟块级机制，
不是 Jacobian 方向失败。处置：保留机制、不调收缩/clamp/gate 邻域；下一候选若启动，应把
相同机制作为单变量移植到 v186。该提交计为第 8 个，配额 `8/10`、剩余 2。
