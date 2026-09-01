# v138：v86 级静态 Attention 时间父版本

v130 的官方 timeout（本地 API `295.437s`）触发了 Attention 时间回溯审计。v138 从已验证
的 v134 Linear 路径出发，仅压缩 Attention：

- 候选只保留少量 reciprocal balance、block-Hadamard 和 GQRB 静态变换；
- proxy/shortlist 均使用 128 token；
- Attention calibration 关闭逐候选 Gram 坐标精修；
- 动态 Q/K/V 不再执行 Gram sweep，在线只做静态变换和普通合法 HiF4 编码；
- Linear 的输出监督 `Q(W)^T W` / `Q(W)^T W_t` cross64 保持不变。

## 本地完整复测

| run | Linear | Attention | W-cal | A-dyn | Attn-cal | Q/K/V-dyn | API total | wall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| first | 0.5073195 | 0.7159420 | 134.501 | 16.915 | 36.253 | 5.327 | 192.996 | 216.324 |
| rerun2 | 0.5073195 | 0.7159420 | 129.982 | 16.046 | 36.785 | 5.123 | 187.935 | 210.855 |

结果逐位一致。Attention 均值接近 v86 的 `0.719696`，但本地 API 明显低于 v134；这只是
时间风险降低的证据，不能替代官方平台计时。v138 提升为当前根和后续 Linear 优化父版本。
