# LC0 Linear result

| 项目 | 结果 |
|---|---|
| 候选 | `continuous_linear_lc0-correctness-hardened` |
| 父版本 | `continuous_linear_l28-proj-vectorized` |
| 源码 SHA256 | `E702D7C1F087097906E752DEFA5CBFF05C2E41585883B0EC8D5E4F937AA4AF3D` |
| 官方状态 | `unregistered/NA` |
| 官方分数 / 时间 | 未提交 |
| 本地协议 | `proxy-v3`, 4B, Linear-only, shard0 |
| 本地 case 数 | 56 |
| 本地状态 | `ok` |
| LC0 local proxy mean | `0.2884021593` |
| L28 local proxy mean | `0.2910968103` |
| paired delta | `-0.002695` |
| reasonableness issues | 0 |

真实 shard 证据位于：

```text
artifacts/proxy_v3/linear_lc0-correctness-retry-shard0/candidate/candidate-linear-shard0.json
artifacts/proxy_v3/linear_lc0-correctness-retry-shard0/candidate/candidate-linear-shard0.md
artifacts/proxy_v3/linear_lc0-correctness-retry-shard0/analysis.json
artifacts/proxy_v3/linear_lc0-correctness-retry-shard0/analysis.md
```

结论：LC0 的 correctness battery 和真实 evaluator 接口 smoke 通过，但单 shard local proxy 相对 L28 回退，故只作为独立修复审计候选归档，不替换 L28，也不进行官方晋级声明。
