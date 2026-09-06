# Attention reciprocal per-KV-head temperature：REJECTED

- 日期：2026-09-06
- 状态：`CLOSED / T1_NOOP_REJECTED`
- 父版本：v189，SHA256 `261202248a0146a2ee45f3df60bd1979bb8171b7c162921013b0024c848617af`
- 候选源码 SHA256：`162CFCF2C0403D38176B16328FCE1A4D5DED010E196924286B9588D7757FC013`
- 根正式版本：v186，未修改

## 结果

候选仅打开已有 `_ATTN_OUTPUT_HEAD_SCALE=True`，使用固定 reciprocal factor 候选池，
没有改变 A1/D1、center/permutation/pair-smooth、Linear/V 或在线路径。T0 的六 API、
合法 state、Q/K multiplier 和有限输出检查通过。

T1 使用固定 proxy-v2 cache、CUDA、v189 baseline 的 Attention eval-v3 前两片：

| shard | cases | candidate mean | delta mean | L1 | 正/负/零 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 8 | 0.777742184 | 0 | 0 | 0/0/8 |
| 1 | 8 | 0.805886376 | 0 | 0 | 0/0/8 |

16 个配对 case 全部逐位一致，factor 候选没有进入最终 state。按 no-op 早停规则，
不运行 T2 六片、OOD 或 fresh default，不提交官方，不扫描 factor 或其邻域。

原始 eval-v3 证据：

- `artifacts/proxy_v3/attention-reciprocal-head-temperature-20260906/t1/`
- `logs/execution/2026-09-06-attention-reciprocal-head-temperature-plan.md`

官方分数/时间：`unregistered/NA`。本地 proxy 不换算官方分数。
