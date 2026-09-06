# Attention Q/K 独立 headwise permutation：REJECTED

- 日期：2026-09-06
- 状态：`CLOSED / P1_NOOP_REJECTED`
- 父版本：v189，SHA256 `261202248a0146a2ee45f3df60bd1979bb8171b7c162921013b0024c848617af`
- 候选源码 SHA256：`5F94099DAFD5F44201430A943DDEC75EBCE3AB8F735C90EED5774D92ECA69683`
- 根正式版本：v186，未修改

## 结果

候选仅打开已有 `_ATTN_OUTPUT_HEADWISE_PERMUTATION=True`，没有改变 Linear/V、父版本
scale/center/pair-smooth 或在线搜索。P0 的六 API、合法 state 和有限输出检查通过。

P1 使用固定 proxy-v2 cache、CUDA、v189 baseline 的 Attention eval-v3 前两片：

| shard | cases | candidate mean | delta mean | L1 | 正/负/零 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 8 | 0.777742184 | 0 | 0 | 0/0/8 |
| 1 | 8 | 0.805886376 | 0 | 0 | 0/0/8 |

16 个配对 case 全部逐位一致，实际独立 permutation 分支没有产生部署 state。按预注册
no-op 早停规则，不运行 P2 六片、OOD 或 fresh default，不提交官方，不扫描排列参数。

原始 eval-v3 证据：

- `artifacts/proxy_v3/attention-headwise-permutation-20260906/p1/`
- `logs/execution/2026-09-06-attention-headwise-permutation-plan.md`

官方分数/时间：`unregistered/NA`。本地 proxy 不换算官方分数。
