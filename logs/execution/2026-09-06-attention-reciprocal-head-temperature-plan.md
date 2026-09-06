# Attention reciprocal per-KV-head temperature 执行记录

日期：2026-09-06

计划：`2026-09-06-attention-reciprocal-head-temperature-plan`

状态：**CLOSED / T1_NOOP_REJECTED**

父版本：v189，SHA256 `261202248a0146a2ee45f3df60bd1979bb8171b7c162921013b0024c848617af`

候选源码：`workbench/attention_reciprocal_head_temperature_solution.py`

候选 SHA256：`162CFCF2C0403D38176B16328FCE1A4D5DED010E196924286B9588D7757FC013`

## 执行

- T0：`py_compile`、六 API 独立导入、合成 GQA calibration/dynamic 调用、state/参数合法性
  和有限输出检查全部通过。
- T1 shard0：8 个 Attention case，candidate mean `0.777742183683`，与 v189 逐 case
  一致，delta mean `0`、L1 `0`、正/负/零 `0/0/8`。
- T1 shard1：8 个 Attention case，candidate mean `0.805886376364`，与 v189 逐 case
  一致，delta mean `0`、L1 `0`、正/负/零 `0/0/8`。

T1 原始 eval-v3 结果：

`artifacts/proxy_v3/attention-reciprocal-head-temperature-20260906/t1/`

结论：固定 reciprocal factor 候选在实际 16 个配对 case 上没有改变最终 state/输出，
按 no-op 早停规则关闭。不运行 OOD/default，不提交官方，不扫描 factor 或邻域；根
`solution.py` 仍为 v186。
