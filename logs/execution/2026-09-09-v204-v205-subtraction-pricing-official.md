# v204/v205 减法定价候选官方结果（2026-09-09 用户回传）

根：v202 Linear + v195 Attention，`18053/281s`，SHA `56DC805D...EFCB2BD`。

## 结果

| 候选 | 机制 | 官方分数/时间 | 相对根 | 定价结论 |
|---|---|---|---|---|
| v204 linear-no-rank2-residual | 关 L-R2 rank-2 残差段 | `18053/286s` | `0 / +5s` | rank-2 残差官方贡献 **0 分**，时间成本在噪声内 |
| v205 attn-no-c764-rotation-search | 关 C76.4 H16/H32 旋转搜索 | `17969/275s` | `−84 / −6s` | C76.4 官方价值 **+84 分 / ~6s** |

## 结论

1. **C76.4 旋转搜索必须保留。** 6s 换 84 分 = 14 分/秒，是全方案已测最高效机制
   （对比 v195 的 4.2 分/秒）。根时间审计（`workbench/full_solution/root-time-audit/report.md`）
   中"C76.4 官方未定价"的疑问解除。
2. **rank-2 残差段是零分死代码。** 可安全移除以简化，但官方时间 +5s（去掉计算不可能更慢，
   属官方计时噪声 ≥±5s），移除不换回可用余量，根不切换。
3. **"砍校准换时间"路线收益极小。** 砍掉本地占 Attention 校准 ~30% 的计算，官方端到端仅 −6s
   （v194 同类：calibration API −22.5% → 官方 +5s）。官方时间由校准以外环节主导。
   推论：v192（+22，需 −40s）无法通过裁剪校准回收，维持关闭。
4. 官方计时噪声实测 ≥±5s（v197 285 / 根 281 / v204 286 / v205 275），
   后续 ±5s 内的时间差不作任何裁决依据。

## 证据

- `solutions/20260909_v204_linear-no-rank2-residual_scoreNA_timeNA/result.md`
- `solutions/20260909_v205_attn-no-c764-rotation-search_scoreNA_timeNA/result.md`
- 候选 SHA：v204 `11d7bbcd...041ca2`，v205 `52a79bd2...e391c7`
