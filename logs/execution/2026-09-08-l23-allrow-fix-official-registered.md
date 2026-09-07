# L23 全校准行修正版：官方探索注册（取代 33D1DA51 超时版）

2026-09-08。按用户修正指令（全部 Qwen-4B 校准数据 A@W 低维拟合）与
[超时后继路径](2026-09-07-l23-official-timeout.md)完成 L23 修正并归档新 SHA。
旧版 33D1DA51 官方 TIMEOUT 的复杂度实现关闭；本版为指定后继，不继承旧回传。

## 修正内容

1. **彻底删除奇偶行拆分**：`_l23_fold_split` 已从源码删除；构造低维基、求系数、
   fold 权重（ω_f = 1/(F·max(‖Y_f‖²,1e-12))，全部行）与接受判定使用同一
   全校准加权数据。旧版"配置声明无 fit/select、代码仍用偶数行求解"的不符已消除。
2. **拟合机制不变**：真实动态量化激活 Xh、原始解码 XWᵀ teacher、rank-8 白化
   残差闭式解、合法五字段投影；不扫秩/正则/参数。
3. **消除重复计算**：全局残差 `R = Y − Xh Wᵀ` 一次计算、增量维护；块提案损失用
   `R_cand = R − Xh_B ΔW_Bᵀ`；接受后 `R ← R_cand`、`W[:, sl]` 原位更新。
   旧版逐块 `W.clone()`（[o,in] 全权重复制）与逐块完整 `L_all` 矩阵乘积
   （每权重矩阵 ~432 次 [N,in]@[in,o]）全部消除，matmul 量约降 144×——
   官方 CPU 超时的主因被移除；本地 CUDA 时间持平（1399.15s vs 旧版 fresh
   等效 ~1369s，均只记录）。
4. **验证修正真实生效**（`workbench/continuous_linear/l23-residual-subspace/verify_direct_fit.py` 全 PASS）：
   - 全部行参与求解：spy 记录 `_l23_block_solve` 收到 192/192 行（3 fold×64），
     无拆分；`_l23_fold_split` 不存在。
   - 增量残差与直接重算一致：最终五字段解码后的实际拟合损失（同 ω 加权、
     直接重算）== 接受判定的最终跟踪损失（rel 1.1e-05，float 噪声级）。
   - 冻结激活路径：L23 on/off 两次校准的 activation_state 逐位一致，动态激活
     输出逐位一致；scale/lv2/lv3 与父逐位一致；仅接受块 sign/mant 改变
     （改变块数 == accepted 计数）。
   - Attention control：标准 attention 校准返回空 state 且合法（L4 冻结侧）；
     diff 范围仅 L23 区（无 attention 文件改动）。
   - 脱离仓库单文件导入（check_import.py）与 CPU contract fuzz
     （fuzz_contract.py：形状/extreme/inference_mode/no_grad/确定性）全 PASS。
5. **不设泛化门**：独立窗口统计只记录；本地时间只记录；官方 300s 唯一裁决。

## 4B 六 shard paired（record-only）

- 父 L4（ACB16F76…F5263），336 Linear 例，cache qwen3.5-4b-proxy-v2.pt，
  scenario linear，shards 0–5，未提前截断，reasonableness 0 问题。
- candidate gain +0.3395 vs 父 +0.5243 → Δmean −0.1848，pos/neg/zero 1/335/0。
- 拟合质量（校准数据合法部署 L_all）：宽层 144/144 块接受、L_all 降 90–99%
  （1.2643e-8→2.8037e-10；3.2230e-9→3.2940e-11；1.0920e-8→1.8136e-10）；
  窄层 40/40、62/64 接受、降 74–91%。远强于旧偶数行版（50–62%），
  机制可达、非 no-op。

## 归档与状态

- 新 SHA：`13639FB22976B2C7C838EE2E69AE9C2C58123E03D9BDEC106623D93B29510FE0`
  （workbench 候选 = 归档 solution.py）。
- 归档 `solutions/continuous_linear_l23-residual-subspace/` 更新：config/
  mechanism/manifest/verification/official-result 均指向新 SHA；旧 33D1DA51
  的 TIMEOUT 记录保留在 official-result.json 的 superseded 字段。
- 父 L4 与根 v189 不变；attention 侧文件未触碰。

## 裁决待回传

- 官方 > L4（4607）且 < 300s：登记 Linear 新侧父，组合 L23+A22-2 单独验证。
- 负向或再次超时：只关闭该具体实现；不扫 rank/基/teacher/ridge 邻域；
  不以泛化性否定 A@W 低维拟合族。