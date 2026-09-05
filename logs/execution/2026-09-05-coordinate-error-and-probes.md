# 同坐标系误差诊断与官方探针计划 · P0–P5 总账

> 计划：[`docs/superpowers/plans/2026-09-05-coordinate-consistent-error-and-official-probes-plan.md`](../docs/superpowers/plans/2026-09-05-coordinate-consistent-error-and-official-probes-plan.md)
> 根 manifest：`artifacts/proxy_v3/coordinate-diagnostics-v186/manifest.json`
> 开始：2026-09-05。规则：官方提交无次数限制；分数不换算（本地只做 Δmean>0 且 L1<0.02
> 符号门禁）；时间用分解模型、预测 <280s 才提交；官方硬限 <300s。

## P0 冻结证据（DONE，2026-09-05）

### 身份核验

- 根 `solution.py` SHA256 = `F8495DCA…7EB8`（v186 官方计分版本，RETAINED 17599/272s）。
- `artifacts/proxy_v3/official-audit-smoke3/v186/candidate-both-shard0.json` 的
  `source_sha256 = f8495dca…7eb8` 与根逐字节一致 → 已有审计产物可复用（无需重跑 shard0）。
- git 工作区干净；本次 P0 只读审计 + 新增 manifest/日志，未改算法与评测器。

### 端点归档清单（见 manifest.endpoints）

v160/v162/v163/v164/v166/v168/v180/v182/v186/v187/v188 归档目录名已逐一核对
（修正 manifest 中 v180/v182 目录名）。v168 的两个重复目录（`_re_… copy` 与标准目录）
solution.py SHA 相同（`5988AE47…`），无选择歧义。

### v186 变换链审计（两个只读子代理 + 人工核对关键行）

**Attention（动态 Q/K/V，实际顺序见 `_nvfp4_to_hif4` L4152）**

1. `dequant_nvfp4`（L4182）：NVFP4 块=16，`x = quant*scale` 后 flatten。
2. K 特有 `center_mode∈{0,2,4}`（L4184-93，`_center_attention_k` L2799）：K 减行常数
   （mode2 = (amax+amin)/2；mode4 = 校准 center_value），softmax 不变。
3. `multiplier` 逐通道乘（L4194-96）。**A1 logits gain 已折入 multiplier**
   （calibration 返回前 L10675-76：q_mult×√g^(1−α)、k_mult×√g^(1+α)，D1 α=_ATTN_A1_ASYMMETRIC_ALPHA；
   α=0 即 A1 对称基线）。state["logit_gain"] 仅记录。
4. `permutation` 通道重排（L4197-4203）。
5. `attention_rotation`（L4204-21，`_apply_attention_rotation` L4710）：head 内 ±1 符号 +
   规范 H_block 旋转，保 Q·K。
6. `block_smooth`（L4222-39）：有 signs 走旋转路径（`_apply_attention_rotation`），否则
   `_block_hadamard_transform`（L4472）。
7. `attention_pair_transform`（L4240-48，`_apply_attention_pair_transform` L2911）：
   相邻通道对独立 2×2 行变换（M=√S 由 Q/K 协方差平衡解）。
8. `_dense_to_hif4`（L4271）：importance 加权误差的码字搜索 → HiF4 五字段。

Q/K state 字段：multiplier、permutation、importance、offsets、error_threshold、
accept_margin、max_refine_ratio、max_refine_blocks；当选才追加 block_smooth_size/seed/signs、
rotation/rotation_block、pair_transform；K 追加 center_mode/center_value。
**V state 无连续变换字段**（`hif4_dynamic_quantize_v` 只传 importance/offsets/refine 五元组）。
镜像重建入口：`_attention_state_transform_dense`（L2935-3001）与上述同序，可直接产出
编码前连续张量（Q_t/K_t）。

**Linear（activation 动态链，`hif4_dynamic_quantize_activation` L8727）**

编码前张量 = R·H·P·(x ⊙ s_inv)，其中：
1. `dequant_nvfp4`（L8740）。
2. `smooth_inv` 逐通道乘（L8741-45）：s_inv = 1/best_d。
3. `permutation` index_select（L8746-53）。
4. `block_hadamard_transform`（L8754-59）。
5. rank 残差（L8761-75）：有 `residual_u/residual_v`（rank-2，v186 `_WEIGHT_RESIDUAL_RANK=2`）
   走 `dense + (dense@U)@Vᵀ`；否则 rank1 `dense + (dense@rank1_u)·rank1_v`。连续域
   `A'W' = (AR)(R⁻¹W)`，VᵀU≈0 保证 R⁻¹=I−UVᵀ。

activation_state 字段：smooth_inv、permutation、block_smooth_size/seed、importance、gram、
h_inv、offsets、rank1_u/rank1_v、residual_u/residual_v（组装 L8696-8719）。
**坐标共享**：weight 与 activation 用同一 permutation/smooth/block-smooth/rank 残差坐标系。

**HiF4 编码（`_dense_to_hif4` L3713）**：每 64 块 `reshape(…, 8, 2, 4)`；amax → E6M2
scale；lv2/lv3 为层级指数位（lv2 作用于 8 通道子块，lv3 作用于 4 通道子块，∈{1,2}）；
解码 `dense = sign*mant*scale_lv3*scale_lv2*scale_factor`。

**当前常量**：`_DYNAMIC_OFFSETS=(-1,1,2,3,4)`（v186 保留的 +4 单码窗）；
`_WEIGHT_RESIDUAL_RANK=2`；`_ATTN_ROTATION_ENABLED=True`、`_ATTN_BLOCK_SMOOTH_ENABLED=True`、
`_ATTN_PAIR_MATRIX_SMOOTH=True`、`_ATTN_LOGIT_GAIN=True`、`_ATTN_SCALE_AWARE_CENTER=True`、
`_ATTN_A1_ASYMMETRIC_FOLD`(D1) 打开。启用层由校准期数据+门控运行时决定，无常量列表。
注意（审计发现）：`_WEIGHT_PRODUCT_SELECTOR/_JDRQ_*/_WEIGHT_HEADROOM/_CAT64` 等开关在
Linear 校准路径中**没有调用点**，不产生 state 字段——只读残留，不构成活动变换。

### 协议记录（供 P1 后引用）

- default panel：168 Linear（24×7 role）+ 120 Attention（24×5 长度）；仅此口径六 API 新鲜
  计时可套官方时间模型。
- eval-v3 六 shard：336 + 48；shard0 = layers [0,6,12,18]，Linear 56 = 4 层×7 role×2
  window，Attention 8 = 4 层×2 window；test_splits = [test, validation]。
- Linear 校准 folds = [0,1]（2 折）；Attention calibration_indices = [0..4]（5 个长度桶）。
- 诊断与分片耗时不代入官方时间模型；校准缓存命中（`calibration_cache_hit=true`）计时无效。

### P0 验收

- 变换表有序、state 字段与来源已标注 ✅（见上 + manifest）。
- identity 一致（SHA/panel/folds/case keys）✅；无 ERROR。
- 待 P1 决定：**P3-B 长度桶共同键**——官方 Q/K 独立调用、无跨调用场景键，唯一可读长度是
  各自 shape[0]。若官方允许 Q_len≠K_len（合同未排除），本块记 `DESIGN_BLOCKED`。

### P1 待办（下一步）

1. 实现 `evaluator/coordinate_diagnostics.py`：导入根 solution 与 reference codec；按上述
   变换链重建编码前浮点张量（可直接用 `_attention_state_transform_dense` 与 activation 链镜像）；
   Attention 8 臂 / Linear 4 臂 MSE 分解 + B/E 分解。
2. `tests/test_coordinate_diagnostics.py`：FP64 合成数据（配对缩放/旋转/K-center 行平移/
   GQA/causal mask/Linear 残差），残差 ≤1e-10；真实链 ≤1e-4。
3. 先跑目标侧 shard0（复用 official-audit-smoke3 的 v186 产物做 111 臂复现对照）。

## P1 同坐标误差分解（DONE，2026-09-05）

专项报告见 [`2026-09-05-coordinate-diagnostics-v186.md`](2026-09-05-coordinate-diagnostics-v186.md)。
六 shard 全层完成：48 Attention + 336 Linear case。

正确性：Linear X_tW_t vs ref = 0.0 全 case；B/E 分解最大残差 2.3e-10；111 臂 player gain
与官方审计逐位一致（layer0 case0 = 0.9233616316569982）。`tests/test_coordinate_diagnostics.py`
5 passed（FP64 恒等式、GQA/causal、镜像一致性、残差 shape 修正为 1-D 向量后全绿）。

关键结论：
- 误差几乎全部是纯量化扰动（Attention mean(E²)/mean(B²) ≈ 4000×；Linear 连续偏差恒为 0）。
- Attention：Q/K 量化影响 ≈ V 的 2.5–2.8×，Q/K 与 V 可加（无抵消）；误差随层单调增大
  （浅层 1e-5 → 深层 8e-3，L21 最差）。
- Linear：权重静态编码误差 ≈ 激活动态编码的 1.9×（proj 4.4×、qkv 2.4×）；两侧近似可加。
- 新机制优先级：深层 Q/K 与深层权重编码质量；浅层余量已很低。
- 新增文件：`evaluator/coordinate_diagnostics.py`、`tests/test_coordinate_diagnostics.py`；
  产物 `artifacts/proxy_v3/coordinate-diagnostics-v186/run-001|run-all-1-5/`（不入库）。

## P2 格式误差定位（DONE，2026-09-05）

专项报告见 [`2026-09-05-format-error-location-p2.md`](2026-09-05-format-error-location-p2.md)。
运行：`coordinate_diagnostics.py --relax` 六 shard 全层（48 Attention + 336 Linear）。

结果（输出 MSE，均值）：
- **R1 mantissa 连续化是唯一有系统余量的字段约束**：Q/K 单侧量化误差下降 82%、
  V 61%、Linear W 77%、X 51%。改善率 100%（Q 24/24 层、W 168/168 pairs），跨
  test/validation 无混合符号，深层同样成立。
- **R2 scale 连续化、R3 lv2/lv3 连续化无余量**（放宽≈player 或更差，≤2% case 差）：
  E6M2 scale 与层级求解器已饱和。
- 判定：R1 通过研究筛选但**无合法实现路径**（3-bit 0.25 网格依法固定）→
  `DIAGNOSTIC_FINDING`；R2/R3 → `NO_MARGIN`。P4 注册条件不满足 → 记为
  `NO_SUPPORTED_MECHANISM`。官方 4166 分差距归因于 3-bit mantissa 表示能力代际差，
  不是本机剩余合法自由度。不重启已关闭家族。
- 按计划继续 P3：官方贡献探针获取分桶证据。


