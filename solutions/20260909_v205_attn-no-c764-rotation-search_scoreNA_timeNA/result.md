# v205 — attn-no-c764-rotation-search（定价用 ablation）

## 状态

- 机制：减法诊断——关闭 C76.4 variable H16/H32 旋转搜索（`_ATTN_ROTATION_ENABLED`
  True→False，单行开关，`solution.py:433`）。审计发现该搜索占 Attention 校准 ~28–30%
  （3 块尺寸 × 4 seed × deployed-MSE 门控），官方从未定价；关闭后 gate 输入语义不变，
  v189 栈与 R3 训练逐位保留。
- 父（当前根 v202+v195）SHA256：`56dc805d6e5a3aef896db8021045740292735725d688b48e3d4393e55efcb2bd`
- 候选 SHA256：`52a79bd20cd3848d5ee7c932e0d535f47c54e11869d0573ab7248b1be4e391c7`
- 官方状态：`unregistered/NA`。

## 检查

- `verify.py`（真实 4B cache layer 0）全 PASS：`_attention_deployed_mse` 调用数 16→4
  （恰好砍掉全部 C76.4 候选评估），非 no-op；合法 state、有限输出、脱离仓库六 API
  导入通过；本层根未选中任何旋转 winner，输出逐位相同（差异纯为省去的搜索成本）。

## 4B shard0

- eval-v3、Qwen3.5-4B proxy-v2、CUDA、Attention-only、shard 0、calibration-cache-mode auto：12 cases。
- candidate mean `0.5721770802`，父 mean `0.5721770802`，paired delta `0`（12/12 全零）——
  shard0 覆盖的层根均未选中 C76.4 旋转；分数影响只可能来自其他层/官方隐藏输入。
- calibration API：父 `8.82s` → 候选 `6.28s`（−2.5s，与审计 ~30% 份额方向一致；
  本地时间不预测官方时间）。

## 证据位置

- 归档：`solutions/20260909_v205_attn-no-c764-rotation-search_scoreNA_timeNA/`
- 评测：`artifacts/proxy_v3/full_solution/attn-no-c764-rotation-search-shard0/candidate/manifest.json`
- 工作脚本：`workbench/full_solution/attn-no-c764-rotation-search/`

## 官方结果

- **17969 / 275s**（2026-09-09 用户回传）。相对根 `−84 分 / −6s`。
- 定价结论：C76.4 H16/H32 旋转搜索官方价值 **+84 分**，时间成本仅 ~6s
  （14 分/秒，是全方案已测最高效机制，对比 v195 为 4.2 分/秒）。
  根时间审计中"官方未定价"的疑问解除：**必须保留，不得为省时间砍掉**。
- 附带事实：砍掉本地占 Attention 校准 ~30% 的计算，官方端到端仅 −6s——
  官方时间由校准以外的环节主导，砍校准换时间的路线收益极小（与 v194 一致），
  v192 回收所需的 −40s 无法通过裁剪校准实现。
