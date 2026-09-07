# L23 直接拟合版：官方探索注册

2026-09-07，按用户指令「直接拟合，用官方评测裁决」将 L23 残差交叉子空间
A@W 低维拟合注册官方探索。本日志登记候选、验证与六 shard 记录，不覆盖旧
[b52dbee 报告](../solutions/continuous_linear_l23-residual-subspace/mechanism.md)。

## 用户指令与政策

- 用户明确：linear 直接在 qwen-4B 校准数据拟合，不考虑泛化；holdout/split/
  负向损失不拦截；官方 300s 与分数唯一裁决。政策记录
  [2026-09-07-linear-direct-4b-fitting-policy](2026-09-07-linear-direct-4b-fitting-policy.md)
  与[4B指引](../../docs/4b-panel-testing-guide.md) §2.4。

## 候选与验证

- 源码：`workbench/continuous_linear/l23-residual-subspace/candidate/solution.py`
  经 git 修改（相对 L4 父仅增加 162 行 L23 逻辑，其余逐字节无 diff）。
- 相对 L4 父 diff：全部新增，`git diff --no-index` 仅 1 hunk +162 行；
  Attention 侧仍为 L4 冻结（v162 standard）。
- 脱离仓库导入检查 PASS（六 API、合法五字段）；
  CPU contract fuzz PASS（窄/宽/非2幂/大形状、extreme ×330000、
  inference_mode/no_grad、repeat determinism）。
- 数学检查 `math_check.py` 全 PASS；去重 `dedup-r0.md` 确认与旧激活 Gram/
  权重 SVD/L21 族不等价。

## 4B 六 shard paired（record-only）

- Cache `qwen3.5-4b-proxy-v2.pt`；父 `solutions/v162_linear_l4-.../solution.py`；
  scenario linear，shards 0–5，未提前截断。
- 336 例：candidate gain +0.3426 vs 父 +0.5243 → Δmean **−0.1817**，pos/neg/zero
  **0/336/0**；最坏 role family `o`（shard0 −0.517）、`proj`（−0.236）。
- 独立窗口统计按直接拟合政策**只记录**，不阻止官方探索（官方为裁决者）。
- 拟合质量（校准数据合法部署 L_all）：各层相对父降 50–62%，accepted
  88–110/144（宽 144 块）/ 24–29/40（窄 40 块），每层 accepted>0，机制可达、
  非 no-op。api_total 1123.97s（1 次校准缓存命中）vs 父 1055.71s（0 命中），
  仅记录无门。

## 归档与状态

- 归档 `solutions/continuous_linear_l23-residual-subspace/`（solution.py 与归档
  同 SHA `33D1DA51…E35D`），含 config.json/mechanism.md/manifest.json/
  verification.json/official-result.json（PENDING_OFFICIAL）。
- workbench state.json、总计划进度已更新；未修改 attention 侧文件。

## 裁决待回传

- 官方 > L4（4607）且 < 300s：登记为 Linear 新侧父，组合 L23+A22-2 单独验证。
- 负向：只关闭该具体实现；不扫 rank/基/teacher/ridge 邻域；不以泛化性否定
  A@W 低维拟合族（工作包 §4 与 AGENTS 已关闭边界）。