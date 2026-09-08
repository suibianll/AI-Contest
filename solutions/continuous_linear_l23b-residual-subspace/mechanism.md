# L23：残差交叉子空间 A@W 低维拟合（全校准行直接拟合，官方裁决）

登记于 2026-09-07；2026-09-08 修正归档（新 SHA `13639FB2…10FE0`，取代
`33D1DA51…E35D`）。一个机制、一个配置，见 config.json。
隶属 [持续研究循环](../../../docs/superpowers/archive/plans/2026-09-08-continuous-research-loop.md)。

## 官方状态：TIMEOUT（2026-09-08 用户回传）

- **L23b（13639FB2）官方 TIMEOUT**（>300s，精确耗时/分数未知）。关闭该复杂度实现
  （全校准行 rank-8 残差交叉子空间逐块求解 + 合法五字段投影）；不机械减
  rank/样本/步数重试，不重复同 SHA 提交。
- 机制族（残差交叉子空间 A@W 低维拟合）**仍 OPEN**：仅实质算法/复杂度变化的新源码
  允许再探索（AGENTS §3）。本地时间归因：L23b 六 shard API 1399.15s（父 1055.71s，
  +343s 本地），增量残差重构不足以把校准期成本降到官方 300s 预算。
- **fit_gain 0.9486 ≥ 0.9 研究目标有效**（校准折叠，168/168 states）；独立窗口
  gain 0.339487 只记录。Linear 侧父 L4 与根 v189 不变。

## 修正内容（用户指令 2026-09-07/08 + 超时回传后继路径）

1. **彻底删除奇偶行拆分**：`_l23_fold_split` 已移除；构造低维基、求系数、
   计算 fold 权重（ω_f = 1/(F·max(‖Y_f‖²,1e-12))，全部行）与接受判定
   全部使用同一全校准加权数据。
2. **拟合机制不变**：真实动态量化激活 Xh、原始解码 XWᵀ teacher、rank-8
   白化残差求解、合法五字段投影；不扫秩/正则/参数。
3. **消除重复计算**：全局残差 `R = Y − Xh Wᵀ` 一次计算后增量维护；
   块提案用 `R_cand = R − Xh_B ΔW_Bᵀ` 判损失，接受后 `R ← R_cand`；
   `W` 原位更新，无逐块 `W.clone()`、无逐块完整 `Xh@Wᵀ` 矩阵乘法
   （相对旧版约 144× matmul 削减——超时回传指定的 time-safe 重构）。
4. **验证修正生效**（`verify_direct_fit.py` 全 PASS）：
   - 全部行参与求解（192/192，无拆分）；`_l23_fold_split` 不存在。
   - 增量残差与直接重算一致：五字段解码后的实际拟合损失 == 接受判定的
     最终跟踪损失（rel 1.1e-05）。
   - 冻结激活路径：L23 on/off 的 activation_state 逐位一致，动态激活输出一致；
     scale/lv2/lv3 与父逐位一致；仅接受块 sign/mant 改变（计数==accepted）。
   - Attention control：标准空 state 合法（L4 冻结侧）。

## 机制（对每 64 输入块 B，父顺序一遍）

- 全校准行构造残差交叉子空间：`G = Σω Xh_BᵀXh_B + λI`；`H = Σω Xh_BᵀR`；
  Cholesky 白化 `Hw = L⁻¹H` → rank-8 截断 SVD → 解回 `ΔW_B`。
- 与旧"激活 Gram top-8"的区别：子空间依赖输出残差交叉（数学检查 C：残差主导
  例子白化下降 85.8% vs 旧激活 Gram 0.01%）。
- `W_current,B + ΔW_B` 投影到父合法五字段格点（固定 scale/lv2/lv3，只改
  sign/mant），全校准行目标 `L = Σω‖Y−Xh W‖²` 严格下降才接受；平局保留父块。
- 低维限制作用于连续拟合参数；离散投影后不声称仍 rank≤8，分别记录
  连续提案与合法部署结果（核算 `L_all` 变化与 accepted 计数）。

## 与旧实现去重（dedup-r0.md）

旧逐块激活 Gram top-8/16、权重 SVD 低秩、L21 逐列/块一次、L21-2 精简体重建、
JDRQ 均不相同；L23 子空间 = 残差交叉 + 白化 + rank-8，数学检查确认不等价。

## 4B 结果（记录；官方裁决）

- 六 shard paired（父 L4，336 Linear 例）：candidate gain +0.3395 vs parent
  +0.5243；Δmean −0.1848，pos/neg/zero 1/335/0。按直接拟合政策只记录。
- 拟合质量（全部校准层）：宽层 144/144 块接受、L_all 降 90–99%
  （如 1.2643e-8 → 2.8037e-10、3.2230e-9 → 3.2940e-11）；窄层 40/40、
  62/64 接受，降 74–91%。远强于旧偶数行版（50–62%）。机制可达、无 no-op。
- 合法性：六 API 导入、legal state（评测器强制）、coverage、finite、
  reasonableness 0 问题；Cache：qwen3.5-4b-proxy-v2.pt。
- api_seconds 仅记录：本地 CUDA 1399.15s（六 shard fresh）与旧版
  fresh 等效 ~1369s 基本持平；官方 CPU 的削减来自消除逐块全矩阵乘积与
  全权重复制（旧版超时主因），官方 300s 仍为唯一时间裁决。

## 官方探索规则

- 拟合误差改善（合法部署后同校准数据 L_all 严格下降，accepted>0）+ 合法/可达/
  control 满足 → 已注册官方探索；官方分数与 300s 为唯一裁决。
- 失败只关闭本实现；不扫 rank/基/teacher/ridge 邻域；不以泛化性否定 A@W 低维拟合族。
- 旧 SHA `33D1DA51…` 官方 TIMEOUT（2026-09-07 用户回传）：关闭该复杂度实现，
  不机械减 rank/样本/步数重试；本修正版为指定后继，不继承旧回传。

## 复现入口

- 源码：`workbench/continuous_linear/l23-residual-subspace/candidate/solution.py`
  （= 本归档 solution.py，SHA `13639FB2…10FE0`）。
- 评测：`artifacts/proxy_v3/continuous/linear/l23-residual-subspace/allrow-six-shard/`
  （parent L4：`solutions/v162_linear_l4-v189-linear-exact_officialNA_timeNA/solution.py`）。
- 验证：`verify_direct_fit.py`（修正生效五项检查）；`math_check.py`（数学）；
  `check_import.py`（脱离仓库导入）；`fuzz_contract.py`（CPU contract fuzz）。