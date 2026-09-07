# L23：残差交叉子空间 A@W 低维拟合（直接拟合，官方裁决）

登记于 2026-09-07；一个机制、一个配置，见 config.json。官方状态 PENDING。
隶属 [21071工作包](../../../docs/superpowers/plans/workpackages/21071-evidence-driven-research.md) §4。
用户指令（2026-09-07）：在全部 Qwen3.5-4B 校准数据上直接做 A@W 低维拟合，
不考虑泛化、不拆 fit/select、不用 holdout/split/负向损失拦截；官方分数与 300s 裁决。

## 机制（对每 64 输入块 B，父顺序一遍）

- 真实 dynamic activation 解码 `Xh`（父 `hif4_dynamic_quantize_activation` 输出），
  teacher `Y = X · W_origᵀ`（非量化原始 NVFP4 权重解码）。
- 用 fold 加权 ω_f 拼 fit 行（偶数 token）：
  `G = Σω Xh_BᵀXh_B + λI`；`H = Σω Xh_Bᵀ R`（R 为全局残差 Y − Xh W）；
  Cholesky 白化 `Hw = L⁻¹H` → rank-8 截断 SVD → 解回 `ΔW_B`。
- 与旧"激活 Gram top-8"的区别：子空间依赖输出残差交叉（数学检查 C：残差主导
  例子白化下降 85.8% vs 旧激活 Gram 0.01%），不是只取 Xh 能量方向。
- 将 `W_current,B + ΔW_B` 投影到父合法五字段格点（固定 scale/lv2/lv3，
  只改 sign/mant，code = codes·sf·lv2·lv3 15 点最近），用**全部校准行**
  目标 `L_all = ||Xh_all W − Y||²` 严格下降才接受；平局保留父块；更新真实残差。
- 低维限制作用于连续拟合参数；离散投影后不声称仍 rank≤8，分别记录连续提案
  与合法部署结果（本项目核算 `L_all` 变化与 accepted 计数，不混连续/离散）。

## 与旧实现去重（dedup-r0.md）

- 旧逐块激活 Gram top-8/16（仅激活能量）、权重 SVD 低秩、L21 逐列/块一次、
  L21-2 精简体重建、JDRQ 均不相同：L23 子空间 = 残差交叉 + 白化 + rank-8，
  数学检查确认与旧激活 Gram 不等价；不更名重试旧实现。

## 合法性与约束

- 固定父 scale/lv2/lv3；只改已接受块 sign/mant；未接受块保留父逐位字段。
- 父 activation_state 与动态输出逐位不变；不保存浮点残差旁路；
  最终只部署合法 weight_params。在线动态 API 无新增矩阵/候选循环。
- 六 API 单文件自包含；脱离仓库导入检查 PASS；CPU fuzz contract 全 PASS。

## 4B 结果（记录；官方裁决）

- 六 shard paired（父 L4，336 Linear 例）：candidate gain +0.3426 vs parent
  +0.5243；独立窗口 Δmean −0.18。按用户指令与工作包 §4 只记录，不阻止官方探索。
- 拟合质量（全部校准层）：`[L23] accepted` 普遍 88–110/144（宽 9216 块）、
  24–29/40（窄）；每层 L_all 相对父下降约 50–62%（如 2.64e-8 → 1.38e-8、
  6.90e-9 → 2.56e-9、2.01e-8 → 9.56e-9）。机制可达、无 no-op。
- 合法性：六 API 导入、legal state（评测器强制）、coverage、finite、
  reasonableness 0 问题；Cache：qwen3.5-4b-proxy-v2.pt。

## 官方探索规则

- 拟合误差改善（合法部署后同校准数据 L_all 严格下降，accepted>0）+ 合法/可达/
  control 满足 → 已注册官方探索；官方分数与 300s 为唯一裁决。
- 失败只关闭本实现；不扫 rank/基/teacher/ridge 邻域；不以泛化性否定 A@W 低维拟合族。

## 复现入口

- 源码：`workbench/continuous_linear/l23-residual-subspace/candidate/solution.py`
  （= 本归档 solution.py，SHA `33D1DA51…E35D`）。
- 评测：`artifacts/proxy_v3/continuous/linear/l23-residual-subspace/direct-fit-six-shard/`
  （parent L4：`solutions/v162_linear_l4-v189-linear-exact_officialNA_timeNA/solution.py`）。
- 数学检查：`workbench/continuous_linear/l23-residual-subspace/math_check.py`（全 PASS）。
- 脱离导入：`workbench/continuous_linear/l23-residual-subspace/check_import.py`；CPU fuzz：
  `workbench/continuous_linear/l23-residual-subspace/fuzz_contract.py`（均 PASS）。