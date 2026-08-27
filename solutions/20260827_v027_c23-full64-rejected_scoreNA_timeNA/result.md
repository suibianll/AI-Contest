# v027 — C23 Full-64 Weight Schur/GPTQ (REJECTED)

- Date: 2026-08-27
- Candidate ID: `C23`（26000 计划 §6）
- Parent: `C21-C` / v025，SHA256
  `83AB4864254F80D221BB491BDEF89F8C9AB8E83534FD62D4DD5E0C1C292FEA12`
- Source SHA256: `DD80CBBF43CD13D7AE6D5AD32399B91A64BB9EF49CA124DC4D526263F2766069`
  （本归档保存 flag=True 的候选版本；根目录 solution.py 随后默认关闭
  `_WEIGHT_FULL64`，行为回到 C21-C，由
  `test_weight_full64_disabled_matches_c21c` 逐位验证）
- Local status: `rejected`（§6.9 晋级门 6 项中 3 项未达）
- Official status: `unavailable`

## 唯一机制（如预注册实施，含实施期修订）

- `_full64_hessian_blocks`：完整 64×64 协方差块（不截断到 4/8/16 对角块）；
- `_cholesky_inverse_factor`：`H+λ·diag_mean·I`，λ 依次 `0.01/0.03/0.1`，
  全失败回退父参数（done 掩码保证首个可分解阻尼生效）；
- `_gptq_initialize64`：`diag(H)` 降序 + Cholesky 逆因子的 GPTQ 序贯
  mantissa 初始化（合法 E6M2 码字，`|code|<=7` 且为 0.25 的整数倍）；
- `_coordinate_descent64`：full-H 64 坐标精确离散下降（e/g 增量更新）；
- `_hierarchy_toggle_refine64`：16 个 lv3 + 8 个 lv2 toggle 批量掩码精修；
- `_refine_weight_blocks64`：scale beam（`standard+{-2..3}` 保留 4）+
  上述流程 + 逐 block fallback（五字段合法且 full-H loss ≤ 父版本）。
- 实施期修订常量：`_WEIGHT_FULL64_MAX_RATIO=0.25`（按父版本 full-H loss
  选 top-25% 64 列进入 beam solve，未选列保持父参数）；
  `_WEIGHT_FULL64_CHUNK_ROWS=1024`（小 B 层调度开销主导，生产 chunk 调大
  但仍满足分块内存硬性要求）。两者已在任何评测前登记于执行日志。

## 开发评测（cuda amax6 offset 0 both）

| Component | C21-C | C23 | Delta |
|---:|---:|---:|---:|
| q | 0.6008 | 0.6220 | +2.12pp |
| k | 0.5936 | 0.6297 | +3.61pp |
| v | 0.5940 | 0.6032 | +0.92pp |
| o | 0.5178 | 0.5389 | +2.11pp |
| fc | 0.4749 | 0.4936 | +1.87pp |
| proj | 0.4058 | 0.4153 | +0.95pp |
| Linear mean | 0.5311 | 0.5504 | **+1.93pp** |

- fc/proj/o 均值：`0.4662 → 0.4826`（**+1.64pp**，门 `>=+3pp` 未达）。
- Linear mean 增量 +1.93pp 低于 `>=+2pp` 门（本日两次独立评测分别为
  +2.07pp / +1.93pp，CUDA 归约抖动约 ±0.2pp，按最新测量记为未达；即便
  计为踩线，仍不改变下面时间门的拒绝结论）。
- Attention 逐位不变（causal `0.4497` / non-causal `0.4944`，weight-only
  候选，结构上不触碰动态激活路径；CPU 串行对照中父子 Attention
  mean/min/max 全部逐位一致）。
- Timing（§10.4 规定口径：父子串行、同环境、CPU）：C21-C
  algorithm-stage `61.32s` vs C23 `95.17s`（ratio `1.55`，门 `<=1.15`
  未达）；CUDA 口径 `24.03s → 32.77s`（ratio `1.37`）作为参考。
  按 `173.8s × 1.55` 推算官方时间约 `269s > 225s`，且已逼近 270s
  硬上限（超限即无条件不得晋级）。

## 固定回归矩阵（§10.2，6/6 正向，但不足以晋级）

| Case | C21-C Linear mean | C23 Linear mean | Delta |
|---|---:|---:|---:|
| amax6 offset 0 | 0.5311 | 0.5504 | +1.93pp |
| amax6 offset 97 | 0.5148 | 0.5372 | +2.24pp |
| amax6 offset 193 | 0.5319 | 0.5538 | +2.19pp |
| amax6 offset 389 | 0.5235 | 0.5469 | +2.34pp |
| amax4 offset 0 | 0.4663 | 0.4929 | +2.66pp |
| pow2 offset 0 | 0.5454 | 0.5644 | +1.90pp |

- amax4/pow2 offset 0 尾部检查：无任何 case 相对指标低于父版本（§6.3 条 4
  满足，扩展 offset 的 Attention 教训未在 Weight beam 上复现）。

## 机制有效性诊断（真实 GPT-2 数据，amax6，instrumented refine）

```text
 comp     parent_H    refined_H     drop  replaced  blocks
    q      49.7668      40.1335   19.36%       36     144
    k      69.6231      43.3462   37.74%       36     144
    v      30.7904      27.7721    9.80%       36     144
    o      28.4401      18.9749   33.28%       36     144
   fc      94.3843      82.8369   12.23%       36     144
 proj     102.1383      83.4890   18.26%      144     576
TOTAL full-H drop: 20.95%
```

- Weight full-H normalized error 下降 `20.95% >= 20%`（门达标，机制真实有效）。
- 块替换率恰为 25%，与 `_WEIGHT_FULL64_MAX_RATIO=0.25` 一致（选择器
  只用 W 自身统计与 H，规则零白名单允许）。

## 合规验证（flag=True 归档版）

- `evaluator/linear_compliance_guard.py` static + runtime 全过：
  `violations=[]`、`review=[]`、`contraction_count=210`、
  `state_tensor_count=5`。einsum `e^T H e` 属 weight-side Hessian 白名单。

## §6.9 晋级门逐项

| 门 | 要求 | 实测 | 结果 |
|---|---|---:|---:|
| Linear mean | >= +2pp | +1.93pp | 未达（踩线，±0.2pp 抖动） |
| fc/proj/o 均值 | >= +3pp | +1.64pp | 未达 |
| full-H 降幅 | >= 20% | 20.95% | 达标 |
| 固定矩阵 | 6/6 正向 | 6/6 | 达标 |
| CPU ratio（§10.4） | <= 1.15 | 1.55 | 未达 |
| 推算官方时间 | < 225s | ~269s | 未达（逼近 270s 硬上限） |

## Decision

`rejected` per §6.9。机制本身有效（full-H 降幅达标、6/6 正向、合规全过），
但绝对计算成本超预算：每层 64×64 Cholesky + 864 步/beam 批量求解使
CPU algorithm-stage 增加 33.9s（§10.4 串行口径 ratio 1.55），推算官方
时间 ~269s 超出 <225s 门并逼近 270s 硬上限。分数增益（+1.9~2.0pp）
不足以 justify 该成本，且 fc/proj/o 主目标分项（+1.64pp）远低于 +3pp
期望。根目录 solution.py 默认关闭
`_WEIGHT_FULL64`（行为与 C21-C 逐位一致，由
`test_weight_full64_disabled_matches_c21c` 验证）；Champion 仍为 C21-C
（v025）。

## 战略后果与遗留发现（供后续候选使用）

- C24 前置条件为「C23 晋级且 weight residual 明显下降」，C25 前置 C24。
  C22/C23 相继 rejected 后，计划 §7–§9 的链条按前置条件全部受阻；继续
  推进必须以新 candidate ID 重新预注册（计划规则 3/4），不得放宽本门。
- 可行方向（如未来重新预注册，须换 candidate ID）：
  1. 收窄覆盖（如 `_MAX_RATIO=0.10`）换 CPU ratio ≤1.15：本诊断显示
     q/k/o 的 drop 远高于均值（33–38%），top 列选择已按 loss 聚合排序，
     减半覆盖预计保留大部分增益，但需重跑全部门；
  2. 用 4×16 或 8×8 分块协方差近似 full-64 H 以降低 Cholesky 成本；
  3. 只对 fc/proj（3072 宽、层均收益最大）启用精修，跳过 q/k/v/o。
- 真实数据 full-H 降幅按组件差异大（v 9.8% vs k 37.7%）：协方差谱越
  平滑的组件，full-64 相对 4/8/16 对角近似的增益越小。
- CUDA 上 128 行 chunk 的 kernel launch 开销主导小 B 层成本（B=12 时
  调度开销与覆盖率无关）。§10.4 CPU 串行口径 ratio `1.55` 劣于 CUDA
  `1.37`：CPU 上 Cholesky/求解的浮点吞吐更低，纯张量运算时间随覆盖率
  实际增长，两口径均远超 1.15 门，拒绝结论跨设备一致。

Holdout 台账：未消耗（`0/3`），seed_hash `96dd4ed7…` 不变。
