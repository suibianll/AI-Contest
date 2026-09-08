# A23 执行结果：4B paired 负向，乘积目标卡按预登记分支关闭

2026-09-07。隶属 [21071 机制证据驱动下一轮计划](../../../docs/superpowers/archive/plans/21071-evidence-driven-research.md) §5。
候选源码 SHA256：`8714ac2a044779465c5e406ef0768be7071ac626f3a2171cdc5083350be92dbf`。
父为 **A22-2**（`solutions/continuous_attention_anchor22-a2/solution.py`，
SHA `4686ad81…`，官方 14424/271s）。评测：4B 面板（`qwen35-4b-panel-v1`，
16Q/4KV/hd256，Attention 72 cases）paired，`--baseline-solution` 指向 A22-2
归档源码；**A22-2 的 4B 基线由本次 paired run 首跑补齐**（此前无任何
anchor 系 4B 运行，符合工作包 R0"缺同口径父基线时在第一个新候选 paired
run 补齐一次"）。

## 关于"父版本测过 4B 吗、为什么 layer0 完全一样"

- A22-2 此前从未在 4B 上评测（全部历史结果是 0.5B 时代产物）；本次
  candidate/baseline 两侧都是本次 fresh 跑的，基线读数以本次 baseline
  manifest 为准。
- layer 0 上 **A22-2 的可加目标 gate 与 A23 的乘积目标 gate 都拒绝了**
  残余提案：两边的最终部署状态都是同一个完整父 P（R3 原始训练产物，
  两侧逐位相同——verify 证明 S=0 强制回退时 A23 与 A22-2 输出 45 个
  非审计字段 bit-identical）。因此该层 12 个 case 的 gain 逐位相同，
  paired Δ 恰好为零。这不是缺测，是 gate 决策在该层收敛到同一回退。
- 差异只出现在 gate 接受的层（L8、L22，见下）。

## 验证（全部通过）

- 乘积目标手工梯度 vs autograd：3 个几何 max 1.5e-8；回退布局 3.7e-9。
- **Q×c / K÷c 乘积不变性**：mean ratio == 1.000000000（可加目标不可见的
  对冲方向，乘积目标正确不看穿为"改善"）。
- S=0 强制回退 = A22-2 官方源码逐位（合成 37 字段、真实 4B L0 45、
  L22 43 非审计字段）。
- frozen prefix（R3 前缀逐字节）、V/Linear 冻结、坐标链拦截、
  inference_mode、隔离导入。
- fuzz 合同：1 个失败与 **A22-2 父逐字相同**（mixed-q 校准 ValueError；
  官方合同不产生 mixed-q 校准输入，父已官方 14424 验证；R3 存活仅因
  吞异常）。继承行为，非本卡回归。

## 4B paired 结果（72 case，A23 − A22-2）

| 指标 | 全部 | validation | test |
|---|---:|---:|---:|
| Δmean | **−0.004750** | +0.000753 | **−0.010253** |
| 负向 L1 | 0.005676 | — | — |
| 正/负/相同 | 10/14/48 | 6/6/24 | 4/8/24 |

- A22-2 4B 基线 mean 0.536715，A23 0.531965。
- 层级分解：L0/L1/L5/L15 零变化（两侧 gate 均拒绝）；**L8 gate 接受但
  9/12 case 负向（层 Δmean −0.0290）**——最后单校准窗口 gate 的过拟合；
  L22 接受后 +0.0005（7/5 混合）。
- **专项门 FAIL**：Δmean<0 且 test split 为负（门要求双 split 均正）。

## 机制判读与裁决

乘积目标行为与预登记的失败模式完全一致：乘积比降到 0.45–0.55
（目标强烈"改善"）而真实 readout 不跟随（L0 拒绝、L8 接受后恶化）。
**按工作包 §5 预登记分支：本目标卡关闭**——"scale 乘积下降而输出
不改善，只关闭此目标"。不扫 loss 权重、窗口或 gate 阈值；
A22-2 父线（14424/271s）不变，本卡负结果不扩写为附加式残余族关闭
（A22-2 的可加目标仍有官方 +19 正证据）。

**LOCAL_NEGATIVE / OBJECTIVE_CLOSED / official NA。**
候选归档 `solutions/continuous_attention_anchor23-a1/solution.py`。

## 下一步

下一卡若考虑"实际量化 QK 输出目标"（而非 scale 代理），须先与旧
Jacobian/动态 Gram 关闭族去重（AGENTS §7：per-call 动态 Gram 族不缩
sweeps 重试；gate 复杂计算须留在校准侧）。另一方向是把本卡的
gate 过拟合教训用于后继卡设计（gate 聚合多窗口的代价须先估算）。
等待协调者/用户登记下一张卡；本侧无待官方包。

## 复现入口

CUDA venv：`build.py` → `verify.py` → `check_math_and_import.py` →
`fuzz_check.py`；`run.py screen` → `run.py full`。
结果目录：`artifacts/proxy_v3/continuous/attention/anchor23-a1/`。
