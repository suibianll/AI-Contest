# L21-2 探针报告：块级一次输出求解（精简对照）

> run_id：`l21-2-lean-output-fit`。侧：Linear。日期：2026-09-07。
> 契约：[linear-output-followthrough.md](../../docs/superpowers/plans/workpackages/linear-output-followthrough.md) §5 L21-2。
> 父：L4 `ACB16F76...F5263`（官方 4607/247s）。局部诊断，不提交官方。

## 1. 动机与假设

L21-1 逐列条件求解 3/3 holdout 退化。需要判定退化原因是：
(a) 逐列 OBQ 循环的高自由度过拟合，还是 (b) 输出感知目标本身的分布依赖。
L21-2 将求解改为**块级一次 LS + 一次格点化**（无逐列 F 更新循环），其余
（真实 activation_state、固定 scale/lv2/lv3、块两臂接受、fold 加权）不变。

## 2. 结果（3 个代表 state，真实 API 闭环）

| state | L21-2 once changed | L_val_end vs parent | test holdout 父→once | rel | L21-1 rel（对比） |
|---|---|---|---|---|---|
| L0-o | 14/14 | 3.9e-8 vs 1.4e-7 | 2.08e-7 → 2.73e-7 | **1.314 DEGRADE** | 1.28 |
| L11-proj | 76/76 | 1.2e-6 vs 1.5e-4 | 1.83e-4 → 3.27e-4 | **1.783 DEGRADE** | 1.78 |
| L0-fc_up | 14/14 | 1.31e-4 vs 7.2e-4 | 9.44e-4 → 1.48e-3 | **1.563 DEGRADE** | 1.56 |

## 3. 解读

- **块级一次求解与 L21-1 逐列结果几乎逐位一致**（rel 差 <0.04）：
  退化**不**由逐列循环复杂度引起，而由**输出感知目标本身**
  `min ||Q(A) W − X W_orig^T||²`（权重吸收激活/文本分布特定模式）引起。
- fold-1（同校准域）验证大改善而独立 test holdout 退化：校准窗口内拟合、
  分布外不泛化——再次与旧 JDRQ、AGENTS"A@W 拟合隐藏分布依赖偏移"警告一致。
- 按 L21-2 §5：**拟合组件不可迁移**；"精简父+拟合"相对 L4 的综合不可用，
  与 A21-1（删旧训练抵消新收益）教训一致——但这里不是删训练抵消，而是
  拟合组件本身在 holdout 上为负。

## 4. 裁决

- **L21-2 拟合增量：REJECTED (local)**（3 代表 state 统一负向）。
- 不注册组合候选、不提交官方、不扫 λ/步数/offset。父 L4 不变。
- 不扩展为"输出感知权重拟合全族关闭"：本结论仅覆盖"逐块 LS→原始 Y"的
  实现（工作包明确定义的主干）；未来若有不同目标（如保持可逆性/跨层一致
  变换）或外部直接机制证据，可另立卡。

## 5. 产物

- `workbench/continuous_linear/l21-2-lean-output-fit/config.md`
- `workbench/continuous_linear/l21-2-lean-output-fit/probe_once_solve.py`
- 本报告