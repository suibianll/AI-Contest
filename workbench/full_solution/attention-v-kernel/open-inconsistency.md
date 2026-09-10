# 未解不一致：候选与诊断给出相反符号（2026-09-11）

**在解决之前，候选不进 shard。** 本文记录定位到的边界，供下一步接手。

## 事实

同一层（L0）、同一窗（w3, T=1024）、同一父参数：

| 路径 | V-only loss 比值 |
|---|---|
| 父 | 1.0000（2.144172e-04） |
| **vk3_rule.py（VK-3 诊断）** | **−1.81%** ← 改善 |
| **候选 `_vk_correct`** | **+98.5%** ← 显著变差 |

## 已排除

**候选的编排没有错。** 用**候选自己的** `_vk_shifted` / `_vk_row_energy` 重搭一遍 vk3_rule 的贪心
（同一父参数、同一核、同一窗）：

```
REBUILD : +98.4623%
CANDID  : +98.4623%
两者码完全一致：0 / 1048576 处不同
```

→ **循环、索引、每通道取最优步、施加方式全部一致。** 分歧不在 `_vk_correct`。

## 已排除（部分）

**核的拟合口径已对齐**（散布抽样 128 行 + 完整 key 轴，取代此前的"前 128 token × 128×128 子块"）。
核确实变了（loss 从 +100.23% 变到 +98.46%），但**没有改变符号**。

→ 拟合口径不是全部原因。

## 已排除（新增两条）

**积木一致。** `helpers_crosscheck.py` 在同一随机输入上对撞两组：
```
apply_W      vs _vk_shifted        max|gap| = 1.776e-15   OK
row_energy   vs _vk_row_energy     max|gap| = 0.000e+00   OK
apply_W(rev) vs _vk_shifted(rev)   max|gap| = 3.553e-15   OK
```

**拟合口径一致。** 候选 `_vk_fit` 已改为与 vk3_rule 逐字同口径（48 行 linspace 散布 + 完整 key 轴 +
同一 `totals/counts` 归一化）。**结果仍为 +98.19%，符号未翻。**

## 已排除（再两条）

**核数值一致。** 同口径（`VK3_RADIUS=7`）dump 两边核对比：
```
max|w_cand - w_diag| = 3.133e-03   (核量级 6.320e-01, rel 4.96e-03)
cand h0: [0.01967, 0.02602, 0.02831, 0.02649, 0.03234, 0.03623, 0.06816, ...]
diag h0: [0.01963, 0.02594, 0.02857, 0.02742, 0.03244, 0.03631, 0.07098, ...]
```
（第一次对比报出 `(16,16)` vs `(16,32)` 是**我 dump 时忘了设 `VK3_RADIUS=7`** 的对照错误。）

**码合法、收益不是未校验造成的。** vk3_rule 的 `true_mse` 原本走手写步长模型 `to_dense`，
**绕过了 `reference_hif4.validate_hif4_params`**——一个产生非法 mantissa 的规则会被静默打分。
已改为经 `dequantize_hif4`（带校验）解码，**−1.8096% 一字不变**。

## 剩下的唯一动作

在同一进程里并排 dump 中间量：`residual` → `sig` → `grad` → `qii` → `best_row/best_dir` → 最终码，
逐项 `allclose`。四条外围已全部排除，分歧必然在其中一项的**输入**上（最可能是 `pv` 的来源：
vk3_rule 用根 `solution.py` 的校准 state，isolate 用**候选的**校准 state——候选的
`hif4_calibration_attention` 会往 `v_state` 里写 `vk_kernel` 等键）。

## 原"剩下的唯一未对照量"（已被上面两条取代）

**核的数值本身。** 两条路径的核从未 dump 出来对比过。

下一步（一句可执行的话）：在候选 `_vk_fit` 与 vk3_rule 的拟合循环里各加一行
`torch.save(w, ...)`，同一层同一 fit 窗跑一次，`assert torch.allclose(w_cand, w_diag, atol=1e-6)`。
不成立 → 差异就在这两个拟合循环的某行；成立 → 说明"同一核给出相反符号"，
那是不可能的，必然是某处输入不同（此时逐项对照 `qd/kd` 与 `keep`）。

## 原诊断（保留）

1. **核仍不同**：vk3_rule 用 `FIT_TOKENS=48` 散布行；候选用 128 行。行数差异本身不该翻转符号，
   但两条 fit 代码的**桶累加细节**（`counts` 只按 h==0 累加、`totals/counts` 的归一化）从未逐位对照过。
2. **积木不同**：`_vk_shifted` / `_vk_row_energy`（候选）与 `apply_W` / `apply_WT` / `row_energy`（诊断）
   **从未互相校验过**。各自内部有检查（诊断侧：banded ≟ Toeplitz ≤ 2e-15），但**两者之间没有**。

## 下一步（唯一需要的动作）

把 vk3_rule.py 的 `apply_W` / `row_energy` 与 `_vk_shifted` / `_vk_row_energy` **在同一输入上对撞**：

```
delta 固定随机张量，weights 固定随机 16 维
assert allclose(_vk_shifted(delta, w, 7), apply_W_unfold(w, delta))
assert allclose(_vk_row_energy(w, T, 7, dev), row_energy_loop(w, T))
```

任一条不成立，分歧即定位。两条都成立，则问题在核的拟合代码，届时逐桶对照 `totals`/`counts`。

## 为什么值得继续

VK-3 的收益（−0.986% 均值、10/12 窗、管线双检查通过）**没有被否证**——被否证的是"候选复现了它"。
两者码一致却结果相反，说明**至少有一方的中间量是错的**，而这是一个可以一次对撞解决的定位问题，
不是机制问题。
