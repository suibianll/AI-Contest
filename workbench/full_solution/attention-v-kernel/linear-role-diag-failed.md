# 失败记录：Linear role 分解脚本的读数与面板矛盾两个数量级（2026-09-11）

**结论：该脚本的读数不可用，未进入任何结论。**

## 现象

`linear_role_diag.py`（layer 0，一窗）报出：

| role | act_rel | w_rel | out_rel | STD act | STD w | STD out |
|---|---|---|---|---|---|---|
| fc_up | **1.3624** | **1.4209** | 0.0926 | 0.0876 | 0.0883 | 0.1186 |
| fc_gate | **1.3804** | **1.4327** | 0.0695 | 0.0876 | 0.0878 | 0.0926 |
| q | **1.2159** | **1.4592** | 0.0435 | 0.0918 | 0.0817 | 0.0817 |

`act_rel`/`w_rel` **大于 1**，即重建与参考几乎不相关——**对一个能拿到 0.43–0.82 增益的编码器不可能成立**。

**与面板矛盾**：面板上 fc_up 的 player 相对误差是 **0.755%**（`candidate-linear-shard0.json`），
本脚本的 `out_rel` 是 **9.26%**——**差约 12 倍**。面板那个来自评测器，以它为准。

## 最可能的原因（未验证）

调用图的构造与评测器不一致：

```python
calib = [tuple(...) for s in range(2)]                    # 只传 2 个校准窗
cal = root.hif4_calibration_and_quantize_weight(wq, ws, calib)
xp  = root.hif4_dynamic_quantize_activation(xq, xs, dict(state))
```

校准窗数量/索引方式（`[s][layer]`）与评测器的 `linear_calibration_activations` 调用图可能不同，
导致 `activation_state` 是在错误的数据上编译的（例如 `multiplier`/`importance` 的统计对象不对），
于是 `x_hat` 与 `x_ref` 处于不同的坐标。

**这与缺陷 #19 同型**：为了分析而重建了一条与被评测函数不同的调用路径。

## 已定位：不是调用图，是坐标系（2026-09-11 补）

`act_rel = 1.36` 与 `out_rel = 0.0926` **并存**，只有一种解释：
**`x_hat` 与 `x_ref` 不在同一个通道坐标系里。**

根在 Linear 侧使用**块序/置换**机制（日志中的 `static actorder`、`COMBINED-LINEAR-SAMPLE-ENERGY`、
`[L-R2] rank2`）：激活按置换后的列序编码，权重的列做对应置换。
于是 **`X̂·Ŵᵀ` 保持不变，而 `‖X̂ − X_ref‖` 与 `‖Ŵ − W_ref‖` 毫无意义。**

**`out_rel` 是自洽的**，且与评测器面板吻合：

```
我们 out_rel 0.0926  vs  标准 out_rel 0.1186
  →  1 − (0.0926/0.1186)² = 0.39   ≈   面板 gain 0.4268  ✓
```

所以这一跑里**唯一有效的读数是 `out_rel`**；operand 级拆分必须做**坐标系感知**的比较
（例如比较解码后的有效重建、或在同一置换下比较），本脚本没有做。

**这与核差异那次（#33）同型**：都是"两个量在不同坐标/归一化下相减"。本轮的第六次。

## 处置

- 本脚本的读数**不写入任何结论**，不进入 Linear 归因。
- Linear 归因**仍然只有面板读数**（role 级 mean_gain、mse_std/mse_player/ref_energy），
  那部分来自 `artifacts/proxy_v3/20260911-root-linear/`，是评测器口径。
- 若要重做分解，**必须先核对评测器如何构造 `calib_activation_list`**（`evaluator/official_eval.py`
  的 linear 分支），照抄其调用图，并**先做一条自检**：`x_hat` 在父 state 下应逐位等于评测器的
  `player_activation`。

## 这条失败本身说明的事

"我们在这三个 role 上太弱"这个判断**只依赖面板读数**，不依赖本脚本，所以它不受影响。
但"弱在哪一步（激活侧还是权重侧）"**尚未有任何可信读数**。
