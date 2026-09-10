# Linear 三 role 的两条具体线索（2026-09-11）

来源：读 `activation_state`（layer 0，两校准窗），不跑评测。角色对照 k/q（好）与 proj/fc_up/fc_gate（差）。

## 线索一：`proj` 被 L-EM2 编译度量显式跳过

日志：

```
[L-EM2] compile arm=compiled     in=2560 gram_diag_mean=1.74e-01 h_norm=3.50e+01   ← k
[L-EM2] compile arm=compiled     in=2560 gram_diag_mean=6.66e-01 h_norm=9.78e+01   ← q
[L-EM2] compile arm=out-of-scope in=9216 gram_diag_mean=0.00e+00 h_norm=0.00e+00   ← proj
[L-EM2] compile arm=compiled     in=2560 gram_diag_mean=6.63e-01 h_norm=5.55e+01   ← fc_up
[L-EM2] compile arm=compiled     in=2560 gram_diag_mean=8.13e-01 h_norm=8.69e+01   ← fc_gate
```

`proj` 的 `activation_state` **没有 `em1` / `em1_arm` 键**（其余四个 role 都有），
`gram`/`h_inv` 为空。即：**`proj` 没有拿到 L-EM2 这条编译度量机制**——
而 L-EM2 是本项目**唯一有官方收益记录、且本地读数可预测官方**的机制类别。

`proj` 是三个差 role 之一（gain 0.4846）。**它缺的恰好是根里最有效的那条机制。**

**已定位判据**（`solution.py:11933 / 11962`）：

```python
_EM1_MAX_CHANNELS = 4096
...
if in_features <= 0 or in_features > _EM1_MAX_CHANNELS:
    diagnostics["em1_arm"] = "out-of-scope"
```

**L-EM2 的编译度量只覆盖 `in_features ≤ 4096`；`proj` 的 9216 被排除。**

**为什么是 4096——成本护栏，不是疏忽。** 度量需要 `in_features × in_features` 的逆：

| in_features | 逆的 flops (≈n³) | 单层估算 |
|---|---|---|
| 4096 | 6.9e10 | ~1 s |
| **9216** | **7.8e11** | **~12 s** |

× 24 层 = **~288 s**，远超 300 s 预算。**取消上限不可行。**

**所以"给 `proj` 补上 L-EM2"必须走结构化/近似度量**——块对角、低秩、
或按块的局部度量（`proj` 的 9216 输入正好是 144 个 64 块，日志里也显示
`[COMBINED-LINEAR-SAMPLE-ENERGY] blocks=144`）。**这正是计划 §6.2 的 LK-1 范围。**

**这是目前唯一一条"差距有 0.32、且缺的机制有官方收益记录、且障碍是已知的成本而非未知的机制"的线索。**

## 线索二：`fc_up` 的通道重要性跨度 114 倍

| role | importance min | max | **跨度** | gain |
|---|---|---|---|---|
| **fc_up** | 0.050 | 5.705 | **114×** | 0.4268 |
| k | 0.518 | 8.076 | 15.6× | 0.8178 |
| q | 0.606 | 6.904 | 11.4× | 0.7555 |
| proj | 0.620 | 4.540 | 7.3× | 0.4846 |
| fc_gate | 0.374 | 2.649 | 7.1× | 0.5228 |

`fc_up` 的重要性跨度是 `k` 的 **7 倍**，存在极端低重要性通道（0.050）。
与"统计被少数极端通道支配"的假设方向一致，**但跨度大本身不证明它有害**——
也可能是数据本身就那样。**未验证。**

## 关于设备（如实记录）

用户两次指出设备问题。核实结果：`.venv` 是 `torch 2.6.0+cu124`，`cuda.is_available()=True`，
GPU 空闲；我脚本里的 `dev` 确实解析为 cuda。

**但成本大头不在脚本自身**：根自己的 `hif4_calibration_and_quantize_weight` /
`hif4_dynamic_quantize_activation` 是为官方 **CPU** 判题器写的，内部可能把 state 与中间量
固定在 CPU；`torch.load(map_location="cpu")` 也把 6.28 GB 的 pack 留在主存。
**我没有核实过根内部是否在 GPU 上执行**——这一条只作如实记录，不下断言。

## 下一步

1. **查 `out-of-scope` 的判据**（`in=9216` 触发什么），判断能否扩展 L-EM2 覆盖 `proj`。
   这是唯一一条"差距有 0.32、且缺的机制有官方收益记录"的线索。
2. 其余两条假设（编码器参数退化 / 格式接近饱和）**均未验证**。
