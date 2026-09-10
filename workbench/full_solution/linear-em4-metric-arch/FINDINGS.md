# L-EM4 侦察结论：校准钩子的成本被高估，动态 API 才是目标

> 2026-09-10。本目录是 L-EM3/v231 归档后的**重新规划侦察**工作区。不产生候选、不改动根。

## 0. 一句话结论

计划卡 L-EM3 登记的"校准钩子折算 +5.8 ~ +6.2 s"**不成立**，真实值约 **+2.6 s**；
误差来自探针自身——分量用 GPU params 计时、整钩子用 CPU params 计时，而官方路径上
params 在 **GPU**。修正后，Linear 线真正的每调用大项是**动态 API**（~0.45–0.51 s/call），
且它是**派发受限**的（单次调用约 19k 次 `as_strided`、1.5k 次 `bmm`、750 次 `einsum`）。
下一张卡应当打动态 API 的**固定开销**，而不是钩子。

## 1. 钩子成本修正（本工作区的第一个产物是一个否定结果）

`profile_hook_breakdown.py` 曾给出：钩子分量之和 ≈ 2.4 s（折 144 次），整钩子 5.59 s，
缺口 ≈ 3.2 s 无法归因；消融掉一整个 matmul 加两个 n² 诊断扫描只改变 4.6 ms。
三轮外部探针（分量计时、分配假设、消融）全部落空。

`probe_torch_profiler.py` 直接问 profiler 哪些 kernel 在跑，答案指向 6 个 `aten::mul`
（占 39%）与 2 个 `aten::nan_to_num`（9%）——正好是两个反量化链的乘法次数：

```
_dequantize_nvfp4_float32   2 mul
_dequantize_hif4            sign * mant * lv3 * lv2 * scale_factor = 4 mul
```

`probe_dequant_device.py` 给出设备 A/B：

| | CPU | GPU | 比 |
|---|---:|---:|---:|
| `_dequantize_hif4` | 20.97 ms | **0.74 ms** | **28×** |
| 整钩子 in=2560 | 39.11 ms | **15.94 ms** | |
| 整钩子 in=4096 | 52.05 ms | **26.87 ms** | |

`probe_real_calibration_path.py` 按 `official_eval.py:2560-2583` 的原始调用形状调用真实候选，
打印返回值设备：**`mant/scale_factor/scale_lv2/scale_lv3/sign` 全部 `cuda:0`**。

> 结论：官方路径上 params 在 GPU，钩子跑的是 15.94 / 26.87 ms。
> `profile_hook_breakdown.py` 的分量用 GPU params（≈2.4 s）而整钩子用缓存里的 CPU params
> （5.59 s），**"缺口"就是这一次设备错配**。修正后 144 次折算 **≈ 2.56 s**。

**可复用教训**：同一个函数的分量与整体必须在**同一设备配置**下计时；本地探针若从缓存
（`_cpu_params` 落盘的格式）取 params，就复现不了官方路径的设备状态。

## 2. GPU 配置下的钩子真实构成（profiler，in=4096，每调用）

| kernel | ms/call | 占比 | 说明 |
|---|---:|---:|---|
| `aten::mm` | 13.58 | 47% | 两个 n² 矩阵乘 |
| `aten::copy_` | 7.53 | 26% | D2H + dtype 转换 |
| `aten::nan_to_num` | 4.61 | 16% | **`_cpu_state_tensor` 在 CPU 拷贝上做的 n² 扫描** |
| `aten::mul` | 1.04 | 4% | 两条反量化链 |
| 其它 | ~2.1 | 7% | |

- `_cpu_state_tensor`（solution.py:6468）= `nan_to_num(x.detach().to("cpu", fp32), 0,0,0).contiguous()`，
  即 **D2H 之后在 host 上对整个 n² 矩阵扫一遍再整份拷一遍**。钩子在调用它之前已经用
  `torch.isfinite(h_matrix).all()` 验过有限性，故该 `nan_to_num` 在此路径上**可证为空操作**。
- 调用点 `_cpu_state_tensor(h_matrix.contiguous())` 的 `.contiguous()` 与函数尾部的
  `.contiguous()` 至少有一个是冗余的。
- `aten::_local_scalar_dense` 15.45 ms CPU/call（3 次 `.item()` 强制同步）——是 CPU 侧最大项。

## 3. 动态 API：真正的目标

`profile_dynamic_api.py` 按 `official_eval.py:2619` 的形状调用：

| case | wall（5 次中位） | profiler device/call |
|---|---:|---:|
| layer0/q in=2560 rows=128 | **501.76 ms** | 2570 ms |
| layer0/o in=4096 rows=128 | **446.10 ms** | 1866 ms |

与 L-EM3 计划卡"父"列的 `0.5107 s` / `0.4469 s` **吻合**。

kernel 构成（in=2560，每调用）：

| kernel | 次数/调用 | 自身 CUDA |
|---|---:|---:|
| `aten::as_strided` | **19 269** | 513.6 ms |
| `aten::view` | 6 852 | 201.5 ms |
| `aten::permute` | 6 768 | 406.1 ms |
| `aten::unsqueeze` | 6 554 | 382.2 ms |
| `aten::reshape` | 5 598 | 311.5 ms |
| `aten::copy_` | 4 149 | 365.5 ms |
| `aten::mul` | 3 798 | 309.5 ms |
| `aten::bmm` | 1 504 | 700.0 ms |
| `aten::einsum` | 752 | 540.2 ms |
| `aten::nonzero` | 384 | 219.7 ms |

**平均单 kernel 5–90 µs，是典型的派发受限**：时间花在几万次小算子派发上，不在算力上。

### 与 L-EM3 登记值的表面冲突（已澄清，记录以免重犯）

L-EM3 的 `dynamic_k2_seconds_per_call_in2560: 0.054` 与这里的 0.50 s 差约 9×。
**两者不矛盾**：L-EM3 的 54 ms 是 **Δk2（K=2 减父的一次差分）**，而 0.50 s 是**绝对调用成本**。
该卡"父"列 `0.5107 s` 与本探针一致。本工作区曾据此一度误判为 10× 误差，特此记录。

### 现场核对：K=1 与 K=2 的绝对成本几乎相同

```
0/q in=2560  ROOT(v230 K=1) 530.04 ms    v231 (K=2) 528.68 ms
0/o in=4096  ROOT(v230 K=1) 433.86 ms    v231 (K=2) 423.05 ms
```

即**每调用的固定开销主导**，第二遍 pass 只增加约 3.2 s（折 144 次）。
**推论：削减固定开销既降低基座、又直接降低 K 的边际价格**——这正是"下一个精度台阶"缺的那笔预算。

## 4. 本地六 shard 的 API 分解（`artifacts/proxy_v3/linear-em3-cand/candidate/manifest.json`）

每 shard：28 次校准 = 135.70 s，56 次动态 = 44.46 s（**0.79 s/call**）。
全量 `api_total_seconds` 1048.12 s。

注意：该跑的校准是**冷缓存**（787.06 s），而官方 292 s 明显带缓存，故此处的
"校准 3× 动态"比例**不能外推到官方**。能外推的是**每次调用的绝对成本**与**派发结构**。

## 5. 本工作区产出的可复用结论

1. **分量与整体必须同设备计时**；从 `_cpu_params` 落盘缓存取 params 会复现不了官方路径。
2. **区分"绝对调用成本"与"差分成本"**；L-EM3 的 54 ms 是差分，误读为绝对会产生 10× 假象。
3. **动态 API 是派发受限**（~19k `as_strided`/call），不是算力受限；优化方向是减少小算子
   派发与中间张量重排，而非减少 FLOP。
4. **每调用的固定开销主导，K 的边际很小**（K=1 与 K=2 绝对成本在噪声内相同），
   所以固定开销是唯一同时改善"基座时间"与"K 价格"的杠杆。
