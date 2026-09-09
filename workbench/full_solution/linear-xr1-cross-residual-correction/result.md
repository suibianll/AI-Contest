# L-XR1 带 Weight 量化交叉项的 rank-4 动态 Activation 输出纠码（计划卡 1）

活动计划：`docs/superpowers/plans/2026-09-10-linear-cross-residual-correction-plan.md` §3/§4。
父编码完成后对少量 mantissa 错码做**一遍冻结梯度**纠码，优化目标是最终 `XW^T` 输出而非 operand MSE。
本地结论：**REJECTED**（shard0 配对 `delta_mean=-0.040082`，`0/40/16`，全角色为负），
不分配版本号、不提交官方；候选源码、配置、验证与运行结果保存在本目录与 `artifacts/proxy_v3/linear-xr1-*`。

- parent SHA256：`56dc805d6e5a3aef896db8021045740292735725d688b48e3d4393e55efcb2bd`（485072 B）
- candidate SHA256：`23c36c4288d9add613c82663b62c40a239120869acf4ce0110fe6014db194967`（505412 B）
- 工作目录：`workbench/full_solution/linear-xr1-cross-residual-correction/`
- 运行：`artifacts/proxy_v3/linear-xr1-shard0/`、`artifacts/proxy_v3/linear-xr1-sixshard/`
- 日志：`logs/execution/2026-09-10-linear-xr1-cross-residual-correction.md`、`logs/execution/linear-xr1-shard0.out`

## 实现

`build.py` 把 `implementation.py` 逐字节追加到父根之后（分隔符固定 `\n\n\n`），只在模块尾部 rebind
两个 Linear API；父编码器、全部共享 helper、Attention 四 API 与所有归档文件逐字节相同
（`control_a` 用 `inspect.getsource` 逐字符核对）。

| hook | 作用 |
|---|---|
| `hif4_calibration_and_quantize_weight` | 父校准返回后编译 rank-4 `G/C` 因子（`state["xr1"]`） |
| `hif4_dynamic_quantize_activation` | 父动态编码返回后做一遍冻结梯度纠码 |

目标二次型（计划 §2）：`DeltaL = 2 <R W_hat, DeltaX> + tr(DeltaX G DeltaX^T)`，
`R W_hat = (X_hat-X)G + X C`，`G = W_hat^T W_hat`，`C = (W_hat-W)^T W_hat`；
运行期 `g = E G_local + (E U_g) diag(lambda_g) U_g^T + (X U_c) diag(S_c) V_c^T`。

预注册固定选择（不搜索 rank/阻尼/层名单/覆盖率/阈值）：

- rank 4；确定性截断子空间迭代（起始基 = 列 2-范数最大的 4 列，24 次迭代 + Rayleigh–Ritz，
  无 RNG，不做全量 `eigh`/`SVD`）；
- 候选集 = 自然 4 元素组内单个元素、mantissa 码 `±1`、固定 `scale_factor/lv2/lv3`，
  每组 8 个候选、每 64 块 16 组；
- 每行每 64 块最多改一个组，一遍，不刷新梯度、不动 hierarchy、Weight 五字段不变、零码保持规范零 sign；
- 接受规则 = 精确单体二次型 `2 g[i] dx + diag(G4)[i] dx^2 < 0`；
- 只有 `state["gram"] is not None`（`in_features <= 3072`）的层编译因子，宽层保持父实现。

冻结梯度下"按父 block order 反向逐块"与向量化逐块独立取最小严格等价（块间无交互、梯度不刷新），
故实现为一次张量化比较，动态 API 无候选轮询、无分解、无矩阵求逆。

## 验证（`verify.py`，CPU，`rows=64/channels=2560` 合成层）

- [A] 单文件独立导入六 API；候选是父根的纯追加（前缀逐字节相同），Attention 四 API 源码逐字符相同，
  Linear 两 API 确实被 rebind；
- [B] 父关闭 control：`state` 去掉 `xr1` 后动态输出与父逐位相同；`in_features=3200` 宽层
  `arm=wide-no-gram`、不存因子、动态输出逐位相同；父五字段未被校准改动；
- [C] 因子合法（形状/CPU/finite/正交/`lam_g` 降序），`capture_g`/`capture_c` 由独立重算复现，
  rank-4 因子真实改动 480 个硬码（= 12 行 × 40 块，恰好每行每块一个组），码步严格 ±1、范围 [0,7]、
  零码 sign 规范；
- [D] 合成可达性：把因子换成精确满秩 `G/C` 分解后，冻结梯度**精确等于**真实残差 `R W_hat`
  （`predicted` 与独立重算的 `linear+diagonal` 一致到 1e-6 相对误差），每个被移动元素的精确单体
  代价 `< 0`，合并移动的真实输出平方误差变化 `total=-2.52e+03 < 0`。

[D] 只证明**梯度管道**在因子精确时无误；把同一层换成真实 rank-4 因子后，同一批移动的
真实输出平方误差变化为 `total=+7.94e+03 > 0`——**近似误差的方向性代价**正是 4B shard0 上
全角色回退的原因（见下）。

## 4B shard0（`artifacts/proxy_v3/linear-xr1-shard0/`，v202 父根配对）

56 个 Linear case（4 layer × 7 role × 2 length），baseline 先评测、candidate 同进程配对：

| 指标 | baseline | candidate | 配对 delta |
|---|---:|---:|---:|
| linear mean | `+0.516432` | `+0.476350` | `-0.040082` |
| `+/-/0` | — | — | `0 / 40 / 16` |
| L1 / tail | — | — | `0.040082` / `-0.030324` |

逐角色配对均值（每个角色 8 case）：

| role | mean delta | min | max | 说明 |
|---|---:|---:|---:|---|
| `fc_gate` | `-0.080172` | `-0.178788` | `-0.037017` | 最差 |
| `q` | `-0.067030` | `-0.097651` | `-0.043749` | |
| `k` | `-0.066337` | `-0.107896` | `-0.044882` | |
| `v` | `-0.049073` | `-0.064990` | `-0.032586` | |
| `fc_up` | `-0.017962` | `-0.023046` | `-0.013604` | |
| `o` / `proj` | `+0.000000` | `0` | `0` | 宽层未触及（`arm=wide-no-gram`） |

40 个被触及的 case **全部**变差，16 个宽层 case 逐位不变；`split`（test/validation）与 `length`
（128/512）两个维度同号，没有任何子群为正。分析器判定 `reject`，blockers：
`delta_mean=-0.040082 <= 0`、`L1=0.040082 >= 0.02`，warning `worst-20% tail -0.030324`。

### 诊断读数（`logs/execution/linear-xr1-shard0.out`）

校准编译 28 次 = 20 次 `arm=compiled` + 8 次 `arm=wide-no-gram`（每 block 2 个宽层）；
动态 40 次 `arm=applied`（20 个编译层 × 2 次调用），每次改动 `4987–20465` 个组
（合计 509600 次组移动，全部为 `±1` 步）。

| 读数（20 个编译层） | min | median | max | mean |
|---|---:|---:|---:|---:|
| `capture_g`（rank-4 对块外 G 的捕获率） | 0.0537 | 0.3752 | 0.8443 | 0.3866 |
| `capture_c`（rank-4 对 C 的捕获率） | 0.0115 | 0.0937 | 0.1836 | 0.0913 |
| `offblock_rel`（块外分量占 G 的相对大小） | 0.4977 | 0.8164 | 0.9026 | 0.7746 |
| `cross_rel`（C 相对残差的相对大小） | 0.0173 | 0.0357 | 0.0645 | 0.0365 |
| `local_gram_delta`（块内 Gram 与父已有 4×4 gram 的差） | 0.1399 | 1.424 | 8.656 | 2.229 |

动态侧冻结梯度分解：`grad_mean` 0.0424 ≈ `local_mean` 0.0375（块内项，父已有精确 4×4 gram）
+ `lowrank_mean` 0.0130 + `cross_mean` 0.0081。即**块外 G 分量主导**（`offblock_rel` 中位 0.82），
而 rank-4 只捕获其中 39%，`C` 只捕获 9%——冻结梯度系统性地低估纠码代价，40 次调用的
`pred_cost` 合计 `-1.38e+04`（预测改善），真实部署输出误差却上升。

## 结论

L-XR1 本地关闭为 **REJECTED**：

1. 机制合法且非等价（40 次调用真实改动 509600 个硬码，Attention/Weight 逐位不变），
   验证控制 A–D 全部通过；
2. 但 4B shard0 上 40/40 被触及 case 变差，全角色、全 split、全 length 同号，
   没有任何可解释为正的子群；
3. 归因是**近似误差的方向性**：块外 G 分量主导（中位 0.82），rank-4 只捕获 39%，
   固定 rank 的冻结梯度对纠码代价系统性低估，单体接受规则因此接受了一批真实变差的移动。
   [D] 控制直接证明：因子换成精确满秩后，同一批移动严格改善（`total < 0`）。

按计划 §4/§6：不缩 rank、不减少覆盖、不改邻码范围重试；不分配版本号、不提交官方；
根保持 R0 `18053/281s`。六 shard 记录见 `artifacts/proxy_v3/linear-xr1-sixshard/`
（`--shards 0,1,2,3,4,5 --stop-after-nonpositive 6`，shard0 复用本次配对结果）。
