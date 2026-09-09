# 2026-09-10 L-XR1 执行记录

活动计划：`docs/superpowers/plans/2026-09-10-linear-cross-residual-correction-plan.md`（卡 1）。
机制：带 Weight 量化交叉项的 rank-4 动态 Activation 输出纠码。父根 v202 Linear + v195 Attention
`18053/281s`，SHA256 `56dc805d6e5a3aef896db8021045740292735725d688b48e3d4393e55efcb2bd`。

- 工作目录：`workbench/full_solution/linear-xr1-cross-residual-correction/`
- candidate SHA256：`2ac7b749813f854946ba0f17f0227a28ac27b857031578b2c0e9c30bca7eb6de`
- 运行：`artifacts/proxy_v3/linear-xr1-shard0/`、`artifacts/proxy_v3/linear-xr1-sixshard/`

## 实现

`build.py` 把 `implementation.py` 逐字节追加到父根之后（分隔符固定为 `\n\n\n`），只在模块尾部
rebind 两个 Linear API，父编码器、全部共享 helper、Attention 四 API 与所有归档文件保持逐字节相同：

| hook | 作用 |
|---|---|
| `hif4_calibration_and_quantize_weight` | 父校准返回后编译 rank-4 `G/C` 因子（`state["xr1"]`） |
| `hif4_dynamic_quantize_activation` | 父动态编码返回后做一遍冻结梯度纠码 |

预注册固定选择（不搜索 rank/阻尼/层名单/覆盖率/阈值）：

- rank 4；确定性截断子空间迭代（起始基 = 列 2-范数最大的 4 列，24 次迭代 + Rayleigh–Ritz，
  无 RNG、不做全量 `eigh`/`SVD`）；
- 候选集 = 自然 4 元素组内单个元素、mantissa 码 `±1`、固定 `scale_factor/lv2/lv3`，
  每组 8 个候选、每 64 块 16 组；
- 每行每 64 块最多改一个组，一遍，不刷新梯度、不动 hierarchy、Weight 五字段不变、零码保持规范零 sign；
- 接受规则 = 精确单体二次型 `2 g[i] dx + diag(G4)[i] dx^2 < 0`。

冻结梯度下"按父 block order 反向逐块"与向量化逐块独立取最小严格等价（块间无交互、梯度不刷新），
故实现为一次张量化比较。

## 验证（`verify.py`，CPU，`rows=64/channels=2560` 合成层）

- [A] 单文件独立导入六 API；候选是父根的纯追加（前缀逐字节相同），Attention 四 API 源码逐字符相同；
- [B] 父关闭 control：`state` 去掉 `xr1` 后动态输出与父逐位相同；`in_features=3200` 宽层
  `arm=wide-no-gram`、不存因子、动态输出逐位相同；父五字段未被校准改动；
- [C] 因子合法（形状/CPU/finite/正交/`lam_g` 降序），`capture_g`/`capture_c` 由独立重算复现，
  rank-4 因子真实改动 480 个硬码（= 12 行 × 40 块，恰好每行每块一个组），码步严格 ±1、范围 [0,7]、
  零码 sign 规范；
- [D] 合成可达性：把因子换成精确满秩 `G/C` 分解后，冻结梯度**精确等于**真实残差 `R W_hat`
  （`predicted` 与独立重算的 `linear+diagonal` 一致到 1e-6 相对误差），每个被移动元素的精确单体
  代价 `< 0`，合并移动的真实输出平方误差变化 `total=-2.52e+03 < 0`。

## 4B shard0（`artifacts/proxy_v3/linear-xr1-shard0/`）

v202 父根同进程配对，56 个 Linear case（4 layer × 7 role × 2 length）：

| 指标 | baseline | candidate | 配对 delta |
|---|---:|---:|---:|
| linear mean | `+0.516432` | `+0.476350` | `-0.040082` |
| `+/-/0` | — | — | `0 / 40 / 16` |
| L1 / tail | — | — | `0.040082` / `-0.030324` |

逐角色配对均值（各 8 case）：`fc_gate -0.080172`、`q -0.067030`、`k -0.066337`、
`v -0.049073`、`fc_up -0.017962`、`o/proj ±0.000000`（宽层未触及）。40 个被触及 case 全部变差，
16 个宽层 case 逐位不变；`split`、`length` 两个维度同号。分析器 `decision=reject`，
blockers `delta_mean<=0`、`L1>=0.02`，warning `worst-20% tail -0.030324`。

API 时间（诊断值）：baseline calibration `954.52s`/28 calls + dynamic `99.20s`/56 calls；
candidate calibration `920.04s`/28 calls + dynamic `99.95s`/56 calls；两次调用合计 `1019.99s`。

### 校准编译读数（20 个 `arm=compiled` 层；8 个宽层 `arm=wide-no-gram`）

| 读数 | min | median | max | mean |
|---|---:|---:|---:|---:|
| `capture_g`（rank-4 捕获块外 G） | 0.0537 | 0.3752 | 0.8443 | 0.3866 |
| `capture_c`（rank-4 捕获 C） | 0.0115 | 0.0937 | 0.1836 | 0.0913 |
| `offblock_rel`（块外分量占 G） | 0.4977 | 0.8164 | 0.9026 | 0.7746 |
| `cross_rel`（C 相对残差） | 0.0173 | 0.0357 | 0.0645 | 0.0365 |
| `local_gram_delta` | 0.1399 | 1.424 | 8.656 | 2.229 |

动态侧 40 次 `arm=applied`，每次改动 `4987–20465` 个组（合计 509600 次 ±1 组移动）；
冻结梯度分解 `grad_mean` 0.0424 ≈ `local_mean` 0.0375 + `lowrank_mean` 0.0130 + `cross_mean` 0.0081，
即块外 G 分量主导（`offblock_rel` 中位 0.82）而 rank-4 只捕获 39%、`C` 只捕获 9%，
`pred_cost` 合计 `-1.38e+04`（预测改善）与真实变差方向相反。

## 六 shard（`artifacts/proxy_v3/linear-xr1-sixshard/`）

按计划 §4 步骤 5 固定运行 `--shards 0,1,2,3,4,5 --stop-after-nonpositive 6`（禁用默认早停），
shard0 复用本次配对结果（`--reuse-existing`，source SHA / dense cache / scope 三项匹配），
shards 1–5 在 GPU 空闲时串行评测。

## 结论

本地关闭为 **REJECTED**，不分配版本号、不提交官方，根保持 R0：

1. 机制合法非等价（509600 次真实 ±1 组移动，Attention 与 Weight 五字段逐位不变），控制 A–D 全通过；
2. 4B shard0 上 40/40 被触及 case 变差，全角色/split/length 同号；
3. 归因：块外 G 分量主导（中位 0.82），固定 rank-4 只捕获 39%，冻结梯度系统性低估纠码代价，
   单体接受规则接受了一批真实变差的移动；[D] 控制证明因子换成精确满秩后同一批移动严格改善
   （`total=-2.52e+03 < 0`，真实 rank-4 因子下 `total=+7.94e+03 > 0`）。

按计划 §4 不缩 rank、不减少覆盖、不改邻码范围重试。详细实现/验证/读数见
`workbench/full_solution/linear-xr1-cross-residual-correction/result.md`。
