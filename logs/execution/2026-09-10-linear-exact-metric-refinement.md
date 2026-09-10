# 2026-09-10 L-EM1 执行记录

活动计划：`docs/superpowers/plans/2026-09-10-linear-exact-metric-refinement-plan.md`。
机制：精确度量下、理想输出目标上的 Activation mantissa 码序贯下降。父根 v202 Linear + v195 Attention
`18053/281s`，SHA256 `56dc805d6e5a3aef896db8021045740292735725d688b48e3d4393e55efcb2bd`。

- 工作目录：`workbench/full_solution/linear-em1-exact-metric-refinement/`
- candidate SHA256：`d9bffd7b2953b74a93550b27a2d577e6e4555e42799cd5c7bcce9802707affe2`（501216 B）
- 运行：`artifacts/proxy_v3/linear-em1-shard0/`

## 实现

`build.py` 把 `implementation.py` 逐字节追加到父根之后（分隔符固定 `\n\n\n`），只在模块尾部 rebind
两个 Linear API；父编码器、全部共享 helper、Attention 四 API 与所有归档文件保持逐字节相同。

| hook | 作用 |
|---|---|
| `hif4_calibration_and_quantize_weight` | 父校准返回后编译 `H = (W_hat - W)^T W_hat` 与标量 `c = mean(diag(h_inv^{-1} - W_hat^T W_hat))`（`state["em1"]`） |
| `hif4_dynamic_quantize_activation` | 父动态编码返回后重建 `G`、一遍精确序贯组下降 |

预注册固定选择（不搜索 rank/阻尼/层名单/覆盖率/阈值）：理想目标 `J(X)=||(X-T)W_hat^T||^2`、
`T = X_ref C G^{-1}`、pm1 候选（每组 8 个）、自然升序块与组序、接受后精确刷新梯度
（`g += delta G[sl,:]`）、接受规则 `2<delta,g> + delta^T G_gg delta < 0`、一遍、`in_features <= 4096`。

## 口径判定（本次新增，替换原“重建 dense W”的写法）

`L` 的参考权重必须取输入 weight 对的 nvfp4 解码（评测器 `ref_weight = dequantize_nvfp4(*weights[...])`
同一口径）。父对 Weight 做变换、对 Activation 做逆变换，两侧解码后仍在模型口径下，因此
`_em1_reconstruct_dense_weight`（还原到部署口径）是错的：

| 口径 | `||W - W_raw||/||W_raw||` | `L - J` 在 `X` 微扰下 |
|---|---:|---|
| `W_raw`（nvfp4 解码） | 0 | `-5.18e-11 -> +1.82e-11`（常数） |
| 父口径重建 | 1.415 | `-4.0959e+06 -> -4.0848e+06`（非常数） |

非常数即证明该口径下 `J` 的下降不保证 `L` 下降，故候选使用 `W_raw`。

## 验证（`verify.py`，CPU）

- [A] 候选是父根纯追加（前缀逐字节相同），单文件隔离导入六 API 齐全，Attention 四 API 源码逐字符相同，
  两个 Linear API 已 rebind 到 L-EM1 hook；
- [B] 父关闭 control：`state` 去掉 `em1` 后动态输出与父逐位相同；`in_features=4160` 宽层
  `arm=out-of-scope`、不存 metric、动态输出逐位相同；父五字段未被校准改动；
- [C] 度量管线：`H` 与独立重算逐位一致（`H_gap/|G|=0`），`G` 重建相对误差合成层 `1.25e-05`、
  真实层 `1.16e-06`；
- [D] 单调性 + 代价恒等式（真实层 `layer0/q` window1 rows128）：
  `L_before=1.932880e+03 -> L_after=1.511509e+03`，`dL=-21.8002%`（与 headroom probe 的
  `ideal/seq` 读数逐位一致）；机制自报 `predicted_cost=-4.213714e+02` 与真实 `dL=-4.213710e+02`
  相对差 `8.8e-7`；梯度有限差分 `analytic=+2.508742e-01` vs `numeric=+2.508769e-01`（rel `1.1e-05`，
  fp32 相对偏差 `2.2e-04`）；合成层 `dL=-99.41%`、`gap_rel=2.0e-05`；
- [E] 确定性（两次调用逐位相同）与 `torch.save/load` 状态往返（输出逐位相同）。

`ALL L-EM1 CONTROLS PASSED`。

## 4B shard0（GPU，串行）

见 `artifacts/proxy_v3/linear-em1-shard0/`。baseline 校准缓存命中（`cache load 9.526s`），
candidate 冷启动（`calibration API 132.331s`）。

| 项 | baseline | candidate |
|---|---:|---:|
| Linear mean（56 case） | 0.517349 | 0.625140 |
| scoring API | 42.141s | 73.807s |
| API total | 42.141s | 206.138s |

配对：linear `mean +0.107791`、`median +0.103636`、`+/-/0 = 48/0/8`（over layers 0/6/12/18）。
scoring 段两进程差 `(73.807 − 42.141)/56 = +0.5655 s/call`。

## 时间核算（判定性：全量覆盖必然是官方 TIMEOUT）

### 同进程配对（`time_paired.py`，8 次取中位，`cuda.synchronize` 在计时区内）

| case | rows | in | groups | parent | candidate | delta | changed mant |
|---|---:|---:|---:|---:|---:|---:|---:|
| layer0/q | 128 | 2560 | 640 | 0.5117s | 1.1091s | **+0.5974s** | 38533 |
| layer0/o | 512 | 4096 | 1024 | 0.4728s | 1.4922s | **+1.0193s** | 388887 |
| layer0/fc_gate | 128 | 2560 | 640 | 0.4941s | 1.0818s | **+0.5877s** | 27617 |
| layer0/proj（超范围对照） | 128 | 9216 | — | 0.9678s | 0.9684s | +0.0006s | 0 |
| layer6/q | 512 | 2560 | 640 | 0.8675s | 1.4725s | +0.6050s | 183689 |
| layer12/v | 128 | 2560 | 640 | 0.5030s | 1.0955s | +0.5924s | 39884 |

`proj` 是超范围层：`changed_mant=0`、`delta=+0.0006s`，证明全部开销来自下降本身而非包装/导入。

单位成本 **0.93–1.00 ms/group，且与 rows 基本无关**（rows 128→512 只让 `q` 从 0.5974 升到
0.6050，+1.3%），符合 CPU 逐组派发受限的模型（每组约 30 个算子 + 2 次 host sync）。

### 官方折算（168 次 Linear 动态调用）

官方调用图为每个 `(layer, role)` 一次：`in=2560` 的 5 个角色（q/k/v/fc_gate/fc_up）共 120 次、
每次 640 组；`o`（in=4096）24 次、每次 1024 组。

`120 × 0.5955 + 24 × 1.0193 = +95.9 s`。

校准钩子（`time_calibration.py`，同进程配对）：`in=2560 +0.0139 s/call`、`in=4096 +0.1289 s/call`，
超范围对照 `in=9216 +0.0875 s`（纯噪声），折算官方 144 次在范围内校准 `+4.8 s`。

合计 **≈ +101 s**。根官方 `18053/281s`、300s 硬限余量 19 s → 预计官方 **≈382s，必然 TIMEOUT**。

### 计划 §3.3 被证伪

§3.3 用 `bench_seq_loop.py`（CPU）估计"折算 GPU 每调用约 50–100 ms … 新增 ~2–2.5 s"；
实测 `+0.567 s/call → +95.9 s`，**低估约 40 倍**，根因是把 CPU 逐组调度时间当作 GPU 时间外推。

### 微优化上限（`bench_seq_gpu.py`，同 GPU、真实 shape）

| 实现 | in=2560 rows=128（640 组） | in=4096 rows=512（1024 组） |
|---|---:|---:|
| `impl`（镜像 `implementation.py`） | 482.1 ms/call | 780.8 ms/call |
| `impl` 去掉 2 次 host sync | 434.8 ms/call | 654.3 ms/call |
| 解析闭式 + 块内梯度 + 无 sync | 280.9 ms/call | 458.8 ms/call |
| 整层向量化（1 pass） | **1.0 ms/call** | **4.4 ms/call** |

1. 两次 host sync 只占 ~10%（47/126 ms）；开销是**派发受限**（每组约 30 次小算子派发、
   ~25 µs/次），不是同步受限；
2. 即使逐组循环做到解析闭式，`0.44 ms/group` 仍折算官方 `120×0.281 + 24×0.459 = +44.7 s`，
   **仍超 19 s 余量 2.4 倍**。逐组 Python 循环这条路在 300s 门下不可行，与实现细节无关；
   要保留机制只能减少顺序迭代次数。

因此 L-EM1 全量覆盖**不提交官方、不占版本号**（`official_status: not-submitted/time-infeasible`），
转 `probe_frontier.py` 测同一机制在 blockseq / jacobi 迭代结构下的精度前沿。

## 精度前沿（`probe_frontier.py`，真实 4B 数据，只读 CPU）

四个变体共享全部预注册固定选择（理想目标 `J`、pm1 候选、自然升序、精确二次接受规则），
只改**下降调度**（每个 pass 内的顺序迭代次数）。`--passes K` 表示重复 K 遍扫描、
扫描之间重算精确梯度，故比较是等扫描数下的比较。`dL_true` 是真实输出平方误差的下降。

### layer0/q window1 rows=128（in=2560，640 组，`L0=1.932880e+03`）

| 变体 | 每 pass 顺序迭代 | p1 | p2 | p4 | p8 | p16 |
|---|---:|---:|---:|---:|---:|---:|
| `seq`（L-EM1 已实现） | 640 | −21.8002% | −28.1631% | −32.4193% | −34.5628% | — |
| `blockseq`（每 64 块一次） | 40 | −3.1037% | −3.6038% | −3.6756% | — | — |
| `jacobi`（整层一次） | 1 | **+0.0000%** | +0.0000% | +0.0000% | +0.0000% | +0.0000% |
| `groupstep`（组号主序 ×16） | 16 | **−19.0998%** | **−26.0625%** | −31.2284% | −34.0637% | −35.1415% |

### layer0/o window1 rows=128（in=4096，1024 组，`L0=1.763695e+00`）

| 变体 | p1 | p2 | p4 |
|---|---:|---:|---:|
| `seq` | −42.0623% | −57.3535% | −69.4334% |
| `groupstep` | −26.9464% | −37.3016% | −47.7262% |

三个结论：

1. **`seq` p1 逐位复现已归档读数**（layer0/q `−21.8002%` 与 `verify.py` 控制 D 一致，
   layer0/o `−42.0623%` 与 headroom probe 一致），探针与候选口径无误；
2. **`jacobi` 恒为 0**（`accepted_rows=0`，每个 pass 都如此）：整层 640/1024 组同时提议，
   联合代价被整行精确校验一律否决 —— 机制的价值确实来自顺序刷新，不是来自提议本身；
3. **`groupstep` 用 1/40 的顺序迭代拿回 `seq` p1 的 88%（in=2560）/ 64%（in=4096）**：
   in=2560 上 K=2（`−26.06%`）已超过 `seq` p1（`−21.80%`）；in=4096 上 K=2（`−37.30%`）
   仍不及 `seq` p1（`−42.06%`），要到 K=4（`−47.73%`）才超过。
   `blockseq` 粒度太粗（块内 16 组同时提议，p1 只有 `−3.10%`），
   只有 `groupstep` 是原生 4 元组的直接加粗。

## 调度成本（`bench_seq_gpu.py`，同 GPU、真实 shape）

| 实现 | in=2560 rows=128 | in=4096 rows=512 |
|---|---:|---:|
| `impl`（逐组，L-EM1） | 507.6 ms | 838.5 ms |
| `analytic`（逐组解析闭式） | 319.2 ms | 446.4 ms |
| `jacobi_1pass` | **1.6 ms** | **4.4 ms** |
| `groupstep_p1` | **19.4 ms** | **52.6 ms** |
| `groupstep_p2` | **29.9 ms** | **95.7 ms** |

折算官方 168 次动态调用（120 次 in=2560 + 24 次 in=4096）：

| 调度 | 动态新增 | + 校准 4.8s | 官方合计 | 余量 |
|---|---:|---:|---:|---:|
| `seq` p1（L-EM1） | +95.9 s | +4.8 s | ≈382 s | **−82 s（超限）** |
| `groupstep` p1 | **+3.6 s** | +4.8 s | ≈289.4 s | +10.6 s |
| `groupstep` p2 | **+5.9 s** | +4.8 s | ≈291.7 s | +8.3 s |

`groupstep` 把机制的代价从"必然超时"压到 8–11 s 余量内，且 K=2 的精度已超过已归档的
L-EM1 候选。据此重规划为 **L-EM2（`groupstep` 调度）**，见
`docs/superpowers/plans/2026-09-10-linear-groupstep-schedule-plan.md`。
