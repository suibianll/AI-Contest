# L-AD1：去掉 16 模式枚举里的整数物化（v235）

计划卡 [`docs/superpowers/plans/2026-09-10-linear-correctness-and-runtime-plan.md`](../../docs/superpowers/plans/2026-09-10-linear-correctness-and-runtime-plan.md) §4.3（固定卡）。

父根：v231 完整根（Linear L-EM3 groupstep K=2 + v195 Attention），官方 **18518 / 291 s**（硬限 300 s），
SHA256 `ea79a1c12dc667142c620975aab188920fae7b29988c804f41c1a696cc5754f1`，505762 B。
本卡不改父根；候选是父根的**纯追加**（append-only），父根字节一个未动。

工作区：`workbench/full_solution/linear-ad1-adaround-materialization/`。
候选：`fe8aec1989189ad9c20cdc9e9c3717ad509c98da987cb556b1e1428fc13bc702` / 510664 B（纯追加 +4903 B）。

## 0. 中途 rebase

本卡初稿是对着 **v230 根**（`0f1af6db…`，505496 B，K=1）写的。开发期间官方回传 v231 得
18518 / 291 s 且被晋级，工作区 `solution.py` 随之变成 v231（差异就是 `_EM1_PASSES` 1→2）。

若继续在 v230 上构建，提交的候选会**把官方刚花 1 s 换来的 K=2 那一臂悄悄回退掉**，
而本地配对会把它报成这张卡的收益。所以本卡在 v231 上重建，并且：

- `build.py` 的 `EXPECTED_PARENT_SHA256` / `EXPECTED_PARENT_BYTES` 换成 v231 值，
  `build.json` 同时记录被取代的 v230 sha 与 rebase 原因；父根 sha 不符时构建直接失败。
- `verify.py` / `timing.py` 的标定缓存键**不再写死**，改为从父根文件现算 sha 前缀。
  写死的前缀在 rebase 后会**静默命中上一个根**的标定 state，让激活对比变成"两个不同 state 互比"并通过。
- `pair_sixshard.py` 的基线从 `linear-em3-v230base` 换成 `linear-em3-cand`
  （该次运行记录的 `source_sha256` 就是 v231 根），否则基线差里会混进 K=2 的增量。
- v230 上的旧证据全部作废重跑。v230 读数（helper +3.93/+6.39/+11.1%，调用 +4.66/+4.91%）
  与新读数一致，构成一次独立复现。

## 1. 定位：成本在算子个数，不在 FLOPs

前序卡 [`linear-parent-path-split`](../../workbench/full_solution/linear-parent-path-split/result.md) 把成本定到
**编码器 `_dense_to_hif4`**（占一次动态调用 69.8–90.8%），再定到 gram 层 `_adaround_mantissa` 的
**16 模式枚举**（占该层编码器 51–68%）。成本与行数无关（425 个 aten op / 次编码器调用，
1060–1068 / 次动态调用），所以这是算子计数问题。

父根那一段：

```python
floor_code = torch.floor(raw_code).clamp(0, 6).to(torch.int64)
ceil_code  = torch.ceil(raw_code).clamp(0, 7).to(torch.int64)
...
masks = ((bits_all.unsqueeze(1) >> torch.arange(4, device=...)) & 1).bool()
...
all_codes    = torch.where(mask_expanded, ceil_code.unsqueeze(0), floor_code.unsqueeze(0))
all_mantissa = all_codes.to(torch.float32) * 0.25
```

`[K,N,8,2]` 的码被**以 int64 物化**成 `[16,K,N,8,2]`，再整块转 float32、乘 0.25。

## 2. 机制与为什么逐位相等

```python
floor_code = torch.floor(raw_code).clamp(0, 6).to(torch.int64)   # 逐字不变
ceil_code  = torch.ceil(raw_code).clamp(0, 7).to(torch.int64)    # 逐字不变
floor_mant = floor_code.to(torch.float32) * 0.25                 # 在小张量上做
ceil_mant  = ceil_code.to(torch.float32) * 0.25                  # 在小张量上做
masks = _AD1_pattern_masks(x_abs.device)                         # 与输入无关，缓存
all_mantissa = torch.where(mask_expanded, ceil_mant.unsqueeze(0), floor_mant.unsqueeze(0))
```

**先于任何测量的论证。** 两条 int64 码表达式逐字未动；唯一移动的是两个逐元素算子相对于一次
`torch.where` 的位置。`.to(torch.float32)` 与 `* 0.25` 都是逐元素算子，而逐元素算子与 `where`
的交换是**定义上的相等**——选择先取出每个位置的一个值，再把函数作用到那个值上——所以两种写法
对任意 int64 对、任意掩码都相等，**不需要对转换做任何精度假设**；非有限输入经 int64 转换得到的
值也一样（两侧拿到同一个值，受同一个逐元素转换）。

候选数（16 个 floor/ceil 模式）、loss、`argmin`、`gather`、覆盖率、refine 轮数、接受规则
全部未动——本卡删的是枚举**内部**的物化，不是候选。

`masks` 只依赖 device，提到调用外并按 device 缓存；只被读（`.reshape` 返回视图，无人写穿）。

### 2.1 构建方式：影子定义

候选 = 父根字节 + 追加的一个模块（3 条顶层语句：`_AD1_PATTERN_MASK_CACHE` 赋值、
`_AD1_pattern_masks`、`_adaround_mantissa` 的第二份定义）。追加的定义在调用时从模块全局解析，
自然覆盖前面那一份，**不需要 hook**。父根自己那份定义逐字还在文件里，是死的。

函数体 17 条语句中前缀 4 条、后缀 7 条相同，改动是中间连续的 4..9 六条（6 换 6，语句数不变）。
被删的名字 `bits_all` / `all_codes` 出现 0 次；新引入的 `floor_mant` / `ceil_mant` 各 2 次。

## 3. 证据

### 3.1 等价性（`verify.py` / `verify.json` = EQUIVALENT）

| 段 | 覆盖 | 结果 |
|---|---|---|
| `unit` | 父子 `_adaround_mantissa` 直接对比：5 种形状（1–5 维）× 4 种 dtype × 10 组取值 × 2 种 scale，另加 3 种非连续布局 | **403 组，0 组不同，0 组抛错** |
| `activation` | 真实 `hif4_dynamic_quantize_activation`，layer0 `q`(2560, gram) 与 `o`(4096, 无 gram) × 128/512 行 | **4/4 组，五字段逐个逐位相同** |
| `weight` | 真实 `hif4_calibration_and_quantize_weight`，layer0 `q` / `o` 各一次完整校准 | **weight_params 与 activation_state 全部逐位相同** |

取值集合是照着重写的**边界**选的：编码正好取整（floor==ceil）、6/7 clamp 的不对称、
码落在 0 以下与 7 以上、`inf` / `nan` / 大数 / 极小 / 混合。
比较走 `uint8` 视图的原始字节（`numpy().tobytes()` 对 bfloat16 不可用，字节视图还顺带比 NaN 载荷）。

`weight` 段另计数：layer0/q **父子都是 511 次调用、每次 `group_gram` 都非空**（改动确实被走到、
且走的是被改的分支）；layer0/o 两侧 0 次（无 gram 路径根本不经过）。调用次数相同本身是不变量：
改动若碰了控制流，这里会先露出来。

### 3.2 计时（`timing.py` / `timing.json`，三臂含 sham 零假设，31 轮）

`sham` = 同一个父根文件**再加载一次**，与 parent 逐字节相同；它与 parent 的配对差就是
"什么都没改时本方法会报出的数"。顺序每轮轮转，`torch.cuda.Event`，不挂 profiler。

| 对象 | parent | 配对效应 | 占 parent | null | \|效应\|/\|null\| | 更快轮数 |
|---|---:|---:|---:|---:|---:|---:|
| helper `[6,128,8,2,4]` | 0.9134 ms | **+0.0391 ms** | +4.28% | −0.0017 | 22.7× | **31/31** |
| helper `[6,509,8,2,4]` | 3.4023 ms | **+0.2246 ms** | +6.60% | −0.0055 | 40.7× | **31/31** |
| helper `[1,128,8,2,4]` | 0.4663 ms | **+0.0632 ms** | +13.56% | +0.0053 | 11.9× | 29/31 |
| 调用 layer0/**q** 2560ch,128 行（**有 gram**） | 548.00 ms | **+23.35 ms** | **+4.26%** | +0.69 | 33.9× | 24/31 |
| 调用 layer0/**q**,512 行（**有 gram**） | 894.46 ms | **+41.41 ms** | **+4.63%** | +0.79 | 52.4× | 30/31 |
| 调用 layer0/**o** 4096ch,128 行（无 gram） | 495.56 ms | +0.59 ms | +0.12% | +3.09 | **0.19×** | 16/31 |
| 调用 layer0/**o**,512 行（无 gram） | 557.13 ms | −0.78 ms | −0.14% | −0.90 | **0.86×** | 15/31 |

**无 gram 的两行是本卡自带的阴性对照，且按预期为零**：`o` 路径根本不调用 `_adaround_mantissa`
（0 次调用），所以这里**必须**测不出效应。有 gram 的两行是 null 的 34–52 倍、24–30/31 轮一致。

量级自洽：每次动态调用约 367 次 `_adaround_mantissa`（40 块 × 9.18 次/块），单次省 0.04–0.22 ms，
合起来 25–40 ms，与 512 行实测 41.4 ms 同量级。

### 3.3 六 shard（`pair_sixshard.py` / `paired_sixshard.json`）

`--linear-only` 六 shard，两侧各跑一遍独立进程，按 `case_id` 配对 `candidate.gain − baseline.gain`。
**候选侧 336 例全部恰好 0**（`positive 0 / negative 0 / zero 336`，等权 shard 均值 +0.000000，
逐 role 全 0）。

配对不是"两边都跑了就算数"：脚本核对每份 shard json 的 `source_sha256`，**同一侧跨 shard 必须一致、
两侧之间必须不同**。候选 `fe8aec19…`、基线 `ea79a1c1…`（= v231 根）通过校验——这同时排掉了
"基线其实是 v230 K=1"这一 rebase 污染。

**时间侧只做弱参考**（`api_total_seconds` 两侧不可比：含权重校准，缓存命中/未命中差约 128 s，
本轮 shard4 候选侧 201.96 s 就是这个离群）。只比 `hif4_dynamic_quantize_activation`
（两侧都每 shard 56 次、都无标定）：候选 41.818/41.567/47.461/41.639/41.703/43.187 合计 **257.374 s**，
基线 44.460/43.345/45.318/42.574/42.681/42.679 合计 **261.056 s**，差 **−3.682 s**（−11.0 ms/调用）。
方向与进程内配对一致，但**六个 shard 里两个为正**，且本轮没有 sham 臂，所以它只是"不矛盾"，
**不是本卡的时间证据**——时间证据是 §3.2 的带 null 三臂配对。

### 3.4 脱离仓库单文件导入（`check_standalone_import.py` / `standalone_import.json`）

`verify.py` 自己就要 import 评测器、读缓存，答不了"这个文件能不能单独提交"。
额外脚本回答 4B 指南提交前清单第 2 条：

- **谱系**：候选以父根字节开头；父根那份 `_adaround_mantissa` 在候选里 AST 逐字还在；
  实际绑定的是追加的影子（按加载后函数对象的 `co_firstlineno` = 12366 > 父文件行数 12325 判定，
  不是读源码猜的）。脚本对着工作区根**现算** sha，不符直接拒绝。
- **隔离执行**：候选单独拷进空目录，`python -I`、cwd 在该目录、`sys.path` 不许出现仓库路径，
  在**这个进程里把六个 API 全部跑起来**（不只检查存在）：权重校准、动态激活编码器、
  Attention 校准、Q/K/V 三个动态编码器。驱动脚本由本文件 `smoke` 源码经 `inspect.getsource`
  生成，两处不会漂移。
- **legal state**：回到本进程，把同一次 smoke 的 4 个 state 交给评测器自己的 `validate_state`，
  5 份参数交给 `validate_hif4_params`——隔离运行不自证。

结果：`starts_with_parent_bytes: true`、`parent_definition_unchanged_in_candidate: true`、
`live_helper_is_the_appended_one: true`、`bindings_of_helper_name: 1`；
两侧 `apis_executed` 一致（activation/q/k/v 各五字段），
`states_accepted_by_validate_state` 4/4、`params_accepted_by_validate_hif4_params` 5/5。

**如实记录一处**：合成随机输入下 `hif4_calibration_attention` **返回空 identity state**
（它没拟合出东西；`_check_attention_state` 随后会因缺 `num_heads` 拒绝空 dict）。这是合成输入的
性质、不是候选的性质，所以 Q/K/V 改用一个显式按合法契约构造的同几何 state 驱动，
空 state 这一事实留在 json 里（`attention_calibration_identity_on_synthetic_input: true`）。
因此**隔离那次运行没有覆盖"真实标定结果 → Q/K/V"这一段**，那一段由 `verify.py` 与面板覆盖。

## 4. 结论与归档

`LOCAL_OUTPUT_EQUIVALENT_LESS_WORK_PENDING_OFFICIAL`。归档 **v235**
（`solutions/20260910_v235_linear-ad1-adaround-materialization_scoreNA_timeNA/`），
官方状态 `unregistered/NA`，**不写本地秒数预测**——计划 §4.3 的接受条件是"官方秒数可分辨地下降"，
本机读数换不成官方秒数。

**根未切换**：工作区 `solution.py` 仍是 v231 根（`ea79a1c1…`，505762 B）。
v231 的官方余量是 9 s（291 / 300），所以本卡的动机是余量而不是分数；它逐位等价、精度风险为零，
是纯粹的余量买回。官方结果由用户统一评测回传后再登记。

## 5. 复现

```bash
.venv/Scripts/python.exe workbench/full_solution/linear-ad1-adaround-materialization/build.py
.venv/Scripts/python.exe workbench/full_solution/linear-ad1-adaround-materialization/verify.py --device cuda
.venv/Scripts/python.exe workbench/full_solution/linear-ad1-adaround-materialization/timing.py
.venv/Scripts/python.exe workbench/full_solution/linear-ad1-adaround-materialization/pair_sixshard.py
.venv/Scripts/python.exe workbench/full_solution/linear-ad1-adaround-materialization/check_standalone_import.py
```
