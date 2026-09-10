# L-AD1：去掉 16 模式枚举里的整数物化（计划 §4.3 的固定卡）

活动计划：`docs/superpowers/plans/2026-09-10-linear-correctness-and-runtime-plan.md` §4.3。
父根：**v231 完整根**（Linear L-EM3 groupstep K=2 + v195 Attention），SHA256
`ea79a1c12dc667142c620975aab188920fae7b29988c804f41c1a696cc5754f1`（505762 字节），
官方 **18518 / 291 s**（RETAINED；v230 为 18428 / 292 s）。
设备：NVIDIA GeForce RTX 3060 Ti。候选：`implementation.generated.py`
（SHA256 `fe8aec1989189ad9c20cdc9e9c3717ad509c98da987cb556b1e1428fc13bc702`，510664 字节）。

## 根基线变更（这张卡在开发中途被 rebase 过）

本卡初稿是对着 **v230 根**（`0f1af6db…`，505496 字节，K=1）写的。在开发期间官方结果回传：
v231 得 18518 / 291 s 且被晋级，工作区 `solution.py` 随之变成 v231（`ea79a1c1…`，505762 字节，
差异就是 `_EM1_PASSES` 1→2）。**若继续在 v230 上构建，提交的候选会把官方刚刚花了 1 s 换来的
K=2 那一臂悄悄回退掉**，而本地配对会把它报成这张卡的收益。所以本卡在 v231 上重建：

- `build.py` 的 `EXPECTED_PARENT_SHA256` / `EXPECTED_PARENT_BYTES` 已换成 v231 值，
  并在 `build.json` 里同时记录被取代的 v230 sha 与 rebase 原因；父根 sha 不符时构建直接失败，不静默吸收。
- `verify.py` / `timing.py` 里标定缓存的键**不再写死**，改为从父根文件现算 sha 前缀
  （`hashlib.sha256(PARENT.read_bytes()).hexdigest()[:16]`）。写死的前缀在 rebase 后会**静默命中
  上一个根**的标定 state，让激活对比变成"两个不同 state 互比"并通过——这条已按构造消除。
- `pair_sixshard.py` 的基线从 `linear-em3-v230base` 换成 `linear-em3-cand`
  （该次运行记录的 `source_sha256` 就是 v231 根 `ea79a1c1…`），否则基线差里会混进 K=2 的增量。
- v230 上的全部旧证据已作废并重跑：候选、等价性、计时、六 shard 全部是本节记录的 v231 读数。
  v230 读数（helper +3.93/+6.39/+11.1%，调用 +4.66/+4.91%）与新读数一致，构成一次独立复现。

## 这张卡做什么

上一轮定位（[`../linear-parent-path-split/result.md`](../linear-parent-path-split/result.md)）把成本定到
**编码器 `_dense_to_hif4`**（占一次动态调用 69.8–90.8%），再定到 gram 层里
`_adaround_mantissa` 的 16 模式枚举（占该层编码器 51–68%）：它把 `[K,N,8,2,4]` 张成 16 倍、
**以 int64 物化**，再整块转 float32、乘 0.25。

§4.3 允许的唯一改动就是去掉这条链上的物化，**候选一个不减**：

```python
# 父根
floor_code = torch.floor(raw_code).clamp(0, 6).to(torch.int64)
ceil_code  = torch.ceil(raw_code).clamp(0, 7).to(torch.int64)
...
masks = ((bits_all.unsqueeze(1) >> torch.arange(4, device=...)) & 1).bool()
...
all_codes    = torch.where(mask_expanded, ceil_code.unsqueeze(0), floor_code.unsqueeze(0))
all_mantissa = all_codes.to(torch.float32) * 0.25

# 候选
floor_code = torch.floor(raw_code).clamp(0, 6).to(torch.int64)   # 逐字不变
ceil_code  = torch.ceil(raw_code).clamp(0, 7).to(torch.int64)    # 逐字不变
floor_mant = floor_code.to(torch.float32) * 0.25                 # 在小张量上做
ceil_mant  = ceil_code.to(torch.float32) * 0.25                  # 在小张量上做
masks = _AD1_pattern_masks(x_abs.device)                         # 与输入无关，缓存
all_mantissa = torch.where(mask_expanded, ceil_mant.unsqueeze(0), floor_mant.unsqueeze(0))
```

**为什么这是逐位相等的，先于任何测量。** 两条 int64 码表达式**逐字未动**；
唯一移动的是两个逐元素算子相对于一次 `torch.where` 的位置。`.to(torch.float32)` 与 `* 0.25`
都是逐元素算子，而逐元素算子与 `where` 的交换是**定义上的相等**——选择先取出每个位置的一个值，
再把函数作用到那个值上——所以两种写法对任意 int64 对、任意掩码都相等，**不需要对转换做任何精度假设**，
非有限输入经 int64 转换得到的值也一样（两侧拿到同一个值，受同一个逐元素转换）。
候选数、loss、`argmin`、`gather`、覆盖率、refine 轮数、接受规则全部未动。

`masks` 只依赖 device，提到调用外并用 device 作键缓存；它只被读（`.reshape` 返回视图，无人写穿）。

## 构建证据（`build.py` / `build.json`）

候选是**父根字节 + 追加的一个模块**，一个父字节都没改：追加的定义在调用时从模块全局解析，
自然覆盖前面那个，不需要 hook。构建脚本逐条算出并且要求满足：

- 父 SHA256 与字节数符合预期（`ea79a1c1…` / 505762）；
- 父模块的 **473 条顶层语句在候选里 AST 逐字相同**，其后**恰好追加 3 条**
  （`_AD1_PATTERN_MASK_CACHE` 赋值、`_AD1_pattern_masks`、`_adaround_mantissa` 影子定义）；
- 函数体 17 条语句中，**前缀 4 条相同、后缀 7 条相同、改动是中间连续的 4..9 六条**
  （语句数不变，6 换 6）；
- 被删掉的名字 `bits_all` / `all_codes` 在候选函数里出现 0 次；新引入的 `floor_mant` / `ceil_mant`
  各出现 2 次（一次绑定、一次读取）；
- 替换次数 = 1（执行前先确认该块在父函数里唯一）。

v230 与 v231 两次构建的追加字节数相同（+4903），父模块语句数、函数语句数、改动窗口全部相同——
被改的函数与 `_EM1_*` 那一行没有任何关系。

## 等价性（`verify.py` / `verify.json`）

共享助手的硬线要求**对全部调用者**证明。三段证据，在 v231 父根上重跑，结论与 v230 时相同：

| 段 | 覆盖 | 结果 |
|---|---|---|
| `unit` | 父子 `_adaround_mantissa` 直接对比：5 种形状（1–5 维）× 4 种 dtype × 10 组取值 × 2 种 scale，另加 3 种非连续布局 | **403 组，0 组不同，0 组抛错** |
| `activation` | 真实 `hif4_dynamic_quantize_activation`，layer0 `q`(2560, gram) 与 `o`(4096, 无 gram) × 128/512 行 | **4/4 组，五字段逐个逐位相同** |
| `weight` | 真实 `hif4_calibration_and_quantize_weight`，layer0 `q` / `o` 各一次完整校准 | **weight_params 与 activation_state 全部逐位相同** |

`verify.json` 另记 `candidate_starts_with_parent_bytes: true`、`parent_definitions_of_helper: 1`、
`candidate_definitions_of_helper: 2`、`parent_definition_unchanged_in_candidate: true`——
即父根那一份定义在候选里逐字还在，改动是影子而不是编辑。

取值集合是照着重写的**边界**选的，不是为了一般性：编码正好取整（floor==ceil，两个候选重合）、
6/7 两个 clamp 的不对称、码落在 0 以下与 7 以上、以及 `inf` / `nan` / 大数 / 极小 / 混合。

比较一律走 **`uint8` 视图的原始字节**（`numpy().tobytes()` 对 bfloat16 不可用，字节视图还顺带比较 NaN 载荷），
不用 `torch.equal`、不用容差。

`weight` 段还数了 `_adaround_mantissa` 的调用次数：layer0/q **父子都是 511 次、且每一次 `group_gram` 都非空**——
这条路径确实走到了被改的函数，而且走的是被改的那条分支；layer0/o 两侧都是 0 次，
说明无 gram 的权重路径根本不经过它（与 `encoder_probe.py` 的读数一致）。调用次数相同本身也是一条不变量：
改动若碰了控制流，这里会先露出来。

## 计时（`timing.py` / `timing.json`）

三臂同状态同输入配对：`parent`（v231 发布根）、`candidate`（v231 + L-AD1）、`sham`（**同一文件再加载一次**，
与 parent 逐字节相同）。sham 臂是重点：它与 parent 的配对差就是"什么都没改时本方法会报出的数"，
是本机、本时刻、本热漂移下的**经验零假设**。顺序每轮轮转，避免某一臂总是最后跑；
`torch.cuda.Event`，不挂 profiler、不重放；每臂先做无计时预热。31 轮。

**函数级（`_adaround_mantissa` 单独计时，200 次/样本，取真实批量形状）：**

| 形状 | parent | 配对效应 | 占 parent | null | \|effect\|/\|null\| | 更快轮数 |
|---|---:|---:|---:|---:|---:|---:|
| `[6,128,8,2,4]`（批量解） | 0.9134 ms | **+0.0391 ms** | +4.28% | −0.0017 | 22.7× | **31/31** |
| `[6,509,8,2,4]`（批量解，N=509） | 3.4023 ms | **+0.2246 ms** | +6.60% | −0.0055 | 40.7× | **31/31** |
| `[1,128,8,2,4]`（边扩展） | 0.4663 ms | **+0.0632 ms** | +13.56% | +0.0053 | 11.9× | 29/31 |

**整调用级（真实 state、真实激活）：**

| | parent | 配对效应 | 占 parent | null | \|effect\|/\|null\| | 更快轮数 |
|---|---:|---:|---:|---:|---:|---:|
| layer0/**q** 2560ch,128 行（**有 gram**） | 548.00 ms | **+23.35 ms** | **+4.26%** | +0.69 | 33.9× | 24/31 |
| layer0/**q**,512 行（**有 gram**） | 894.46 ms | **+41.41 ms** | **+4.63%** | +0.79 | 52.4× | 30/31 |
| layer0/**o** 4096ch,128 行（无 gram） | 495.56 ms | +0.59 ms | +0.12% | +3.09 | **0.19×** | 16/31 |
| layer0/**o**,512 行（无 gram） | 557.13 ms | −0.78 ms | −0.14% | −0.90 | **0.86×** | 15/31 |

**无 gram 的两行是这张卡自带的阴性对照，而且它按预期为零**：`o` 路径根本不调用
`_adaround_mantissa`（见上节 0 次调用），所以这里**必须**测不出效应；实测效应确实落在 null 之内
（0.19× 与 0.86×，更快轮数 15–16/31 就是抛硬币）。有 gram 的两行效应是 null 的 34–52 倍、
24–30/31 轮一致，方向一致。

两级的数量级自洽：每次动态调用约 367 次 `_adaround_mantissa`（40 块 × 9.18 次/块，
9.18 = 1 + 3 + 3×边扩展次数，见 `encoder_probe.py`），单次省 0.04–0.22 ms，
合起来约 25–40 ms，与 512 行实测 41.4 ms 同量级。

## 六 shard 本地结果（`pair_sixshard.py` / `paired_sixshard.json`）

`--linear-only` 六 shard，两侧各跑一遍独立进程（不做进程内配对），按 `case_id` 逐例配对
`candidate.gain − baseline.gain`。**候选侧 336 例，全部恰好 0**：

| shard | 例数 | Δ=0 | 正 | 负 | \|Δ\|max | 按 role 的 Δ |
|---|---:|---:|---:|---:|---:|---|
| 0–5（各） | 56 | **56** | 0 | 0 | 0.0 | q/k/v/o/fc_gate/fc_up/proj 全 0.0 |
| **合计** | **336** | **336** | **0** | **0** | **0.0** | 等权 shard 均值 **+0.000000** |

逐位相同的字节 ⇒ 逐位相同的分数；这里连一次"末位抖动"都没有，`case_count` 336 / `zero_cases` 336。

配对不是"两边都跑了就算数"：`pair_sixshard.py` 会核对每份 shard json 里记录的
`source_sha256`，**同一侧跨 shard 必须一致、两侧之间必须不同**，否则拒绝配对。
本轮的候选侧 `fe8aec19…`、基线侧 `ea79a1c1…`（即 v231 根）都通过了这道校验——
这同时排掉了"基线其实是 v230 K=1"这一 rebase 污染（见上节）。

**时间侧：这条运行只做弱参考。** `api_total_seconds` 两侧不可比（含
`hif4_calibration_and_quantize_weight`，缓存命中与未命中差约 128 s；本轮 shard4 候选侧
weight 校准就是 201.96 s 的离群），所以只比 `hif4_dynamic_quantize_activation`——
两侧都是每 shard 56 次调用、都不含标定：

| shard | 0 | 1 | 2 | 3 | 4 | 5 | 合计 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 候选 | 41.818 | 41.567 | 47.461 | 41.639 | 41.703 | 43.187 | **257.374** |
| 基线 | 44.460 | 43.345 | 45.318 | 42.574 | 42.681 | 42.679 | **261.056** |
| Δ | −2.642 | −1.778 | **+2.143** | −0.936 | −0.978 | **+0.508** | **−3.682** |

跨进程、跨时刻的每例差是 **−0.0110 s/调用**，方向与 `timing.py` 的进程内配对一致，
但**六个 shard 里有两个是正的**，所以这条读数只是"不矛盾"，不是本卡的时间证据——
本卡的时间证据是 `timing.py` 里带 sham 零假设的三臂配对（有 gram 的层 34–52 倍于 null），
本表没有对应的 null，不足以分辨。按 `timing.py` 的效应量（约 25–40 ms/调用）折算，
336 次调用预期省 8–13 s；实测 −3.68 s 落在这个量级内但偏小，与上表的噪声一致。

## 脱离仓库的单文件导入检查（`check_standalone_import.py` / `standalone_import.json`）

`verify.py` 自己就要 import 评测器、读缓存，答不了"这个文件能不能单独提交"；4B 指南
提交前清单第 2 条（legal state + 脱离仓库单文件导入）由本脚本单独回答：

- **谱系**（对着工作区根现算，sha 不符直接拒绝）：候选以父根字节开头，
  父的一份 `_adaround_mantissa` 在候选里 AST 逐字还在，且**实际绑定的是追加的那一份影子**
  （按加载后的函数对象 `co_firstlineno` = 12366 > 父文件行数 12325 判定，不是读源码猜的）；
- **隔离执行**：把候选单独拷进空目录，用 `python -I`、cwd 设在该目录、`sys.path` 里
  不许出现仓库路径，在**这个进程里**把**六**个 API 全部**跑起来**（不只是存在）：
  权重校准、动态激活编码器、Attention 校准、以及 Q/K/V 三个动态编码器，
  各自用合成输入；驱动脚本由本文件 `smoke` 的源码经 `inspect.getsource` 生成，两处不会漂移；
- **legal state**：回到本进程，把同一次 smoke 返回的 4 个 state 交给评测器自己的
  `validate_state`，5 份参数交给 `validate_hif4_params`——隔离运行不自证。

结果：`starts_with_parent_bytes: true`、`parent_definition_unchanged_in_candidate: true`、
`live_helper_is_the_appended_one: true`、`bindings_of_helper_name: 1`；
隔离侧与进程内两侧的 `apis_executed` 完全一致（activation/q/k/v 各五字段
`mant, scale_factor, scale_lv2, scale_lv3, sign`），
`states_accepted_by_validate_state` = 4/4，`params_accepted_by_validate_hif4_params` = 5/5。

一处如实记录：合成随机输入下 `hif4_calibration_attention` **返回空 identity state**
（它没拟合出东西；评测器的 `_check_attention_state` 随后会因缺 `num_heads` 拒绝空 dict）。
这是合成输入的性质、不是候选的性质，所以 Q/K/V 改用一个显式按合法契约构造的同几何 state
来驱动，空 state 这一事实本身留在 json 里（`attention_calibration_identity_on_synthetic_input: true`），
真实拟合 state 上的 Q/K/V 由 `verify.py` 与面板覆盖。

## 边界（这份结果不主张什么）

- **本地秒数不是官方秒数。** 上面的毫秒是本机 CUDA event 读数，不能换算官方时间；
  官方 300 s 硬限是唯一时间门，官方分数与秒数由用户统一评测回传。计划 §4.3 的接受条件是
  "官方秒数可分辨地下降"，因此本卡的官方状态在归档时写 null，不写预测。
- **这不是"整模型 4.6%"**：+4.26%/+4.63% 只对 **有 gram 的层的动态激活调用**成立。
  无 gram 层（o、proj）测不出效应也不应测出。按调用点加权，gram 层约占动态激活宽度 76%
  （120/168 个调用点），整 API 的收益要按这个权重折算，本目录不做这个折算。
- **v231 的官方余量是 9 s（291 / 300）**，所以这张卡的动机是余量而不是分数；
  它逐位等价、精度风险为零，是纯粹的余量买回。
- `masks` 缓存是本卡新增的**模块级状态**：以 device 字符串为键，只增不减，进程内只读。
  它不是纯函数意义上的无副作用改动，虽然值上与被替换的表达式逐位相同。
- 等价性证据覆盖：单元 403 组、真实激活 4 组、真实权重校准 2 个 state。
  没有扫全部 24 层的全部 role；"全部调用者"的主张建立在
  **逐元素算子与 `where` 交换**这条对任意输入成立的等式上，实测是它的对照，不是它的替代。
- **六 shard 的时间读数不是对照实验**：它是两次独立进程的先后运行，没有 sham 臂，
  六个 shard 里两个为正；本卡的时间结论只由 `timing.py` 的三臂配对支撑（见上）。
- **单文件导入检查是契约级 smoke，不是面板**：合成张量、小形状，只证明"能导入、能跑完、
  产出过评测器校验"，对精度与秒数不置一词。
- `hif4_calibration_attention` 在合成随机输入上返回空 state；Q/K/V 是用显式构造的
  合法 state 驱动的，因此隔离那次运行**没有**覆盖"真实标定结果 → Q/K/V"这一段，
  那一段由 `verify.py` 与面板的 72 例/shard 覆盖。

## 复现

```bash
.venv/Scripts/python.exe workbench/full_solution/linear-ad1-adaround-materialization/build.py
.venv/Scripts/python.exe workbench/full_solution/linear-ad1-adaround-materialization/verify.py --device cuda
.venv/Scripts/python.exe workbench/full_solution/linear-ad1-adaround-materialization/timing.py
.venv/Scripts/python.exe workbench/full_solution/linear-ad1-adaround-materialization/pair_sixshard.py
.venv/Scripts/python.exe workbench/full_solution/linear-ad1-adaround-materialization/check_standalone_import.py
```
