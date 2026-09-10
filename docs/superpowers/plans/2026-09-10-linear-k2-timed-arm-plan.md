# Linear groupstep 的 K = 2 臂：用官方数据裁决换算口径（L-EM3）

> 状态：ACTIVE，2026-09-10。L-EM2/v230 之后的**重新规划卡**。
> 当前完整根：v202 Linear + v195 Attention，官方 `18053/281s`，SHA256
> `56DC805D6E5A3AEF896DB8021045740292735725D688B48E3D4393E55EFCB2BD`。
> 前一张卡 [L-EM2 组号主序调度](2026-09-10-linear-groupstep-schedule-plan.md) 已归档为
> v230（K=1，本地六 shard `288/0/48`、等权 `+0.079454`，`official_status: PENDING`）。

## 1. 为什么是"抬 K"，而不是别的

L-EM2 把机制的成本从 L-EM1 的 `+95.9 s` 降到 `+12.5 s` 本地（动态 `+6.5`、校准钩子 `+6.0`），
机制本身**逐条未变**：精确度量 `G`、交叉矩阵 `H`、理想目标 `T`、pm1 候选集、精确整行联合接受。
唯一剩下的、计划卡**自己预先声明过**的自由度就是 passes `K`：

| K | 本地动态 | in=2560 p1 相对损失下降 | 依据 |
|---:|---:|---:|---|
| 1 | +6.5 s | −19.10% | v230 实测 |
| 2 | ≈+10.0 s | **−26.06%** | `probe_frontier` 真实 4B 数据实测 |
| 4 | ≈+20 s | −31.23% | 同上 |

机制**由构造单调**（每步整行联合移动都过精确二次复验，被否决的行本步完全不动），
所以 K 增大**只有时间风险、没有精度风险**。L-EM1 的逐组序贯（等价 K→∞ 的一遍）在同一
shard0 面板上是 `+0.107791`，即 K=1 只拿到最终可达增益的 82%，剩余的那 18% 全部锁在 K 里。

### 为什么这一轮才做

L-EM2 的计划卡规定了一条**时间驱动的回退**：真实配对折算官方 > 296 s 则 K 降为 1。该规则在
卡片自己的**朴素可加**口径下触发（K=2 折算 297.5 s），于是 v230 归档为 K=1。

归档时同时记录了一个事实：本项目对"本地秒 → 官方秒"另有一个**实测分解回归**
（`docs/official-local-fitting-analysis-2026-09-04.md` §4，21 个版本，`R²=0.799`、`MAE=10.1 s`）：

```
T_official ≈ 170.3 + 0.1154·W_calib + 0.6939·A_calib + 0.7344·dyn_act − 1.5837·dyn_qkv
```

`dyn_act` 与 `W_calib` **恰好就是本机制改动的两个 API**。按它差分折算，K=2 是
`281 + 0.7344×10.0 + 0.1154×6.0 ≈ 289 s`，**不会触发 296 s 线**。两个口径对同一个候选给出
相反结论，而**本地无法裁决**：官方时间是唯一的门，本项目的历史也明确记录官方时间不能由本地
边际可靠外推（v190 侧隔离 246 s 而完整包 TIMEOUT）。

**因此本卡不试图在本地判定哪个口径对，而是把两个 K 都交给官方定价。**
官方提交无限制（`docs/superpowers/plans/README.md`），K=1（v230）已在册，本卡补上 K=2（v231）。
两个数据点同时回答两件事：机制是否过门，以及 **Linear 动态 API 的本地→官方换算系数**——
后者对后续每一条 Linear 机制都是可复用的公共资产。

## 2. 预注册（在测量之前写定，不得事后更改）

| 项 | 值 |
|---|---|
| 候选 | L-EM2 逐字节不变，**只把 `_EM1_PASSES` 由 1 改为 2** |
| 版本号 | **v231**（v230 的兄弟，同一父根 `56dc805d`）。注意 **v230 已被两条线同时占用**（Linear `20260910_v230_linear-em2-groupstep-schedule_…` 与 Attention `20260910_v230_attention-afix1-train-deploy-align_…`），引用必须写全目录名；v231 未占用 |
| 机制 | 与 v230 完全相同的 groupstep；K=2 即 pass 之间重算精确梯度 |
| 精度期望 | 真实 `layer0/q` window1 rows128：`dL = −26.0625%`（probe `ideal/groupstep/pm1/p2`） |
| 时间 | 配对实测 K=2 动态 + 校准钩子，**记录不设门** |
| 归档条件 | **不论折算结果是否越过 296 s，v231 一律归档为 `PENDING`** |
| 不做的 | 不得因为 v231 更快或更慢而回头修改 v230 的 K；不得扫描 K∈{3,4,…}（本卡只做 K=2 这一个**已预先声明**的点） |

> **"不设门"为什么不是放松纪律。** 计划卡的 296 s 线是为了防止提交一个**可能超时**的候选。
> 本卡的目的相反：主动让官方**测量**这条线在哪。官方超时得 0 分，但本项目的历史表明超时
> 只关闭该实现、不损伤根（v223/v224/v225/v229 均如此），而**一次超时换来的换算系数**会
> 改变后续所有 Linear 卡的时间判断。收益大于代价。两个臂都是计划卡原文预先声明的（K=2 是
> 主选，K=1 是回退），本卡不是在事后重新挑参数，而是把主选臂补交。

## 3. 顺带的等价降时（可选，实测决定是否采纳）

groupstep 每步构造一个**稠密**的 `(rows, channels)` `row_delta`，其非零列只有本步的
`blocks×4` 列（占 `1/16`），再与 `(channels, channels)` 的 `metric` 相乘。改用本步列积在
数学上恒等。`bench_groupstep_gpu.py`（合成 shape，10 次中位）实测：

| shape | dense | 列积 | 比值 |
|---|---:|---:|---:|
| in=2560 rows=128 | 9.47 ms | 10.43 ms | **1.10（更慢）** |
| in=2560 rows=512 | 21.87 ms | 11.68 ms | 0.53 |
| in=4096 rows=512 | 42.41 ms | 17.88 ms | 0.42 |

**结论：不作为本卡的改动。** rows=128 时单步已是纯派发受限（~300 次小算子发射 ≈ 9.5 ms），
列积多出 2–3 次 `index_select` 反而更慢；而官方面板以 rows=128 为主，总体只值 <1 s。
记录备查，不做形状分支（那会引入阈值选择）。

## 4. 执行步骤

1. 复制 L-EM2 workbench 为 `linear-em3-k2-arm/`，`implementation.py` 只改 `_EM1_PASSES = 2`，
   `build.py` 的 `REQUIRED_TOKENS` 同步改为 `"_EM1_PASSES = 2"`；
2. `verify.py` 沿用 L-EM2 的 A–F，把 control D 期望值与 control F 的 `expected_passes`
   换成 K=2 的（`_EM2_REAL_REL_BY_PASSES[2] = -0.260625`，`accepted_steps = 32`）；
3. `time_paired.py` 配对实测 K=2（校准钩子不变，直接沿用 L-EM2 的 `profile_compile_hook.py` 结论）；
4. 跑 shard0，然后六 shard（**先 `nvidia-smi` 确认 Attention 线未在跑，单卡串行**）：

   ```
   .venv\Scripts\python.exe evaluator\eval.py ^
     --solution workbench\full_solution\linear-em3-k2-arm\candidate\solution.py ^
     --baseline-solution solution.py --linear-only ^
     --shards 0            --name linear-em3-shard0   ^
     --stop-after-nonpositive 99 ^
     --cache artifacts\official_eval\cache\qwen3.5-4b-proxy-v2.pt ^
     --calibration-cache-mode write --algorithm-device cuda
   ```

   六分片同上，`--shards 0,1,2,3,4,5 --name linear-em3-sixshard`；
   （本卡初稿写的 `run_shard.py` 并不存在，实际入口是 `evaluator/eval.py`，输出落在
   `artifacts/proxy_v3/<name>/candidate/`。此段是**执行细节的事后订正**，§2 预注册表逐字未动。）
5. 归档 v231 并提交推送；v230 保持不变，两臂并列 `PENDING` 等用户批量官方评测。

## 5. 与 Attention 计划的协调

- 开发与 verify 阶段不占 GPU；任何 4B 评测前先 `nvidia-smi` 确认，单卡串行；
- calibration cache 继续按 SHA 修剪，`--min-age-hours 2`；
- 官方组合顺序：两条线各自独立消融，组合顺序以官方实测为准。
