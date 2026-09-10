# A-GR1 落到 v231 完整根：执行记录（2026-09-10）

计划卡 [`2026-09-10-attention-agr1-on-v231-plan.md`](../../docs/superpowers/plans/parallel/2026-09-10-attention-agr1-on-v231-plan.md)。
归档 `solutions/20260910_v236_attention-agr1-on-v231_scoreNA_timeNA/`。

## 1. 为什么开这张卡

A-GR1 的官方价值已经由侧隔离测出（`standard-linear_v234-attn` = `14455/263.7s`，
相对 v195 侧基准 `14426/243s` 为 **+29**，继 C76.4 +84、A1 +60 之后第三大 Attention
正向机制），**但它从未落到完整根上**——v234 的父是 v230，而根此后晋级为 v231。
Attention 侧"已官方定价但未兑现"的正向机制，这是唯一的一个。

本卡动作只有一个：把 v234 的 A-GR1 块原样重新挂到 v231 根上，取一次完整包官方读数。

## 2. 构建与字节级证据

候选 = v231 根 `ea79a1c1…`（505762 B）**纯追加** A-GR1 块（15193 B），得 520955 B、
SHA `3319fc354af1ff0f35075498f26886bdf43a078d32a27852f575a4c50d32ba07`。

两条独立测量（不是构建脚本自述）：

1. `head -c 505762 candidate | cmp - solution.py` → **完全相同**，即纯追加。
2. `diff(v234归档候选, 本候选)` 与 `diff(v230根, v231根)` → **同一个 hunk**：
   只有 Linear 的 `_EM1_PASSES = 1 → 2` 及其注释。

`_EM1_PASSES` 全文件引用点只有三处：定义、`for _pass in range(_EM1_PASSES)`
（`_em1_dynamic_descent` 内，Linear）、和一个审计字段 `"em1_passes"`。
**Attention 段不读它。**

shard0 运行时账本显示 attention-only 场景下
`hif4_calibration_and_quantize_weight: 0 calls`、`hif4_dynamic_quantize_activation: 0 calls`
——评测的 attention-only 路径不调用任何 Linear API。于是推论：**本候选的 Attention
行为与 v234 必然逐位相同**。

**推论必须实跑证伪，不能替代证据。** 见 §4。

## 3. Control

`.venv\Scripts\python.exe workbench\full_solution\attention-agr1-on-v231\control.py` →
`OVERALL: PASS`（输出存 `control_results.txt`）。

| 控件 | 结果 |
|---|---|
| 1 脱离仓库单文件导入 | 六 API 全部可调用，`missing_apis=[]` |
| 2 校准 + 合法 state | `validate_state`/`validate_hif4_params` 通过；`agr1_attempted=1`；loss 2.0 → 1.2170；逆误差 2.98e-07 |
| 3 Linear 与 v231 父逐位 | `weight_params` 与动态 activation 逐位相同 |

控件 2 顺带打印 `[L-EM2] dynamic arm=applied groups=32 passes=2 accepted_steps=32`，
机械确认候选确实建在 **K=2 的 v231 根**上，而不是误建在 v230 上。

## 4. 六 shard：同构推论被独立证实

首次运行**只出了 2 条记录**——评测器 `--stop-after-nonpositive` 默认值是 **2**，
而本候选 shard0/shard1 恰好都是精确零，于是它停在 shard2 之前，**正好藏掉唯二两个
真正 arm 的层（15/22）**。显式传 `--stop-after-nonpositive 6` 后跑满（v234 当时也传了 6）。

```powershell
.venv\Scripts\python.exe evaluator/eval.py `
  --solution workbench/full_solution/attention-agr1-on-v231/candidate/solution.py `
  --baseline-solution solution.py --attention-only --shards 0,1,2,3,4,5 `
  --stop-after-nonpositive 6 `
  --cache artifacts/official_eval/cache/qwen3.5-4b-proxy-v2.pt `
  --calibration-cache-mode auto --algorithm-device cuda `
  --output-dir artifacts/proxy_v3/attention-agr1-on-v231-sixshard-20260910
```

| shard | 层 | Δmean | L1 | Δtail | ±/0 | 判决 |
|---|---:|---:|---:|---:|---|---|
| 0 | 0 | +0.000000 | 0.000000 | +0.000000 | 0/0/12 | gate parent（逐位） |
| 1 | 1 | +0.000000 | 0.000000 | +0.000000 | 0/0/12 | gate parent（逐位） |
| 2 | 8 | +0.000000 | 0.000000 | +0.000000 | 0/0/12 | gate parent（逐位） |
| **3** | **15** | **+0.017449** | 0.021696 | +0.027302 | 10/2/0 | **接受** |
| **4** | **22** | **+0.005618** | 0.005659 | +0.012536 | 11/1/0 | **接受** |
| 5 | 5 | +0.000000 | 0.000000 | +0.000000 | 0/0/12 | gate parent（逐位） |

等权 **`+0.003845`，21/3/48**。72 case 绝对分布 mean `0.5378421259`、median `0.5568569856`、
min `0.24300402`、max `0.79707225`。

**与 v234 六 shard 逐项完全相同**：不只是 Δmean，L1、Δtail、计数、绝对分布全部一样。
同构推论成立。

**这张卡因此不产出新的机制证据**——它是"同一 Attention 行为 + 新 Linear 父"。
如实登记，不把复现当增益。

## 5. 时间与官方

- 侧隔离形态 +20.7s（相对 v195 侧）；当前完整根 291s / 300s，**余量 9s**。
- **不写官方秒数预测**：侧时间对完整包无预测力（v194 侧 −4s / 完整 +5s；v190 侧 246s /
  完整 TIMEOUT，两个方向都错）。
- 本次 attention-only 运行为校准缓存命中，api 秒数是纯计分口径，不跨缓存状态比较（AGENTS §5）。
- 官方 `unregistered/NA`，交用户统一评测。晋级规则事前固定：**官方分数高于 18518 且在
  300s 内才晋级；超时或更低则不晋级。**

### 5.1 归档之后：兄弟卡的完整包超时

本卡建好并归档之后，用户回传 **v234 完整包（v230 根 + 同一块 A-GR1）官方
`TIMEOUT (>300s)`**——见 [`2026-09-10-v234-agr1-official-timeout.md`](2026-09-10-v234-agr1-official-timeout.md)。

这条证据直接落在本卡上：A-GR1 的额外时间在**校准侧**、是**机制属性**，基本不随 Linear 父变化，
而 v231 只比 v230 快 **1s**（291 vs 292）。所以本卡预期总时约为 v234 的 −1s，
**只有 v234 超限幅度 ≤1s 时本卡才可能进限**。超限幅度官方未给出 → 仍不写秒数预测。

它**不改变本卡事前固定的规则**，只是把**超时分支**推成主要分支；也**不能替本卡下结论**
（不同实现、不同父）。

**产生的判断：卡住 A-GR1 的是时间余量本身（需约 20s，手上 9s），不是父、也不是机制。**
Attention 侧最高价值的动作因此不是再出 A-GR1 变体——该卡规则禁止缩步/缩窗/减候选（那是削弱
机制）——而是**先把秒数腾出来再落地 A-GR1**，与 Linear 侧正在做的运行成本回收同向。

## 6. 血统

v236 是 **v231 的后代**（含 K=2），**不含** v235（Linear L-AD1）与 v233（L-TF1），
与二者不可相加。Attention 机制本身是 v234 的，v234 建在 v230 根上；
v234 归档与其父保留不动。

## 7. 给后续 Attention 卡的执行提醒

`--stop-after-nonpositive` 默认 2 会在**精确零的连续 shard** 上提前终止。
Attention 侧的候选经常出现多层 gate parent（逐位零），所以这个默认值会系统性地
**藏掉真正 arm 的层**。跑 Attention 六 shard 一律显式传 `6`。
