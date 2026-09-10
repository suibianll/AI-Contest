# v239 — Attention A-CT2：`_agr1_train` 末尾统计复用的训练尾部消除

活动计划 [`docs/superpowers/plans/2026-09-10-retained-root-gradient-reuse-plan.md`](../../docs/superpowers/plans/2026-09-10-retained-root-gradient-reuse-plan.md) §4 的 A-CT2 固定实现卡。
执行记录 [`logs/execution/2026-09-10-attention-act2-train-tail-reuse.md`](../../logs/execution/2026-09-10-attention-act2-train-tail-reuse.md)。
停滞诊断 [`docs/attention-stall-analysis-2026-09-10.md`](../../docs/attention-stall-analysis-2026-09-10.md)。

| | |
|---|---|
| 正式父 | v231 完整根（Linear L-EM3 K=2 + v195 Attention），官方 **18518 / 291 s**（余量 9 s） |
| 正式父 SHA256 | `ea79a1c12dc667142c620975aab188920fae7b29988c804f41c1a696cc5754f1` / 505762 B |
| 装配来源 | v236（= 该根 + A-GR1，逐字节验证）`3319fc35…` / 520955 B，**非晋级父** |
| 候选 SHA256 | `55103e8ba530bf2cc03fc07bf24358ada4bda8d54ccc976468f5cbd985ecdac9` / 531018 B |
| 改动 | 两次子串替换：`final_loss` 循环带上两个按角色的标量和；两个 ratio 由这两个和算出，删掉第二次遍历（−74 B） |
| 本地六 shard | 对**同父 A-GR1 旧实现 v236**：**72/72 精确零**（0/0/72） |
| 调用计数 | 每次 `_agr1_train`：`_agr1_scale_loss_grad` **204 → 198**、`_a2_apply_group_rotation` **476 → 470**；**训练段 192 次未动** |
| 时间 | **本机测不出**：四行效应都落在同字节 sham 空对照之内（0.58× / 0.32× / 2.37× / 19.83×，**方向一致性全部 ≤11/15**） |
| 官方 | **`unregistered/NA`**。**量级已测定不足**——见下文"边界" |

## 机制：训练尾部对同一批 fold 走了两遍

`_agr1_train` 收尾时先算 `final_loss`：

```python
final_loss = float(n.square().mean().item()) * _AGR1_REG_WEIGHT
for fold in prepared:
    for (coordinate, denominator, heads), matrix in zip(fold, (m, p)):
        value, _ = _agr1_scale_loss_grad(
            _a2_apply_group_rotation(coordinate, int(heads), matrix),
            denominator,
        )
        final_loss += float(value.item()) / float(len(prepared))
```

它已经为每个 fold 分别用 `m`（q 角色）和 `p`（k 角色）算过一次 `..._scale_loss_grad(...)[0]`。
随后 `info` 里的两个字段**把同一批调用又做了一遍**，只取回第一次丢掉的标量：

```python
"agr1_q_scale_ratio2": 0.0 if not prepared else float(
    sum(float(_agr1_scale_loss_grad(
        _a2_apply_group_rotation(fold[0][0], int(fold[0][2]), m), fold[0][1])[0].item())
        for fold in prepared) / float(len(prepared))),
"agr1_k_scale_ratio2": ... 同上，用 fold[1] 与 p ...
```

候选把这两个和直接带出来：

```python
q_scale_sum = 0.0
k_scale_sum = 0.0
for fold in prepared:
    ((q_coordinate, q_denominator, q_heads), (k_coordinate, k_denominator, k_heads)) = fold
    q_value, _ = _agr1_scale_loss_grad(_a2_apply_group_rotation(q_coordinate, int(q_heads), m), q_denominator)
    final_loss += float(q_value.item()) / float(len(prepared))
    k_value, _ = _agr1_scale_loss_grad(_a2_apply_group_rotation(k_coordinate, int(k_heads), p), k_denominator)
    final_loss += float(k_value.item()) / float(len(prepared))
    q_scale_sum += float(q_value.item())
    k_scale_sum += float(k_value.item())
...
"agr1_q_scale_ratio2": 0.0 if not prepared else q_scale_sum / float(len(prepared)),
"agr1_k_scale_ratio2": 0.0 if not prepared else k_scale_sum / float(len(prepared)),
```

**为什么两个累加顺序能逐位复现父**：`final_loss` 仍是逐 fold 先 q 后 k、每项先除以 `len(prepared)` 再累加；
两个 ratio 仍是按 fold 顺序左折叠后一次除以 `len(prepared)`——`sum()` 从 0 起步，而 `0 + x` 是精确的。

**展开成两个角色是必需的，不是风格问题**：float 不可变，若写成
`zip(fold, (m, p), (q_scale_sum, k_scale_sum))` 只会重绑定局部名，和在循环外丢失。

## 审计先于开发（计划要求的顺序）

计划 §4 规定"先验证同 dtype、顺序和异常行为，明确可复用的每 fold 标量；只在确认后，在本节补一张固定实现卡再开发"。
`audit.py` 的三问都由 AST 与实测算出，不是断言：

| 问 | 方法 | 结果 |
|---|---|---|
| Q1 两个表达式是不是同一个 | AST 上按 `zip(fold, (m, p))` 的位置对应 | 迭代 i 的通用调用 == ratio 的角色 i 调用（同 fold 项、同矩阵） |
| Q2 第二次是否真的逐位重复 | 仪器化真实 `_agr1_train` 调用，记录入参与返回值字节 | **6/6 重复调用入参逐位相同、返回标量逐位相同**（六层全同） |
| Q3 带出来的标量能否重建 ratio | 用循环内的标量重算两个 ratio | 与上报值**精确相等**（六层全 True） |

另记录在案的事实：全链路 float32；`force_zero` 在全部 5 个归档调用点**均不可达**（该分支会跳过 `final_loss` 循环、
令标量未绑定，若可达则本卡前提不成立）。每次 `_agr1_train` 的 204 次调用构成：
192（32 步 × 2 角色 × 3 fold）+ 6（final_loss）+ 6（ratio）。

## 构建证据（`build.py` / `build.json`）

候选 = v236 装配字节 + 追加模块，父字节一个未动；被追加的 `_agr1_train` 影子在调用时从模块全局解析生效。

- 两次子串替换，**每次替换前断言该块在父函数里唯一、替换后断言其已消失**；
- 候选文本必须等于 `父文本.replace(块1).replace(块2)`——在唯一性成立的前提下，
  这就是"除这两块外什么都没动"的陈述；
- 字节级：改动 36 行、**−74 B**（原帖里的空行与多层缩进随之消失）；
- `_agr1_scale_loss_grad` 在该函数内的调用点数由 4 降为 3（训练 1 + 尾部 2）。

## 等价性（`verify.py` / `verify.out`，CPU，全 PASS）

**比只比那三个 float 更强**：`agr1_final_loss`、`agr1_q_scale_ratio2`、`agr1_k_scale_ratio2` 都在返回的 state 里，
所以**逐字节相同的 state** 已经蕴含这三个字段逐位相同——脚本两者都查。

| 控制 | 内容 | 结果 |
|---|---|---|
| A | 血统与唯一改动 | 候选以根字节、以装配字节为前缀；Linear/Attention/Q/K/V 字节码相同；影子 `_agr1_train` 确为生效的那份 |
| B | 端到端等价 | 六层真实数据全量校准 **q/k/v state 逐字节相同**、审计字段相等 |
| C | 调用计数 | `_agr1_scale_loss_grad` 204→198、`_a2_apply_group_rotation` 476→470；**训练段 192 未动**，尾部 12→6 |
| D | 覆盖 | 接受（层 0/22）与拒绝（层 1/5/8/15）真实出现；M=I、ineligible、异常回退各一致 |
| E | 确定性 | 装配自比逐字节相同 |

### 一处过程修正（如实记录）

M=I 探针第一次失败，报 `_a2_apply_group_rotation` 差 264 而不是 6。原因**在我的补丁设计**：
替换函数闭包捕获的是 assembly 的原始校准函数，于是候选模块的计数漏掉了校准内部那 266 次调用，
**输出仍然正确、只有按模块的计数会暴露它**。改为"每个模块拿自己的原函数"后，实测归因显示
`_agr1_train` **内部** 210→204（差 6 ✓）、**外部** 266/266 相同。这个错误已写进 `verify.py` 的注释。

## 六 shard 本地结果（`pair_sixshard.py` / `paired_sixshard.json`）

`--attention-only` 六 shard，基线 **v236（同父 A-GR1 旧实现）**，逐例重算 `candidate.gain − baseline.gain`：

| shard | 0 | 1 | 2 | 3 | 4 | 5 | 合计 |
|---|---:|---:|---:|---:|---:|---:|---:|
| n | 12 | 12 | 12 | 12 | 12 | 12 | **72** |
| Δ mean | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | **+0.000000** |
| +/-/0 | 0/0/12 | 0/0/12 | 0/0/12 | 0/0/12 | 0/0/12 | 0/0/12 | **0/0/72** |

**处处为零是预期读数**：基线就是要复现的实现，非零才是缺陷。两侧 `source_sha256` 已核对
（候选 `55103e8b…` 跨 shard 一致、基线 `3319fc35…` 即 v236）。

### `stopped_early: true` —— 显式标注，不是截断

停止检查在 shard 结果 append **之后**（`evaluator/eval_system.py:487` 与 `:509`），等零候选使计数器
**在最后一个被请求的 shard** 上触顶。`results` 6 条、全 `ok`、各 12 例、合计 **72**、`analysis-*.json` 6 份，
**无截断**；`verification.json → stopped_early_flag` 记了独立覆盖核验。

## 对正式父的实际输出变化

与 v236 在 72 例上逐例为零，故 A-GR1 相对 v231 根的变化（等权 `+0.003845`，21/3/48；层 15 `+0.017449`、
层 22 `+0.005618`）原样继承。**本卡不产出新的机制证据。**

## 时间（`timing.py` / `timing.json`）：本机测不出，且比值单独看会骗人

三臂配对（assembly / candidate / **同字节 sham**），15 轮，轮转顺序，CUDA event 无 profiler。分两段：

| 段 | 层 | assembly 中位 | 候选效应 | 占父 | null(sham) | \|effect\|/\|null\| | 候选更快轮数 |
|---|---|---:|---:|---:|---:|---:|---:|
| 整校准 | 0 | 6005.435 ms | +17.948 ms | +0.299% | +31.183 ms | **0.58×** | 10/15 |
| `_agr1_train` | 0 | 1304.192 ms | +1.104 ms | +0.085% | −3.440 ms | **0.32×** | 11/15 |
| 整校准 | 22 | 6008.985 ms | +19.981 ms | +0.333% | +8.439 ms | 2.37× | **9/15** |
| `_agr1_train` | 22 | 1328.064 ms | +3.995 ms | +0.301% | +0.201 ms | **19.83×** | **9/15** |

**layer22/`_agr1_train` 那行的 19.83× 不能当作效应证据**：比值大只是因为空对照的中位恰好贴近零
（+0.201 ms），而**方向一致性只有 9/15，是抛硬币**。四行合起来读：层 0 两段都在 null 之内，
层 22 两段虽过比值线但轮数不过线。**结论是本机分辨不出**，与"省 6 次调用"的计数证据并不矛盾——
计数证明工作量确实少了，墙钟证明这点工作量在这台机器上测不出来。

## 边界（这份结果不主张什么）

- **量级已测定不足，归档不改变这个判断。** 逐位等价基线 v236 在同一根上**官方 TIMEOUT**
  （[超时记录](../../logs/execution/2026-09-10-v236-agr1-on-v231-official-timeout.md)），
  停滞诊断把整个 Attention 去重方向的收益测为**约 0.2 s**、而该机制需要**约 12 s** 才可能进 300 s 门，
  并明确写了 A-CT2 更小、不值得开卡（`docs/attention-stall-analysis-2026-09-10.md` §6.3 第 4 条）。
  本卡按用户指示照计划跑完并归档，**归档不主张它能改善任何官方结局**，也不因此回收诊断结论。
- **不换算官方秒数**：官方 300 s 是唯一时间门，本地毫秒不能换算官方。
- **A-GR1 的官方记录不被继承**：侧隔离 `+29` 与完整包两次 TIMEOUT（v234 / v236）各自保留。
- `api_total` 与校准秒数两侧不可比（缓存 identity 不同），按 AGENTS §5 只记录不解读。
- 等价性覆盖六个真实 attention 层 × 5 窗，加上 M=I / ineligible / 异常回退 / 确定性；
  没有扫其他面板或别的根。

## 复现

```powershell
.venv\Scripts\python.exe workbench/full_solution/attention-act2-train-tail-reuse/audit.py
.venv\Scripts\python.exe workbench/full_solution/attention-act2-train-tail-reuse/build.py
.venv\Scripts\python.exe workbench/full_solution/attention-act2-train-tail-reuse/verify.py
.venv\Scripts\python.exe evaluator/eval.py --solution workbench/full_solution/attention-act2-train-tail-reuse/candidate/solution.py --baseline-solution solutions/20260910_v236_attention-agr1-on-v231_scoreNA_timeNA/solution.py --attention-only --shards 0 --cache artifacts/official_eval/cache/qwen3.5-4b-proxy-v2.pt --calibration-cache-mode auto --algorithm-device cuda --output-dir artifacts/proxy_v3/attention-act2-shard0-20260910
.venv\Scripts\python.exe evaluator/eval.py --solution workbench/full_solution/attention-act2-train-tail-reuse/candidate/solution.py --baseline-solution solutions/20260910_v236_attention-agr1-on-v231_scoreNA_timeNA/solution.py --attention-only --shards 0,1,2,3,4,5 --stop-after-nonpositive 6 --cache artifacts/official_eval/cache/qwen3.5-4b-proxy-v2.pt --calibration-cache-mode auto --algorithm-device cuda --output-dir artifacts/proxy_v3/attention-act2-sixshard-20260910
.venv\Scripts\python.exe workbench/full_solution/attention-act2-train-tail-reuse/pair_sixshard.py
.venv\Scripts\python.exe workbench/full_solution/attention-act2-train-tail-reuse/timing.py
```
