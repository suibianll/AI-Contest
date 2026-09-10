# A-CT2：`_agr1_train` 末尾统计复用的训练尾部消除（v239）

计划卡：[`docs/superpowers/plans/2026-09-10-retained-root-gradient-reuse-plan.md`](../../docs/superpowers/plans/2026-09-10-retained-root-gradient-reuse-plan.md) §4 的 A-CT2 固定实现卡。
归档：[`solutions/20260910_v239_attention-act2-train-tail-reuse_scoreNA_timeNA/`](../../solutions/20260910_v239_attention-act2-train-tail-reuse_scoreNA_timeNA/result.md)。
工作目录：`workbench/full_solution/attention-act2-train-tail-reuse/`。
设备：NVIDIA GeForce RTX 3060 Ti。

本日志记**过程与判定**；逐项证据表在归档的 `result.md`，不在此重复。

## 0. 开工前置与一条必须说明的时序

| 项 | 值 |
|---|---|
| 正式父 | v231 完整根，官方 **18518 / 291 s**，余量 9 s，SHA `ea79a1c1…` / 505762 B |
| 装配来源 | v236 = 该根 + A-GR1，SHA `3319fc35…` / 520955 B（`cmp` 确认前缀逐字节等于根） |
| 对照定位 | v236 是**同父、同算法的实现对照**；不设为晋级父、不继承其官方结果 |

**时序说明（如实记录）**：本卡按计划 §4 的裁决顺序执行——A-CT1 归档（v238）后进入审计、确认后注册固定卡再开发。
开发期间，另一个会话交付了 **Attention 停滞诊断报告**（`docs/attention-stall-analysis-2026-09-10.md`），
其 §6.3 第 4 条结论是"去重方向收益已测定、可以收手……**A-CT2 量级更小，不值得开卡**"；
同一时期 **v236 官方 TIMEOUT** 回传（`logs/execution/2026-09-10-v236-agr1-on-v231-official-timeout.md`）。

即：**本卡的开发请求与"它不值得开卡"的新证据同时到达。** 已把该冲突提交用户裁决，用户明确选择
**"照计划跑完并归档 v239"**。本卡因此完成并归档，但**归档里如实写明量级已测定不足、不主张改善官方结局**，
不回收诊断结论，也不把这次归档当作对该方向的背书。

## 1. 机制与改动点

`_agr1_train` 收尾时，`final_loss` 循环已经为每个 fold 用 `m`（q 角色）与 `p`（k 角色）各算过一次
`_agr1_scale_loss_grad(_a2_apply_group_rotation(fold[i][0], int(fold[i][2]), m|p), fold[i][1])[0]`；
`info` 里的 `agr1_q_scale_ratio2` / `agr1_k_scale_ratio2` 又把同一批调用做了一遍，只取回第一次丢掉的标量。

改动：在 `final_loss` 循环里带上两个按角色分开的标量和，两个 ratio 由这两个和算出，删掉第二次遍历。
**展开成两个角色是必需的**——float 不可变，用 `zip(..., (q_scale_sum, k_scale_sum))` 只会重绑定局部名。

**不改**：32 步训练循环与其梯度、`_agr1_project`、被调的两个辅助函数、gate、窗口/候选数/接受逻辑、
最终 state 编译。

## 2. 审计（先于开发，`audit.py` / `audit.out`）

计划要求"先验证同 dtype、顺序和异常行为，明确可复用的每 fold 标量；只在确认后……再开发"。三问都由 AST 与实测算出：

- **Q1** AST 上按 `zip(fold, (m, p))` 的位置对应：循环的通用调用在迭代 i 就是 ratio 的角色 i 调用（同 fold 项、同矩阵）。
- **Q2** 仪器化真实 `_agr1_train`：每次共 **204** 次 `_agr1_scale_loss_grad`（192 训练 + 6 final_loss + 6 ratio），
  六层实测 **6/6 的 ratio 调用入参逐位相同、返回标量逐位相同**。
- **Q3** 用循环内标量重建两个 ratio，与上报值**精确相等**（六层全 True）。
- **FACTS** 全链 float32；`force_zero` 在全部 5 个归档调用点**均不可达**（该分支跳过 `final_loss` 循环、
  令标量未绑定；若可达则本卡前提不成立）。

结论：`DUPLICATION CONFIRMED`。

## 3. 构建（`build.py` → `build.json`）

候选 = 装配字节（520955 B）+ 追加模块 = 531018 B。两次子串替换（`final_loss` 循环、两个 ratio 条目），
每次替换前断言该块唯一、替换后断言已消失；候选文本必须等于父文本经这两次替换的结果——在唯一性成立时，
这就是"除这两块外什么都没动"。改动 36 行、**−74 B**。

候选 SHA256 `55103e8ba530bf2cc03fc07bf24358ada4bda8d54ccc976468f5cbd985ecdac9`。

## 4. CPU 验证（`verify.py` / `verify.out`，全 PASS）

- **B**：六层真实数据全量校准 **q/k/v state 逐字节相同**、审计字段相等——**比只比那三个 float 更强**，
  因为三个字段就在返回的 state 里。
- **C**：`_agr1_scale_loss_grad` 204→198、`_a2_apply_group_rotation` 476→470；**训练段 192 次未动**，尾部 12→6。
- **D**：接受（层 0/22）与拒绝（层 1/5/8/15）真实出现；M=I、ineligible、异常回退各一致。
- **E**：装配自比逐字节相同。

### 4.1 一处过程修正（我的补丁缺陷，如实记录）

M=I 探针首次失败：`_a2_apply_group_rotation` 差 **264** 而不是 6。根因在**我的补丁设计**——
替换函数闭包捕获的是 assembly 的原始校准函数，于是候选模块的计数漏掉了校准内部那 266 次调用。
**输出仍然正确，只有按模块计数会暴露它。** 改为"每个模块拿自己的原函数"后，实测归因显示
`_agr1_train` 内部 210→204（差 6 ✓）、外部 266/266 相同。该教训写进了 `verify.py` 的注释：
这类错误在输出上不可见，只在按模块计数时可见。

## 5. GPU 评测

父侧用 v236 归档（对照是同父 A-GR1 旧实现）。

- **shard0**：`records=1`、`reasonableness_issues: 0`、12 例 `mean_delta_gain=0.0`（0/0/12）。
- **六 shard**：`{'protocol': 'eval-v3', 'records': 6, 'reasonableness_issues': 0}`。

### 5.1 `stopped_early: true` —— 显式标注，不是截断

同 v237 / v238 的结构性读数：停止检查在 shard 结果 append **之后**，等零候选使计数器在**最后一个被请求的 shard**
上触顶。`results` 6 条、全 `ok`、各 12 例、合计 **72**、`analysis-*.json` 6 份，**无截断**。

### 5.2 逐例配对（`pair_sixshard.py`）

六 shard 全部 12/12 恰好零，合计 **72 例 0/0/72**；两侧 `source_sha256` 已核对
（候选 `55103e8b…` 跨 shard 一致、基线 `3319fc35…` 即 v236）。

### 5.3 对正式父的变化：继承

因与 v236 逐例为零，A-GR1 相对 v231 根的变化（等权 `+0.003845`，21/3/48）原样继承。**本卡不产出新的机制证据。**

## 6. 计时（`timing.py`）：本机测不出，且比值单独看会骗人

三臂配对（assembly / candidate / **同字节 sham**），15 轮，轮转顺序，CUDA event 无 profiler。分两段测：

| 段 | 层 | assembly 中位 | 候选效应 | 占父 | null | \|effect\|/\|null\| | 更快轮数 |
|---|---|---:|---:|---:|---:|---:|---:|
| 整校准 | 0 | 6005.435 ms | +17.948 ms | +0.299% | +31.183 ms | **0.58×** | 10/15 |
| `_agr1_train` | 0 | 1304.192 ms | +1.104 ms | +0.085% | −3.440 ms | **0.32×** | 11/15 |
| 整校准 | 22 | 6008.985 ms | +19.981 ms | +0.333% | +8.439 ms | 2.37× | **9/15** |
| `_agr1_train` | 22 | 1328.064 ms | +3.995 ms | +0.301% | +0.201 ms | **19.83×** | **9/15** |

**layer22/`_agr1_train` 的 19.83× 不是效应证据**：比值大只因空对照中位恰好贴近零（+0.201 ms），
而**方向一致性只有 9/15**。四行合读：层 0 两段都在 null 之内，层 22 两段过比值线但轮数不过线。
**结论：本机分辨不出。**

计数证据与墙钟证据不矛盾：**计数证明工作量确实少了 6 次，墙钟证明这点工作量在这台机器上测不出来。**
两条都照实记，不取好看的那一半。

## 7. 裁决

**LOCAL：与同父 A-GR1 逐位等价（72/72 精确零、六层 state 逐字节相同）、训练尾部少执行 6 次调用、
墙钟本机不可分辨。官方 `unregistered/NA`。**

计划 §4 的 A-CT2 卡按用户指示执行完毕。**归档不主张能改善官方结局**：逐位等价基线 v236 同根官方 TIMEOUT，
停滞诊断把整个去重方向测为约 0.2 s、而机制需要约 12 s 才可能进 300 s 门（`docs/attention-stall-analysis-2026-09-10.md` §6.3）。
是否花费一次官方提交由用户决定；本卡不因此被抬价，也不因此关闭诊断结论。

## 8. 产物

- 工具：`workbench/full_solution/attention-act2-train-tail-reuse/`（`audit.py` / `build.py` / `verify.py` /
  `timing.py` / `pair_sixshard.py` / `archive.py`）；
- 证据：`audit.out`、`build.json`、`verify.out`、`timing.json`、`timing.out`、`paired_sixshard.json`；
- 评测：`artifacts/proxy_v3/attention-act2-shard0-20260910/`、`artifacts/proxy_v3/attention-act2-sixshard-20260910/`；
- 归档：`solutions/20260910_v239_attention-act2-train-tail-reuse_scoreNA_timeNA/`；
- 官方字段未知记 `null`。
