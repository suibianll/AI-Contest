# Attention 序列自适应精化计划（v128 余量回收，已归档）

> 状态：**ARCHIVED / CONCLUDED — S1 官方 timeout，per-call 动态族关闭**
>
> 创建：2026-09-03；归档：2026-09-03（同日执行完毕）
>
> 官方父版本：v160，`17532 / 232s`，归档源码 SHA256
> `33B1D061CE6BFCD92659C597BE4830BB9B910E646FF518433DA67B925AE8680D`
>
> 所有候选从 v160 归档源码分支；Linear（v159 GPTQ）与 v158/v160 Attention 既有机制全部
> 冻结；父版本 JSON 固定复用。

## 0. 最终结论（归档时写入）

- **S1 REJECTED / TIMEOUT**：v161（SHA `27EEE471...1848`）本地全漏斗通过（Qwen default
  120 paired `+0.052502`、106+/14−、touch 88.3%；GPT-2 `+0.0678` 同号；D1 本地满足），
  但官方回传 **timeout（`>300s`，无分数）**。按 §5 解释表：时间外推门禁失效，REJECTED，
  不缩 sweeps 重试。证据：`logs/execution/2026-09-03-v161-official-timeout.md`；

- **归因**：官方机（鲲鹏）上动态 per-call 小张量算子成本远超本地 CUDA 外推
  （`0.092s/call` CUDA → 官方耗尽 v160 的 68s 余量）。家族证据链：v138（无 dyn refine）
  官方 `208s` 通过，v128/v129/v130/v131/v161（含 dyn refine）全部官方 timeout——
  **修正 §1.2 的核算：超时元凶不只是校准搜索，动态精化本身在官方硬件上即超预算**；

- Step 0 判读行"v128 ≥ +0.03 且 v138 ≈ 0 → 进入 S1"按本地口径成立（v128 `+0.0636`、
  v138 `−0.014`），但该余量在官方 300s 内无法回收；

- S2 不启动（前置条件"S1 官方正向"不满足）；D1 维持 3/3；P9 无法记录；

- 本计划归档后，本地已知机制族全部闭环（Linear 结构 full64/Householder、Attention
  解析静态族、Attention per-call 动态族）；下一动作是外部材料搜索或用户指定新机制。

## 1. 背景与依据

### 1.1 为什么此前路线关闭

- **Linear 结构封闭（结构性，非工程性）**：本地校准 token（128+512=640）少于输入维度
  （q/k/v/o 896、fc/proj 4864），经验 Gram 秩亏。全宽耦合变换可以把量化误差旋进校准
  协方差零空间使训练损失趋零，但真实分布满秩，误差在官方数据上全部回归——这是
  v134 家族本地 +0.10 / 官方 −165 的精确机制（A3 transform-off 控制已证实）。可转移的
  低自由度 block-64 族已被 v159 推到下限（full64、Householder 六变体全族否定）。

- **Attention 解析静态族饱和**：Matrix-Smooth 2×2 是偏差-方差甜点，4×4 已越界
  （A1a REJECTED）；深层 K 无一致结构病因（A2 诊断方向相反）。

### 1.2 未开采的余量：v128 家族的 per-call 序列自适应

v128 家族（v121–v133 一系）的官方裁决是 **timeout，不是 wrong answer**——其精度从未被
官方否证。本地证据（official-shape-v1 口径）：

| 版本                         | attention mean |         本地 attention 侧成本 | 官方        |
| -------------------------- | -------------: | -----------------------: | --------- |
| v128 全量                    |     **0.8378** | calib 199.8s + dyn 33.5s | timeout   |
| v129 sweep1（搜索缩减）          |         0.8366 | calib 126.4s + dyn 35.1s | timeout   |
| v138（搜索删除 + dyn refine 删除） |         0.7159 |   calib 36.3s + dyn 5.4s | 208s 通过   |
| v86（参照）                    |         0.7197 |                        — | 222.7s 通过 |

关键时间事实（`workbench/v128_timing_compare.py`，legacy-v1 JSON）：

- **超时元凶是校准期候选搜索**：`hif4_calibration_attention` 8.3s/call × 24 = 199.8s
  （160+ 候选 × 代理前向排序）；

- **动态 per-call 精化很便宜**：`dyn_q/k` 0.08s/call（含 3-sweep 坐标精化），
  400 次调用共 32s；v138 删除后降至 0.010s/call；

- v129 证明搜索可砍半而精度几乎不变（−0.0012）；v138 证明全砍后掉 −0.121（搜索残余

  - dyn refine 混合，贡献未分离）。

### 1.3 机制描述（为什么它没有 Linear 的过拟合问题）

v128 的动态 Q/K 路径 = 编译 state 变换 + `_dense_to_hif4` 编码 +
**`_refine_activation`** **3-sweep 有界坐标精化**：

- 精化目标 `error^T · G64 · error`（按 64-channel block），其中 Q 的 `G64` 来自
  **K 的校准块 Gram**、K 的来自 Q 的——交叉算子 Gram 恰是 QK logits 误差对量化扰动
  的二阶 Hessian，即直接优化真实目标而非重建误差；

- 逐调用作用于**当前序列的实际码字**（per-call 自适应）：不存在"校准拟合-测试暴露"
  的分布偏移通道，泛化风险被转换为计算成本；

- 有界（sweeps 固定 3、每坐标闭式 argmin、无候选循环），state 只增 CPU 有限张量
  gram64，与 v128 相同的合法格式（v128 官方回传 timeout ≠ WA 已证明合法性）；

- v100/v107 的 WA 教训是 PAWV dense 路线的 state/数值问题，与本机制无关。

官方预算：v160 官方 232s，余 68s。dyn refine 估算（按官方 attention 动态调用 \~576 次
Q+K、0.08s/call、官方机速比不确定）约 30–50s，**紧但可行**；时间门禁见 §4。

## 2. Step 0：零实现同协议消融（先测奖金大小）

v128/v129/v138 归档的六个 API 与当前评测器接口相同，可直接在 proxy-v2 attention
compact 面板上运行（`--attention-only --compact-panel`，NVFP4 cache 命中）：

```powershell
.venv\Scripts\python.exe -u evaluator\official_eval.py --solution solutions\20260901_v128_fixed-attn-budget_timeout\solution.py --attention-only --compact-panel --cache-mode read --nvfp4-cache-mode auto --capture-device cuda --algorithm-device cuda --baseline-json artifacts\official_eval\a1-parent-v160-attn-compact.json --output artifacts\official_eval\s0-v128-attn-compact.json --report logs\official_eval\s0-v128-attn-compact.md
```

（v129/v138 同理，产物 `s0-v129-*`、`s0-v138-*`。）

### 判读（相对 v160 parent 0.797462）

| 观测                            | 结论                                 | 动作                        |
| ----------------------------- | ---------------------------------- | ------------------------- |
| v128 ≥ +0.03 且 v138 ≈ 0       | 余量真实且属自适应族                         | 进入 S1                     |
| v128 ≥ +0.03 且 v138 也 ≥ +0.02 | 余量主要来自 v128 的静态变换族（非 per-call）     | 跳过 S1，评估 S2'（v128 静态变换移植） |
| v128 < +0.01                  | 余量是 official-shape-v1 协议假象         | 关闭本计划                     |
| 接口/状态 ERROR                   | v128 state 与现行 validate\_state 不兼容 | 记录后按 v128 源码最小适配重试一次      |

同时记录三个版本的 attention 侧 API 时间（compact 调用图下），作为 S1 时间外推基线。
零 API 成本：每版约 1 分钟。

## 3. S1 候选：交叉算子 Gram64 per-call 精化（唯一算法候选）

从 v160 归档分支，`workbench/s1_qk_gram_refine.py`，两处修改：

### 3.1 校准期（`hif4_calibration_attention` 末尾追加）

在 v160 全部既有拟合（A1 终验门、A2、Matrix-Smooth、V importance）完成后：

1. 用**最终部署坐标**的变换链（multiplier → permutation → rotation → block smooth →
   pair\_transform，与 `_attention_state_transform_dense` 一致）重放全部校准 Q/K 样本；
2. 移植 v128 的 `_qk_gram64`（L973–996）：Q 每 head 每 64-block 用同组 K 的块 Gram，
   K 每 head 每 block 用对应 Q 组的块 Gram；ridge 与 v128 相同常量；
3. `q_state["gram64"]` / `k_state["gram64"]` 存 CPU 张量。

不改任何既有 state 键、gate 或候选选择——gram64 是纯追加。

### 3.2 动态期（`hif4_dynamic_quantize_q/k`）

在 `_nvfp4_to_hif4` 产出 params 后，移植 v128 的 `_refine_activation`
（sweeps=3，v128 固定值，不扫描）：对解码码字做 3-sweep 块内坐标下降，最小化
`error^T · gram64 · error`。**V 路径不动**（bit-exact control）。

### 3.3 不变量与纪律

- 连续域变换不变（精化只动离散码字，不碰变换）；

- Gram 在最终部署坐标计算（§2 提交代码约束）；

- 在线路径无候选循环、无矩阵求逆（每坐标闭式 argmin，向量化按 token 维）；

- 无 seed/alpha/offset/threshold/sweeps 搜索；单一机制；一次官方提交。

## 4. 验证漏斗（失败即停）

| 阶段 | 内容                                                                                                                                                                             |        预算 | 停止条件       |
| -- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------: | ---------- |
| A  | 六 API 隔离导入；V bit-exact；单元检查（gram64 对称半正定、refine 单调不增、3-sweep 有界）                                                                                                               | `≤10 min` | 任一失败       |
| B  | Qwen attention compact 4 哨兵 vs v160 parent：mean/median Δ、touch、QK-only/V-only 分解、**attention API total ≤ parent + 15s**（compact 口径）                                            |  `≤5 min` | 负向、V 变化或超时 |
| C  | GPT-2 attention compact 同号；Qwen attention default 120 执行 D1 判别器（touch≥50%、improved>regressed、median≥0；OPA-1 账本预注册，现有 3/3）+ 各长度分组不为负 + **default attention API ≤ parent + 40s** | `≤10 min` | 任一失败       |
| D  | JSON replay、单文件隔离导入、执行日志、SHA                                                                                                                                                   |  `≤5 min` | 证据不完整      |
| E  | 一次官方提交（用户执行）                                                                                                                                                                   |         — | 见 §5       |

时间外推门禁：default 120 case 的 dyn Q/K 增量 ×（官方/本地调用数比，按 v160 官方 232s
反推）+ calib gram 增量 ≤ 60s 官方预算；不满足即 REJECTED（时间），不调 sweeps。

## 5. 结果解释

| 结果              | 结论                                | 后续                                       |
| --------------- | --------------------------------- | ---------------------------------------- |
| S1 通过 D1 且官方正向  | per-call 自适应精化可迁移，D1 增至 4/4       | S2：校准搜索的解析化（闭式 alpha/offset 回归），仍单机制     |
| S1 通过 D1 但官方零/负 | 本地 attention proxy 对 per-call 族失效 | D1 降级记录；关闭本族，不邻域调参                       |
| S1 本地 B/C 失败    | v160 坐标系下 gram 精化无余量或时间不可行        | 记录机制归因；若 Step 0 余量大，转 S2'（v128 静态变换移植评估） |
| 官方 timeout      | 时间外推门禁失效                          | REJECTED，不缩 sweeps 重试                    |
| Step 0 即否定      | v128 余量是协议假象                      | 立即归档本计划                                  |

官方回传同时记录 P9 检验（17816 的 +284 缺口归因），不作为调参输入。

## 6. 执行顺序与停止条件

1. **Step 0**：v128/v129/v138 归档 attention 同协议消融（零实现，\~5 min）；
2. 按判读表决定 S1 或 S2' 或关闭；
3. S1 实现 → 漏斗 A–D → 通过则一次官方提交；
4. 官方结果无论正负只记录；失败不邻域调参（sweeps/ridge/block 不动）；
5. S1/S2/S2' 结束或连续阻塞后归档本计划，同一提交更新状态文档。
