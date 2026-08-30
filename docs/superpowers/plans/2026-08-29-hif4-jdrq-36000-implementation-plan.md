# HiF4 JDRQ 36000+ 算法实施计划

日期：2026-08-29（方案原始日期；状态更新 2026-08-30）
状态：Superseded in root / 保留为研究与回滚参考
目标：跳出局部 HiF4 重建框架，以固定在线 `Q(A)` 下的离线输出蒸馏、完整合法 HiF4 离散残差求解和 Attention 端到端结构化校准，持续逼近官方榜 `36000+`  
实施对象：其他 AI、开发者或后续 Codex 任务可按需复现实验；新实现以根 `solution.py` 和当前状态报告为准
权威设计说明：[`华为算法大赛-HiF4量化赛题完整解析与算法方案.md`](../../../华为算法大赛-HiF4量化赛题完整解析与算法方案.md)  
官方硬约束：六个 API 不变、HiF4 五字段合法、state 合规、不得用 `A@W` 拟合/选择/反推在线逐元素 `Q(A)`、最终总时间严格 `<420s`。当前根实测与本计划的差异见 [`docs/current-solution-status.md`](../../../docs/current-solution-status.md)

---

## 0. 本计划覆盖范围与执行原则

本计划曾取代 `2026-08-29-hif4-linear-22000-optimization-plan.md` 作为算法实施主线；
随后根目录改为 clean Gram-hierarchy 单一路径。本文件及旧计划都保留用于历史追溯，
不直接描述当前提交文件。

### 0.1 固定事实

| 项目 | 当前事实 |
|---|---|
| 官方本地冠军 | C66：`22557 / 217.2s` |
| 当前根版本 | clean Gram-hierarchy：BOAT + cross-fold Weight-HSDQ + Gram-hierarchy Activation-HSDQ + Attention top-4 部署复评；本地实验版本，无新官方分 |
| 外部参考 | `youxilee/hif4`：用户提供 `24153 / 239s` |
| 官方榜上限信号 | 用户确认已有超过 `36000` |
| 官方面板 | 250 Linear + 200 Attention |
| 最终时间上限 | `<420s` |
| 已完成归档 | v084 / C84、v086 / C86（历史快照）；clean 根版本尚未产生新的 vNNN 官方归档 |
| 下一可用归档号 | v087 / 下一次经过完整评测并确认的发布候选 |

若面板每 case 以百分制累加，`22557 -> 36000` 需要把当前剩余 MSE 再降低约 60%。因此本计划不把 offset、coverage、固定 headroom 等千分位微调作为主线。

### 0.2 唯一主算法链

```text
C66 官方冠军父版本
  -> D0 JDRQ ceiling 与量化鸿沟诊断
  -> C72 结构化连续蒸馏目标
  -> C73 输出残差 mantissa 坐标下降
  -> C74 完整 HiF4 hierarchy 离散求解
  -> C75 变换/激活质量升级后重新 JDRQ
  -> C76 Attention 两阶段结构化 Q/K
  -> C77 Q/K/V 策略级交替
  -> C78 精度—时间压缩与发布候选
```

候选号是建议编号。若仓库在执行前已有新归档，顺延编号，但保持阶段名称和唯一机制不变。

### 0.3 不过度防御的硬规则

开发过程中只有以下条件是硬失败：

1. HiF4 参数非法；
2. API、shape、dtype、device 或 state 不合规；
3. 产生 NaN/Inf、运行异常或不可复现；
4. `A@W` 派生量进入 `activation_state`，或在线路径用输出反推 `Q(A)`；
5. 发布候选官方路径总时间 `>=420s`；
6. 明显的数据泄漏、模型名/官方分数/测试标签硬编码。

以下项目**不是硬失败**：

- 某一个 calibration fold 轻微回退；
- 某一个非主模型回退；
- 某个层或算子没有提升；
- 开发阶段运行超过 420 秒；
- 改进没有达到任意预设百分比；
- 第一次实现没有超过当前冠军。

所有非硬失败用连续指标记录。不得因为一次负实验否定整个算法族。

### 0.4 允许停止一个方向的证据

只有同时满足下列条件，才可停止一个机制方向：

1. 实现通过独立公式/单元测试，排除代码错误；
2. 已测它的连续松弛上限或 oracle 上限，而不只是一个离散实例；
3. 至少尝试了一个低自由度版本和一个与 HiF4 层级对齐的版本；
4. 在 Qwen 主模型及至少一个结构不同的模型上都没有可复现信号；
5. 结果日志准确说明“哪个实现失败”，不写“整个理论方向无效”。

若上限高、离散实现低，结论必须是“求解器未兑现上限”，不能停止方向。

### 0.5 父版本恢复协议

正式 C72 实现从官方冠军 C66 开始：

```text
solutions/20260829_v066_c66-activation-ratio100_scoreNA_timeNA/solution.py
```

执行 AI 不得直接覆盖一个含未归档修改的根文件。顺序必须是：

1. `git status --short` 检查用户修改；
2. 计算根 `solution.py` 与 C66、C69 归档的 SHA256；
3. 确认当前根已对应某个不可变归档；若没有，先归档当前根；
4. 再把 C66 快照恢复到根 `solution.py`；
5. 运行语法、release、compliance 和一层 smoke test；
6. 将恢复后的 SHA 写入 JDRQ execution log；
7. C69 只作为评测对照，不把其 CAT/Gram8/product selector 自动带入新父版本。

若 D0 需要诊断 C69，可加载 C69 归档为独立候选；不要因此改变 C72 的 C66 父版本。

---

## 1. 合规数据流

### 1.1 Linear 数据流

```text
W, calibration A
  │
  ├─ 构造 activation_state
  ├─ 冻结 activation_state
  ├─ 调真实在线函数得到 Z = Q(A)
  ├─ 计算教师 Y = A @ W.T
  ├─ 只优化静态 weight_params
  └─ 返回 frozen activation_state + optimized weight_params
```

`A@W` 可以用于：

- 计算教师输出 `Y`；
- 优化和选择静态 `Q(W)`；
- 选择 ridge、结构化蒸馏和离散求解超参；
- 评估完整 Linear 管道，但不得用它改变已冻结的在线逐元素激活输出。

为了最清晰地通过代码审核，本计划默认：

1. `activation_state` 在 JDRQ 前完成并冻结；
2. JDRQ 前后 state 的递归内容完全一致；
3. `Y`、`R`、`Z`、Gram 和候选分数均为校准期局部变量；
4. JDRQ 只返回新的合法 `weight_params`。

### 1.2 改善 Q(A) 的合法路线

若需要改变 `Z=Q(A)`，使用不依赖教师输出反推激活元素的路线：

- 源 NVFP4 四个 16-block scale 的 source-aware proposal；
- `W^T W` 导出的 activation metric；
- SmoothQuant、permutation、structured transform 的 operand-local 目标；
- 固定统计策略和最终在线量化器。

改变 activation_state 后，必须重新运行 JDRQ；旧权重不能与新激活直接配对。

### 1.3 Attention 数据流

Attention 没有 Linear 的 `A@W -> Q(A)` 问题。校准阶段允许用真实
`softmax(QK.T/sqrt(d))V` 选择固定、可部署的 q/k/v state，但不得记忆测试答案或在在线阶段访问其他当前张量的输出。

---

## 2. 文件级实施地图

### 2.1 允许修改

| 文件 | 用途 |
|---|---|
| `solution.py` | 唯一活跃算法；实现 JDRQ 与 Attention 新路径 |
| `evaluator/jdrq_diagnostics.py` | 新增 ceiling、量化鸿沟和 fold 诊断 CLI |
| `evaluator/linear_error_decomposition.py` | 扩展 W4A4/teacher/student 分解 |
| `tests/test_jdrq.py` | ridge/structured regression、残差增量、合法参数和单调性测试 |
| `tests/test_linear_compliance_guard.py` | state 冻结和 A@W 数据流回归 |
| `tests/test_release_candidate.py` | 最终 API、合法性和确定性 |
| `logs/execution/2026-08-29-jdrq-execution.md` | 持续实验账本 |
| `solutions/...` | 每个候选的不可变归档 |

### 2.2 不允许直接修改

- `solutions/` 中已有历史 `solution.py`；
- 历史官方结果 JSON；
- 冻结 real-model cache；
- evaluator 中标准 HiF4 codec，除非发现独立可复现的协议错误。

### 2.3 建议新增的 solution.py 常量

先加入但默认关闭，逐阶段启用：

```python
_JDRQ_ENABLED = False
_JDRQ_MAX_CALIB_ROWS = 256
_JDRQ_FOLDS = 4
_JDRQ_FOLD_CHUNK = 8
_JDRQ_LAMBDA_RATIOS = (0.03, 0.1, 0.3, 1.0)
_JDRQ_ETAS = (0.25, 0.5, 0.75, 1.0)
_JDRQ_STRUCTURED_BLOCKS = (1, 4, 8, 16, 64)
_JDRQ_ROW_CHUNK = 128
_JDRQ_ACTIVE_BLOCK_RATIO = 0.20
_JDRQ_SWEEPS = 2
_JDRQ_SCALE_BEAM = 4
_JDRQ_MANTISSA_COORDINATE = False
_JDRQ_HIERARCHY_TOGGLE = False
```

这些值是首轮计算预算，不是永久最优参数，也不是晋级门槛。实验发现信号后可以扩大搜索，再压缩时间。

---

## 3. Phase D0：JDRQ ceiling 与量化鸿沟诊断

目标：在修改正式算法前回答两个问题：

1. 固定当前 `Q(A)`，连续静态权重最多还能降低多少测试误差？
2. 现有 HiF4 量化器能兑现多少连续收益？

### Task D0.1：新增诊断 CLI

创建 `evaluator/jdrq_diagnostics.py`，支持：

```powershell
.\.venv\Scripts\python -u evaluator\jdrq_diagnostics.py `
  --models gpt2-small opt-125m qwen2.5-0.5b `
  --roles q k v o fc proj `
  --layers 0 1 2 `
  --solution solution.py `
  --cache-mode read `
  --output artifacts\jdrq\d0-ceiling.json `
  --report logs\evaluations\d0-ceiling.md
```

CLI 复用 `real_model_suite.py` 的 cache loader、NVFP4 encoder、独立 HiF4 decoder 和 solution loader，不复制评测协议。

### Task D0.2：输出七项指标

每个 model/layer/role/fold 输出：

| 字段 | 计算 |
|---|---|
| `reference_zero` | `MSE(XW.T, XW.T)` |
| `weight_only` | `MSE(XW.T, XQW.T)` |
| `activation_only` | `MSE(XW.T, QXW.T)` |
| `current_w4a4` | `MSE(XW.T, QXQW.T)` |
| `continuous_target` | `MSE(XW.T, QXW*.T)` |
| `requantized_target` | `MSE(XW.T, QXQ_HiF4(W*).T)` |
| `parent_normalized_ratio` | candidate / current |

额外输出：

- train 与 holdout 分开；
- mean、median、worst quartile CVaR；
- ridge λ、η、target family；
- 运行时间和矩阵 shape；
- `activation_state` fingerprint。

### Task D0.3：fold 定义

只有两条 calibration sequence 时，不按 sequence 做 2-fold。把每条序列切成长度 8 的连续 chunk，再按

```python
fold_id = (global_row // 8) % 4
```

分成 4 个交错 fold。三 fold 拟合、一 fold验证，轮换四次。

### Task D0.4：连续目标族

必须比较：

1. 以变换后的精确 `Wt` 为中心的 full-dual ridge；
2. 以父版本解码 `QW_parent` 为中心的 full-dual ridge；
3. diagonal `X_t≈ZC`；
4. block-4/8/16/64 `X_t≈ZC`；
5. structured + 小 rank residual，可在后续加入。

full-dual：

$$R_0=Y-ZW_0^T$$

$$\Delta W^T=Z^T(ZZ^T+\lambda I)^{-1}R_0$$

$$W_*=W_0+\eta\Delta W.$$

structured：

$$C_b=(Z_b^TZ_b+\lambda I)^{-1}Z_b^TX_{t,b}$$

$$W_*=W_tC^T.$$

### Task D0.5：数学测试

新增 `tests/test_jdrq.py`（后续若拆分文件，保持同一断言集合）：

1. 小矩阵上 dual 解与 primal 解一致；
2. `Z=X` 时 `W*` 收缩回 `W0`；
3. structured identity 映射不改变 W；
4. block mapping 的乘法方向验证：`X≈ZC -> W*=WC.T`；
5. λ 增大时 correction norm 非增；
6. 训练 loss 不高于中心解；
7. fold 无行重叠；
8. 所有输出 finite。

### D0 完成条件

D0 是诊断，不以“必须提升某百分比”为完成条件。只要：

- 三模型数据成功输出；
- 数学测试通过；
- 能明确区分连续 ceiling 与 requantized 结果；
- 报告能指出每类算子的上限和量化鸿沟；

即可进入 C72。

已知快速诊断锚点：GPT-2 small layer-0 q 连续测试误差比约 `0.799～0.817`，proj 约 `0.956～0.970`；直接重新 HiF4 量化后约 `0.992～1.000`。新 CLI 应能在相同配置复现数量级。

---

## 4. C72：结构化蒸馏候选池

目标：实现可审核、低自由度、跨 fold 的 `W*` 候选生成，并接入当前合法量化器；本阶段用于建立正式代码骨架和选择 target family，不期待直接跨越大分差。

### Task C72.1：新增 helper

在 `solution.py` 中增加：

```python
def _jdrq_sample_calibration(...): ...
def _jdrq_fold_ids(rows: int, folds: int, chunk: int, device): ...
def _jdrq_quantized_calibration(calib_pairs, activation_state): ...
def _jdrq_dual_target(z, y, center, lambda_ratio, eta): ...
def _jdrq_structured_target(xt, z, wt, block_size, lambda_ratio): ...
def _jdrq_candidate_score(y_folds, z_folds, qweight): ...
def _jdrq_select_target(...): ...
```

helper 要求：

- 全部 `@torch.no_grad()`；
- FP32 求解；
- state 只读；
- Cholesky 失败时增加 damping，不抛弃整个层；
- correction 做 per-row norm clip；
- 返回 Tensor 或 Python 标量，不返回教师输出到 state。

### Task C72.2：接入点

在 `hif4_calibration_and_quantize_weight` 中：

1. 完成当前 transform 和父权重量化；
2. 完整构造 `activation_state`；
3. 对 state 做递归 fingerprint；
4. 调用 `_jdrq_select_target`；
5. 用现有 `_dense_to_hif4` 量化 target；
6. 比较 parent、structured、dual 候选的 fold robust loss；
7. 返回 winner；
8. 再次 fingerprint，必须一致。

不要让 JDRQ 重算或修改：

- smooth scale；
- permutation；
- CAT transform；
- activation Gram；
- activation refine ratio。

### Task C72.3：候选排序目标

每 fold：

$$r_f=\frac{MSE(Y_f,Z_fQW^T)}{MSE_{STD,f}+\epsilon}.$$

robust 排序：

$$L=0.75\,mean(r_f)+0.25\,CVaR_{25\%}(r_f).$$

这只是连续排序目标，不设“每 fold 必须改善”或固定 min-gain。parent 始终作为候选，因此校准选择本身有自然回退。

### Task C72.4：候选预算

第一版：

- block size `{1,4,8,16,64}`；
- λ ratio `{0.03,0.1,0.3,1.0}`；
- full-dual η `{0.25,0.5,0.75,1.0}`；
- 两个 center：`Wt`、`QW_parent`；
- 先用 64 个输出 row 快筛；
- top-3 target 在完整输出 row 上量化与评分。

若开发时间超过 420 秒，不在本阶段立即缩小网格；先记录精度信号，C78 再压缩。

### Task C72.5：测试

新增/更新：

- state fingerprint before/after 一致；
- `_JDRQ_ENABLED=False` 时与 parent 位级一致；
- target 候选非法时只回退该候选；
- parent 候选永远存在；
- 同输入两次确定性一致；
- release/compliance tests 通过。

### C72 结果解释

- 连续 target 好、合法量化后不变：进入 C73，说明离散求解器是瓶颈；
- structured target 比 full-dual 泛化好：C73 以 structured 为中心；
- full-dual 某些算子好：保留按层自适应选择，不做角色名硬编码；
- 某模型回退：记录 fold/role，不用一个模型否决整个阶段。

---

## 5. C73：输出残差 mantissa 坐标下降

目标：不再把 `W*` 简单舍入，而是从父合法 HiF4 参数出发，用最终输出残差决定 signed mantissa 的移动。

### 5.1 精确目标

固定 `Z=Q(A)`，当前合法权重 `QW`：

$$R=Y-ZQW^T.$$

对 input 64-block `b`：

$$G_b=Z_b^TZ_b,\qquad A_b=Z_b^TR.$$

候选从旧块变为新块，定义：

$$\delta=QW_{old,b}-QW_{new,b}.$$

则：

$$\Delta L=2A_b^T\delta+\delta^TG_b\delta.$$

接受后：

$$R\leftarrow R+Z_b\delta^T.$$

### Task C73.1：残差上下文

新增：

```python
def _jdrq_build_residual(y, z, params): ...
def _jdrq_block_statistics(z): ...
def _jdrq_predict_change(a_block, gram_block, delta): ...
def _jdrq_apply_delta(residual, z_block, delta): ...
```

测试精确公式：随机小矩阵上，预测 Δ 与重算 loss 差值相对误差 `<1e-4`。

### Task C73.2：signed code 表示

将当前 block 解码成：

```text
signed_code = sign * round(mant * 4)  # [-7, 7]
denominator = scale_factor * scale_lv2 * scale_lv3
q = denominator * signed_code / 4
```

对每个坐标候选：

- `c-1`；
- `c+1`；
- `0`；
- 仅当当前幅值很小且梯度支持时尝试符号翻转。

所有 code clamp 到 `[-7,7]`。mant 为 0 时 sign 强制为 0。

### Task C73.3：候选优先级

不要全量逐元素 Python 循环。先计算一阶近似 gain：

$$gain_i\approx -2g_i\Delta q_i-H_{ii}(\Delta q_i)^2.$$

对每个 row-block 只精确评估 top 坐标。首版：

- row chunk 128；
- 每 block top-8 坐标；
- 每坐标最多 3 个新 code；
- active row-block 由预计 gain 排序，先处理 top 20%；
- 两次 Gauss-Seidel sweep。

20% 是首轮时间预算，不是收益阈值。若 top 20% 仍有大量正 gain，下一实验扩大 coverage。

### Task C73.4：候选中心的用途

`W*` 不直接决定最终 code，只用于：

- 给坐标移动方向先验；
- 给活跃 row-block 排序；
- 在 `c-1/c+1` 打平时选择更接近 target 的候选。

最终接受仍以精确输出 `ΔL` 为准。

### Task C73.5：单调性与 holdout

- 训练残差每个接受步骤必须非增；
- 不要求 holdout 每步非增；
- sweep 完成后才计算 holdout robust loss；
- parent、C72、C73 三者作为最终候选选择；
- 不设置每 fold 否决。

### C73 测试

1. `ΔL` 公式；
2. 训练 loss 单调；
3. sign/mant 合法；
4. zero sign 归零；
5. disabled 位级一致；
6. activation_state 不变；
7. 一行一块穷举小例与坐标选择一致。

---

## 6. C74：完整 HiF4 hierarchy 离散残差求解

目标：让 E6M2、E1_8、E1_16 和 mantissa 都能响应输出梯度，超过外部 v2.6 的 scale-only joint refine。

### Task C74.1：E1_16 leaf toggle

对每个 4-element leaf：

1. 保持顶层 scale 和 lv2；
2. 尝试 lv3 `1↔2`；
3. 对 target `W*` 生成最近 signed code；
4. 加入当前 code、C73 code 和 `±1` 邻居；
5. 用完整 64×64 `G_b` 精确计算 Δ；
6. 每 leaf 保留 top-2 候选。

### Task C74.2：E1_8 branch toggle

对每个 8-element branch：

- 尝试 lv2 `1↔2`；
- 两个 lv3 组合共 4 种；
- 每个 leaf 使用上一步 top-2；
- beam 合并，不穷举全部 mantissa 组合；
- branch beam 首版宽度 4。

### Task C74.3：E6M2 scale beam

scale 候选来自：

1. 当前 parent code；
2. `W*` amax 对应 code；
3. 当前 code 邻域 `[-4,+6]`；
4. C72/C73 winner code；
5. 如 beam 边界仍胜，再向该方向扩 2 档。

每个 scale 运行 branch/leaf solver，按精确 Δ 保留 top-4。

### Task C74.4：Gauss-Seidel

每个 block 接受 winner 后立即更新 `R`。首版两 sweep：

1. sweep 1 按预计 gain 降序；
2. sweep 2 只回访发生改变的 block 和与其相关性最高的邻块；
3. 若 sweep 2 仍有大量改善，记录为后续扩大 sweep 的信号，不因当前预算停止方向。

### Task C74.5：向量化与内存

- candidate 维批量；
- row chunk 64/128；
- 不物化 `[all_rows, all_blocks, all_scales, 64,64]`；
- `G_b` 按 block 缓存；
- `A_b=Z_b.T@R` 在 block 更新前计算；
- candidate weight block 使用 FP32，pack 时投影合法字段。

### C74 对照组

正式报告必须同时跑：

```text
parent C66
external-style scale-only joint
C72 target requantization
C73 mantissa coordinate
C74 full hierarchy
```

这样才能知道收益来自连续目标、mantissa 还是 hierarchy。

---

## 7. C75：改变 Z 后重新运行 JDRQ

固定当前 `Q(A)` 的 ceiling 不足以解释 36000。C75 的目标是合法改善在线激活量化，使后续 JDRQ 获得更高上限。

### Task C75.1：source-aware proposal

一个 HiF4 64-group 对应四个 NVFP4 16-block。新增动态候选：

- 从四个 E4M3 scale 的 log2 中位/最大/分位生成 E6M2 code；
- 按 16-block scale 比例初始化 lv2/lv3；
- 加入现有 amax code 和 offset code；
- 只改变候选生成，不改变合法 solver。

验证：所有旧候选仍在池中，source-aware 只是增加 proposal，不构造硬替换。

### Task C75.2：activation gram64

从与当前激活配对的静态权重构造：

$$H_W=W^TW.$$

按 64-group 保存：

```python
gram64: Tensor[K // 64, 64, 64]
```

先用官方 self-check 验证一个大 Tensor 是否只计一个 state 节点。若通过：

- 普通块走当前快解；
- hard block 使用 full64 beam-2；
- 一轮 mantissa coordinate；
- hard block 按实际二次损失排序。

若 state/时间不通过，降级为：

- diagonal + rank-8；或
- 8 个 8×8 block diagonal。

不要因一种 state 表示失败而停止 full64 metric 方向。

### Task C75.3：变换候选

按 operand-local 目标生成和选择：

- identity；
- 当前 Smooth/P/CAT；
- H4/H8/H16/H32/H64；
- 两层 sparse butterfly 只在 down-projection/高 ceiling 层进入第二阶段。

每个 state winner 固定后重新生成 Z，并跑 D0 ceiling。先比较 ceiling，再决定是否进入完整 C74 JDRQ。不得用旧 QW 评价新 state。

### Task C75.4：successive halving

1. operand-local 快筛所有变换；
2. top-4 计算连续 JDRQ ceiling；
3. top-2 跑 C73；
4. winner 跑 C74 full hierarchy；
5. parent 始终保留。

这里的 top-k 是计算预算，不是固定收益门槛。

---

## 8. C76：Attention 两阶段结构化 Q/K

目标：以真实最终 Attention 输出选择结构化 Q/K 状态，扩大当前单层 Smooth/Permutation/Hadamard 的自由度。

### 8.1 严格等价变换

每个 KV head：

$$Q'=QT,\qquad K'=KT^{-T}.$$

GQA 中对应同一 KV head 的所有 Q head 共享 T。

结构：

```text
T = D1 · P · B1 · D2 · B2
```

- `D1/D2`：对角 reciprocal scale；
- `P`：head_dim 内排列；
- `B1/B2`：signed butterfly/Hadamard；
- 条件数限制只用于数值稳定，不设收益门槛。

### Task C76.1：统一最终量化器 scorer

重构 `_attention_candidate_metrics`：

1. 候选变换 Q/K；
2. 使用最终 `_nvfp4_to_hif4` 路径量化 Q/K；
3. V 第一阶段保持精确；
4. 计算 causal 和 non-causal 输出；
5. 按 fold `0.75 mean + 0.25 CVaR` 排序；
6. 不再用逐 Q/K reconstruction 做最终 gate。

### Task C76.2：阶段 1

枚举：

- D 候选；
- P 候选；
- B1 size `{4,8,16,32,64}`；
- seed/角度小池；
- K center policy `{none, midpoint, mean, trimmed_mean}`。

用轻量最终量化器筛 top-4。

### Task C76.3：阶段 2

对 top-4：

- 加 D2/B2；
- 使用完整动态 refine；
- V 使用当前真实量化路径；
- 完整 causal/non-causal fold 复核；
- parent A1 state 作为候选。

不因一个 fold 轻微回退否决；最终按 robust loss 排序。

---

## 9. C77：Q/K/V 策略级交替

目标：在不逐测试样本反推元素的前提下，学习低自由度、可泛化的动态量化策略。

### Task C77.1：V policy

固定 C76 的 Q/K state，用真实 `Qh/Kh` 搜索：

- V offset 子集；
- per-head headroom；
- hard-block refine ratio；
- causal position bucket 的预算；
- E[A²] importance 的 shrink 系数。

policy 是固定 state 参数。最终以真实 Attention 输出选型。

### Task C77.2：回访 Q/K

固定 V winner，重新搜索：

- reciprocal Q/K balance；
- 第二层 butterfly 的少量角度；
- K center；
- 轻微 temperature 候选。

temperature 若改变未量化 logits，必须：

- 低自由度；
- 跨 fold；
- parent temperature=1 保留；
- 不按测试样本动态选择。

### Task C77.3：只做一轮交替

第一版：

```text
QK state -> V policy -> QK revisit -> final
```

若一轮后仍有显著校准下降且 holdout 方向稳定，再开第二轮。不要预先无限迭代，也不要因为首轮不大就停止整个策略方向。

---

## 10. C78：时间压缩与发布候选

C78 之前允许诊断版本超过 420 秒。只有确认有效机制后才压缩。

### 10.1 精度—时间 Pareto

记录每个可调预算：

- calibration rows；
- target family 数；
- λ/η 数；
- active row-block 比例；
- scale beam；
- sweeps；
- Attention candidate 数；
- online hard-block 比例。

从最高精度版本逐项降低预算，保留完整 Pareto，不设置旧 300 秒内部硬门。

### 10.2 优先优化顺序

1. 缓存 `Y`、`Z`、`G_b`；
2. candidate/row/offset 向量化；
3. successive halving；
4. 只回访 changed block；
5. row chunk 调优；
6. 缩小无贡献 candidate family；
7. 最后才降低 active ratio 或 beam。

不要先砍搜索空间再证明算法无效。

### 10.3 发布检查

```powershell
git diff --check
.\.venv\Scripts\python -m py_compile solution.py evaluator\jdrq_diagnostics.py
.\.venv\Scripts\python -m pytest -q tests\test_jdrq.py
.\.venv\Scripts\python -m pytest -q tests\test_linear_compliance_guard.py tests\test_release_candidate.py
.\.venv\Scripts\python -m pytest -q
```

完整 Qwen panel：

```powershell
.\.venv\Scripts\python -u evaluator\real_model_suite.py `
  --models qwen2.5-0.5b `
  --candidates c39 c41b c47b c66 `
  --solution solution.py --candidate-name active-jdrq `
  --panel-profile qwen-official --primary-model qwen2.5-0.5b `
  --device cpu --algorithm-device cpu --cache-mode read `
  --seq 128 --calib 2 --test 4 `
  --output artifacts\real_model_suite\active-jdrq.json `
  --report logs\evaluations\active-jdrq.md
```

多模型 guardrail：

```powershell
.\.venv\Scripts\python -u evaluator\real_model_suite.py `
  --models gpt2-small opt-125m pythia-160m qwen2.5-0.5b `
  --candidates c66 `
  --solution solution.py --candidate-name active-jdrq-multi `
  --panel-profile qwen-official --primary-model qwen2.5-0.5b `
  --device cpu --algorithm-device cpu --cache-mode read `
  --seq 128 --calib 2 --test 4 `
  --output artifacts\real_model_suite\active-jdrq-multi.json `
  --report logs\evaluations\active-jdrq-multi.md
```

发布候选必须 `<420s`。诊断版本超过 420 秒只说明需要 C78 优化，不说明精度机制失败。

---

## 11. 每阶段统一实验记录

每个实验在 `logs/execution/2026-08-29-jdrq-execution.md` 追加：

```markdown
## Cxx / mechanism

- Parent SHA:
- Active SHA:
- Unique change:
- Hypothesis:
- Mathematical upper bound:
- Models/roles/layers:
- Cache and revisions:
- Train fold ratios:
- Holdout fold ratios:
- Qwen panel Linear/Attention/total:
- Guardrail panel:
- Runtime:
- Legality/compliance tests:
- Result: continue / revise / archive
- Exact conclusion:
- Next falsifiable experiment:
```

禁止使用这些模糊结论：

- “没用”；
- “过拟合”；
- “不稳定”；
- “理论无效”。

必须写成：

- 哪个模型、role、layer、fold；
- continuous 还是 quantized；
- train 与 holdout 各是多少；
- target family、λ、η、beam、sweep；
- 是上限不足还是求解器未兑现；
- 下一步如何区分两种解释。

---

## 12. 候选归档规则

每个 C72+ 候选，无论成功、失败或超时，都按现有归档流程保存：

```text
solutions/YYYYMMDD_vNNN_cNN-topic_scoreNA_timeNA/
  solution.py
  result.md
```

根 `solution.py` 是唯一活跃实现。历史归档不修改。

建议主题：

```text
C72 jdrq-structured-targets
C73 jdrq-mantissa-residual
C74 jdrq-full-hierarchy
C75 source-aware-gram64-jdrq
C76 attention-two-stage-transform
C77 attention-qkv-policy-alternating
C78 jdrq-release-pareto
```

归档前检查 SHA、测试命令、cache、模型 revision、panel 分、时间和唯一变化。

---

## 13. 给执行 AI 的逐轮指令

### 每轮开始

1. 读取本计划、根 `README.md`、目标 helper 周边代码；
2. 检查 `git status --short`，保留用户已有修改；
3. 确认当前父版本 SHA 和候选唯一机制；
4. 更新计划状态，只允许一个 implementation step in progress；
5. 不直接修改历史归档。

### 实现过程中

1. 先写数学/不变量测试，再接主路径；
2. 新功能有总开关，关闭时位级不变；
3. 每个 candidate 保留 parent；
4. 使用完整部署量化器做最终评分；
5. proxy 只做筛选；
6. 发现高 ceiling 时优先完善求解器，不增加防御阈值；
7. 发现局部回退时先做误差分解，不按模型名硬门控；
8. 精度机制完成前不急于压到 420 秒。

### 每轮结束

1. 跑本阶段单元测试和合规测试；
2. 跑至少一个 Qwen 主评测和一个异构模型；
3. 保存 JSON/Markdown 报告；
4. 写清上限、兑现率和量化鸿沟；
5. 归档候选；
6. 决定继续扩展、修改求解器或进入下一阶段；
7. 不以任意百分比作为自动停止条件。

---

## 14. 最终验收标准

算法完成不是“某个组件写完”，而是满足：

1. D0 能稳定报告连续 ceiling 和合法量化鸿沟；
2. C74 能让 mantissa、lv3、lv2、E6M2 全部响应精确输出残差；
3. activation_state 在 Linear 教师输出优化前冻结且无泄漏；
4. C75 改善 Q(A) 后自动重新求 Q(W)；
5. Attention 使用最终 Q/K/V 量化器和真实 softmax 输出选型；
6. 多模型报告中没有非法/非 finite/异常；
7. 发布版本总时间 `<420s`；
8. 每个收益都有单机制消融；
9. 官方提交文件 SHA 与归档一致；
10. 若仍未达到 36000，报告能量化剩余差距位于：activation ceiling、HiF4 离散兑现率、Attention 或评测迁移中的哪一项，而不是继续盲调。

本计划的核心执行心法：

> 先测上限，再完善求解器；先让算法兑现精度，再压缩时间。负实验只约束被测实现，不能用过严门控把尚未完成的结构级优化提前终止。

---

## 15. 2026-08-29 首轮实现状态（已落地）

本轮没有覆盖用户已有的根父版本；在当前根 C69 系列代码上增量实现并保留开关，后续归档时再按父版本协议生成不可变快照。

### 已实现

| 阶段 | 代码/测试 | 当前策略 |
|---|---|---|
| C72 | `solution.py::_jdrq_make_target`、`_jdrq_ridge_projection` | 完整 dual / 64-block structured ridge 已实现为研究臂，默认关闭，便于跨模型复核 |
| C73 | `solution.py::_jdrq_refine_mantissa_coordinates` | 固定 `Z=Q(A)` 后，对高杠杆 64-block 做合法 signed-mantissa 坐标下降 |
| C74 | `solution.py::_jdrq_refine_hierarchy_offsets` | 固定 `Z` 后，对 E6M2 offset、lv2、lv3 和 mantissa 做精确输出残差二次变化搜索；默认启用，最多 4 个 block |
| 合规 | `solution.py` 返回路径 | JDRQ 位于 `activation_state` 完成之后，只返回静态五字段 weight；未把 product/residual 写入 state |
| 单测 | `tests/test_jdrq.py` | dual/primal 等价、层级参数合法性、产品损失不增 |

### 实测结果

配置：`amax6 / seq128 / calib2 / test4 / cache_mode=read / algorithm-device=cuda`（GPT-2 单项另有 CPU 对照）。分数为本地 evaluator native total，仅用于相对比较，不映射官方绝对分。

| 模型 | 当前 C74 total | 当前 C74 API | 对照（C66/C69 归档） | 结论 |
|---|---:|---:|---:|---|
| GPT-2 small | 160.572 | 61.88s CUDA | 155.604（根关闭 JDRQ） | Linear 有可复现增益 |
| OPT-125M | 85.581 | 59.56s CUDA | 85.120（C66） | 未出现 C71 式灾难回退 |
| Pythia-160M | 179.059 | 59.71s CUDA | 178.939（C66） | 基本持平、略升 |
| Qwen2.5-0.5B | 356.606 | 163.41s CUDA | 350.152（C66） | 主模型有增益且远低于 420s |

双重 ridge 研究臂在 GPT-2 的 calibration product 上可降低损失，但 hidden validation 反而退化；因此当前默认先用低自由度 C74，不是删除 C72。下一轮应继续做结构化 target 的正则/交叉窗口实验，而不是直接把整个 JDRQ 理论判死。

D0 诊断工具 `evaluator/jdrq_diagnostics.py` 进一步显示：GPT-2 layer-0
down-proj 的 parent/continuous/legal/hierarchy 相对损失为
`0.006693/0.000082/0.003706/0.006481`，Qwen 对应为
`0.002686/0.000012/0.001883/0.002648`。连续上限与合法投影之间仍有很大
离散兑现鸿沟，后续应优先完善结构化离散求解器，而不是把当前小幅 C74
增益误认为已接近上限。

### C75.1/C75.2 增量状态（v073）

已把 C75 的前两项接入活动根版本并归档为 v073：

1. source-aware proposal 从每个 NVFP4 64-group 的四个 E4M3 source scale
   生成 median/q75/max 三类合法 E6M2 候选，保留旧 amax/offset 候选；
2. gram64 从静态变换权重构造每个 64-group 的 `W.T@W`，动态激活只对
   高损失 block 做一次精确 signed-mantissa 格点下降；
3. v073 当时的 gram64 默认按形状启用在 `out_features < in_features` 的
   down-projection，避免 q/k/v/o 方阵路径的迁移噪声，未使用模型名门控；该
   project-only 策略已由 v076 的 all-shape 复测结果 supersede，下面保留作
   历史对照；
4. 激活 state 完成后重新运行 C72--C74，禁止复用旧 Q(W)。

GPT-2 project-only gram64 实测 native total `158.561896`、API
`59.64s CUDA`；它优于同一 source-aware arm 的 `160.597330`，且优于
全形状 gram64 的 `159.690510`。形状门控后的 Qwen/OPT/Pythia 已重新
生成 Q(W)，native total 分别为 `360.658419`、`85.772835`、
`179.473083`，API 为 `168.72s`、`61.14s`、`61.98s`，均未出现灾难
回退。全形状诊断值仍只用于归因，不作为 v073 分数。

对应单元/发布测试：`25 passed, 1 deselected`。当前活动根及 v073
归档 SHA256 为 `A0DCE5D79DA931D5B67FACCBA47226B6C8FCE9FC9551200ED86A3693A1E464DA`。

### C75.3/C75.4/C75.5/C75.6 增量状态（v074）

在 v073 的 source-aware + project-only gram64 基础上，本轮完成并归档了四项
结构级改动：

1. **rowwise JDRQ hierarchy**：每个输出行独立选择高杠杆 64-block，默认
   两块预算；`channels <= 4096` 才启用，4864-wide 层继续用 global 版本，
   这是宽度派生的预算而不是模型名门控。选择器将全窗口 robust loss 与留出
   calibration window 做软混合，避免单折记忆。
2. **wide gram64**：将 gram64 上限提升到 8192；v074 曾只对
   `out_features < in_features` 的 down-projection 构造静态 `W.T@W` 64×64
   block。v076 已完成 all-shape 复测并启用同一合法 hierarchy/mantissa 求解，
   state 仍只保存合法 CPU gram64 张量。
3. **H32/H64 candidate pool**：H32/H64 仅进入 operand-local 的廉价候选池，
   不强制替换父 transform。曾尝试输出乘积 reranker，但运行时 provenance
   审计会将其中间张量误判为残差交叉项，故发布路径关闭该 reranker；这不是
   放弃 A@W 离线校准，JDRQ 的合法静态 `Q(W)` 目标仍然启用。
4. **source-aware + hierarchy interaction**：NVFP4 的四个 E4M3 source
   scale 继续只作为动态激活候选 proposal，之后每次 state 变化都会重新
   运行固定 `Q(A)` 下的 JDRQ，禁止复用旧 `Q(W)`。

四模型本地 evaluator（CUDA、`amax6/seq128/calib2/test4`）的最佳已部署路径
观察为：GPT-2 `158.550907`、OPT `85.736733`、Pythia `179.446007`、
Qwen `361.503707` native total；Qwen panel proxy `242.505358`，API
`179.27s`，均低于官方 `<420s`。这些值是相对回归证据，不是官方分数换算。

对应测试：

```text
python -m py_compile solution.py evaluator/jdrq_diagnostics.py tests/test_jdrq.py
python -m pytest -q tests/test_jdrq.py tests/test_linear_compliance_guard.py \
    tests/test_reference_hif4.py tests/test_release_candidate.py \
    -k "not local_holdout_offsets"       # 48 passed, 1 deselected
```

v074 历史发布旋钮为 project-only wide gram64；当前 v084 旋钮为
source proposal=on、all-shape gram64=on、Gram-64 `ratio=1.0`/
`max_blocks=128`/`sweeps=5`、wide hierarchy offsets=`-4..4`、static JDRQ
offsets=`(-2,-1,1,2,3)`、rowwise max-blocks=2、rowwise width cap=4096、
H32/H64 output reranker=off。`solutions/v074` 已保存
与根文件相同的源码和 SHA256。

### C76.4 增量状态（v075）

在 v074 的 Qwen 主模型上，固定真实输出 scorer 搜索 head-local signed
Hadamard 旋转：H16/H32/H64 × 4 个确定性 sign seed。GQA 中同一 KV head
对应的 Q heads 共享旋转，连续 `QK^T` 严格保持不变；动态 state 只增加
`rotation` 和 `rotation_block`，均为 CPU、finite、可验证字段。为避免 MHA
迁移噪声，首版只在 `q_num_heads != kv_num_heads` 的 GQA 结构启用，不依赖
模型名称。

Qwen 直接复测得到 native total `369.344509`、panel proxy `258.840363`、
Linear `298.383991`、Attention `70.960519`、API `188.06s`；相对 v074
Attention `63.119717` 有明显提升。MHA 的 rotation-gated 路径保持 v074
逐位结果。C76.1 独立 Q/K permutation、C76.2 Fisher importance、C76.3
reciprocal temperature 已完成消融但在 Qwen test fold 回退，保留为关闭的
研究开关，不能与 v075 的 GQA rotation 混用。

对应单元/发布测试仍为 `48 passed, 1 deselected`，v075 根与归档 SHA256 为
`DCA23116D76033A7EB5A04C5CC7EF003A52995905261699B2D06883D4C0BE4A4`。

### C77 all-shape gram64 增量状态（v076）

在 v075 的 GQA rotation 父版本上重新评估了 `gram64` 形状门。关闭
`_ACTIVATION_GRAM64_PROJ_ONLY` 后，所有满足宽度上限的 Linear 都能用静态
`W.T @ W` 64-block metric 驱动动态激活 refinement；没有将 `A@W`、residual
或测试输出写入 state。Qwen native 从 `369.344509` 升至 `372.623675`，panel
从 `258.840363` 升至 `260.060290`，Linear 从 `298.383991` 升至
`301.663157`，Attention 保持 `70.960519`。GPT-2、OPT、Pythia 也分别达到
`159.774232`、`87.248114`、`182.160394`，均高于 v075，API 时间最高
`207.72s`，低于 420s。

叠加 full-width JDRQ 的 C78 交互实验为 Qwen panel `260.050784`，略低于
v076，因此当前继续保留 projection-only JDRQ；GQA group reciprocal scale
和 non-causal A1 重排也已测为回退，均不启用。

v076 根/归档 SHA256：
`C87B61C8A4A9F869A43EFDEECF7734A0A810EA0E5621D51826EC5E56A31ED0E4`。

### C80 full gram64 coverage 增量状态（v080 历史基线）

在 v076 all-shape gram64 的基础上，连续扩大动态 64-block refinement 覆盖
预算：`0.08/8 -> 0.16/16 -> 0.32/32 -> 0.64/64 -> 1.0/128`。每一级
都先在 Qwen 主模型验证，再以 GPT-2、OPT、Pythia 做结构 guardrail；四级
均正向，最终 full coverage 达到 Qwen native `386.903134`、panel
`265.372589`、Linear `315.942615`、Attention `70.960519`、API
`208.70s`。异构模型对应 native 为 `164.221204`、`91.605403`、
`188.695479`，均高于前一级。中间版本分别由 `877db7d`、`07cf5f6`、
`50782a8`、`45179eb` 提交，最终归档为 v080。

该方向说明先前的 8-block 预算是求解器预算瓶颈而非理论上限；目前不再
把固定 coverage 上限作为防御门。state 仍只有静态 CPU gram64 统计和合法
HiF4 字段，未引入在线 `A@W` 或测试输出。

v080 根/归档 SHA256：
`62EC3DB74933986886D01751E5307E58DDC8F4007E56D9A484C239F74AE69813`。

### C84 full gram64 coordinate sweep 增量状态（v084 当前根）

在 v080 的 full-coverage 配置上，保持 `ratio=1.0/max_blocks=128` 不变，
只增加每个 64 维 block 的确定性坐标扫描轮数。sweep=2、3、4、5 依次由
`b702c93`、`6b8925e`、`7e9464a`、`e5fd172` 提交；Qwen 主模型的结果如下：

| sweep | native total | panel total | Linear | Attention | API time |
|---:|---:|---:|---:|---:|---:|
| 1 (v080) | 386.903134 | 265.372589 | 315.942615 | 70.960519 | 208.70s |
| 2 | 390.780409 | 266.815028 | 319.819890 | 70.960519 | 238.30s |
| 3 | 391.684115 | 267.151228 | 320.723597 | 70.960519 | 256.34s |
| 4 | 391.956412 | 267.252529 | 320.995893 | 70.960519 | 285.40s |
| 5 (v084) | **392.055970** | **267.289567** | **321.095451** | 70.960519 | **309.09s** |

GPT-2、OPT、Pythia 在 sweep5 的 native total 分别为 `167.049503`、
`92.848854`、`190.277968`，相应 Linear 分别为 `145.743266`、`73.201252`、
`149.630088`；四模型均比 sweep4 正向。Qwen API 时间距官方 `420s` 仍有
约 `110.91s`，但单轮增益已明显递减，所以当前先停在 sweep5，把剩余时间预算
留给新的 V/QK 策略或输出级离散求解，而不是盲目增加 sweep6。

state 仍只包含静态 CPU Gram-64 统计和合法 HiF4 字段；所有 sweep 都不把
`A@W`、输出残差或测试输出写入在线 activation state。v084 根/归档 SHA256：
`A8A4427DBA95723570FBDEBCDA1E4EDDBF152A3693CC851E30A87368A02CA284`。

### C86 attention block-H final-lattice 增量状态（v086 当前根）

在 v084 的 Qwen GQA rotation 之上，加入共享 Q/K 的 head-local Hadamard
候选。每个 Q head group 与对应 KV head 使用同一 4/8/16 block-Hadamard
和 deterministic sign，因此连续 QK 内积严格保持不变；候选排序改用最终
offset/refinement lattice 的真实 Attention 输出误差，避免“候选量化器”和
“部署量化器”目标错位。state 只保存 block size、seed 和静态 CPU signs，
不保存 A@W、输出残差或测试输出。

Qwen 全层结果为 native `392.064774`、panel `267.307909`、Linear
`321.095451`、Attention `70.969323`、API `313.58s`，相对 v084 panel
`+0.018342`，仍低于 420 秒。异构 guardrail：GPT-2 native `169.829549`
（Attention `24.086283`）、OPT `92.579685`（Attention `19.378433`）、
Pythia `190.239876`（Attention `40.609788`）；OPT 小幅回退、Pythia 近持平，
按 Qwen-primary 规则晋级但继续观察跨模型稳定性。

实现提交为 `2c1cf85`、GQA 符号修复为 `31b99d6`、最终量化器对齐为
`90844fe`；v086 归档 SHA256：
`E7A16D6991DBB70A593FBE87D0C5D1D8FD38F801665354A01FFAF2F0A96F03CD`。

### C85 rejected audit

C85 二次 JDRQ pass 已在 `ff8861f` 中实现并以 `4e9861e` 回退：Qwen
panel `267.262237`、API `313.57s`，低于 v084。直接 `Q(W)^TQ(W)` Gram
虽然局部 proxy 上升，但运行时 provenance 检出 `R_A R_W` 交叉残差，违反
在线 Q(A) 合规边界；W-only Gram 及 blend 均低于 v084，因此不晋级。

### 下一轮唯一优先级

1. 以 v086 为不可变父版本，继续测 C76.2 Fisher/V importance 的
   validation-fold 选择和 C76.4 rotation 的 block/seed 单机制消融；只在
   真实 Attention 输出的跨窗口收益稳定时晋级更多结构；
2. 保持 `A@W` 只服务静态 `Q(W)`，任何候选变换改变后都重建 `Z` 并重跑
   JDRQ，不将输出或 pooled derivative 写入 `activation_state`；
3. 每个机制至少保留一个可回退开关、单测、局部消融和不可变 `vNNN` 快照，
   不以单一模型或单一 fold 的轻微回退停止结构方向；
4. 继续使用 `scoreNA_timeNA` 记录无官方评测的候选，提交前只宣称已实测
   的本地指标和通过的合规/时间门槛。
