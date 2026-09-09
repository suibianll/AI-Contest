# L-JRB1 A/W 双量化器联合边界（计划卡 3）

活动计划 §5。一轮固定次序的 block coordinate descent：冻结父残差上求 activation 6 个 sign 共享边界 →
整表重新编码一次全部 `X_hat`/`H` → 在新的 `X_hat` 上求 weight 12 个 sign 分离边界（L-RB1 精确二次型）。
不回到 activation、不加第二轮、不拆粒度、不做 role 路由。

- parent SHA256：`56dc805d6e5a3aef896db8021045740292735725d688b48e3d4393e55efcb2bd`
- candidate SHA256：`a70337421909c2e3adf7e6b9ba58408d15e914e0ef2c5b9e76f444fcfda1fa4b`
- 工作目录：`workbench/full_solution/linear-jrb1-joint-aw-boundaries/`
- 运行：`artifacts/proxy_v3/jrb1-shard0/`、`artifacts/proxy_v3/jrb1-sixshard/`

## 实现

`build.py` 用 13 处精确文本补丁把 activation 阈值穿进**两条活动 mantissa 生成路径**及其全部调用点：

| # | 补丁 | 作用 |
|---:|---|---|
| 1–2 | `_solve_exact_hierarchy` 签名 + mantissa | `mantissa = _jrb1_boundary_mantissa(x_abs, local_scale, boundaries)` |
| 3–4 | `_dense_to_hif4` 签名 + 初始编码 mantissa | 同上（`denominator`） |
| 5–7 | `_dense_to_hif4` 内 candidate / edge-extension / L1 三处 `_solve_exact_hierarchy` 调用 | 全部传 `boundaries=boundaries` |
| 8–10 | `_activation_gptq_quantize` 签名 + 单块/逐块 `_dense_to_hif4` 调用 | 全部传 `boundaries=boundaries` |
| 11–12 | `hif4_dynamic_quantize_activation` 的 GPTQ 入口与 dense 入口 | `boundaries=activation_state.get("activation_boundaries")` |
| 13 | v202 block-order wrapper `_combined_dynamic_fast_reordered_gptq` | `boundaries=state.get("activation_boundaries")` |

规则为双边：`c=m (frac<tau)`、`c=m+1 (frac>tau)`、`c=round(u) (frac==tau)`，因此全 0.5 表逐位等于父
（精确半整数仍走 half-to-even）。weight 侧沿用 L-RB1 的部署坐标系后处理（合法五字段，不改 encoder）。
动态 state 只新增一条长度 7 的 CPU float32 表；weight 仍只返回合法五字段。

### 结构性发现：activation 舍入边界只存在于 gram-free 路径

`_dense_to_hif4` / `_solve_exact_hierarchy` 在 `group_gram is not None` 时改由 `_adaround_mantissa`
产生 mantissa，rounding boundary 完全不被查询。`need_activation_gram = _ACTIVATION_QUADRATIC and
in_features <= 3072`，而 4B 面板是 hidden 2560 / FFN 9216：

- `q/k/v/gate/up`（in_features 2560）→ 有 gram → **activation 边界结构上不可达**；
- `o_proj`（4096）与 `down_proj`（9216）→ 无 gram → activation 边界可达。

`probe.py` 直接证明：128 通道（有 gram）时 `_jrb1_boundary_mantissa` 调用 0 次、`_adaround_mantissa`
14 次、合成表改动 0 个硬码；3200 通道时两条路径都被穿过。因此 `_jrb1_joint_fit` 对
`state["gram"] is not None` 的层直接记 `arm=activation-unreachable` 并**整体回退父**，不退化成
weight-only（那是已关闭的 L-RB1/v226 坐标）。

## 验证（`verify.py`，随机张量、CUDA、rows=64/channels=3200）

- 六 API 可独立导入；脱离仓库单文件导入通过。
- `validate_state` / `validate_hif4_params` 通过。
- 规则 control：全 0.5 表与无表都逐位等于父公式；合成 `tau[3]=0.25` 改动数**精确等于**解析预期 19。
- 穿透 control：合成非 0.5 表在 refined 路径改动 5743 个硬码、initial-only 路径 11871 个，证明阈值
  不被 v202 wrapper / GPTQ / 后续 refinement 忽略或覆盖（A-RB1 的教训）。
- 端到端：`arm=joint`，activation 改动 1037 码（真实重编码），weight 改动 10426 码，
  `ΔL_joint=-9.98e-01 < 0`，case 等权真实 MSE `23.69211 -> 23.66731`。

## 4B shard0 可达性诊断（`artifacts/proxy_v3/jrb1-shard0-diag/`）

28 次 calibration（4 block × 7 Linear），逐层读数：

| 层型 | `in_features` | `act_path` | `arm` |
|---|---:|---|---|
| `q/k/v/gate/up` | 2560 | `adaround-gram-unreachable` | `activation-unreachable`（20 次） |
| `o_proj` | 4096 | `rounding-boundary` | `no-activation-proposal`（4 次） |
| `down_proj` | 9216 | `rounding-boundary` | `no-activation-proposal`（4 次） |

可达的 8 次调用并没有出现资格塌陷或求解器未执行：

| block | 层 | eligible/elements | up/down switchable | up/down 单体负代价 | 最小单体代价 |
|---:|---|---:|---:|---:|---:|
| 0 | `o_proj` | 303652/565248 (53.7%) | 157217/146435 | 50347/44492 | `-1.43e-03` |
| 0 | `down_proj` | 857238/1271808 (67.4%) | 446324/410914 | 140135/123910 | `-1.01e-03` |
| 1 | `o_proj` | 268482/565248 (47.5%) | 136852/131630 | 40268/36418 | `-2.07e-03` |
| 1 | `down_proj` | 867141/1271808 (68.2%) | 451119/416022 | 135551/123321 | `-6.67e-04` |
| 2 | `o_proj` | 321164/565248 (56.8%) | 166158/155006 | 43983/39974 | `-1.12e-03` |
| 2 | `down_proj` | 846953/1271808 (66.6%) | 442059/404894 | 132199/118701 | `-2.50e-03` |
| 3 | `o_proj` | 319469/565248 (56.5%) | 164949/154520 | 43094/37986 | `-2.49e-02` |
| 3 | `down_proj` | 849953/1271808 (66.8%) | 443343/406610 | 133447/121359 | `-3.91e-03` |

即 47–68% 的元素进入拟合、每层 30 万元素级可翻码、其中约 30% 单体代价为负，但
`act_proposals=0`。原因是聚合代价在**每一个**桶位都非负：父表下 `total_cost[32] ≡ 0`（上行元素
`frac<0.5`、下行元素 `frac>0.5`，两者都不进 32 号桶），因此 `proposals=0` 等价于
"没有任何共享阈值位置得到负的总代价"——单体负收益被同码类中正收益元素抵消。这不是资格塌陷，
而是 sign-shared 层内共享边界在真实校准数据上不存在可接受方向。

shard0 结果（`artifacts/proxy_v3/jrb1-shard0-diag/candidate/manifest.md`）：

- candidate linear mean `+0.517349` == baseline `+0.517349`（56 cases），`+/-/0 = 0/0/56`；
- `all_outputs_finite=True`、`expected_case_coverage=True`、API 合计 `191.913s`；
- 候选与父**逐位相同**：`w_changed=0`（weight 侧虽有 9–11 个提案，但 activation 提案为 0 时按固定
  次序直接回退父，不退化成 weight-only）。

## 六 shard（`artifacts/proxy_v3/jrb1-sixshard/`）

24 block × 7 Linear = 168 次 calibration（48 次可达 / 120 次 `activation-unreachable`；shard0 的 8 次
可达读数取自 `jrb1-shard0-diag`，本次六 shard 运行复用其缓存，故只打印 shards 1–5 的 140 行）。

| shard | cases | delta_mean | +/-/0 | decision |
|---:|---:|---:|---:|---|
| 0 | 56 | `+0.000000` | `0/0/56` | `reject` |
| 1 | 56 | `+0.000000` | `0/0/56` | `reject` |
| 2 | 56 | `+0.000000` | `0/0/56` | `reject` |
| 3 | 56 | `+0.000000` | `0/0/56` | `reject` |
| 4 | 56 | `+0.000000` | `0/0/56` | `reject` |
| 5 | 56 | `+0.000000` | `0/0/56` | `reject` |

candidate linear mean `+0.529266` == baseline `+0.529266`（336 cases），`all_outputs_finite=True`、
`expected_case_coverage=True`、API `949.120s`（诊断值）、calibration cache hits `1`、未提前停止。

48 次可达调用的结果：

- **47 次 `act_proposals=0`**：聚合代价在每个桶位都非负（单体负代价约 3–14 万元素，但被同码类正代价
  抵消），候选直接回退父；
- **1 次 `act_proposals=1`**（shard2 block1 `o_proj`）：解析表给出 `tau[6]=0.0`（floor-6 元素全部上翻），
  `min_cost=-5.29e-01`，整表重编码改动 **295701** 个硬码，但真实部署 MSE 由 `8.244593e-05` 升到
  `8.349849e-05`（×1.0128），接受门正确回退父五字段。

因此六 shard 候选与父**逐位相同**，`delta_mean=+0.000000`、336 case 全零。

## 结论

按计划 §5.4 / §9 关闭为 `NO_JOINT_REACHABILITY`，不分配版本号、不提交官方：

1. 5/7 层型（`q/k/v/fc_gate/fc_up`，`in_features=2560`）的 activation 舍入边界被自适应 hierarchy
   （`_adaround_mantissa`）完全吸收，联合坐标在这些层上不存在；
2. 2/7 层型（`o_proj` 4096、`down_proj` 9216）可达且求解器确实执行（47–68% 元素入拟合、约 30% 单体
   代价为负），但 48 次可达调用里 47 次没有负聚合桶位、1 次唯一的负桶位在真实重编码下反而恶化
   1.28%，被接受门回退；
3. 净结果：六 shard 输出与根逐位相同。

与 L-RB1/v226、A-RB1 互为第三个同族证据：**sign-shared 的层内共享舍入边界不是"低维、受结构约束"
的安全自由度**——要么不存在可接受方向，要么一次性同向翻转数十万元素后对角曲率模型失效。按计划
不再拆 per-head/per-block/per-sign 表。

