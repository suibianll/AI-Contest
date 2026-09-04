# Attention 解析式宽域实验计划（已归档）

> 状态：**ARCHIVED / CONCLUDED — A1/A2/A3 全部关闭**
>
> 创建：2026-09-03；归档：2026-09-03（同日执行完毕）
>
> 官方父版本：v160，`17532 / 232s`，归档源码 SHA256
> `33B1D061CE6BFCD92659C597BE4830BB9B910E646FF518433DA67B925AE8680D`
>
> 所有候选从 v160 归档源码分支，Linear（v159 GPTQ）与 A1/A2/L1 全部冻结；
> 父版本 JSON 固定复用，不重复运行父版本。

## 0. 最终结论（归档时写入）

- **A1a REJECTED**（阶段 B Qwen compact 哨兵即停）：4×4 组内扩展 paired mean Δ
  `-0.078643`、`0/2/2`、`consistent_regression`；机制归因为每 block 自由度 3→10 在
  少量校准折上拟合方差大，gate 验证折通过但真 holdout 回归；同时间接否定 A1b（自由度
  更高）。证据：`logs/execution/2026-09-03-a1a-matrix-smooth-4x4-rejected.md`；
- **A2 SKIPPED**（第 0 步零 API 诊断未找到一致病因）：K 通道 outlier/集中度与 K-only
  gain 秩相关 +0.63（方向与病因假设相反），head_ratio 无解释力（layer 1/8 失衡大但误差
  小），深层负增益更可能是参考能量分母效应。证据：
  `logs/execution/2026-09-03-a2-deep-k-diagnosis-no-cause.md`；
- **A3 不启动**：前置条件（A1/A2 至少一个通过 D1）不满足；
- 按 §5 解释表最后一行：Attention 解析宽域族在 v160 上饱和，本地已知机制族
  （Linear 结构 full64/Householder、Attention 解析 Matrix-Smooth 扩展）全部闭环，
  下一动作是外部材料搜索或用户指定新机制，不再从当前族内继续微调。

## 1. 背景与方向依据

Linear 侧两个正交结构假设均已闭环否定（full64 REJECTED；Householder 六变体全负，见
归档计划 `../archive/plans/2026-09-03-official-pattern-and-linear-structure-experiments-superseded.md`）。
本计划转向 Attention 侧，依据是
[`OPA-1 Stage 1 官方证据账本`](../../../logs/execution/2026-09-03-opa1-stage1-official-evidence-ledger.md)
的四条结论：

1. Attention 轴三次干净的官方迁移全部正向：+227（v84→v86）、+117（v86→v158）、
   +741（v140→v147），轴向余量远未饱和；
2. 本地 Attention 与官方 Spearman `0.846`（Linear 为 `−0.61`）：QK 误差耦合是
   `softmax(QK^T)V` 的算术性质，跨模型成立，不依赖模型特异几何；
   > **[2026-09-04 作废]** 「0.846 / −0.61」来自 5 折 cluster（reeval5），不外推；跨模型探针
   > 与官方排序 ρ = −0.071。见
   > [修订清单 §1 / §8](../../../stale-information-inventory-2026-09-04.md)。
3. 误差集中点已被 v160-a2 default 分解定位：深层 K 是最大剩余单侧源
   （layer 22/23 K-only `−66/−201`，QK interaction `+142/+221`）；短序列 regime 误差更大
   （len10 Q-only `−36` vs len1024 `−13`）；V 近中性（`−0.0007`）不是目标；
4. v159 = linear.txt Linear + v158 Attention，若合成忠实，17816 的 `+284` 缺口只能在
   Attention 侧（P9 假设）；任何 Attention 候选的官方回传同时检验该归因。

## 2. 目标与边界

只测 Attention 侧解析式宽域机制，冻结清单不变：

- 禁止 seed/alpha/offset/threshold 扫描、数据依赖门控、模型/layer/role/length 路由；
- 禁止 ROAB、PAWV、动态 Q/K Gram 搜索等已关闭路线；
- 候选必须解析闭式、全层统一规则、宽覆盖（touch 大、跨层跨长度），与 v158
  Matrix-Smooth 同族；
- 每个假设最多一个候选、一次官方提交；官方失败后不邻域调参，不把官方分数回填 loss；
- Linear 侧 control 必须逐位一致（compact 即可），不重复 Linear 校准。

## 3. 验证漏斗（失败即停）

| 阶段 | 内容 | 预算 | 停止条件 |
| --- | --- | ---: | --- |
| A | 单文件导入、六 API 合法性、Linear 侧与 v160 compact 逐位一致 | `≤2 min` | 接口或 control 失败 |
| B | Qwen `--attention-only --compact-panel` 四哨兵 + GPT-2 attention compact 同号 | `3–5 min` | 哨兵负向或 GPT-2 反号 |
| C | Qwen `--attention-only` default 120 case，执行 D1 判别器与 QK/长度分解 | `≤2 min` | D1 或分解门禁失败 |
| D | JSON replay、单文件隔离导入、报告与 SHA | `≤1 min` | 证据不完整 |
| E | 一次官方提交（用户执行） | — | 见 §6 解释表 |

Attention default wall 约 `69s`（v160-a2 实测），整个漏斗远快于 Linear 侧；首个候选需
一次性建立 GPT-2 attention compact parent。

### 3.1 default 门禁（D1 判别器，预注册于 OPA-1 账本 §5）

- touch `>= 50%` 且 improved `>` regressed 且 median Δgain `>= 0`；
- Q/K/V 分解：QK-only 不恶化，V-only 变化 `|Δ| <= 0.005`（V 冻结优先）；
- logits MSE 与 probability MSE 不恶化；
- 各长度分组 mean Δ 不为负（重点 len10）；
- Linear compact control 逐位一致。

现有 D1 证据 3/3（v84→v86、v86→v158、v140→v147）。候选满足 D1 → 预测官方正向并提交；
这是预测不是保证，官方 Δ≤0 时 D1 降级为"v158 族特有"并记录，不触发邻域调参。

### 3.2 GPT-2 门禁

- GPT-2 attention compact 与 Qwen 同号（mean Δ 同向）；
- OP-125m/pythia-160m 仅在最终晋级候选时作封存 holdout；
- 跨模型只否决不调参。

## 4. 候选（按序执行，每个单机制）

### A1：Matrix-Smooth 组内扩展（首选）

v158 已验证 GQA 组内 2×2 解析平滑。同族扩展二选一（按 v160 归档中现有结构的自然延伸
选择，不并行）：

- **A1a**：组内平滑 block 从 2×2 扩大到 4×4（保持解析闭式、固定规则）；
- **A1b**：Q/K 跨 head 联合解析均衡（对 K 投影的 head 维度做与 v158 同型的
  smooth 量构造）。

连续域乘积不变；Hessian/Gram 在最终部署坐标计算；state 只存解析参数，动态路径不加算子。

### A2：深层 K 结构诊断 → 统一解析规则

第 0 步是零 API 诊断（读 `v160-a2-attn-default-candidate.json` 分解 + 检查 layer 22/23
K 投影权重/校准激活的结构特征，与浅层对比：outlier 通道、scale 失衡、能量集中）。仅当
找到跨层一致且跨模型可验证的解析病因时，构造**全层统一**规则（规则本身无层路由，误差
最大的层自然收益最大）；找不到一致病因则跳过 A2。

### A3：短序列 regime

len10 的 Q 误差是 len1024 的 2.7×。统一规则版本：对所有长度一致的解析缩放/居中结构
（校准长度 `[10,128,512,1024,1024]` 天然覆盖两个极端）。仅当 A1/A2 至少一个通过 D1 后
才启动，避免并行多机制。

## 5. 结果解释

| 结果 | 结论 | 后续 |
| --- | --- | --- |
| A1 通过 D1 且官方正向 | Matrix-Smooth 族仍有余量，D1 增至 4/4 | 同族下一个解析扩展 |
| A1 通过 D1 但官方零/负 | 本地 Attention proxy 对该子族失效 | D1 降级记录，停止本族 |
| A1 本地 D1 失败 | 组内扩展无本地余量 | 转 A2/A3 |
| A1/A2/A3 全部本地失败 | Attention 解析宽域族在当前父版本上饱和 | 归档本计划，等待外部材料或新机制 |
| 官方正向 ≈ +284 量级 | 支持 P9（17816 缺口在 Attention 侧） | 记录归因，不作为调参依据 |
| 官方正向但 ≪ +284 | P9 部分成立或 linear.txt 合成有保真度损失 | 记录，两个假设不分离 |

## 6. 执行顺序与停止条件

1. 建立 GPT-2 attention compact parent（一次性）；
2. A1 候选实现 → 漏斗 A–D → 通过则一次官方提交；
3. A2 第 0 步诊断（零 API）→ 有一致病因才实现候选；
4. A3 仅在前序通过后实现；
5. 官方结果无论正负，本计划只记录，不做邻域调参；
6. 所有候选结束或连续阻塞后，本计划归档并在同一提交指定后继计划。

时间余量参考：v160 官方 `232s`，距 `300s` 余 `68s`；Attention 动态 Q/K/V 官方成本极低
（本地 3.4s），候选不改变调用图，复杂度不构成约束，但仍须在阶段 A 记录校准增量。
