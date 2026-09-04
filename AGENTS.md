# HiF4 竞赛工作记忆

> 最后整理：2026-09-04。这里只保留当前可执行规则和不可误读的状态；旧实验细节、失败版本
> 和历史分数只在 `docs/current-solution-status.md`、`solutions/README.md`、`artifacts/`、
> `logs/` 中保留，不复制到本文件。读历史文档前先看过期信息清单（两版都要读）：
> **[2026-09-04 修订版](docs/stale-information-inventory-2026-09-04.md)**（本地↔官方换算类
> 结论：跨模型探针、时间换算、Spearman 相关性、ρ=0.224）→
> [2026-09-02 版](docs/stale-information-inventory-2026-09-02.md)（旧权重 / 420s /
> 36000 目标 / 协议退役 / A@W 边界）。新版不取代旧版。
>
> **⚠️ 2026-09-04 判定的三条硬约束**（详见修订清单）：① 本地数值**不能**换算官方绝对分
> （LOO MAE ≈ 1108 分 vs 单步增益 1~123 分）；② 跨模型 GPT-2/opt 探针与官方排序
> ρ = −0.071 / −0.200，**不得作为晋级旁证**；③ 官方时间**可以**预测，
> `T ≈ 170.3 + 0.115·W_calib + 0.694·A_calib + 0.734·dyn_act − 1.58·dyn_qkv`
> （R² = 0.799），**预测 < 280 s 才提交**。

## 1. 当前状态

- **v186（v182 + 在线 scale 窗口 +4 单码）官方 `17599/272s`，相对 v182 `+1/−1s`，
  RETAINED 成为新完整官方父版本**（SHA `F8495DCA...7EB8`）。step_gain `+1`；时间
  −1s 验证了时间模型的校准中性预测（预测 274.0s vs 实测 272s，MAE 带内）。
  本地 Δmean +0.010344（A1 后最大信号）→ 官方 +1：再次确认本地均值不换算官方分，
  但符号门禁（Δmean>0、L1<0.02）零误。**+4 窗口族官方正裁决；不扫其他码邻域**。
  当前距榜首 21765 差 4166，时间余量 28s。v186 是完整组合版本；独立父侧仍为
  `P_L=v166（4590/226s）`、`P_A=v168（14005/210s）`。oracle 分解方法论
  （穷举→二分定位→最小产物）首次产出官方晋级。

- **v187（v185 clean-room + Attention Jacobian 坐标敏感度）官方 `9167/169s`，
  相对 v185 `+721/+4s`，机制 RETAINED、但不作为完整父**：唯一新机制把完整 Attention
  输出的一阶 Jacobian 压缩为 KV-group 共享的
  Q/K `KV-head×64` importance，经 leave-one-fold-out gate 后在线只应用固定权重；Linear/V
  不变。7/24 层可达；相对 v185 default `Δmean +0.015187`、L1 `0.016199<0.02`、
  `32+/3-/85=`；官方大幅正向确认机制有效且时间代价仅 4s。但总分仍比 v186 少 `8432`，
  所以只保留为 clean-room 研究父，不移根文件；下一步应把相同解析机制单独移植到 v186，
  不扫描收缩/clamp/gate 邻域。

- **v185 clean-room 稳健算子量化重写官方 `8446/165s`，REJECTED**：不继承历史
  `solution.py`，从六 API/HiF4 合法域实现解析 Linear 对角平衡与 Attention K-center、
  KV-head Q/K balance、收缩 logits gain、门控 `+4`。官方相对 v186 少 `9153` 分；165s
  说明失败来自算法表达能力而非超时。原 balance/gamma/refine 邻域关闭，作为第 7 个提交
  计入配额；根 `solution.py` 保持 v186。

- **v183（Attention block-smooth final-quantizer refine 全覆盖）官方 `17598/279.7s`，
  与 v182 同分、慢 6.7s，REJECTED。** `step_gain=0` 且时间通过 `<300s`，因此失败归因
  于覆盖率扩展没有官方收益而非超时；按预注册规则覆盖率族关闭，不扫 ratio/blocks 邻域。
  根 `solution.py` 与完整官方父继续保持 v182，提交账本 `4/10`、剩余 6。

- **v182（v180 + L-R2 融合 rank-2 残差重分布）官方 `17598/273s`，相对 v180 `+1/+31s`，
  RETAINED 成为新完整官方父版本**（SHA `F3E39E99...A438`）。step_gain `+1`（计划 §11
  `0<G_L≤20`：残差低秩族接近饱和），**rank-3/系数/fold 邻域明确关闭**。273s 在 300s
  硬限内，但时间余量收窄至 27s；rank-2 校准 power iteration 是时间增量主因。
  当前距榜首 21765 差 4167。v182 是完整组合版本；独立父侧仍为
  `P_L=v166（4590/226s）`、`P_A=v168（14005/210s）`。Attention 与 v180 逐位一致，
  +1 可归因 Linear rank-2 第二残差方向。它只闭合原差距约 `0.024%` 却增加 31s，
  因而确认现有局部扩展与榜首之间是机制代际差距；v180 保留为时间预算父。

- v180（v175 + D1 A1 Q/K 非对称折叠）官方 `17597/242s`，相对 v175 `+3/−3s`，
  RETAINED；现为 v182 的精确父版本（SHA `2BA40122...8AA3`）。D1 只改 Attention
  state multiplier、Linear 与 v175 逐位一致，故 +3 可归因 D1；无在线新增算子，
  −3s 只记实测不宣称稳定加速。v175 的 interaction=0 证明侧向可加性。

- 低复杂度扩展计划全部裁决完毕（A1-A4 + L1-L4 + 组合 v175）。官方 2026-09-04 批测
  回传：**v175（组合 v166+v168）`17594/245s`** **RETAINED**——
  interaction = 0，侧向可加性在官方总分上精确成立；**v171（A4）`13657/214s`、v174
  （L4）`4508/190s`、v176（C1）`13964/205s`** **均官方负 REJECTED**（step\_gain −348/−82/−41，
  时间均 <300s，负向归因算法而非超时）。A4/L4/C1 家族关闭。v175 的 interaction=0
  证明侧向可加性，现为 v180 的精确父版本。

- 计划候选清单 C1/C2/C3 全部裁决完毕：**C1 官方负关闭**（v176 REJECTED，本地
  default −0.004450 / GPT-2 −0.002753 / opt −0.021851 方向一致）；C2 双分支
  > **[2026-09-04 修订]** 「三模型方向一致」**不再视为佐证**：GPT-2 / opt 探针与官方排序的
  > Spearman ρ = **−0.071 / −0.200**，不携带官方排序信息。C1 关闭的**唯一有效依据是官方 −41**，
  > 三模型同号只是当时的记录事实，不得再用它支持或否决候选。见
  > [修订清单 §1](docs/stale-information-inventory-2026-09-04.md)。
  （v177/v178）、C3（v179）本地 REJECTED。五轮 SOTA 搜索收敛，无新的零动态、HiF4
  字段兼容、非已闭合家族的独立机制方向。当前唯一官方正向 Attention 机制为 A1
  （v168）。

- **官方提交配额（用户约束，2026-09-04 目标设置起）**：提交通过版本 ≤10。v171/v174/
  v175 为目标设置前已排期队列（不占配额）；**v176 为第 1 个（1/10，官方负不退还），
  v180 为第 2 个（2/10，官方 +3 RETAINED；GPT-2 轻微负风险记录保留），
  v182 为第 3 个（3/10，官方 +1 RETAINED 成为新完整父），
  v183 为第 4 个（4/10，官方 `17598/279.7s` 与父持平 step_gain 0 → REJECTED，
  覆盖率族关闭；官方父保持 v182），
  v184 为第 5 个（5/10，官方 >300s TIMEOUT——双窗×2 校准增量 +36.6s
  （时间模型 0.694×52.7s），从 v182 父预测 309.6s 与实际吻合；超时来自
  双窗架构而非 5 码窗口本身——单窗 5 码校准与 4 码同速（本地 66.0s vs
  66.1s）；按预注册规则 TIMEOUT，不缩窗重试，+4 窗口族待单窗重构评估），
  v186 为第 6 个（6/10，官方 `17599/272s` +1 RETAINED 成为新完整父；
  oracle 分解方法论首产官方晋级），
  v185 为第 7 个（7/10，官方 `8446/165s`，算法大幅负向 REJECTED），
  v187 为第 8 个（8/10，官方 `9167/169s`，相对 v185 +721，机制 RETAINED），
  剩余 2**。每新增一个候选扣 1
  配额，官方负向不退还。SOTA 搜索（二至五轮：KVLinC/
  VecInfer/ResQ/OTT、ScaleSweep/H-Scale、MXFP4 误差三分量/HCP/ARCQuant、QuantVLA
  温度匹配/SageBwd/谱界）均落入已闭合域或 A1 已覆盖域，不注册新候选。官方裁决后
  A1-freedom 计划已经归档：D1 v180 官方 +3 RETAINED，D2 v181 本地 REJECTED，D3
  由 v180 完成。**L-R2 计划（post-v180-linear-rank2）已裁决完毕**：v182 官方
  `17598/273s` RETAINED 为新父，step_gain `+1`（残差低秩族接近饱和），
  **rank-3/系数扫描/fold 邻域明确关闭**；计划更新状态后可移入 archive。

- **V 侧方向结构性关闭（2026-09-04 穷尽审计）**：V 的量化自由度全部排除——
  per-head importance 无法改变 HiF4 64 块内离散解（64 块恰 = 1 head 1 token 的 64 维，
  per-head 常量只整体缩放块损失，A3/v170 实测 −0.06pp 且单层退化）；per-channel
  multiplier 在当前编码器语义下是「编码前缩放、解码不逆缩放」（`_dequantize_hif4`
  无 multiplier），对 Q/K 是 logits 缩放（softmax 容错），对 V 直接破坏输出 O（无
  softmax 容错）；per-token scale 因 HiF4 五字段无 per-token 表不可行（C4）；A2
  V-bias、C1 K 侧 per-channel 等化均官方负。V 侧不注册新候选。

- v159 原始 SHA `0508045A...4242` 的官方分数为 **17532**、时间未知；v159 修正归档 SHA
  `13C9CF0B...5EC79` 只增加数学等价的 GPU device 修复与中间量复用，尚未官方复测。v158
  **16861 / 223s** 仍是时间与源码均完整的安全父版本，其 Attention Matrix-Smooth 继续冻结。

- v160 归档 SHA `33B1D061...680D` 的官方结果为 **17532 / 232s**，相对 v159 分数 no-op；
  它保留为历史实验父版本。根 `solution.py` 已同步为 v186 官方计分源码，SHA
  `F8495DCA...7EB8`。

- 用户已确认官方评测稳定；禁止为验证确定性、估计时间噪声或单独研究批处理而提交相同 SHA
  或逐位等价时间 A/B。候选时间只作为算法验证的附带门禁。

- 用户确认的当前官方榜首是 **21765 / 290s**，源码和配置未知；它只作为外部目标锚点，
  不能伪造归档或替代本地实验结果。当前可复现最高 v186 距榜首 `4166` 分、官方时间余量
  `28s`，不能靠已有局部调参族填补。v180 快 30s 仅少 2 分，保留为高复杂度新机制的
  时间预算父；v186/v180 共同构成当前分数—时间 Pareto 前沿。

- 根目录 `solution.py` 与 v186 归档同步，SHA `F8495DCA...7EB8`。低成本候选从 v186
  构造；需要超过 28s 余量的新机制可从 v180 时间预算父构造。新候选必须单独归档并同时
  保留官方计分 SHA 与候选归档 SHA，不能把未复测 SHA 写成官方结果。

- 本地 proxy 只用于同机机制诊断和时间记录，**不能换算官方分数**；已知历史中
  存在本地排序与官方排序反转，任何本地正向都必须等待官方回传确认。
  > **[2026-09-04 细化]** 分数与时间要**分开说**：
  > ① **分数**——禁止换算（逐 case 特征 + LOO 的 MAE ≈ **1108 分**，而单步官方增益只有
  > 1~123 分）。本地只做**符号/风险判读**，硬门禁为 `Δmean > 0 且 L1 < 0.02`
  > （17 个官方配对上 5/5 零误提交，并拦下 −1163/−1164/−165 三次最大损失）；
  > ② **时间**——**可以预测**，用六 API 分解模型（R² = 0.799 / MAE 10.1 s），
  > **预测 < 280 s 才提交**。见
  > [修订清单 §5](docs/stale-information-inventory-2026-09-04.md)。

- v161（S1 交叉算子 Gram64 per-call 精化，v160 分支，SHA `27EEE471...1848`）官方
  **timeout（>300s，无分数）**，已按 `_timeout` 归档。本地全漏斗通过（Qwen default 120
  paired `+0.0525`、106+/14−、GPT-2 同号、D1 满足、attention API +28.0s CUDA 在
  +40s 门禁内），但官方鲲鹏机上动态 per-call 小张量算子成本远超本地 CUDA 外推。
  > **[2026-09-04 修订]** 这段里的「GPT-2 同号」「D1 满足」「本地 +40s 门禁」三条**均已失效**：
  > ① 跨模型探针与官方排序 ρ ≈ −0.07 ~ −0.20，不携带官方信息；② 本地 CUDA 时间门禁对官方
  > 时间的预测能力记为失效；③ 官方时间改用**分解模型**
  > `T ≈ 170.3 + 0.115·W_calib + 0.694·A_calib + 0.734·dyn_act − 1.58·dyn_qkv`
  > （R² = 0.799 / MAE 10.1 s），判据为**预测 < 280 s 才提交**。该模型也解释了 v161 的
  > 超时机制：本地 Attention 动态 1 秒在官方值 0.734 秒。见
  > [修订清单 §1 / §3](docs/stale-information-inventory-2026-09-04.md)。
  **修正时间核算：v128 家族超时元凶不只是校准搜索（199.8s/24 calls），动态精化本身
  （本地 0.092s/call CUDA）在官方硬件即超预算**（v138 无 dyn refine 官方 208s 通过；
  v128/v129/v130/v131/v161 含 dyn refine 全部官方 timeout）。per-call 动态自适应族
  结构性关闭，不缩 sweeps 重试；S2 前置条件不满足不启动；D1 维持 3/3；本地 CUDA
  时间门禁对官方时间的预测能力记为失效。

- 官方两侧分数比重校准已完成：v162 `1001/146s`、v163 `4587/202s`、v164
  `13945/204s`；`17532-4587-13945+1001=1`，standard/v160 两端的官方分数近似按侧可加。
  当前已实现边际为 Linear `3586`、Attention `12944`，约 `1:3.61`；该比值混合官方权重与
  当前算法质量，只指导优先级，禁止作为本地 gain 到官方分的换算率。
  > **[2026-09-04 复核] `1:3.61` 仍然正确，无需修改。** 用当前 v182 口径复算
  > `C_A/C_L = 13007/3590 = 3.62`，与 v160 口径 `12944/3586 = 3.61` 一致。
  > ⚠️ 但 `docs/official-local-fitting-analysis-2026-09-04.md` §3.2 初版曾误把「侧隔离提交
  > 总分 `S`」当侧贡献 `C` 代入比值，得出 `3.05`——**该数字是算术错误（未扣 1001 零点），
  > 已在原文勘误，不得引用**。差距分解用 `C_L: 3590 → 4491（+901）`、
  > `C_A: 13007 → 16273（+3266）`。见
  > [修订清单 §10](docs/stale-information-inventory-2026-09-04.md)。官方时间不具同样的
  可加性：两侧边际和预测 `260s`，v160 实测 `232s`。

- `21765` 双路线计划的本地候选已全部裁决：Attention A 在 compact **REJECTED**（mean
  `-0.007813`、median `-0.004871`、`1+/3-/0=`），B 取消；Linear C 的五折 minimax 在 C1
  接口/control 通过，`1.584×` 单 state 时间只记高风险、不作硬否决，随后 C2 compact 为
  mean/median `-0.088775/-0.088583`、`4+/52-/0=`、worst `-0.216586`，七个 role mean、
  test/validation 和 W-only 均负，故按泛化门禁 **REJECTED**。不运行 C3-C5、不改
  fold/Jacobi/coverage/邻域、不提交官方；根 `solution.py` 未改。

- 低复杂度算法扩展计划已全部裁决/实现（记忆完整性）：A1 v168 官方晋级（`14005/210s`，
  step\_gain `+60`）；A2 v169 REJECTED（跨模型结构性反向）；A3 v170 REJECTED（双模型
  一致回归，动态 refine 是官方 12944 Attention 承重组件）；A4 v171 候选（近中性）；
  L2 v172 / L3 v173 REJECTED（明确负优化）；L4 v174 候选；组合 v175 候选。候选仍从
  v162 双标准零点单侧构造：`P_L = v166 4590/226s`、`P_A = v168 14005/210s`；官方差分
  按计划 §3.3 登记 step\_gain 与相对 `3586/12944` 的固定口径比例；每包一个候选、失败
  换机制不扫邻域；v165 约束（动态 API 无 Gram contraction、无候选循环、复杂计算只在
  calibration）对 Attention 新候选仍强制。**组合条件已满足**：`S_pred = 4590 + 14005 −
  1001 = 17594`（仅比 v160 高 `+62`，距榜首 `4171`）；组合时间余量充裕。
  本地 panel 只作描述性诊断，不再用轻微 mean/median/尾部负向取消首次官方测量；
  硬检查仅为接口、合法 state、有限输出、机制 reachability 和非目标 standard
  control。每个机制仍只允许一个预注册配置，官方负向后不得邻域扫描。

- 2026-09-03 的首次 L3 full64 no-op 结果无效：`_WEIGHT_FULL64_APPLY=True` 被
  `_WEIGHT_E2E_REFINE=False` 外层死分支遮蔽，目标函数未执行。不得引用该结果证明块级收敛；
  后续必须记录 attempted/accepted block 计数验证 reachability。

- 修正 reachability 后已仅重跑一次 L3 compact：attempted `659456`、accepted `657540`，但
  paired mean delta `-0.017920`、`6+/42-/8=`；W-only `+0.107169` 被 interaction
  `-0.118818` 反转。该实验为 `REJECTED`，禁止再次运行或调整 full64 参数。

- Householder 统一 64-block 坐标重分布全族 REJECTED：基础候选 compact `0.705508→0.699190`
  （`8+/48-/0=`），五个 C 源变体（amax/rms/xrms/x-only/w-only）全部低于基线；研究臂在根
  `solution.py` 默认关闭。Linear 侧同坐标码字与坐标几何两个正交假设均无本地余量，Linear
  原 full64/Householder/单折邻域族闭环，禁止复跑或改参数。最后一个预注册例外 cross-fold
  minimax 部署 A\@W 已在 C2 因系统性 holdout 回归关闭；不得改 fold 聚合、Jacobi/Gauss-Seidel、
  coverage、邻域或 role 路由重启。

## 2. 提交代码约束

- 正式提交是根目录 [`solution.py`](solution.py)，必须单文件、自包含，只提供评测器要求的六个
  API：
  `hif4_calibration_and_quantize_weight`、`hif4_dynamic_quantize_activation`、
  `hif4_calibration_attention`、`hif4_dynamic_quantize_q`、
  `hif4_dynamic_quantize_k`、`hif4_dynamic_quantize_v`。

- 正式代码不得通过 `importlib`、相对/绝对路径、归档目录或其他 Python 文件加载实现；脱离
  仓库仍必须能导入六个 API。

- 所有编码器、解码器、E6M2 scale、层级 scale/lv2/lv3、mantissa/sign 和状态逻辑必须在同一
  提交模块内，并通过 `evaluator/reference_hif4.py` 的合法状态检查。

- 在线动态 API 只执行校准阶段编译的规则；禁止把校准搜索、完整矩阵求逆或未限制的 Python
  候选循环带入在线路径。

- Linear 研究目标是实际输出误差
  `XW^T - Q(XR) Q(WR^{-T})^T`，变换必须保持连续域乘积不变；Hessian/Gram 必须在最终
  变换和部署权重坐标系中计算。

- Attention 与 Linear 分开改、分开归因。当前侧向计划中 Linear 实验冻结 v162 standard
  Attention，Attention 实验冻结 v162 standard Linear；只有两侧均取得独立官方结果后，才构造
  一个组合候选检查 interaction。

### 2.1 编码原则

- 默认只处理用户明确提出的问题，先给结论并保持简短；不得自行扩展为额外评测、实现、文档整理
  或长篇分析。只有完成当前问题确实需要时才运行工具或展开细节，用户要求深入时再补充。

- 不追求过度防御：只写能解决当前问题的最小代码，不做猜测性的扩展设计。

- 只修改必须改的地方；不顺手改动与任务无关的代码、格式或注释。

- 只清理自己产生的问题；不重构或清理他人遗留、无关的问题。

- 禁止为不可能发生的场景添加错误处理、回退、空值检查或校验；只信任框架保证，仅在系统
  边界（用户输入、外部 API）进行必要校验。

- 禁止为一次性操作创建辅助函数、工具类或抽象；三行相似代码优于提前抽象，不为假想的未来
  需求设计参数、标志或兼容层。

## 3. 唯一评测口径

- 唯一本地主评测器是 [`evaluator/official_eval.py`](evaluator/official_eval.py)，协议是
  `proxy-v2`。`official-shape-v1`、GPT-2 和外部 hif4 只作历史/跨结构诊断，不能与当前结果
  混排。

- 本地固定结构假设：Qwen2.5-0.5B、24 层、WikiText-2 raw v1、Attention calibration
  lengths `[10,128,512,1024,1024]`，以及独立的 HiF4 validation。说明书没有公开这些隐藏
  结构，因此它们不是官方模型证据。

- 默认 panel 是 168 Linear（24 层 × 7 role）+ 120 Attention（24 层 × 5 长度）。
  `--full-cases` 的 2016 + 288 只作 stress；`--linear-cases/--attention-cases` 是顺序前缀
  smoke，不能用来判断泛化或晋级。

- `--compact-panel` 是低成本机制筛选：Linear 选 layer `0/8/15/23`、7 role、两组
  validation/test 同长度 holdout，共 56 cases，只建立 28 个 Weight state；Attention
  compact 只保留四个深度/长度哨兵。它只做父子机制和跨 holdout 泛化诊断，不能冒充 default
  panel 或官方调用图。

- `--ood` 是 OOD 泛化面板（`data/ood-suite-v1`，code/news/zh 三域 15 个测试窗口）：
  校准保持 WikiText（与部署一致），只换测试文本；case 数与默认面板相同（168 Linear +
  120 Attention，Attention 用 8 层深度铺开 × 15 窗口）。它只做过拟合诊断，与同 SHA 的
  in-dist proxy-v2 运行相减得 `gain_in − gain_ood`（v182 基线：Linear `+0.015903`、
  Attention `+0.020590`），**不参与 proxy 排名、不能与 in-dist 结果混排**。语料由
  `workbench/build_ood_corpus.py` 确定性重建。五版本标定（v158–v176）确认：小幅官方负
  在 OOD 上无特征，OOD 只拦截分布拟合型大失败，**可执行阈值为
  `|Δ(gain_in − gain_ood)| > 0.01`（家族带 ±0.007 之外）**，带内不作门禁；见
  [标定记录](logs/execution/2026-09-04-ood-calibration-five-versions.md)。

- 单侧场景必须隔离：`--linear-only` 不调用 Attention API，`--attention-only` 不调用 Linear
  API；本侧校准仍按共享 state 调用图执行，不按 case 制造 oracle。

### 3.1 Local proxy 的定义

每个 case 的本地分数是：

```text
gain = (MSE_STD - MSE_PLAYER) / MSE_STD
     = 1 - MSE_PLAYER / MSE_STD
```

`STD` 是标准 HiF4 对同一 NVFP4 解码输入的输出，`PLAYER` 是候选 API 的输出；它不是模型
准确率、不是官方总分，也不是官方时间。`overall_mean` 是实际 case 的等权平均，不拟合
Linear/Attention 权重。只有同一 `proxy-v2` cache、同一 panel、同一 device 的
`default-panel` 才能做本地 proxy 横向比较。

### 3.2 防止过拟合（强制）

- 本地 proxy 只用于否定机制、定位误差和比较同机成本，不得凭本地均值正向直接晋级；官方结果
  只验证预先声明的单一假设，失败后不得围绕 threshold、seed、alpha、offset 或候选数量做邻域调参。

- 校准、候选选择和验证必须分离。A\@W/GPTQ 的参数只用 calibration folds 学习，晋级读取独立
  holdout；不得用同一 fold 同时选规则和证明收益。多折选择使用 median、worst-fold 或固定 robust
  聚合，禁止只取第一折或最好一折。

- 每个版本只改变一个可解释机制，候选数量固定且与数据结果无关。优先使用低自由度解析结构、
  block-Schur/块对角/低秩补偿和预先固定的正则；不得通过扩大 permutation、Hadamard seed、搜索
  网格或多机制叠加换取本地分数。

- Linear 必须评估最终部署目标 `Q(A)Q(W)^T`，并在最终变换坐标系计算 Hessian/Gram；operand MSE、
  对角 importance 和 aggregate mean 只能用于诊断，不能代替输出误差与跨折证据。

- 晋级至少同时检查 focus 的 median、worst-quartile、负 case、跨 holdout 同号率和未修改 control。
  收益若集中在少数 layer/role/fold、依赖单一模型形状，或 control 发生变化，按过拟合处理。
  > **[2026-09-04 修订]** 「依赖单一模型形状」这条**不再作为过拟合判据**：跨模型探针与官方
  > 排序 ρ = −0.071 / −0.200，跨模型负向不携带官方风险信息。过拟合的唯一量化判据改为
  > **`L1`（逐 case gain 的平均绝对变化）< 0.02**，且 `L1` 与官方 Δ 负相关（ρ = −0.469）：
  > **本地改得越狠，官方越可能负**。「control 发生变化」仍然有效。见
  > [修订清单 §2 / §9.2](docs/stale-information-inventory-2026-09-04.md)。

- OOD 判据（2026-09-04 起可用）：候选过拟合的补充量化信号是
  **`Δ(gain_in − gain_ood)`**——同 solution 在 `--ood` 与 in-dist proxy-v2 各跑一次，
  与父版本的两侧差值相减；本地增益上升而 OOD gain 同步大幅下降即拟合型机制
  （v140/v155/v156 失败模式）。v182 父基线 gap：Linear `+0.015903`、Attention
  `+0.020590`。五版本标定确认小幅官方负无 OOD 特征，**只在 `|Δgap| > 0.01`
  （家族带 ±0.007 之外）时作为提交禁止门禁**，带内不作门禁。见
  [拟合分析 §6](docs/official-local-fitting-analysis-2026-09-04.md) 与
  [基线执行记录](logs/execution/2026-09-04-ood-suite-baseline-v182.md)。

- 官方 mini 用例只做接口、合法性和真实形状复杂度 smoke，不用于选算法或参数；Qwen/GPT-2 等
  本地结构只作机制压力测试。发生本地与官方排序反转后，立即停止用该 proxy 为同一路线晋级。

## 4. NVFP4 输入缓存

- `--nvfp4-cache-mode auto`（默认）按 scenario/panel/case profile 持久化已编码的 NVFP4
  carrier/scale；缓存只含 evaluator 输入，不含候选 state 或候选输出。

- profile、协议、codec/mode、dense source identity、数据 hashes 或 panel 不一致时，`read`
  拒绝命中，`auto` 重建。`write` 强制重建，`off` 禁用 NVFP4 持久化。

- `--cache-mode auto` 先读已有 dense cache，只有 dense cache 不存在时才重新做模型前向；
  `--cache-mode read` 只读，不允许隐式重新捕获。

- cache 命中只减少输入准备/量化时间，不改变候选 API 数量、输出误差或本地分数。缓存文件是
  本机生成的 ignored artifact，不作为正式源码证据。

## 5. 当前评测步骤（固定）

> 当前 v162 官方侧向隔离计划对下述第 5–7 步作专项覆盖：这些统计仍须记录，但不再作为首次
> 官方提交的准确率硬门禁；只有接口/合法性/有限输出/reachability/control 错误阻止提交，算法
> 提升与否由相对 v163 或 v164 的官方分数裁决。跨模型负向只能标记风险，不能据此调模型路由。

1. **启动读取**：先读本文件，再读 `docs/superpowers/plans/README.md`、唯一活动计划、
   `docs/current-solution-status.md`、`solutions/README.md`、目标父版本和
   `evaluator/official_eval.py`。历史文档先过 stale inventory。
2. **固定父版本**：父版本只运行一次并保存 immutable JSON/report；后续候选使用同一 cache、
   panel、device 和 evaluator，不重复运行父版本。
3. **接口 smoke**：选定一个目标场景，运行目标侧最小 smoke，检查六 API/状态合法性和
   `--nvfp4-cache-mode auto` 是否可用；smoke 只判接口，不判效果。
4. **compact 配对**：

   - Linear：`--linear-only --compact-panel`；

   - Attention：`--attention-only --compact-panel`。
     候选使用 `--baseline-json`，先看 focus 的 mean/median signed delta、正负 case、未修改
     control、W/A 或 Q/K/V 来源、最坏 layer/role/shape/split/length 和 API 时间。必须精确匹配
     `(layer, role, test_window, split, length)`、`mse_standard`、`reference_energy`；已有同 panel
     JSON 用 `--candidate-json` 零 API 重放。
5. **泛化判断**：Linear 至少记录 median、q25/q75、worst-quartile、negative cases、
   validation/test 同号率和 interaction；Attention 至少检查 Q/K/V、QK/QKV interaction、
   logits/probability 误差和最坏长度/层。不得用 aggregate mean 单独宣称本地泛化；当前计划中
   这些结果只作官方回传后的归因证据。
6. **单侧 default audit**：完成 compact、control、尾部和复杂度记录后，运行目标侧
   default panel（Linear 168 或 Attention 120）。旧 `--effect-panel` 只在需要“完整校准图 +
   缩减动态 case”的专项审计时使用，不是默认必经步骤。
7. **跨模型泛化记录**：目标侧 Qwen default 完成后，用其他模型真实前向捕获的 W/A/Q/K/V
   做同 cache、同 device 的父子配对。`gpt2` 为强制验证，最终候选再使用一个不同架构的本地
   `pythia-160m` 或 `opt-125m`。跨模型只作封存 holdout，不能反向调参数；若 Qwen 正向而
   跨模型整体负向，候选标记 `model-specific-risk`，仍由首次官方结果裁决，禁止增加
   模型/layer/role 专属路由。
   > **[2026-09-04 修订]** 本条**降级为记录项**：`gpt2` 不再是强制门禁，跨模型整体负向
   > **不再触发 `model-specific-risk` 标记**（探针与官方排序 ρ = −0.071 / −0.200，
   > 该推理链前提不成立）。跨模型运行仍执行、仍写入日志，但只作机制鲁棒性描述，
   > 不得晋级也不得否决。过拟合判据改用 `L1 < 0.02`。见
   > [修订清单 §2](docs/stale-information-inventory-2026-09-04.md)。
8. **完整端到端审计**：只有明确需要检查集成调用图时，才省略 `--linear-only/--attention-only`
   跑完整 168 + 120 panel，六 API 全部执行；`--full-cases` 仍只作压力测试。完整测试必须
   保存 JSON 和 Markdown report，并把 local proxy、API total、wall time、official 状态分开写。
9. **决策与归档**：接口/环境失败记 `ERROR`；机制证据否定记 `REJECTED`；官方明确超时记
   `TIMEOUT`；官方未知写 `unregistered/NA`，不能用本地秒数填充。没有实质算法/复杂度变化的
   运行不分配版本号。正式版本归档前只做一次脱离仓库单文件导入检查。

推荐命令模板：

```powershell
# Linear compact / cached input（必须使用 CUDA venv；系统 Python 是 CPU-only）
.venv\Scripts\python.exe evaluator/official_eval.py --solution solution.py --linear-only --compact-panel --cache-mode read --nvfp4-cache-mode auto --capture-device cuda --algorithm-device cuda

# Attention compact / cached input
.venv\Scripts\python.exe evaluator/official_eval.py --solution solution.py --attention-only --compact-panel --cache-mode read --nvfp4-cache-mode auto --capture-device cuda --algorithm-device cuda

# Complete default-panel integration audit
.venv\Scripts\python.exe evaluator/official_eval.py --solution solution.py --cache-mode read --nvfp4-cache-mode auto --capture-device cuda --algorithm-device cuda
```

## 6. 证据、比较和 Git

- 原始 `artifacts/official_eval/*.json`、`logs/official_eval/*.md`、`logs/execution/*.md` 不覆盖；
  修正使用独立日志和状态更新。结果先看 `evaluation_scope`：compact/effect/replay/smoke/stress
  都不是官方分数等价物。

- 官方结果优先级最高，其次是活动计划已确认事实，再次是归档 result/log 和本地 JSON/report；
  未验证推测不得写成结论。当前官方事实集中维护在
  [`docs/current-solution-status.md`](docs/current-solution-status.md) 和
  [`solutions/README.md`](solutions/README.md)。

- 每次实质代码或状态更新后运行 `git diff --check`，提交、push，并核验 `git status`；不要把
  ignored 的大 cache、`.codegraph/` 或临时目录加入提交。

<!-- End of current memory. Historical details stay in the linked evidence files. -->
