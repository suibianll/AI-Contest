# 合法编码复核与最终输出优化执行计划

> 日期：2026-09-05。状态：**ACTIVE / DESIGN_ONLY**。本轮只制定计划，未执行下述实验。
> 交接对象：下一位实现 AI。按 R0 → R1 → R2 → R3 → R4 顺序执行，逐门停止，不并行扩展机制。
> 分数父：v186，官方 **17599 / 272s**；SHA256
> `F8495DCA20334ACBDAD16FC18EE41A4970F31E1837FDEEDCEE9C70AEE54E7EB8`。
> 时间预算父：v180，官方 **17597 / 242s**。榜首 21765 / 290s 是用户提供的外部锚点。
> 旧 codebook 计划已结束；其实现缺陷及结论边界见[本次审计说明](../../../logs/execution/2026-09-05-next-plan-evidence-audit.md)。

## 1. 本轮要回答的问题与交付物

本轮目标是找到**能在现有合法字段和调用接口内降低实际输出误差**的一个机制。
先排除错误实验导致的假性饱和，再区分两种问题：

1. 编码实现遗漏：同一输入、同一目标下，现有合法编码器是否漏掉明显更好的解？
2. 优化目标遗漏：即使张量 MSE 已接近最优，联合舍入能否降低
   `||XW^T - X_hat W_hat^T||²` 或 `||Attention(Q,K,V)-Attention(Q_hat,K_hat,V_hat)||²`？

最低交付物是可复现的合法编码证据和有边界的 go/no-go 裁决；只有跨折、部署与时间门通过，
才交付单文件候选。**不得为了交付候选而跳过失败门，也不得把无候选写成数学上无优化空间。**

| 阶段 | 实现内容 | 验收产物 | 后继 |
|---|---|---|---|
| R0 | 冻结输入、重现具体缺陷、核对历史机制 | manifest、缺陷测试、novelty 表 | 合法性测试全过才进入 R1 |
| R1 | 正确的 HiF4 层级精确解与合法块证据 | 合法五字段、参考解码、逐块对比 | 编码缺口进入 R3-A；无缺口进入 R2 |
| R2 | 固定真实量化操作数的输出目标诊断 | Linear 联合舍入证据；Attention 实际输出证据 | 满足 R2 门才进入 R3-B；否则结束 |
| R3 | 只实现一个可部署机制，跨折及全分片验证 | 自包含候选、配对 JSON、control/OOD 报告 | 全部门通过才进入 R4 |
| R4 | 真实协议计时、官方裁决与归档 | SHA、预测、官方分数/时间、结论 | 正式晋级或关闭该机制 |

## 2. 不可误读的现状

- QK/V 官方探针 `12010/203s`、`2974/175s`，以 v162 的 1001 为零点，贡献分别
  11009、1973，交互为 −38。这支持优先研究 Q/K，**不表示剩余差距的 85% 在 Q/K**。
- W2/W3 官方贡献 1818、1767，W0/W1 为 0；表示 v160 机制在这些桶的已实现收益。
  首轮 Linear 聚焦 fc_gate/fc_up/proj，其余 role 是控制；零贡献不证明隐藏空桶或零潜力。
- 同坐标下连续偏差 B 很小，只证明当前变换近似保语义，不证明可选变换耗尽。
- 放宽 mantissa 的收益不能当合法编码可达收益；单操作数放宽不能代替完整部署输出。
- cb1/cb2 的负数值保留，但存在具体实现错误，不能继续作整个编码机制的关闭证据。
- 不采用研究分析中的榜首本地分换算、901/3266 差距归属、非法放宽“合法天花板”推算；
  不将不同算法的 ±4 或小于 100 分差认定为已验证噪声。

## 3. 公共实验契约

### 3.1 仓库、版本与文件

工作目录 `D:\工作内容\AI竞赛`，仅使用 `.venv\Scripts\python.exe`（CUDA 环境）。
先读取 AGENTS、三份 stale inventory、本计划、当前状态、solutions 索引及代码；本计划的
专项修订只针对本次审计列出的无效推论，不撤销已正确执行的历史负结果。

- 根 `solution.py` 全程保持 v186，候选写到独立研究目录；无需为诊断分配 v189 等版本号。
- 新工具：`workbench/legal_codec_output_probe.py`；测试：`tests/test_legal_codec_output_probe.py`。
- 新工具不得复制 cb1/cb2 的错误计数与自定义解码器；旧脚本及旧 run-001 输出不覆盖。
- 实验根：`artifacts/proxy_v3/legal-codec-output-20260905/`，阶段分目录 `r0/`、`r1/`、
  `r2-linear/`、`r2-attention/`、`r3/`、`r4/`；存在且完整则读取，失败重跑用独立 run 编号。
- manifest 必须包含 git HEAD、父/工具/配置/参考编码器 SHA256、cache SHA256、torch/CUDA/GPU、
  实际命令、case keys、dtype、阶段状态。大 cache 的 hash 复用已验证同一文件身份的 manifest。
- 正式候选六 API 自包含。研究工具允许导入参考代码，正式 solution 不允许外部加载实现。

### 3.2 数据、抽样和统计

输入固定为 `artifacts/official_eval/cache/qwen2.5-0.5b-proxy-v2.pt`；不得重新捕获模型。
它是本地机制数据，不能声称与官方隐藏模型一致。

- R1/R2 首轮层固定 `[0,8,15,23]`；Linear 固定 fc_gate/fc_up/proj，其他四 role 作 control。
- 每个矩阵取均匀索引 `floor(k*(n-1)/(m-1))`，去重且排序；n=1 时取 0。
  R1 取 8 行 × 4 个 64 块；R2 Linear 取 32 个输出行 × 4 个 64 块。
- Attention 每层、每个 calibration fold，取 head 0 及最后一个 KV group、4 个均匀 query token；
  GQA 关系从 pack 读取，同组 Q heads 使用对应 K/V。记录实际数量，不能静默跳过失败样本。
- 抽样只控制研究成本，不能当默认完整面板。holdout 使用 evaluator 已定义的独立
  validation/test；选择参数、配对及规则只读 calibration，不能读 holdout 后回调参数。
- 跨 fold 用等权 fold loss；各 fold 内 MSE 按元素数归一，禁止长窗口自动获得更大选择权。
- 同时报告 mean/median、q25/q75、最差四分位、正负 case、分层/role/长度、有效分母。
  主 gate 的 Δgain 使用每 case 的标准误差分母，不以平均 MSE 比替代平均 gain。
- 校准重建可复用已验的 v186 state；必须验证其 SHA、输入和设备身份。

### 3.3 合法性与语义

五字段 shape 必须与参考一致：每 64 块为 `[8,2,4]`，lv2 共享 8 元素、lv3 共享 4 元素。
signed mantissa 码为整数 `[-7,7]`，mant=abs(code)/4，sign 在零码处规范化为 0。
sf 从参考 `e6m2_decode(0..254)` 取值；字段里放解码后的 scale，不能把整数 code 当 scale。
仅使用 `evaluator/reference_hif4.py` 的合法性检查及解码作为裁判。

参考输出、父输出必须来自同一 NVFP4 输入。同坐标分解复用
`evaluator/coordinate_diagnostics.py` 的已验变换镜像，并核对最终 player 输出。
禁止将某些 Q token/块切回原坐标、同时让对应 K 留在旋转坐标；独立动态 API 也不能共享
未声明的运行时 Q/K/V。不能用 holdout 的 K/V 生成提交时 Q 编码规则。

## 4. R0：先修复证据链

**状态：PENDING。实现成本上限：一个工作包，不启动完整模型评测。**

1. 将本次审计的 cb1/cb2 缺陷逐项写成失败回归测试：dense 与 carrier 计数混淆、未乘兼容表、
   负码漏计、cb2 漏乘输入 scale、sf seed 固定为 0、非法层级共享形状、把 operand MSE 当输出 MSE。
   不必修复旧脚本；在新工具中建立正确路径并说明每项如何避免。
2. 用所有有限 E4M3 scale 与 E2M1 carrier 做小型穷举：比较 FP32 乘积与转 BF16 再转 FP32。
   报告逐位差异数和反例；额外把“原始码本乘积”和“经过平滑/旋转后的 dense”分开测试。
   不能把后者破坏码本结构归因于 BF16 本身。
3. 合成反例必须覆盖：负码、零、次正规 scale、非单位 scale、同一 4 元素共享层级冲突、
   E6M2 边界、不同层级下可精确而 lv2=lv3=1 不精确的块。
4. 审查 `_jdrq_refine_mantissa_coordinates`、L3 full64、cross-fold minimax、CAT/Trellis/Babai
   的实际调用与历史结果。输出 novelty 表，列出“目标、变量、联合更新范围、在线成本、父版本”。
   R2-B 只能测试尚未正确验证过的**固定层级、8 个码同时选择、真实量化 X 目标**这一差异；
   若已有等价有效实验，标 `DUPLICATE_CLOSED`，跳过 R2-B/R3-B，不改组大小重启。

**G0：**所有预期 case 执行，参考解码合法，合成测试通过，误差归因可解释，才进入 R1。
异常/漏 case 记 ERROR，不记 REJECTED。修 bug 后可在同一固定配置重跑；禁止同时换算法参数。

## 5. R1：合法 HiF4 精确块解，替代 free-a 乐观表

### 5.1 算法（只作诊断，不直接搬到动态 API）

目标为给定 64 维 x 的**无权 operand SSE**。对每个合法 sf 值 s：

1. 对每个 4 元素组 j、总层级指数 e∈{0,1,2}，令 `a=s*2^e`，
   `z=clamp(round(4*x/a),-7,7)`，`C[j,e]=sum((x-a*z/4)^2)`。
2. 对每个 8 元素组的两个子组 j0/j1：
   `L0=min(C[j0,0],C[j0,1])+min(C[j1,0],C[j1,1])`；
   `L1=min(C[j0,1],C[j0,2])+min(C[j1,1],C[j1,2])`。
   选择较小者决定 lv2，再选择各自 lv3。相等时选较小层级指数。
3. 累加 8 个组损失，遍历全部 255 个 sf 后选最小 SSE；平局优先父编码（若同损失），
   再取较小 sf code。返回真正合法五字段，重新参考解码计算损失。

这是**固定坐标、64 块、无权平方重建目标**的全局最优解；不是 A@W/Attention 最优解。
原因是给定 sf 后各 8 元素组独立，给定层级后逐元素最近码精确求解。用 FP64 算术核验
小例子，接近舍入边界时采用显式候选损失和平局规则，不能仅凭 GPU round 的偶然差异裁决。

原 P0 的 `S={k/4,k/2,k}` 是单元素并集，允许每个元素独立挑层级，不能证明一整个共享层级
的块可达。精确占比只能从最终合法解码张量统计，分母区分全部元素、非零元素和完整 64 块。

### 5.2 对比臂与门槛

同一抽样分别测原始 NVFP4 dense 和父最终变换坐标 dense；分开汇总。每种坐标比较：

- reference standard；
- 父 `_dense_to_hif4` 的相同固定配置（标为 encoder-only）；
- 父完整部署编码（标为 deployed，含其输出导向修改，不能假定 operand MSE 更优）；
- 上述合法精确块解。

用 `improvement=(SSE_parent-SSE_exact)/SSE_parent`；父 SSE=0 单独记录，不填成完美增益。
exact SSE 大于 parent SSE 超出 `1e-10 + 1e-7*SSE_parent` 时是实现错误，不能进入后续。

**G1-A：**在最终坐标 encoder-only 对比中，median improvement≥1%，至少 3/4 层为正，
且存在可序列化的合法 witness → 登记缺口类型，进入 R3-A。
否则写 `NO_MATERIAL_OPERAND_GAP_IN_PANEL`，进入 R2。1% 是预注册研究投入门，非分数预测。
即使原始坐标结果漂亮，也不能据此绕开父变换；仅保留为输入结构诊断。

## 6. R2：把优化目标转到真实输出

### 6.1 R2-L：8 码联合舍入（优先可部署分支）

**前置：R0 novelty 通过。**冻结全部连续变换、激活动态规则、sf/lv2/lv3；只改 W 的 mant/sign。
对每个校准 fold f，按父动态 API 实际生成 `Z_f=X_hat_f`，原始 NVFP4 参考
`Y_f=X_f W^T`。不能用 float X_t 替代 Z_f。

对于输出行 r，残差 `e_f=Y_f[:,r]-Z_f*w_hat_r`。每个已选 64 块按自然顺序分成 8 个
连续的 8 坐标组 S（不学习 permutation，不挑梯度最大组）。设合法步长
`d_i=sf*lv2_i*lv3_i/4`，给最终连续权重 `w_t_i` 的 signed 码各取
`clip(floor(w_t_i/d_i),-7,7)` 和 `clip(ceil(w_t_i/d_i),-7,7)`。

固定枚举最多 2^8=256 个联合码，并始终加上父当前 8 码作为候选；去重不改变排序。
对每个改变量 δ，计算真实输出损失增量：

`Δ_f = δ^T G_f δ - 2*g_f^T δ`，
`G_f=Z_f[:,S]^T Z_f[:,S]/n_f`，`g_f=Z_f[:,S]^T e_f/n_f`。

按 fold 等权均值选择；相等保持父值。以固定块/组次序**只做一遍**，接受后更新残差。
这是有限候选集内的精确联合选择，不是 full64 全局最优证明。记录“单个码变化均不降损失、
联合变化却降低损失”的 barrier witness，证明该机制与单坐标下降确有差别。

训练与验证：对实际可用 calibration folds 做 leave-one-fold-out，训练余下 folds 的候选，
在留出 fold 比较同一父。若少于 2 folds 则跳过该 state 并记录原因；不能复制 fold。
state 开启条件固定为 LOO Δgain 的 median>0 且最差 fold≥0；通过后在全部 calibration
folds 重算一次规则，最终只在 evaluator holdout 检验，不用 holdout 决定 gate。

**G2-L：**12 个 focus state 中至少 8 个 LOO gate 通过、至少 3 层通过；holdout focus
Δmean>0、median>0、两种 split 的 mean 均>0，L1<0.02，且存在至少一个跨折仍获益的
barrier witness → R3-B。否则关闭此固定 8 码分支，不改成 2/4/16 码、不加遍数或选组规则。
该门仅支持部署可行性，不证明能弥补 4166 分。

### 6.2 R2-A：Attention 真实输出条件诊断（不注册在线联合搜索）

冻结 K_hat/V_hat，替换 Q 的合法编码；另做冻结 Q_hat/V_hat、替换 K 的合法编码。
对 R1 抽样块，比较父码与 R1 产生的合法编码；其余块/操作数保持父。目标必须是完整
`Attention(Q_hat,K_hat,V_hat)` 对原参考的 MSE，使用 evaluator 的 mask、缩放和 GQA 语义。
Q 替换可只计算受影响的 query 行；K 替换必须计算受影响的所有 query，不能只测一行。
单独记录 tensor MSE、logits、probability、输出，检查是否出现目标反转。

此处按 calibration 实际输出选的块替换是**离线 oracle**，不准在 holdout 重选，也不准
将运行时 K/V 留在 Q state。holdout 只测固定编码算法的整条部署路径；没有这样的算法时
只输出 `ORACLE_ONLY / NO_DEPLOYABLE_RULE`。

**G2-A：**只有 R1 缺口能归结成所有输入通用、无需联合运行时数据的确定性编码修复，
才允许走 R3-A。只有 oracle 改善时，不重新启动 v161/v165 动态 Gram 或 v167 低秩 importance。
Attention 优先级高不构成强行提交不可部署算法的理由。

## 7. R3：只编译一个候选

### 7.1 选择顺序与机制边界

- **R3-A 优先**：只有 R1 找到明确遗漏时才执行。先出机制卡：缺失条件、最小反例、
  父分支位置、正确通用规则、相对旧 scale 窗口的差异。完整 255-sf 枚举留在研究工具。
  若唯一改动只是扩大已关闭 scale 窗口，写 `DUPLICATE_CLOSED`，不实现候选。
  Linear/Attention 分开测和提交，不因共享编码器一起改两侧。
- **R3-B 后备**：R2-L 通过且 A 无可部署修复时，编译固定 8 码联合更新。
  在 `hif4_calibration_and_quantize_weight` 的**最后一次实际权重编码之后**插入，不能被
  后续编码覆盖。处理全部权重行、全部 64 块、每块 8 组一遍；分批仅影响内存，不影响选择。
  fc_gate/fc_up/proj 只作分析标签；部署使用原代码已有形状信息识别 expansion/contraction，
  不新增 layer/模型专属路由。最终仅序列化正常 W 五字段；激活与 Attention 代码逐位不变。
- R3-B 从 v180 时间预算父构造；这是预先固定的成本选择。R2 的 v186 结果仅验证机制，
  R3 必须在 v180 上重新得到配对证据，不能把 v186 差分搬过去。不得两个父之间择分。
- R3-B 在线零额外计算，校准严禁形成 `T×out×256` 巨张量；按行/组分批计算小二次式。
  初始 microbatch=32 行，OOM 只缩 batch，保持顺序、结果与配置含义一致并记录。
- 每次记录 attempted/accepted groups、LOO gate、改码比例、最终返回权重 hash。
  `attempted=0` 或返回值未改变是 ERROR/NOOP，不是机制被验证。

### 7.2 评测与晋级

1. 合成测试、最小接口 smoke、六 API 可导入与合法状态检查。
2. 首个 shard 做复杂度和正确性诊断；运行一次目标侧六 shard 完整配对，不能因前两片负
   自动提前退出而报告“全层失败”。父已有同协议结果用 `--reuse-existing`，不反复重跑。
3. focus 与全目标侧都必须 Δmean>0，L1<0.02；记录尾部/负 case，两个 split 分别报告。
   未改 control 逐位一致。不满足即关闭，不扫 coefficients、fold、coverage 或候选数。
4. 在同 SHA 下测 OOD，并与同父 in-dist/OOD 成对计算：`|Δ(gain_in-gain_ood)|≤0.01`。
   跨模型仅为历史规则要求的描述性记录，不作为晋级或否决依据。
5. 单侧机制按 v162 standard 对侧隔离。若当前 exact-parent 单侧尚无独立官方测量，
   先构造同对侧 standard 的父锚，再提交子；不可拿 v163/v164 旧端点代替更新后的父。
   本轮低风险且允许的父锚是不同算法组合的信息测量，不重复已测同 SHA。

## 8. R4：官方裁决

时间输入必须来自与既有模型拟合一致的 **proxy-v2 default 168 Linear+120 Attention**
完整新鲜六 API 计时。`evaluator/official_eval.py` 仅在这一专项时间审计中使用，不修改文件。
不启用 compact/effect/full-cases/OOD，不用 v3 shard 或校准缓存命中秒数代入。

`T_pred=170.3+0.115*W_calib+0.694*A_calib+0.734*dyn_act-1.58*dyn_qkv <280s`。
四个量的定义/单位从拟合分析中核对并写入报告，缺项记不可预测；不得填 0。
负 dyn_qkv 系数是回归关联，不能宣称增加 QKV 计算反而省时；结构性新增动态计算另作风险判断。
若候选不满足门，则记时间阻断，不能减少 fold/覆盖率重新扫描；已定 v180 分支也不能继续换父。

官方提交次数无限制；官方结果未知记 NA，不写本地秒数。候选只有在目标侧官方
step_gain>0 时 RETAINED；同分且不更快不晋级；负分 REJECTED，不用“噪声”重新命名。
侧候选通过后可构造一次完整组合验证 interaction 与时间；完整根只接受分数/时间可用的新 Pareto 点。
失败关闭本机制，根 v186 不变。任何相同 SHA 或逐位等价版本不重复官方测量。

## 9. 必须实现的 CLI 与运行顺序

以下 `legal_codec_output_probe.py` 是**待实现接口契约**，当前尚不存在，不得报告已跑通。
工具依次提供 `--stage r0|r1|r2-linear|r2-attention`、`--parent`、`--cache`、`--config`、
`--output-dir`、`--device`。配置 JSON 在第一次读取实验结果前固化，内容按 §3–§6，记录 hash。
阶段间通过 manifest/结果文件交接，不依赖临时 Python 会话。

```powershell
# R0 完成工具与测试后
.venv\Scripts\python.exe -m pytest tests/test_legal_codec_output_probe.py -q
.venv\Scripts\python.exe workbench/legal_codec_output_probe.py --stage r0 --parent solution.py --cache artifacts/official_eval/cache/qwen2.5-0.5b-proxy-v2.pt --config workbench/legal_codec_output_probe_config.json --output-dir artifacts/proxy_v3/legal-codec-output-20260905/r0 --device cuda
# G0 通过后，将 stage/output-dir 改为 r1；之后严格按 G1 决定 r2-linear/r2-attention。

# R3：先将 parent/candidate 路径写入 run_manifest，以下变量赋值用已验证的实际绝对路径。
# 这两个变量由执行者填写，不代表已经存在候选文件。
.venv\Scripts\python.exe evaluator/eval.py --baseline-solution $parentPath --solution $candidatePath --name legal-output-candidate --linear-only --shards 0,1,2,3,4,5 --cache artifacts/official_eval/cache/qwen2.5-0.5b-proxy-v2.pt --calibration-cache-mode auto --algorithm-device cuda --stop-after-nonpositive 7 --reuse-existing --output-dir artifacts/proxy_v3/legal-codec-output-20260905/r3/in-dist
# Attention 分支改 --attention-only；OOD 在同命令加 --ood 并用独立输出目录。

# R4：全调用图专项计时，输出路径不得覆盖已有报告。
.venv\Scripts\python.exe evaluator/official_eval.py --solution $candidatePath --name legal-output-timing --cache artifacts/official_eval/cache/qwen2.5-0.5b-proxy-v2.pt --cache-mode read --nvfp4-cache-mode auto --algorithm-device cuda --output artifacts/proxy_v3/legal-codec-output-20260905/r4/default-timing.json --report artifacts/proxy_v3/legal-codec-output-20260905/r4/default-timing.md
```

已核对 `eval_system.py`：0 会立即触发早停，并非禁用值；这里显式设 7，高于本轮 6 个 shard
总数，确保完整执行。目录是否创建由工具负责。

## 10. 停止条件、后续方向与交接格式

- R1 无明显差距：关闭本 panel 下 operand 编码改良投入，继续 R2 输出目标。
- R2-L 等价旧失败或未过门、R2-A 无可编译规则：本计划以 `NO_SUPPORTED_DEPLOYABLE_MECHANISM`
  结束，给出合法 witness/反例和适用边界，不写格式极限、不宣称榜首不可能。
- 若输出 oracle 有收益而固定规则无收益，后续研究问题是**如何用当前 API 可见信息预测
  联合量化残差**；若缺口在层级共享，后续问题是**共享分组的联合优化**。这两项仅进入
  evidence-backed backlog，需先与历史 CAT/分组、Fisher/低秩 Gram 对照，不能自行重启。
- 不再安排无明确假设的新榜单探针、宽泛文献搜索、相同父复测或局部参数扫描。

每个阶段结束输出一个 JSON 和 Markdown，至少含：`stage/status/hypothesis/parent_sha/
config_sha/input_identity/expected_cases/executed_cases/failed_cases/metrics/gate/reason/next_step`。
每个阶段更新本计划的状态表；保留原 JSON，修正另写日志。完成后移入 archive，更新 AGENTS、
计划索引、current-solution-status、solutions/README、根 README；运行 `git diff --check`，
提交并 push，核验远端与工作区。只做诊断不分配正式版本，不把 oracle 标成可提交方案。

**给下一位 AI 的启动指令：**读取本计划和审计说明，保持根 v186，先实现 R0 测试及 R1
合法层级精确解。不得直接沿用 cb1/cb2 的 CLOSE_W/NO-GAIN 推论跳过修复，不得提前实现
在线搜索；按 gate 推进到最终输出目标，再决定是否形成单文件候选。
