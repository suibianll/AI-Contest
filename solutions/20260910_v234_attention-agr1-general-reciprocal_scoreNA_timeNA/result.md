# v234 Attention A-GR1：一般非对称互逆矩阵残差

> 归档：2026-09-10。官方状态：**完整包 TIMEOUT（>300s），REJECTED**；侧隔离官方正向 +29（见文末）。
> 父：v230 Linear (L-EM2) + v195 Attention 完整根，官方 `18428/292s`，
> SHA256 `0f1af6dbc207ff32b2c6be16987e9c4fe50f3f10747de26782ef52a6f2fab7bc`。
> 候选 SHA256 `4f27fb59252de95c7d8a65db9d742af1bc2ce16cd7ca513d49e8fbc7813a9267`
> （与本地评测所用候选逐位一致）。
> 计划卡：`docs/superpowers/plans/parallel/2026-09-10-attention-general-reciprocal-plan.md`。

## 机制

v192（互逆残差族唯一官方正向成员，侧隔离 +22）的单变量推广：对称零迹 S
（exp(S) 必然正定）→ 一般矩阵 M = I + N（N 无约束），Q 侧乘 M、K 侧乘精确求逆的
M^{-T}。fit windows 0,1,2 / gate windows 3,4、32 步 Adam（lr 0.01、clip 1.0、
reg 1e-3）、逐 64 块 amax 尺度比目标、逐层"全部 gate 窗口严格改善才 arm"全部
与 v192 相同；每步结束 M 奇异值钳到 [1/√2, √2]（v192 谱界 ±log(2)/2 的对应）。
编译进根既有 `learned_rotation`/`learned_center` 字段（含 center 同步编译），
动态路径只有一次 matmul、无求逆。根全部已有 state 冻结（不重训 rotation/center）。

## Control（全部通过，`workbench/full_solution/attention-agr1-general-reciprocal/`）

1. 0 训练步：Q/K/V 五字段与最终输出与根逐位一致（gate 自动判 parent）。
2. 互逆性：编译对 `tq @ tk^T` vs `rq @ rk^T` 最大偏差 2.3e-7（fp64）；
   合成输入量化前 dense logits 最大偏差 1.3e-8（<1e-6）。
3. 六 API 脱离仓库独立导入通过；`validate_state`/`validate_hif4_params` 通过。
4. V/Linear 与根逐位不变；训练分支可达：8/8 合成种子 attempted=1，
   训练损失 2.0 → ~1.21（解析梯度 `grad_N = G_q − P·G_k^T·P` 已经
   有限差分验证，worst relerr 5.3e-8——初版梯度公式缺转置，FD 抓到后修正）。

## 真实 4B 校准审计（6 个 full-attention 层）

| 层 | 根 a2 arm | agr1 arm | 训练损失 2.0→ | gate 父 | gate 候选 |
|---|---|---|---|---|---|
| 0 | rotation | parent | 0.979 | 4.0923e-4 | 4.0530e-4（均值改善但非全窗口） |
| 1 | rotation | parent | 0.954 | 1.6597e-3 | 1.7692e-3 |
| 5 | rotation | parent | 0.919 | 1.2467e-3 | 1.2890e-3 |
| 8 | identity | parent | 0.913 | 3.6691e-3 | 3.8162e-3 |
| 15 | rotation | **accepted** | 0.869 | 2.2210e-3 | 2.1643e-3（−2.55%） |
| 22 | rotation | **accepted** | 1.122 | 1.4342e-2 | 1.4270e-2（−0.50%） |

attempted 6/6、accepted 2/6；inverse_error ~1e-6（fp32 matmul 精度）。

## 本地六 shard（诊断口径，不换算官方分；eval-v3，proxy-v2 dense cache，CUDA）

| shard | 层 | delta_mean | +/-/0 |
|---|---|---|---|
| 0 | 0 | 0.0 | 0/0/12 |
| 1 | 1 | 0.0 | 0/0/12 |
| 2 | 8 | 0.0 | 0/0/12 |
| 3 | 15 | **+0.017449** | 10/2/0 |
| 4 | 22 | **+0.005618** | 11/1/0 |
| 5 | 5 | 0.0 | 0/0/12 |

等权 mean **+0.003845**（21/3/48，72 case）。非逐位相同、非 NO_EFFECT。
shard0 先行接口检查：status ok、reasonableness_issues 0。
Attention 侧 api total delta 约 +5.5s/六 shard（缓存口径混合，仅记录；
注意根官方余量仅 8s，v229/v230-attn 均有 TIMEOUT 先例）。

## 官方状态

**完整包 TIMEOUT（>300s），REJECTED**（2026-09-10 用户回传"v234官方也超时了"）。
v230 根余量只有 8s，而 A-GR1 的额外时间在**校准侧**（6 层 × 32 步 Adam；部署动态路径无新增），
是机制自带代价，进不了限。精确秒数/分数未知，不写预测。**不缩步/缩窗/减候选重试**。
见 `logs/execution/2026-09-10-v234-agr1-official-timeout.md`。

**侧隔离官方读数 `14455/263.7s` 仍成立**（2026-09-10 用户回传），相对同口径 v195 侧基准
`14426/243s` 为 **+29/+20.7s**（相对 R3 基线 14405 为 +50）。它测的是**分数**、且是更小的包；
本次超时关闭的是完整包可落地性，**不推翻**这个分数声明。A-GR1 为继 C76.4（+84）、A1（+60）后
Attention 侧第三大官方正向机制；表达力梯度（对角 0 < 三角 0 < 对称全矩阵 +22 < 一般矩阵 +29）
获官方确认。所以本卡结论是**"机制有分、代价不可落地"**，不是"机制无效"。
见 `logs/execution/2026-09-10-v234-agr1-side-official.md`。

**对 v236（A-GR1 重挂 v231 根）**：v236 的唯一改动是父替换，而校准侧代价基本不随 Linear 父变化、
v231 只快 1s，所以本次超时直接落在 v236 上；其预登记晋级规则的超时分支已被预先回答。
秒数未知不写预测，v236 的登记由该卡执行者串行处理。
