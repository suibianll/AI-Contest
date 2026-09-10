# Attention 一般非对称互逆矩阵残差计划（A-GR1）

> 状态：ACTIVE 执行附录，2026-09-10。
> 从属于当前活动总计划。当前完整根为 v230 Linear (L-EM2) + v195 Attention，
> 官方 `18428/292s`，SHA256 `0F1AF6DBC207FF32…`。本文件只负责 Attention A-GR1；
> 版本登记、组合与根切换由总协调线处理。
> 前序 Attention 卡：A-G1（v227 本地 REJECTED）、A-QB1（v228）、
> A-MC1（v229 官方侧 −2，K 平移类关闭）、A-QC1、A-FIX1（v230 Attention 官方 TIMEOUT）。

## 1. 定位与证据

v192（全对称零迹矩阵残差 S，Q@exp(S)、K@exp(−S)）是互逆残差族唯一官方正向成员：
标准 Linear 侧隔离官方 **+22**（14427/272s），完整包 TIMEOUT。表达力梯度证据：
对角（v190）=0、块三角（v191）=0、全矩阵（v192）=+22。

**A-GR1 是 v192 的单变量推广**：把对称零迹 S（exp(S) 必然正定）推广为一般矩阵
`M = I + N`（N 无约束），K 侧用精确求逆的 `M^{-T}`。其余全部保持 v192 的实现形态
（窗口、步数、lr、gate 口径都不动）。用户已确认此"实质差异"定位。

与 v227（A-G1）的差异：v227 是逐通道对角 gauge 且在 A2 循环内联合训练（重训
rotation，层 15 翻车）；A-GR1 是全矩阵互逆残差，冻结根全部已有 state，在根最终
部署坐标上叠加，rotation/center 只被复合编译、不被重训。

## 2. 机制定义与合法性

- 作用点与 v192 相同：根最终部署坐标（learned rotation + K center 之后、
  `_dense_to_hif4` 之前）。为保持精确 logit 不变性，`M^{-T}` 作用于加 center 后的
  K，编译为 `R_k' = R_k @ M^{-T}`、`c' = c @ M^{-T}`（v192 同款 center 同步编译）。
- 粒度照抄 v192：每个 KV group 一个 `head_dim × head_dim` 矩阵。
- 校准结束时精确求逆 `M^{-T}` 并与根 rotation 复合存入既有
  `learned_rotation`/`learned_center` 字段——**动态 API 只求 matmul、严禁求逆**
  （v192 的动态 Q@exp(S) 是合法先例）。不新增 state 字段类型，五字段格式不变。
- Q 侧乘 M、K 侧乘 M^{-T}：logits 精确不变，收益只来自量化输入分布的移动。

## 3. 固定算法（镜像 v192，无扫描）

1. 从当前根逐位复制候选，追加 A-GR1 段；冻结根全部 state。
2. 训练：fit windows 0,1,2；32 步 Adam（lr 0.01、β 0.9/0.999、clip 1.0、
   正则 1e-3）；目标为 v192 同款逐 64 块 amax 尺度比损失；STE 反向，
   N 的解析梯度 `grad_N = G_q − P·G_k^T·P`（P = M^{-T}，已经有限差分验证）；
   每步结束把 M 的奇异值钳到 `[1/√2, √2]`（对应 v192 的谱界 ±log(2)/2）。
3. gate：与 v192 相同的逐层真实部署路径 MSE 门，gate windows 3,4 全部严格改善
   才 arm，否则该层保持父。记录 attempted/accepted。
4. 实现差异声明：训练设备用 `cuda if available`（对齐当前根 A2 trainer；
   v192 的 CPU 训练是其完整包 TIMEOUT 的因素之一），不改变数学定义。

## 4. Control（全部通过并留证）

1. 0 训练步：Q/K/V 五字段与最终输出与根逐位一致；
2. 互逆性：编译对 `(tq, tk)` 满足 `tq @ tk^T == rq @ rk^T` 到 fp 精度；
   合成输入下量化前 dense logits 不变（<1e-6）；
3. 六 API 脱离仓库独立导入；`validate_state` 通过；
4. V/Linear 与根逐位不变；训练分支真实可达（attempted>0、训练损失下降）。

## 5. 执行步骤

工作目录：`workbench/full_solution/attention-agr1-general-reciprocal/`。
日志：`logs/execution/2026-09-10-attention-agr1-general-reciprocal.md`。
输出：`artifacts/proxy_v3/attention-agr1-<run>/`。

GPU 串行（<2000 MiB 才启动，后台轮询等待）：先 shard0 排接口错误，再
`--shards 0,1,2,3,4,5 --stop-after-nonpositive 6` 跑满六 shard；本地数值只作诊断。
侧隔离探针：`standard-linear_v233-attn`（build + verify + 探针 vs 归档候选
attention-only 六 shard 逐位配对）。

## 6. 完成条件

- 72 case 与根逐位相同或六层全 parent：`NO_EFFECT`，不占版本号，只写日志；
- 否则归档 v233（官方 `unregistered/NA`，用户统一评测）；
- 官方 TIMEOUT 或负向：只关闭该实现，不以缩步/缩窗/改 lr 重试；
- 完成后本文件归档。

## 7. 执行结果

- 候选 SHA256 `4f27fb59252de95c7d8a65db9d742af1bc2ce16cd7ca513d49e8fbc7813a9267`（根纯追加）。
- Control 全过：0 步逐位恢复根；互逆性 fp64 2.34e-7；量化前 dense logits 不变 1.3e-8；
  六 API 独立导入；`validate_state` 通过；V/Linear 逐位不变；训练分支可达
  （合成 8/8 损失 2.0→~1.21；真实 4B 六层 attempted 6/6，损失 2.0→0.87~1.12）。
  过程缺陷：初版 K 侧梯度缺转置（`−M⁻¹G_kM⁻¹`），被 control 的损失上升抓到，
  有限差分验证后修正为 `G_q − P·G_k^T·P`（worst relerr 5.3e-8）。
- 六 shard（2026-09-10，`attention-agr1-sixshard-full-20260910`）：等权 mean
  **+0.003845**（21/3/48）；层15 +0.017449（10/2/0）、层22 +0.005618（11/1/0）接受，
  层 0/1/5/8 gate parent（层 0 gate 均值改善但非全窗口）；非 NO_EFFECT。
- shard0 接口检查 ok；api total delta 约 +5.5s/六 shard（诊断）。
- 侧隔离探针 `standard-linear_v234-attn`：build+verify PASS；与归档候选六 shard
  72/72 精确零（逐位一致）。
- 归档 v234（v233 被 Linear L-TF1 占用，按规则取下一空闲号）：
  `solutions/20260910_v234_attention-agr1-general-reciprocal_scoreNA_timeNA/`，
  官方 **unregistered/NA**，待用户统一评测。
- 执行日志：`logs/execution/2026-09-10-attention-agr1-general-reciprocal.md`。
