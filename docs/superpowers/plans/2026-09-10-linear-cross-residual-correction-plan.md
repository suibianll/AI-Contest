# Linear 完整输出交叉残差纠码与双线协调计划

> 状态：ACTIVE，总协调计划，2026-09-10。
> 当前完整根：v202 Linear + v195 Attention，官方 `18053/281s`，SHA256
> `56DC805D6E5A3AEF896DB8021045740292735725D688B48E3D4393E55EFCB2BD`。
> 本计划负责 Linear 执行和两条优化线的统一父版本、版本登记、组合顺序与最终归档；Attention 算法见
> [Attention 联合仿射 gauge 计划](parallel/2026-09-10-attention-joint-affine-gauge-plan.md)。

## 1. 目标与算法判断

上一轮 L-RB1、A-RB1、L-JRB1 已证明：少量层级共享 threshold 要么被自适应 hierarchy 吸收，
要么一次翻转数百万硬码后使局部近似失真。Linear 本轮不再调整共享 threshold、offset、rank-8
残差基或 hierarchy 邻码，而是实现一个固定算法：

**L-XR1：带 Weight 量化交叉项的 rank-4 动态 Activation 输出纠码。**

它处理父编码完成后的少量 mantissa 错码。校准阶段只编译部署 Weight 的输出度量；动态阶段根据当前
样本已经发生的 Activation 量化误差做一次张量化纠码。它与已关闭的静态 Weight 边界、A@W gain、
无结构 Weight 逐码贪心不同，Weight 五字段完全不变。

## 2. 精确目标

设 dense Activation、父解码 Activation、dense Weight、父部署 Weight 分别为
`X`、`X_hat`、`W`、`W_hat`。父 Linear 输出残差为

`R = X_hat W_hat^T - X W^T`。

若只改变 Activation，变化为 `DeltaX`，真实输出平方误差的变化为

`DeltaL = 2 <R W_hat, DeltaX> + tr(DeltaX G DeltaX^T)`，

其中

`G = W_hat^T W_hat`，

`R W_hat = (X_hat-X)G + X C`，

`C = (W_hat-W)^T W_hat`。

旧 Gram-only Activation 精修只使用 `(X_hat-X)G`，忽略了 `X C`，因此无法感知 Weight 量化误差
与 Activation 错误的抵消关系。L-XR1 同时近似 `G` 的块外部分和 `C`，优化目标仍是最终
`XW^T` 输出，而不是 operand MSE。

## 3. 固定实现

### 3.1 校准编译

只处理父状态已有 4×4 `group_gram` 的 `in_features <= 3072` 层；宽层保持父实现：

1. 完成父 Weight 编码，取得最终 `W_hat`；
2. 计算 `G=W_hat^T W_hat` 和 `C=(W_hat-W)^T W_hat`；
3. 保留父已有的 4×4 block-local Gram；
4. 对 `G-blockdiag4(G)` 做对称特征分解，固定保存绝对特征值最大的 4 个方向
   `U_g`、带符号 `lambda_g`；
5. 对 `C` 做固定 rank-4 SVD，保存 `U_c`、`S_c`、`V_c`；
6. 不搜索 rank、阻尼、层名单、覆盖率或接受阈值，不保存完整 `G/C`。

新增状态为上述紧凑因子；Weight 参数、Linear transform、permutation 和父 activation state 的其他字段
保持不变。

### 3.2 动态纠码

1. 完整执行父动态 Activation 编码并解码得到 `X_hat`，计算 `E=X_hat-X`；
2. 一次计算
   `g = E G_local + (E U_g) diag(lambda_g) U_g^T + (X U_c) diag(S_c) V_c^T`；
3. 按父 `gptq_block_order` 的反向顺序处理自然 64 通道块；
4. 每行每个 64 块只比较当前码及固定 `scale_factor/lv2/lv3` 下 mantissa `-1/+1`，候选一次张量化；
5. 用冻结的 `g` 和精确 4×4 局部 Gram 计算二次变化，每行每块最多应用一个损失严格下降的
   4 元素组；
6. 只走一遍，不刷新梯度、不改变 hierarchy、不做 Python 候选轮询；mantissa 为 0 时写规范零 sign。

动态 API 只执行校准编译的低秩规则，不做校准搜索、矩阵分解或完整矩阵求逆。

## 4. 执行步骤

工作目录：`workbench/full_solution/linear-xr1-cross-residual-correction/`。
日志：`logs/execution/2026-09-10-linear-xr1-cross-residual-correction.md`。
输出：`artifacts/proxy_v3/linear-xr1-<run-id>/`。

1. 从本计划启动时记录的完整根复制单文件候选，不修改根和任何归档源码；
2. 做六 API 独立导入、合法 state、finite、父关闭 control、合成 reachability；
3. 记录各层 `G/C` rank-4 重建误差以及完整校准 `DeltaL`，只用于解释近似误差；
4. 运行 Linear shard0 排除接口错误和死分支；
5. 只要真实调用形成合法、非等价硬输出，即固定运行 Linear 六 shard一次；本地正负和时间只记录；
6. 保存源码、配置、SHA、attempted/accepted、五字段 changed count、336 case 配对结果和单文件导入结果；
7. 将唯一代表候选交官方裁决。只有官方分数提高且时间 `<300s` 才晋级。

无真实码变化记 `NO_REACHABILITY`；官方负向或超时关闭 L-XR1，不缩 rank、不减少覆盖、不改邻码范围
重试。

## 5. 与 Attention 计划的协调

### 5.1 开发隔离

- 两条线都冻结同一份启动根 `R0`；任何一边先完成都不得改变另一边的源码父或结果口径；
- Linear 只修改 `hif4_calibration_and_quantize_weight`、`hif4_dynamic_quantize_activation` 及其私有 helper；
- Attention 只修改 calibration attention、动态 Q/K 及其私有 helper，V 保持父实现；
- 两条线使用独立 workbench、日志、artifact 和校准缓存；单张 GPU 上评测串行，禁止同时跑 4B；
- Linear 候选必须证明 Attention 三 API 在固定输入上与 R0 一致；Attention 候选必须证明 Linear 两 API
  与 R0 一致。

### 5.2 官方定价与组合

两个单侧候选都从同一完整 R0 构建，并分别作为完整六 API 文件提交，不建立 Linear/Attention 侧父：

| Linear 官方结果 | Attention 官方结果 | 动作 |
|---|---|---|
| 非正向 | 非正向 | 两个机制分别关闭，根保持 R0 |
| 正向 | 非正向 | L-XR1 成为新完整根，Attention 机制关闭 |
| 非正向 | 正向 | A-G1 成为新完整根，Linear 机制关闭 |
| 都正向 | 都正向 | 先选官方分更高的单机制完整候选为父，再在该源码上重新应用另一机制，构造一个组合候选 |

事实（2026-09-10）：A-G1 已本地关闭为 `REJECTED`（v227，六 shard 等权 `-0.005294`，未提交官方，
用户将统一做官方评测），根保持 R0；L-XR1 线不受影响，组合路径待 A-G1 官方回传后按上表裁决。
Attention 后续卡由新附录
[A-QB1](parallel/2026-09-10-attention-qk-logit-bias-plan.md) 承接，归属本计划统一登记。

组合候选不是复制粘贴两个归档文件。它必须从较高分完整父重新构建，重算全部 calibration state，先做
目标两侧 control，再运行一次 `--scenario both` interaction audit，最后交官方。组合是否晋级仍只看
完整官方分数和 `<300s`；不得用两个单候选分数相加预测组合收益。

官方结果回传可能异步，但版本号、归档和根切换由本计划串行登记，避免两个执行线抢占同一版本或覆盖
`solution.py`。

## 6. 缓存、提交与结束

- 两条线存活期间，缓存清理同时保留 R0、L-XR1 和 A-G1 前缀，并使用 `--min-age-hours 2`；绝不删除
  `qwen3.5-4b-proxy-v2.pt`；
- 每个机制只有一个固定配置和一个正式代表，不按本地 shard 结果扫描参数邻域；
- 本地 paired、holdout、API 时间均为诊断，不是提交门；官方分数与 `<300s` 是唯一晋级依据；
- 实质实现和结果分别提交，归档目录按实际结果标记，拒绝候选名包含 `rejected`；
- Linear 与 Attention 单机制均获裁决，且必要的组合候选完成裁决后，本计划结束并移入 archive。
