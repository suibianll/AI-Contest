# L 侧背景论文检索去重：2026-09-07 第二批

> 侧：Linear。目的：在 L1-L3、fc/proj 解析旋转全部关闭后，检索是否有
> 非关闭族、可部署（合法五字段/单文件/时间预算/冻结侧）的新机制候选。
> 本轮为检索+公式级去重，不产生候选、不运行评测。

## 检索到的候选与逐项去重

| 论文/方法 | 核心机制 | 与已关闭/已测空间的关系 | 去重结论 |
|---|---|---|---|
| WUSH（ISTA, 2512.00956） | 闭式最优线性块变换 = Hadamard backbone + 数据依赖二阶矩收缩，非正交变换，FP/INT 均覆盖 | 本质 = 随机 Hadamard（已有关 block-smooth/CAT64/perm）+ 逐通道对角收缩（已有 smooth scale）。是既有 CAT64+smooth 组合的解析化，无新自由度；广义仍属坐标变换族（AGENTS §7 Householder 关闭邻域） | **DUPLICATE**（现有 CAT64/block-smooth/smooth 已覆盖） |
| D²Quant（2026-02） | 针对 down-projection 的 Dual-Scale Quantizer（可吸收辅助缩放）+ LayerNorm 均值漂移修正（DAC） | DSQ 属 scale 参数家族（v156 stored-scale 官方 REJECTED、scale 邻域关闭）；DAC 需逐层输出 bias 修正，与 eval-v3"每层激活为真实前向捕获、无跨层传播"协议不兼容 | **DUPLICATE / NOT MIGRATABLE** |
| WaterSIC（MIT MEng thesis, 2026-05） | rate-distortion 视角：per-column 比特分配 + attention-weighted 校准混合 | per-column 变比特需自定义码本/表单，HiF4 五字段每 64 块固定比特 → 格式不兼容；其"协方差条件数大时 GPTQ gap 大"的洞见已被 block-smooth/actorder 的 conditioning 改善覆盖 | **NOT MIGRATABLE**（五字段固定比特） |
| SliderQuant（ICLR 2026?） | 层间滑动窗口（浅层渐扩/深层渐缩）+ 层内增量量化 | 层间差异路由 = 增加 layer 专属处理，AGENTS §3 禁止 layer 专属路由；且格式要求层间变精度 | **NOT MIGRATABLE** |
| LoRDS（PKU, 2601.22716） | 连续低秩缩放 manifold S=BA 替代块量化 scale | 部署 scale 必须为合法 E6M2 码，低秩连续缩放不合法；与 v156 stored-scale（官方拒绝）同族 | **DUPLICATE**（stored-scale 已关） |
| DFRot（COLM 2025） | 旋转矩阵 Procrustes 精化（Hadamard/正交） | 旋转/正交族，Householder 与 fc/proj 解析 T 均关闭 | **DUPLICATE**（旋转族已关） |
| RoLoRA（2407.08044） | 旋转 + 微调 | 旋转族 + 需训练 | **DUPLICATE** |
| AQAS/SLAC/dINT（2311.05161） | 联合 scale、校准长度感知、混合格式 | 校准长度与协议固定；dINT 改格式；AQAS=scale 族 | **NOT MIGRATABLE / DUPLICATE** |

## 结论

本轮检索未发现可迁移的非关闭族新机制：

1. **所有"闭式/解析变换"方法（WUSH/DFRot）** 都落在既有 CAT64/block-smooth/smooth
   组合或旋转族（已关闭）内，无新自由度。
2. **所有"变精度/变比特分配"方法（WaterSIC/SliderQuant）** 与 HiF4 固定
   五字段、每 64 块固定比特的合法格式不兼容，且层间路由被 AGENTS 禁止。
3. **所有"scale 重构/低秩缩放"方法（LoRDS/D²Quant-DSQ）** 与 stored-scale
   官方 REJECTED 同族关闭。
4. **bias/均值修正（DAC）** 与 eval-v3 真实前向捕获协议不兼容。

## 更新后的关闭边界清单（本侧）

- 坐标变换族（Householder、block-smooth 邻域、CAT64 邻域、FlatQuant 训练式
  COST_HOLD、解析特征基/Kronecker T 负向）——全部关闭/无余量。
- scale 族（stored-scale、dual-scale、低秩缩放）——官方/本地关闭。
- 变比特/变精度表——格式不兼容；层间差路由——AGENTS 禁止。
- JDRQ/GPTAQ、合法编码搜索、块序族、importance 槽位——已关。
- bias/均值修正——评测协议不兼容。

结论：L 侧在已关闭边界内无可识别的新机制候选，维持 NEEDS_NEW_HYPOTHESIS；
除非用户重开某一边界或提供新官方线索。父 L4 不变，无候选超过本地最高 0.6368。