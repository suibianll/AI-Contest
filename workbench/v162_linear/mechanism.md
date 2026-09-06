# L 侧机制卡（v162 独立分支）

> 侧：Linear（L）。日期：2026-09-06。契约：
> [总计划](../../docs/superpowers/plans/2026-09-06-v162-independent-linear-attention-plan.md) +
> [L 任务书](../../docs/superpowers/plans/workpackages/v162-linear.md)。
> 零点 v162 SHA `56101559D267D962084CD67A9F9AF8EB924501B17AB408EAF676081876CC000A`。

## L1 机制：迁入 v160 Linear 校准优化栈（RECOVERY，工程基线）

### 1. 从 v162 缺少的哪个明确机制切入

v162 六 API 全部是标准 HiF4 codec（NVFP4→BF16 中间解码→标准编码），state 为空，
Linear/Attention 均无任何校准学习。本侧第一步迁入的唯一机制单元是 **v160 Linear
校准优化栈整体**：等价坐标变换搜索（含 SmoothQuant 型 scale、坐标置换、块内
Hadamard 类变换）+ 变换后协方差驱动的 Weight GPTQ + 部署 Gram 驱动的 Activation
GPTQ + quadratic AdaRound / E6M2 offset / data-driven refinement。

- 来源与官方锚点：`solutions/20260903_v163_v160-linear_standard-attn_scoreNA_timeNA/solution.py`
  （SHA `见 config.json`），即 "v160 Linear + 标准 Attention" 的侧隔离包，官方
  **4587 / 202s**；相对 v162 零点贡献 `C_L = 4587 − 1001 = 3586`。
- 定性：**RECOVERY**（已验证基础机制重现，作为本侧学习/工程基准），不宣称新突破。
- 与历史关闭边界去重：
  - 不是 v189 块序邻域重跑——本迁移不含任何 64-block 块序机制，块序族后续版本才单独处理；
  - 不是 Householder/full64/cross-fold minimax/JDRQ/条件曲率/cross-block Hessian——
    这些是后来在完整父上被否定或关闭的其他族，与本栈不同源；
  - ROAB 结论针对 v157 在 exact-v86 上的新增，不是本栈组成；本栈以 v160 归档源码
    原样迁移，不重调任何参数。

### 2. 变量、乘积不变性与最终解码路径

- 变量：Weight 校准期学习每层每 role 的等价变换与量化参数（scale/lv2/lv3/mantissa/sign
  五字段 + 变换坐标），存入 `activation_state` / weight state；Activation 动态侧用
  校准得到的部署坐标统计做 output-aware 编码。
- 连续变换保持 `XW^T`：所有坐标变换成对可逆（`X→XR, W→WR^{-T}` 型），仅在校准期
  学习、部署期固定；最终解码路径仍为标准五字段 HiF4 合法输出。
- 在线复杂度：`hif4_dynamic_quantize_activation` 保持 O(块) 编码 + 校准期统计使用，
  无候选循环、无 Gram contraction（v165 边界由后续版本自行遵守，本版动态 API 与
  v160 相同，已被 v163 官方 202s 验证时间可行）。

### 3. 校准如何学习、真实 X/W interaction 如何进入最终输出损失

Weight GPTQ 使用变换后真实校准激活的二阶统计（协方差/Hessian）；Activation 侧使用
部署权重输出 Gram。目标即最终输出误差 `XW^T vs Q(XR)Q(WR^{-T})^T` 的部署坐标形式，
与 L2 任务书要求一致（W-only 胜出不能掩盖 X/W interaction，本栈的 deployment Gram
结构直接以输出为目标）。

### 4. API 插入点、成本、候选数量与停止门

- 插入点：`hif4_calibration_and_quantize_weight`（全部校准搜索）、
  `hif4_dynamic_quantize_activation`（部署 Gram 编码）；四个 Attention API 冻结
  v162 standard，state 为空。
- 校准成本：v163 官方 202s 含完整栈校准，时间门 `<280s` 有 78s 余量（官方实测，
  不再用本地外推）。
- 候选数量：1（本卡唯一预注册配置，见 config.json）；失败换机制，不扫邻域。
- 停止门：按总计划 §4——合法性、finite、case 身份、冻结侧逐位 control、
  `L1_negative<0.02`、独立 validation/test mean 均正、OOD 成对风险记录（不作否决门）、时间预测
  `<280s`。

### 5. 校准/选择/holdout 拆分

与 evaluator 协议一致：每 `(layer,role)` 用固定 calibration folds（proxy-v2 面板
`[10,128]` 两折）学习，validation/test 交替 holdout 独立验证；不临时改 fold/seed/
role 路由。v160 栈本身的多折聚合规则随源码冻结迁移，不重新选择。

## 后续版本阶梯（预注册顺序，每版一个机制）

1. **L2（RECOVERY）**：rank-2 残差重分布（v182 Linear 侧，官方 step_gain +1 锚）。
2. **L3（RECOVERY）**：v189 静态部署 Hessian activation-GPTQ 64-block 块序（不含
   +4 码窗，用于单机制归因）。
3. **L4（RECOVERY，阶梯终点）**：+ `_DYNAMIC_OFFSETS` 增加 +4 码（v186 常量；
   本侧活动 Attention 路径不读该常量，只影响 Linear 动态编码）。L4 = v189
   Linear 侧的逐字节精确复现（结构验证：diff(L4, v189) 全部 hunk 仅落在
   attention 区与标准块尾部）。官方锚 v189 17616/275s；eval-v3 参考0.6368。
4. **L5+（新机制）**：仅在完成上述 RECOVERY 阶梯后提出，必须相对强对照
   v166（SHA `9C0EAC6A...B4646`）有材料进展（研究目标 `D_strong≥20%`），且不在
   AGENTS.md §7 已关闭族内。
