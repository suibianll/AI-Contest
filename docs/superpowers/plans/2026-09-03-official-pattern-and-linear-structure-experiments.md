# Linear 快速结构验证计划

> 状态：**ACTIVE**
>
> 更新：2026-09-03
>
> 官方父版本：v160，`17532 / 232s`，归档源码 SHA256
> `33B1D061CE6BFCD92659C597BE4830BB9B910E646FF518433DA67B925AE8680D`
>
> 所有候选从 v160 归档源码分支；父版本 JSON 固定复用，不重复运行父版本。

## 1. 目标与边界

本阶段只回答一个问题：统一、低自由度的 64-block Householder 坐标重分布，能否在不同
shape、holdout 和模型上稳定改善最终 `Q(A)Q(W)^T`，同时不显著增加时间。

- Linear×Attention 官方 2×2 只提供交互信息，不直接优化分数，停止执行；
- 官方稳定性已确认，不重复提交相同 SHA；
- 逐位等价的官方时间 A/B 停止执行；时间只作为算法候选的附带门禁；
- Attention 固定为 v160，不开发新门控；
- 禁止 seed/alpha/rank/block-size/offset sweep，以及模型、layer、role 专属路由；
- 固定机制只做一个候选，失败后不围绕参数调优。

## 2. 已知官方锚点

| 版本 | 主要变化 | 官方分数/时间 |
| --- | --- | ---: |
| v86 | 安全基线 | `16744 / 222.7s` |
| v158 | v86 Linear + Matrix-Smooth Attention | `16861 / 223s` |
| v160 | v159 GPTQ Linear + v158 Attention | `17532 / 232s` |

v155/v156 的 `10^-4` 级局部收益没有迁移；v158 的广覆盖 Attention 变化得到 `+117`，v159
的大范围 Linear 变化得到 `+671`。因此候选必须呈现跨 shape、跨 holdout、跨模型的广覆盖收益，
不能依赖 aggregate mean 或少数正 case。

## 3. 快速验证漏斗

父版本结果直接读取 immutable JSON；每个候选只运行候选侧，按失败即停执行：

| 阶段 | 内容 | 时间预算 | 停止条件 |
| --- | --- | ---: | --- |
| A | 单文件导入、六 API、reachability、连续域乘积不变量 | `≤30s` | 接口、调用或相对误差失败 |
| B | Qwen Linear compact 56 case + Attention 4-case control，使用 NVFP4 cache | `60–75s` | 任一 Qwen compact 门禁失败 |
| C | GPT-2 compact Linear + OPT-125m compact Linear，均读取现有 dense cache | `90–120s` | 任一模型整体负向或结构异号 |
| D | 仅一次 Qwen Linear default 168 case，不跑完整 Attention | `255–300s` | default 分布或时间门禁失败 |
| E | JSON replay、单文件隔离导入、报告与 SHA | `≤60s` | 证据不完整 |

时间依据是当前同机实测：Qwen Linear compact wall `52.2–56.0s`，Qwen Linear default wall
`254.1s`，GPT-2/OPT 完整集成各约 `121–123s`；compact 跨模型预算按各 `45–60s` 保守估计。

- Qwen compact 即失败：约 `1–2 分钟`结束；
- 到跨模型后失败：约 `3–4 分钟`结束；
- 全部门禁通过：代码完成后约 `8–10 分钟`完成本地验证；
- 首个候选需一次性生成 GPT-2/OPT compact parent JSON，额外约 `2 分钟`；后续候选直接复用；
- 不运行每个模型的 full/default，也不重复运行 parent。

### 3.1 Qwen compact 门禁

- paired mean delta `> 0`，median delta `>= 0`；
- 正 case 多于负 case；
- qkv、o、fc、proj 各 family mean delta 均不为负；
- worst-quartile mean delta `>= -1e-3`；
- 非零变化覆盖至少 25% case；
- Attention 4-case control 与 v160 逐位一致。

### 3.2 跨模型 compact 门禁

- GPT-2 和 OPT 各自 paired mean/median delta 不为负，正 case 不少于负 case；
- shape-family 主效应与 Qwen 同号；
- W-only、A-only、Both 和 interaction 的主来源不能在不同模型间反转；
- 任一模型整体负向即 `REJECTED`，不靠另一模型均值抵消。

### 3.3 Qwen default 与复杂度门禁

- 正 case 多于负 case，validation/test 成对同号率 `>=75%`；
- 各 role family mean delta 不为负；
- 未修改的 Attention 通过 compact control 即可，不再跑 120-case default；
- Linear API total 不超过同机 v160 Linear parent `231.4s + 30s`；
- dynamic 路径不允许 Python token/block 循环，六 API 调用数不变。

## 4. 唯一算法候选：统一 64-block Householder

在 v159 已选定的 smooth/permutation/Hadamard 坐标之后，对每个连续 64-channel block：

```text
C_A = X_b^T X_b / trace(X_b^T X_b)
C_W = W_b^T W_b / trace(W_b^T W_b)
C   = 0.5 * (C_A + C_W)
```

从固定向量 `ones/sqrt(64)` 开始做恰好 4 次 power iteration 得主方向 `u`；目标均衡向量
`t = sign(u)/sqrt(64)`，零符号按 +1；令 `v=(u-t)/||u-t||`，构造
`H=I-2vv^T`。若分母小于固定 epsilon，该 block 使用 identity。

- Weight：`W_b' = W_b H`；
- Activation：`X_b' = X_b H`；
- 因 H 正交且对称，`X_b'(W_b')^T = X_b W_b^T`；
- Weight Hessian 在最终坐标使用 `H^T G_b H`；
- Activation Gram 从最终量化 Weight 计算；
- state 每 block 只保存一个 64 向量 `v`；dynamic 使用一次向量化
  `x - 2(x·v)v`。

所有 layer/role 共用同一规则；不增加 parent/proposal 双搜索，也不设置数据依赖 gate。

## 5. 结果解释与停止条件

| 结果 | 决定 |
| --- | --- |
| Qwen compact 失败 | 立即拒绝，耗时控制在 2 分钟内 |
| Qwen 通过但 GPT-2/OPT 失败 | 标记模型特化并拒绝，不跑 default |
| 三模型 compact 通过但 Qwen default 失败 | 拒绝，不提交官方 |
| 全部门禁通过 | 生成唯一正式候选，交给用户进行一次官方评测 |
| 官方零或负向 | 关闭该 Householder 家族，不反向调参数 |
| 官方正向 | 下一计划只研究同一低秩解析族，不用该结果搜索参数 |

实现、验证、归档分开计时。本地验证的目标是快速否定错误机制，不把本地 proxy 换算为官方分数。
