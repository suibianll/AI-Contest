# C40：稳健 Block-LDLQ 跨块条件重求解

## 结论

C40 在 C39-FW 的相同层范围和相同 25% FULL64 预算下，将独立 64 维块求解升级为
相邻 128 维超块内的 Block-LDLQ 条件重求解。六个本地场景全部提升，Linear 平均增益
为 `+0.14pp～+0.60pp`，Attention 路径完全不变。

这不是覆盖率、阈值或拦截策略实验。C39 与 C40 的激活算法、层范围和预算保持一致，
唯一实验变量是权重量化求解器是否利用相邻 64 维块的 Hessian 非对角耦合，并在条件目标
变化后重新搜索完整 HiF4 层级参数。

## 算法

对变换后的静态权重 `W`、其 HiF4 重建 `Q` 和仅由校准激活计算的 Hessian

```text
H = AᵀA / N
```

最小化合法的权重目标：

```text
L(Q) = tr((Q-W) H (Q-W)ᵀ)
```

将相邻两个 64 维块组成 128 维超块。固定另一块误差 `E_o = Q_o-W_o` 时，当前块
`b` 的条件最优目标可写为：

```text
W'_b = W_b - E_o H_ob (H_bb + λI)⁻¹
```

随后不是只在原固定尺度网格上修改 mantissa，而是调用完整 FULL64 求解器重新量化
`W'_b`：

1. 对 E6M2 block scale 做多候选 beam 搜索；
2. 精确初始化 lv2/lv3 层级；
3. 使用 damped inverse-Hessian GPTQ 顺序初始化；
4. 做完整 64×64 Hessian 坐标下降；
5. 重新搜索 lv2/lv3 位翻转；
6. 将候选还原到标准五字段 HiF4 表示。

每个 128 维超块按 `block 0 → block 1` 做一次条件迭代。只有当候选同时降低以下两个
目标时才提交：

```text
pooled_loss = L_mean(Q)
robust_loss = mean(L_f) + 0.5 × (max(L_f) - mean(L_f))
```

其中每个 `H_f` 由不同校准激活项的 1024 个采样 token 独立计算。独立折不参与
Linear 输出拟合，只用于防止单一 pooled Hessian 的窗口过拟合。

## 与 C39 的受控关系

- 仅处理 C39 已启用 FULL64 的 wide FFN `fc/proj` 层；
- 仍使用 `max_ratio=0.25`；
- q/k/v/o、动态激活量化和 Attention 全部不变；
- 激活重要性继续使用 Block-LDLQ 前的 C39 权重重建，避免在同一候选中混入第二机制；
- C39 归档源码不变，可随时按哈希回退。

开发过程中曾发现按输入宽度判断会让 `fc(768→3072)`错误获得 100% 覆盖。该结果虽
提升更大，但混入覆盖扩张，已明确废弃。以下全部数据来自修正后的同预算版本。

## 本地结果

| 场景 | C39-FW | C40 | 增益 |
|---|---:|---:|---:|
| amax6 / offset 0 | 0.5357 | 0.5393 | +0.36pp |
| amax6 / offset 97 | 0.5213 | 0.5248 | +0.35pp |
| amax6 / offset 193 | 0.5385 | 0.5445 | +0.60pp |
| amax6 / offset 389 | 0.5312 | 0.5368 | +0.56pp |
| amax4 / offset 0 | 0.4740 | 0.4754 | +0.14pp |
| pow2 / offset 0 | 0.5521 | 0.5550 | +0.29pp |

默认场景中，`fc` 从 `0.4931` 提升到 `0.5058`，`proj` 从 `0.4152` 提升到
`0.4237`；q/k/v/o 数值与 C39 完全一致。默认 causal Attention 保持 `0.4497`。

同机 CUDA 配对计时：

| 版本 | algorithm-stage |
|---|---:|
| C39-FW | 27.54s |
| C40 | 45.32s |
| 增量 | +17.78s |

C40 的 CPU 全模型 algorithm-stage 为 `100.05s`。以 C39 官方 `159.2s` 为锚点，
即使将本地新增耗时放大 3 倍，估计总时间约为 `212.5s`，仍低于官方确定的 `<300s`
限制；最终仍须以官方返回时间为准。

## 官方规则合规性

C40 不计算任何形式的 `A@W`，不构造 `[tokens, out_features]` 的 Linear 参考输出，
不计算输出 residual，也不利用此类信息拟合或倒推 `Q(A)`。

允许的数据流只有：

```text
A_f → transform(A_f) → H_f = A_fᵀA_f/N
(W, Q(W), H_f) → weight-only Block-LDLQ → Q(W)
```

`H_f` 只影响静态 `Q(W)`。C40 生成 `Q(A)` 的状态沿用 C39 路径，并且刻意冻结为
Block-LDLQ 前的权重重要性。静态 AST 守卫、运行时 taint 守卫和完整测试均通过。

## 验证与提交决策

- 完整测试：`66 passed`；
- 六项 Linear 稳健性矩阵：`6/6` 正增益；
- Attention：无行为变化；
- CPU algorithm-stage：`100.05s < 300s`；
- 当前源码 SHA256：
  `D24BC94F513907CBE97B43865973D1498133D8B9264FAF12661836FF65AAB656`。

决定：将 C40 作为下一官方校准候选提交。若官方分数提升，则下一步继续研究跨越固定
相邻配对的全局 block schedule 或 DHSS scale decoding；若官方回落，则优先消融独立折
稳健目标与条件 full-hierarchy re-solve，不通过调接受阈值掩盖失败。
