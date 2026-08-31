# v100 官方 Attention WA 后的版本边界审计

> 日期：2026-08-31  
> 官方反馈：用户确认 v100 在官方评测中为 Attention `wrong answer`；不是 timeout。
> 官方分数、时间和提交文件 SHA 未提供，保持 `NA`，不把本地数据代填为官方结果。

## 1. 修正后的结论

v100 不包含 v107 的完整 Linear `deployment_gram`，但仍得到同类 Attention WA。
因此以下旧推测不再成立：

- 不能继续把 v107 WA 首要归因于 `deployment_gram`；
- 不能把 v100 当作官方安全候选；
- v100/v106/v107 以及继承其 clean Attention 路径的后续版本，均没有官方通过证据。

本地 Qwen/synthetic 合约矩阵没有复现错误，只能说明当前本地 evaluator 未覆盖官方
失败输入，不能覆盖官方结果。

后续已用 calibration list 内 `seq=32/48` 的合法变长输入稳定复现：v72/v98 通过，
v100/v107 在 B2 PAWV `_build_pawv_metric` 抛出 shape mismatch。最新根因报告见
[`v100/v107 Attention WA 根因`](2026-08-31-v100-v107-attention-wa-root-cause.md)。

## 2. v66→v100 的 Attention 实现边界

对归档源码做 AST 语义哈希和调用闭包检查，以以下四个官方 API 为根：

- `hif4_calibration_attention`
- `hif4_dynamic_quantize_q`
- `hif4_dynamic_quantize_k`
- `hif4_dynamic_quantize_v`

结果：

| 版本段 | 四个公共 API | 可达 helper/相关常量相对 v66 | 官方证据 | 风险裁决 |
|---|---|---|---|---|
| v66 | 基线 | 基线 | `22557 / 217.2s` 通过 | 官方控制组 |
| v67–v68 | 与 v66 相同 | 与 v66 相同 | 未提交 | 低风险，但本地增益不足/被拒绝 |
| v72 | 与 v66 相同 | 45 个可达函数及相关常量均与 v66 语义一致 | `22662 / 226s` 通过 | 前一官方基线 |
| v73 | 公共 API 与 v66 相同 | 共享 `_nvfp4_to_hif4`/`_dense_to_hif4` 及 Gram/source helper 已改变 | 未提交 | 尚无官方结论 |
| **v74** | **公共 API 与 v66 相同** | **共享 helper 已改变** | **`22750 / 239.387s` 通过** | **当前官方通过基线** |
| v75–v84 | Q/K calibration/dynamic 已改变 | 新增 GQA rotation | 未提交 | 首个明确 Attention 变更边界 |
| v86 | 再次改变 Q/K 路径 | block-Hadamard final selector | 未提交 | 更高风险 |
| v100+ | clean Attention 重写 | B1 GQRB、B2 PAWV、Gram refine 等新闭包 | v100/v107 均 WA | 官方无效，停止推荐 |

v72 的 Attention 闭包等价判断比“同一 Qwen 输出相等”更强：不仅四个入口的 AST 相同，
从入口递归可达的 45 个本地 helper 及其引用常量也没有语义差异。

## 3. 新候选裁决

### 绝对保底

- v66：已有官方 `22557 / 217.2s`，保留为控制组。
- v72：用户确认官方 `22662 / 226s`，相对 v66 `+105` 分、`+8.8s`。
- v74：用户确认官方 `22750 / 239.387s`，相对 v72 `+88` 分、`+13.387s`。
- 源码：`solutions/20260829_v066_c66-activation-ratio100_scoreNA_timeNA/solution.py`。

### 分数尽量高且已通过

- v74 / C75 rowwise JDRQ + wide gram64 hierarchy。
- 源码：`solutions/20260829_v074_c75-rowwise-jdrq_scoreNA_timeNA/solution.py`。
- SHA256：`7789a0487915ee1860eeca2736311bdd1e357bf86e5528805472182f51b944cc`。
- 本地 Qwen native：`361.503707`，较 v72 提升 `+4.898105`。
- 本地 Qwen Attention：`63.119717`，与 v66/v72 相同。
- Qwen CUDA API：`179.27s`；四模型记录均 `<420s`。
- 机制只在冻结 activation state 后用 calibration 产品残差优化静态 `Q(W)`；不把
  输出监督写入 Attention 或 activation state。

v74 已修改 Attention 共用量化 helper。此前在没有官方结果时将其排在 v72 之后是合理
的风险控制；现在官方成功证明这些改动在当前评测集合法可运行，首选排序更新为
`v74 > v72 > v66`。

### 当前 evaluator 同场冒烟

使用 Qwen layer 0 固定缓存、`seq=128 / calib=2 / test=4 / amax6 / CPU`，在一次运行中
同时加载 c66 和 v72：

| 候选 | layer-0 panel | Linear panel | Attention panel | API |
|---|---:|---:|---:|---:|
| c66 | `314.731294` | `131.580914` | `183.150380` | `24.23s` |
| v72 | `314.681945` | `131.531565` | `183.150380` | `24.28s` |

两者 Attention 逐分相同；v72 在该单层 Linear 略低 `0.049349` panel。该结果不覆盖
历史四模型全量中 v72 相对 v66 的 Qwen正向，也不用于证明官方通过；它说明 v72 的
推荐依据是“完整 Attention 闭包等价 + 历史全量 Linear 增益”，不是单层冒烟涨分。

## 4. 后续实现纪律

1. 官方提交线从 v74 开始，v72/v66 作为控制组，不从 v100/v125 回退式修补。
2. Linear 优化必须移植到 v74，并保持 v74 Attention 调用闭包的语义哈希不变。
3. 每次只移植一个 Linear 机制；若任何 Attention 可达 helper/常量变化，候选自动降级为
   研究版本，除非获得新的官方通过结果。
4. 本地新增 hidden-shape fuzz 只能提高发现率，不能再把 local pass 写成 official-safe。
