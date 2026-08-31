# v107 Attention 官方 WA 契约审计

日期：2026-08-31  
对象：`v107-l3-global-lrh` 与已经通过本地/时间门的 `v106-l2-cat`，并以 `v100-b2`
作为更早的 Attention 对照。这里的“官方 WA”指用户回传的 Attention 场景
`wrong answer`；仓库没有官方隐藏输入和逐 case 输出，因此不能直接复现服务器的
那条错误消息。

## 1. 静态差异结论

对 v106、v107 做 `git diff --no-index --unified=3` 后，v107 的唯一行为变化是
Linear 的 Global Activation-LRH proposal、`deployment_gram` state 和相关 gate；
没有修改 Attention 常量、校准函数、Q/K/V 动态函数或 Attention forward。

以下函数在 v100、v106、v107 的函数体 SHA256 完全相同：

| 函数 | body SHA256（共同值） |
|---|---|
| `hif4_calibration_attention` | `4b2a7ae732a63cd02edc6b518b1fbbc44f4e6dbf62cc4f42e080043bb736cb13` |
| `_dynamic_attention_operand` | `6130d4818b4a04ec6353b41fefbfb837df9b00ed752a7dedf4ff2ae5288195a3` |
| `hif4_dynamic_quantize_q` | `7c484f8184d5d85f3bf46f4a69aceae345994cd4f977fa9243aaebcedfd06df3` |
| `hif4_dynamic_quantize_k` | `d009fa75bec8db0c02668ff01ff69a6824292fc62dc7691d71504ca8093523b0` |
| `hif4_dynamic_quantize_v` | `e95e278ff6f2546427af3c12f9e09c0dbd81cf575d01530f6f46ad6853e2f18f` |
| `_attention_forward` | `7872fa976ccfa23b79efbd3b3e5229e269eb5c84591582acb78fc0f2b616485f` |
| `_attention_candidate_score` | `1654a8eb5c8a64e138e05997d2ff7cb43959a4ad2de038670e5c08470334d3d5` |

`_ATTN_OFFSETS`、rotation sizes/seeds、GQRB 参数和 PAWV 参数也逐字相同。因此，
不存在“v107 把 Attention 的 Q/K/V 输入变换或输出字段改坏”的代码差异；若官方
输入完全相同，v106/v107 的 Attention 量化输出应相同。

## 2. 输入/输出差分验证

### 合成官方风格契约矩阵

固定 `balanced/heavy_tail/saturated_logits/v_outlier/k_mean_shift` 五类输入，覆盖
MHA/GQA、`head_dim=64/128`、`amax6`/`amax4`/`pow2`，每个版本 15 个校准 case，
并对 Q/K/V 动态输出做五字段、shape、CPU、finite、HiF4 合法性检查：

| 版本 | 契约 case | 非法 state/参数 | 结论 |
|---|---:|---:|---|
| v106 | 15 | 0 | PASS |
| v107 | 15 | 0 | PASS |

在 `amax6` 的 5 个 case 上进一步调用 evaluator 独立的 `validate_state` 和
`validate_hif4_params`，每个版本均验证 30 个 Q/K/V 参数，失败数均为 0。

### 逐位差分

使用相同随机输入和四组拓扑 `(14,2,64)`、`(12,12,64)`、`(8,2,128)`、
`(16,4,64)`：

- `hif4_calibration_attention` 返回的 `q_state/k_state/v_state`：逐 tensor 相等；
- `hif4_dynamic_quantize_q/k/v` 返回的所有五字段：逐 tensor 相等，最大差值 `0`；
- 先执行 Linear calibration + dynamic activation，再执行 Attention calibration，
  v106/v107 的 Q/K/V 输出仍逐位相等，排除了 Linear 调用改变全局 RNG/Attention
  状态的可能。

Qwen 固定 cache 的历史 full-layer 结果也完全印证了这一点：v106、v107、v107b1、
以及 v100 的 Attention mean 都是 `0.8420394884610322`，standard/player sum、
`global_gain=0.85789896108266` 完全相同。

## 3. v107 新增的资源风险

虽然不改变 Attention 数值路径，v107 在 Linear `activation_state` 中为每个
`channels <= 8192` 的权重保存完整 `deployment_gram = W_q^T W_q`。Qwen cache 的
24 层 × 7 role 形状合计约为：

\[
24\times(6\times896^2+4864^2)\times4
=2{,}733{,}637{,}632\text{ bytes}
\approx2607.0\text{ MiB}.
\]

若官方 runner 保留多个 Linear state，这会造成约 2.6 GiB 的额外内存压力；同时
v107 的 Qwen API 时间为 `481.0365s`，已经超过最新官方 `300s` 限制（2026-08-31 修订，已判 timeout）。官方外层若把资源
耗尽、超时或后续 Attention 没有有效结果统一标为 `atten wrong answer`，这条新增
的 Linear state/运行时路径是首要嫌疑，但当前本地 evaluator 的逐层生命周期不会
复现该资源峰值，不能把它写成已证实的官方根因。

## 4. 当前判定与复现动作

当前证据支持：

1. **数值根因未发现**：v107 Attention 与 v106/v100 是同一实现、同一输出；不能
   通过修改 Attention 算法解释该 WA。
2. **优先排查提交包/资源**：用完全相同的 `solution.py` 单文件包分别提交 v106
   和 v107；确认平台 Python/Torch 版本、总时间、内存和错误发生在 Linear 之后
   还是 Attention API 内部。
3. **若 v106 通过而 v107 失败**：先做资源隔离实验，临时关闭 v107 的完整
   `deployment_gram`（或只对实际需要 exact gate 的形状保留），再单独测官方；这
   是验证资源假设的实验，不应直接当成精度修复。
4. **若 v106/v107 都失败**：再针对官方隐藏 Attention 输入形状（2D/sequence、
   GQA divisor、head_dim 对齐、dtype/device）收集最小复现；不能从当前 Qwen
   panel 分数推断官方输出。

本审计没有修改 v107 或当前 Attention 算法，避免在没有服务器输入的情况下引入
不可归因的回退。
