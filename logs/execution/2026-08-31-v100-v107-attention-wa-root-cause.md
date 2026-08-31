# v100/v107 Attention WA 根因分析

> 日期：2026-08-31  
> 官方反馈：v100、v107 均在 Attention 场景 `wrong answer`；用户明确 v100 不是 timeout。
> 结论：已在本地构造官方接口允许的变长校准集，稳定复现 v100/v107 的直接运行异常。

## 1. 官方要求中的关键边界

任务书 2.3、2.4 对 Attention 的要求是：

1. `hif4_calibration_attention` 接收若干组 NVFP4 Q/K/V 和 head 参数；没有规定不同
   calibration sample 必须具有相同 `seq_len`。
2. 动态 Q 的输入 shape 为 `[seq_len, q_num_heads * head_dim]`，K/V 类似；没有声明
   校准/测试的 `seq_len` 固定为某个值。
3. `q_state/k_state/v_state` 可以包含合法 CPU Tensor 等静态校准信息。
4. 任一测试用例出现运行异常、缺失结果或非法 HiF4 参数，整次提交失败。

因此候选代码必须对每个合法输入列表独立处理，不能把本地 evaluator 的固定
`seq=128` 当成 API 前置条件。

## 2. v100 的直接缺陷

v100 相对 v98 的唯一 Attention 增量是 B2 PAWV diag-only。其
`_build_pawv_metric` 实现为：

```python
tokens = int(q_samples[0].shape[0])
metric = torch.zeros(tokens, tokens, ...)
for q, k in zip(q_samples, k_samples):
    probability = _attention_probability(q, k, ...)
    metric += einsum(probability, probability)
```

对第 `i` 个样本，Attention probability 的 shape 是

\[
P_i\in\mathbb{R}^{H_q\times L_i\times L_i},
\qquad
P_i^TP_i\in\mathbb{R}^{L_i\times L_i}.
\]

代码却预先固定

\[
M\in\mathbb{R}^{L_0\times L_0}
\]

并执行 `M += P_i^T P_i`。只要 `L_i != L_0`，张量形状必然不匹配。这个平均在数学上
本来也没有定义，因为不同 token 位置空间不能直接逐元素相加。

v107 的 Attention 函数体与 v100 相同，所以继承同一缺陷；Linear
`deployment_gram` 与该异常无关。

## 3. 最小复现

构造 MHA `q_heads=1, kv_heads=1, head_dim=64`，两组合法 NVFP4 calibration sample，
每组内部 Q/K/V 长度一致，但两组长度分别为 32 和 48。结果：

| 版本 | 结果 |
|---|---|
| v72 | PASS，返回 `q_state/k_state/v_state` |
| v98 | PASS，返回 `q_state/k_state/v_state` |
| v100 | **FAIL**：`RuntimeError: size 32 must match size 48 at dimension 1` |
| v107 | **FAIL**：同一 `RuntimeError` |

这也解释了为什么此前本地 Qwen、五场景 contract matrix 和 24 层逐输出审计全部通过：
本地 cache 将每个窗口统一裁成 `seq=128`，从未覆盖 calibration list 内的变长样本。

## 4. 其他怀疑项的裁决

### Q/K 等价旋转不是直接违规

官方明确允许 state 保存旋转/缩放等信息。若同一正交矩阵 `R` 作用于 Q/K，

\[
(QR)(KR)^T=QRR^TK^T=QK^T,
\]

因此不需要逆变换回原始坐标；只要 GQA head 映射正确，它是合法的 Attention 等价变换。
这类变换可能在量化后回退，但回退应产生负分，不应直接导致运行异常。

### reciprocal scaling 与 K centering 也是合法不变量

逐维 reciprocal scaling 满足 `(QD)(KD^{-1})^T=QK^T`。对所有 key 减去同一向量 `c`，
每个 query 的 logits 只增加一个对所有 key 相同的常数，softmax 不变。这两项不是当前
WA 的首要原因。

### non-causal-only 校准是精度风险，不是 WA 根因

v100 候选排序只计算 non-causal Attention；若官方包含 causal case，候选可能泛化回退。
但官方评分允许负分，因此它更可能降低分数，而不会解释直接 wrong answer。

### 仍存在一个共同的潜在长度假设

v66/v72 和 v100 的内部校准 forward 都默认单个 sample 内 `len(Q)=len(K)=len(V)`。
任务书没有明确保证矩形 decode Attention；不过 v66 已通过当前官方集，所以这不是
v100 相对 v66 新出现 WA 的解释。后续测试仍应覆盖 `L_q != L_k=L_v`，避免评测集扩展。

## 5. 根因置信度与下一步

综合证据，B2 PAWV 的跨 calibration sample 固定 token 维假设是 v100/v107 官方
Attention WA 的**高置信度直接根因**：

- 时间线吻合：v98→v100 首次加入该代码；v107 继承；
- 失败类别吻合：发生在 Attention calibration 的直接 RuntimeError；
- 官方契约吻合：没有等长约束，任一运行异常即整次失败；
- 本地漏检原因吻合：所有缓存窗口固定 `seq=128`；
- 对照版本吻合：v72/v98 对同一变长输入通过，v100/v107 同时失败。

在没有官方逐 case traceback 的情况下不能声称形式上的 100% 证明，但该解释已比
`deployment_gram`、超时或普通精度回退更强。

后续纪律：

1. 官方候选继续使用 v72/v66，不提交 v100+ clean Attention 路径。
2. 若未来重启 PAWV，metric 必须是与长度无关的统计（例如按样本独立使用/按位置 bucket
   聚合/归一化谱摘要），不得直接平均不同 shape 的 `P^TP`。
3. Attention 发布矩阵必须加入 calibration-list 变长、动态长度变化、MHA/GQA、
   `head_dim=64/128`、以及矩形 `L_q != L_k=L_v`。
4. local contract pass 只能记作本地通过，不能再表述为官方安全。

