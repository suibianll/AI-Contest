# v100/v107 Attention WA 根因分析

> 日期：2026-08-31  
> 官方反馈：v100、v107 均在 Attention 场景 `wrong answer`；用户明确 v100 不是 timeout。
> 官方确认：v72 `22662 / 226s` 正常通过；官方 mini sample 在 Attention 校准阶段
> 报出与本地复现完全一致的 shape mismatch。v100/v107 的直接 WA 根因现已确认。

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

官方 mini sample 的长度为 `[10,128,512,1024,1024]`。旧代码先建立 `[10,10]`
metric，第二个样本产生 `[128,128]` 后精确报错：

```text
RuntimeError: The size of tensor a (10) must match the size of tensor b (128)
at non-singleton dimension 1
```

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

## 5. v126 修复

v126 不再构造跨样本固定方阵，而是对每个样本直接计算

\[
d_j=\frac1H\sum_h\sum_i P_{hij}^2
=\operatorname{diag}\left(\frac1H\sum_hP_h^TP_h\right)_j.
\]

随后按 `str(seq_len)` 分组求平均。校准 V 与在线 V 都按当前 `V.shape[0]` 精确查找
diagonal；在线长度没有对应 calibration 组时回退普通 HiF4 量化。state 的 key 保持为
官方允许的字符串。旧代码虽只使用 diagonal，却仍构造完整 `P^TP` 并执行 `eigh`；
v126 一并删除这两项无用计算，把 metric 部分从额外的立方/方阵开销降为直接对
probability 求平方和。

回归覆盖官方长度模式 `[10,128,512,1024,1024]`、公开 calibration API 的 `10/32`
变长输入、匹配长度动态 V 和未见长度回退，均通过。

## 6. 根因结论与下一步

综合证据，B2 PAWV 的跨 calibration sample 固定 token 维假设是 v100/v107 官方
Attention WA 的**已确认直接根因**：

- 时间线吻合：v98→v100 首次加入该代码；v107 继承；
- 失败类别吻合：发生在 Attention calibration 的直接 RuntimeError；
- 官方契约吻合：没有等长约束，任一运行异常即整次失败；
- 本地漏检原因吻合：所有缓存窗口固定 `seq=128`；
- 对照版本吻合：v72/v98 对同一变长输入通过，v100/v107 同时失败。

后续纪律：

1. 当前官方基线使用已通过的 v74 `22750 / 239.387s`；v126 修复在完成时限与官方复测前不替代它。
2. PAWV state 必须按长度分组或采用另经验证的长度无关统计，不得直接平均不同 shape
   的 `P^TP`。
3. Attention 发布矩阵必须加入 calibration-list 变长、动态长度变化、MHA/GQA、
   `head_dim=64/128`、以及矩形 `L_q != L_k=L_v`。
4. local contract pass 只能记作本地通过，不能再表述为官方安全。
