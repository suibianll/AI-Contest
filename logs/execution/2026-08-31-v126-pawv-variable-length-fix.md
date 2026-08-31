# v126 PAWV 变长校准修复

> 日期：2026-08-31  
> Parent：v125 precision-only 研究根  
> 状态：实现完成，合成/API 回归通过；full-layer 与官方评测未运行  
> 根/归档规范 SHA256：`47e2e3ab76c6deaac8de47bbcbd8f689cf5989dc8ff9e9081a887ec89e819b08`

## 修改

1. `_build_pawv_metric` 不再按首个样本建立固定 `tokens×tokens` 方阵。
2. 每个 calibration sample 直接计算 `diag(P^TP)`，按 `str(seq_len)` 分组平均。
3. calibration V 对每个样本选择匹配长度的 diagonal。
4. `v_state["row_diagonals"]` 保存长度到 CPU Tensor 的映射。
5. dynamic V 按当前行数精确查找；未见长度回退普通 HiF4。
6. 删除最终未使用的 full `P^TP`、低秩特征分解和 `_ATTN_PAWV_RANK`。

## 复杂度

旧 metric 路径先计算 full `P^TP`，并对其执行 `eigh`。对序列长度 `L`，除 Attention
probability 本身外还引入约 `O(HL^3)` 的 Gram 乘法和 `O(L^3)` 特征分解，以及
`O(L^2)` metric 存储。v126 直接计算 diagonal：

\[
d_j=\frac1H\sum_{h,i}P_{hij}^2,
\]

额外复杂度为 `O(HL^2)`，输出状态为每个已见长度一个 `O(L)` Tensor。

## 验证

```powershell
.\.venv\Scripts\python.exe -m pytest \
  tests/test_release_candidate.py::test_attention_calibration_supports_variable_sequence_lengths -q
```

结果：`1 passed in 3.28s`。

覆盖内容：官方 mini sample 长度 `[10,128,512,1024,1024]` 的 metric 构建、公开
calibration API 的 `10/32` 变长样本、两种已见长度的动态 V、未见长度 `24` 的回退、
state CPU/finite 合规。`py_compile` 与 `git diff --check` 通过。

另外直接把五个官方长度全部送入公开 `hif4_calibration_attention`（MHA
`q_heads=kv_heads=1, head_dim=64`）：

```text
elapsed_seconds 11.397717099986039
v_state_lengths ['10', '1024', '128', '512']
state_devices ['cpu']
dynamic_v 10   (10, 64)   True
dynamic_v 128  (128, 64)  True
dynamic_v 512  (512, 64)  True
dynamic_v 1024 (1024, 64) True
dynamic_v 24   (24, 64)   True
```

这证明完整校准 API 已跨过原先在第二个样本发生的异常；四种已见长度与未见长度回退
均返回 finite、shape 正确的 HiF4 参数。该 11.40s 是单个本地合成 case 时间，不是
官方总时间估计。

固定长度小样本对 v125 的 Q/K state 与动态 Q/K 五字段逐位一致。V 因 diagonal 求和
顺序变化会在等距量化点出现少量 mantissa tie-break 差异，因此 v125 的 Attention
full-layer 分数不能直接记到 v126；必须重测后再更新。

## 当前裁决

v126 已修复导致 v100/v107 WA 的直接异常，但继承 v125 的高 Linear 运行时间，不能
据此直接提交。正式官方基线已更新为 v74 `22750 / 239.387s`；后续 Linear 候选必须
同时满足变长 Attention 回归与 `<300s`。
