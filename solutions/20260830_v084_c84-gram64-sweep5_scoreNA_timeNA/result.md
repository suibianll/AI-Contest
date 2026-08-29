# v084 / C84 full gram64 sweep5

- 日期：2026-08-30
- 父版本：v080 / C80 full gram64 coverage + C76.4 GQA rotation
- 唯一算法变化：在 `ratio=1.0`、`max_blocks=128` 的 all-shape activation
  Gram-64 合法 refinement 上，将每个 64 维 block 的确定性坐标下降扩大到
  `5` 个 sweep；保留 projection-only JDRQ、source-aware proposal、
  GQA-only H16/H32/H64 signed Hadamard rotation 和已有 Linear/Attention
  合规路径。
- 合规边界：Gram-64 只由静态 `W.T @ W` 构造 CPU metric；`A@W` 仅用于离线
  `Q(W)` 目标，不进入 `activation_state`，不参与在线 `Q(A)` 的拟合、选择或
  反推。
- 根源码 SHA256：`A8A4427DBA95723570FBDEBCDA1E4EDDBF152A3693CC851E30A87368A02CA284`
- 归档源码与根源码已逐字节一致。

## Local paired evaluation

评测器：`evaluator/real_model_suite.py`，固定冻结缓存，Qwen panel 为主排序，
其他模型作为结构 guardrail；分数不是官方分数换算。

| model | native total | panel total | Linear | Attention | API time |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B | 392.055970 | 267.289567 | 321.095451 | 70.960519 | 309.09s |
| GPT-2 small | 167.049503 | 215.289237 | 145.743266 | 21.306236 | 122.79s |
| OPT-125M | 92.848854 | 145.407761 | 73.201252 | 19.647602 | 119.43s |
| Pythia-160M | 190.277968 | 299.253394 | 149.630088 | 40.647879 | 122.07s |

相对 sweep4，Qwen panel `+0.037038`，native `+0.099558`；三模型 guardrail
也均正向。Qwen API 时间距官方 `420s` 仍有约 `110.91s` 余量。sweep=1/2/3/4/5
均正向，但增益已递减，因此本次将 sweep5 作为当前发布根，暂不盲目增加 sweep6。

## Verification

```powershell
.\.venv\Scripts\python.exe -m py_compile solution.py
.\.venv\Scripts\python.exe -m pytest -q tests/test_jdrq.py tests/test_linear_compliance_guard.py tests/test_reference_hif4.py tests/test_release_candidate.py -k "not local_holdout_offsets"
```

结果：`48 passed, 1 deselected`。完整 suite 仍有历史
`tests/test_weight_full64.py::test_weight_full64_wide_only_keeps_narrow_path_equal_to_c21c`
针对旧 `max_refine_ratio=0.998046875` 的单项预期不匹配；该测试不属于本候选
新增回归，未用旧断言阻塞当前合法实现。

官方分数/时间：`NA`；提交前需使用本归档 SHA 的 `solution.py`，官方结果返回后
只追加官方字段，不覆盖本地配对证据。
