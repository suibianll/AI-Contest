# C70 外部 v2.6 X/W 联合残差补偿（本地拒绝）

- 日期：2026-08-29
- 版本：v070 / C70
- 父版本：C69（C66 线上官方冠军之后的本地代理候选）
- 唯一机制：在 `activation_state` 完整冻结后，按 64-channel block 对静态
  `Q(W)` 枚举 E6M2 offset `{-2,-1,+1,+2,+3}`，用固定校准 `Q(A)` 的输出残差
  做最多 3 轮 Gauss–Seidel 更新。候选按 `ΔE < 0` 接受，残差只存在于校准期。
- 合规边界：没有修改六个 API；残差、`Xq` 和候选分数不进入
  `activation_state`，在线 `hif4_dynamic_quantize_activation` 路径不变。
- 根文件/归档文件 SHA256：
  `E897F14878623E9D0A6C700EDE6A870E8335A5456BF64372254AEAC02F476AFF`

## 验证

```powershell
\.venv\Scripts\python.exe -m py_compile solution.py
\.venv\Scripts\python.exe -u evaluator\real_model_suite.py --models gpt2-small opt-125m qwen2.5-0.5b --solution solution.py --candidate-name c70-joint-refine-screen --device cpu --algorithm-device cuda --cache-mode read --seq 128 --calib 2 --test 4 --output artifacts\real_model_suite\c70-joint-refine-screen.json --report logs\evaluations\2026-08-29-c70-joint-refine-screen.md
```

| 模型 | Linear | Attention | Total | API |
| --- | ---: | ---: | ---: | ---: |
| GPT-2 small | 140.600381 | 21.306236 | **161.906618** | 93.43s |
| OPT-125M | 64.856742 | 19.647602 | **84.504344** | 93.93s |
| Qwen2.5-0.5B | 280.040838 | 63.119717 | **343.160555** | 266.79s |

相对 C69：GPT-2 small `+6.270550`，OPT `−0.615957`，Qwen `−6.991865`；
三模型合计 `589.571517`，同时 Qwen 校准时间显著增加。C70 只保留为外部算法
复现与回归证据，不覆盖 C66 的官方结果，也不晋级为 active parent。

## 诊断结论

外部 v2.6 在其 GPT-2 12 层、2-calibration 配置中报告 72/72 Linear 层改善，
但直接叠加到本地 C69 的 CAT、FULL64、静态产品选择后，收益只在 GPT-2 small
出现。Qwen 代理的 `proj` 和 `o` 有改善，`q/k/v` 与两个宽 `fc` 路径回退；
OPT 的 `proj` 也回退。说明联合残差项对校准分布和父版本变换高度敏感，不能
用“每次 `ΔE<0`”替代跨 fold 的稳定性检查。

下一步不增加模型名门控；先做低自由度的软 fold 选择/候选混合实验，并优先
验证 Qwen-30B 类宽 FFN down-proj。官方新增的两个 Qwen-30B 特征用例仍应
单独评测，不能从本地 Qwen2.5-0.5B 回归官方绝对分数。
