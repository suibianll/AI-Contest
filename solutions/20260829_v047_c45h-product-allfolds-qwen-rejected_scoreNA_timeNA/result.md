# C45h 全宽多折 A@W 产品选择（归档，拒绝）

- 日期：2026-08-29
- 版本：v047 / C45h
- 父版本：v043 / C45f
- 唯一机制：将静态 Q(W) 产品选择的维度预算从 4096 放宽到 8192，使 Qwen2.5-0.5B 的 4864 行 FFN 层也参与全部 calibration folds 的 A@W 候选裁判；在线 `activation_state` 与 Q(A) 路径不变。
- 合规边界：A@W 只用于离线静态 `weight_params`，不写入 `activation_state`，不参与在线激活量化。
- 评估运行时根文件 SHA256：`4B4C799C677E39987C2084A78D0AAFE7172C404E91202BBDB9B31964D4762A9B`
- 归档文件 SHA256：`9C94EDECBEEA0E06FD63171705653E6A44E1831B1F8A123A198BD658A4723CD5`

## 评测

- 命令：

  ```powershell
  .\.venv\Scripts\python.exe -u evaluator\real_model_suite.py --models qwen2.5-0.5b --solution solution.py --candidate-name c45h-product-allfolds-qwen --device cpu --algorithm-device cuda --cache-mode read --seq 128 --calib 2 --test 4 --output artifacts\real_model_suite\c45h-qwen.json --report logs\evaluations\2026-08-29-c45h-qwen.md
  ```

- Qwen2.5-0.5B：Linear `285.702496` / Attention `62.862350` / Total `348.564846` / API `131.030s`。
- 相比 C45f 的 `349.036389` 回退 `0.471543`；回退来自 4864 行 `fc_gate/fc_up`，未抵消 `proj` 的小幅收益。
- 数据与协议：real-model-suite scoring protocol v2；本地结果只用于候选相对排序，不冒充官方绝对分数。

## 结论

C45h 的全宽产品选择增加了校准成本且 Qwen 总分下降，因此拒绝晋级；后续改为仅按输出行数控制候选预算，保留可受益的窄输出 `proj` 层。
