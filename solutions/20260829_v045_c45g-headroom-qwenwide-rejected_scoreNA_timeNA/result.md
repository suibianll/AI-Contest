# C45g Qwen 宽层 headroom 放开（归档）

- 日期：2026-08-29
- 版本：v045 / C45g
- 父版本：v043 / C45f
- 唯一机制：对最大维度超过 4096 的宽层也生成离散 headroom FULL64 候选；A@W 仍只用于静态 Q(W) 的 parent/candidate 选择，产品条件更新保持 4096 上限。
- 根文件 SHA256：`8D854104C8FF12E4B391EBA2FB424B0170E642073E92A3C22786DFBBEE97C2BE`
- 归档文件 SHA256：`8D854104C8FF12E4B391EBA2FB424B0170E642073E92A3C22786DFBBEE97C2BE`

## 评测

- 命令：

  ```powershell
  .\.venv\Scripts\python.exe -u evaluator\real_model_suite.py --models qwen2.5-0.5b --solution solution.py --candidate-name c45g-headroom-qwen --device cpu --algorithm-device cuda --cache-mode read --seq 128 --calib 2 --test 4 --output artifacts\real_model_suite\c45g-qwen.json --report logs\evaluations\2026-08-29-c45g-qwen.md
  ```

- Qwen2.5-0.5B：Linear `286.174039` / Attention `62.862350` / Total `349.036389` / API `120.87s`。
- 与 C45f 分数完全相同，额外计算未带来收益。
- 数据与协议：real-model-suite scoring protocol v2；本地结果只用于配对比较，不冒充官方分数。

## 结论

Qwen 4864-wide 层的 headroom 候选在多折 A@W 裁判下均回退到父候选，只增加校准耗时。因此保持 C45f 的规模预算，不将该放开策略作为后续父版本。
