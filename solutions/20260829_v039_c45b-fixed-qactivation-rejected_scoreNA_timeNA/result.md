# C45b 固定激活量化产品目标（归档）

- 日期：2026-08-29
- 版本：v039 / C45b
- 父版本：v037 / C43b
- 唯一机制：在 `activation_state` 完成后，用固定的校准 Q(A) 与 A@W 产品目标更新静态 Q(W)。
- 合规边界：产品输出只用于离线 `weight_params` 选择，不写入 `activation_state`，不改变在线 Q(A) 状态。
- 根文件 SHA256：`8AB3AC383771D98823B4A7CB8062805756EC9E93C6E85D14E2FDC9B45AC8E80A`
- 归档文件 SHA256：`8AB3AC383771D98823B4A7CB8062805756EC9E93C6E85D14E2FDC9B45AC8E80A`

## 评测

- 命令：

  ```powershell
  .\.venv\Scripts\python.exe -u evaluator\real_model_suite.py --models gpt2-small opt-125m qwen2.5-0.5b --solution solution.py --candidate-name c45b-fixed-qactivation --device cpu --algorithm-device cuda --cache-mode read --seq 128 --calib 2 --test 4 --output artifacts\real_model_suite\c45b-3model.json --report logs\evaluations\2026-08-29-c45b-3model.md
  ```

- GPT-2 small：Linear `129.712444` / Attention `21.120464` / Total `150.832908` / API `45.76s`。
- OPT-125M：Linear `29.862380` / Attention `19.581565` / Total `49.443946` / API `44.53s`。
- Qwen2.5-0.5B：Linear `269.025229` / Attention `62.862350` / Total `331.887579` / API `122.64s`。
- 数据与协议：real-model-suite scoring protocol v2；本地结果只用于配对比较，不冒充官方分数。

## 结论

固定 Q(A) 的产品目标仍使三种架构的宽层静态权重发生过大偏移，三模型均低于 C43b，尤其 Qwen2.5-0.5B 回退明显。该版本按实验纪律保留，但不作为后续父版本；后续恢复到原始 A@W→Q(W) 目标并降低覆盖率，或直接转入 MR-GPTQ。
