# C46a CAT 强度 β 网格（归档）

- 日期：2026-08-29
- 版本：v044 / C46a
- 父版本：v043 / C45f
- 唯一机制：将 CAT 强度从固定 `β=0.25` 扩展为 `{0.125, 0.25, 0.375}`，仍用 operand-local 目标选择；Weight headroom 与多折 A@W 路径保持不变。
- 根文件 SHA256：`240E32CA2600CC36CE22F5D3C85710ED9060B29D3B3E3DC7C9828D6BDAFB9012`
- 归档文件 SHA256：`240E32CA2600CC36CE22F5D3C85710ED9060B29D3B3E3DC7C9828D6BDAFB9012`

## 评测

- 命令：

  ```powershell
  .\.venv\Scripts\python.exe -u evaluator\real_model_suite.py --models gpt2-small gpt2-medium qwen2.5-0.5b --solution solution.py --candidate-name c46a-cat-beta-grid --device cpu --algorithm-device cuda --cache-mode read --seq 128 --calib 2 --test 4 --output artifacts\real_model_suite\c46a-3model.json --report logs\evaluations\2026-08-29-c46a-3model.md
  ```

- GPT-2 small：Linear `133.601982` / Attention `21.120464` / Total `154.722446` / API `70.12s`。
- GPT-2 medium：Linear `229.271482` / Attention `43.767156` / Total `273.038638` / API `166.17s`。
- Qwen2.5-0.5B：Linear `286.499658` / Attention `62.862350` / Total `349.362008` / API `154.28s`。
- OPT-125M：评测首个模型即回退为 Linear `-846.212506` / Total `-826.630941` / API `63.88s`，随后中止 Pythia。
- 数据与协议：real-model-suite scoring protocol v2；本地结果只用于配对比较，不冒充官方分数。

## 结论

CAT β 网格在 GPT-2/Qwen 上有局部正向，但 OPT 发生结构性回退，不能作为父版本。恢复 C45f 的固定 β=0.25；后续若继续学习式 CAT，只允许以 β=0.25 的解析变换为中心做极小、可回退扰动。
