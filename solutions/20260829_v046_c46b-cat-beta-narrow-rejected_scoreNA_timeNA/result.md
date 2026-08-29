# C46b 窄 CAT β refinement（归档）

- 日期：2026-08-29
- 版本：v046 / C46b
- 父版本：v043 / C45f
- 唯一机制：将 CAT 强度从 `β=0.25` 改为 `{0.20, 0.25, 0.30}` 的窄候选集，其他 Weight headroom/A@W 逻辑不变。
- 根文件 SHA256：`AE63FC24C8288BEF4A0441B72B5AE8486C5E101A9DE2E8A9C061DE94CB4A5533`
- 归档文件 SHA256：`AE63FC24C8288BEF4A0441B72B5AE8486C5E101A9DE2E8A9C061DE94CB4A5533`

## 评测

- 命令：

  ```powershell
  .\.venv\Scripts\python.exe -u evaluator\real_model_suite.py --models opt-125m --solution solution.py --candidate-name c46b-cat-narrow --device cpu --algorithm-device cuda --cache-mode read --seq 128 --calib 2 --test 4 --output artifacts\real_model_suite\c46b-opt.json --report logs\evaluations\2026-08-29-c46b-opt.md
  ```

- OPT-125M：Linear `31.214825` / Attention `19.581565` / Total `50.796390` / API `67.81s`。
- 相比 C45f Total `62.860582` 回退 `-12.064192`，其余模型未运行。
- 数据与协议：real-model-suite scoring protocol v2；本地结果只用于配对比较，不冒充官方分数。

## 结论

OPT 对 CAT β=0.25 以外的扰动仍不稳定；C46b 不能作为父版本。恢复 C45f 固定 CAT β=0.25，后续仅研究静态 Weight 的可逆候选。
