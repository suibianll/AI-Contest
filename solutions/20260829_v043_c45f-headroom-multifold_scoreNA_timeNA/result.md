# C45f adaptive headroom + 多折 A@W 静态权重选择（归档）

- 日期：2026-08-29
- 版本：v043 / C45f
- 父版本：v042 / C45e
- 唯一机制：在已冻结的 activation_state 之后，用扩展 E6M2 scale beam `{-4..4}` 重跑同一 64-channel FULL64 GPTQ，并用全部 calibration folds 的原始 A@W 选择 parent/headroom 静态 Q(W)；覆盖率仍为父版本 25%。
- 合规边界：A@W 只用于离线 `weight_params`，不写入 `activation_state`，不改变在线 Q(A)。
- 根文件 SHA256：`6137D4BBAC9B7889EE625AC669F291EE65B7C547B33536DE23523C95E6599049`
- 归档文件 SHA256：`6137D4BBAC9B7889EE625AC669F291EE65B7C547B33536DE23523C95E6599049`

## 评测

- GPT-2 small/OPT：

  ```powershell
  .\.venv\Scripts\python.exe -u evaluator\real_model_suite.py --models gpt2-small opt-125m --solution solution.py --candidate-name c45f-headroom --device cpu --algorithm-device cuda --cache-mode read --seq 128 --calib 2 --test 4 --output artifacts\real_model_suite\c45f-2model.json --report logs\evaluations\2026-08-29-c45f-2model.md
  ```

- GPT-2 medium/Pythia：

  ```powershell
  .\.venv\Scripts\python.exe -u evaluator\real_model_suite.py --models gpt2-medium pythia-160m --solution solution.py --candidate-name c45f-headroom-extra --device cpu --algorithm-device cuda --cache-mode read --seq 128 --calib 2 --test 4 --output artifacts\real_model_suite\c45f-extra.json --report logs\evaluations\2026-08-29-c45f-extra.md
  ```

| 模型 | Linear | Attention | Total | API |
| --- | ---: | ---: | ---: | ---: |
| GPT-2 small | 133.226930 | 21.120464 | 154.347394 | 48.28s |
| GPT-2 medium | 229.019937 | 43.767156 | 272.787093 | 115.68s |
| OPT-125M | 43.279017 | 19.581565 | 62.860582 | 46.58s |
| Pythia-160M | 138.246673 | 40.614368 | 178.861041 | 47.20s |
| Qwen2.5-0.5B | 286.174039 | 62.862350 | 349.036389 | 99.21s（沿用 C45e：4864 宽层跳过） |

- 五模型合计 `1017.892499`，较 C43b `1002.004644` 增量 `+15.887855`，较 C45e 增量约 `+10.699`（OPT）。
- 数据与协议：real-model-suite scoring protocol v2；本地结果只用于配对比较，不冒充官方分数。

## 结论

扩展 scale/headroom beam 在 OPT 的 proj 权重上解决了 C43b 的主要负项，同时没有改变其他四个模型的结果。C45f 作为当前本地父版本保留；后续可在此基础上做小步 learned CAT/refinement，但不得把 A@W 结果传入 activation_state。
