# C45e 多折 A@W 静态权重选择（归档）

- 日期：2026-08-29
- 版本：v042 / C45e
- 父版本：v037 / C43b
- 唯一机制：在 `activation_state` 冻结后，用全部 calibration folds 的原始 A@W 产品梯度更新静态 Q(W)；最大矩阵维度 4096 以上的层保留父候选。
- 合规边界：产品输出与梯度只用于离线 `weight_params`，不写入 `activation_state`，不改变在线 Q(A)。
- 根文件 SHA256：`C783224C7C3143D6EF2A490E77079B8E51D0DCA2027057D2C60403777749EAB6`
- 归档文件 SHA256：`C783224C7C3143D6EF2A490E77079B8E51D0DCA2027057D2C60403777749EAB6`

## 评测

- 主命令（GPT-2 small/medium/Qwen）：

  ```powershell
  .\.venv\Scripts\python.exe -u evaluator\real_model_suite.py --models gpt2-small gpt2-medium qwen2.5-0.5b --solution solution.py --candidate-name c45e-product-allfolds --device cpu --algorithm-device cuda --cache-mode read --seq 128 --calib 2 --test 4 --output artifacts\real_model_suite\c45e-3model.json --report logs\evaluations\2026-08-29-c45e-3model.md
  ```

- 补测命令（OPT/Pythia）：

  ```powershell
  .\.venv\Scripts\python.exe -u evaluator\real_model_suite.py --models opt-125m pythia-160m --solution solution.py --candidate-name c45e-product-allfolds-extra --device cpu --algorithm-device cuda --cache-mode read --seq 128 --calib 2 --test 4 --output artifacts\real_model_suite\c45e-extra.json --report logs\evaluations\2026-08-29-c45e-extra.md
  ```

| 模型 | Linear | Attention | Total | API |
| --- | ---: | ---: | ---: | ---: |
| GPT-2 small | 133.226930 | 21.120464 | 154.347394 | 45.20s |
| GPT-2 medium | 229.019937 | 43.767156 | 272.787093 | 104.70s |
| OPT-125M | 32.580090 | 19.581565 | 52.161655 | 47.07s |
| Pythia-160M | 138.246673 | 40.614368 | 178.861041 | 45.95s |
| Qwen2.5-0.5B | 286.174039 | 62.862350 | 349.036389 | 99.21s |

- 五模型合计 `1007.193572`，C43b 同口径为 `1002.004644`，增量 `+5.188928`。
- 数据与协议：real-model-suite scoring protocol v2；本地结果只用于配对比较，不冒充官方分数。

## 结论

多折拟合比单折产品补偿稳定：五模型均不低于 C43b，且 Attention 逐位保持不变。该候选作为当前本地父版本保留；下一步在其上加入轻量 adaptive headroom/LWC，继续只修改静态 Q(W)。
