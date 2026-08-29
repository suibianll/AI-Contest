# C45c 原始 A@W 静态权重选择（规模上限，归档）

- 日期：2026-08-29
- 版本：v040 / C45c
- 父版本：v037 / C43b
- 唯一机制：对已冻结的 CAT-64/Full-H 在线状态，用原始校准 A@W 产品损失更新静态 Q(W)；最大矩阵维度超过 4096 的层保留父候选。
- 合规边界：产品输出只用于离线 `weight_params` 选择，不写入 `activation_state`，不改变在线 Q(A)。
- 根文件 SHA256：`C1E3507CB1E8D023034EFF0386AB961B7EB9CAC6E7A1D41F12ED0BA448A90459`
- 归档文件 SHA256：`C1E3507CB1E8D023034EFF0386AB961B7EB9CAC6E7A1D41F12ED0BA448A90459`

## 评测

- 命令：

  ```powershell
  .\.venv\Scripts\python.exe -u evaluator\real_model_suite.py --models gpt2-small opt-125m qwen2.5-0.5b --solution solution.py --candidate-name c45c-raw-sizecapped --device cpu --algorithm-device cuda --cache-mode read --seq 128 --calib 2 --test 4 --output artifacts\real_model_suite\c45c-3model.json --report logs\evaluations\2026-08-29-c45c-3model.md
  ```

- GPT-2 small：Linear `131.769809` / Attention `21.120464` / Total `152.890273` / API `47.25s`。
- OPT-125M：Linear `31.602006` / Attention `19.581565` / Total `51.183572` / API `45.70s`。
- Qwen2.5-0.5B：Linear `286.174039` / Attention `62.862350` / Total `349.036389` / API `103.99s`。
- 数据与协议：real-model-suite scoring protocol v2；本地结果只用于配对比较，不冒充官方分数。

## 结论

规模上限成功保留了 GPT-2 的产品选择收益，同时跳过 Qwen 4864 宽层的高方差更新；三模型合计 `553.110234`，较 C43b 三模型合计 `552.568041` 小幅上升。OPT 的局部负向仍存在，下一步继续做静态 MR-GPTQ/act-order，不改变 CAT 或 Attention 状态。
