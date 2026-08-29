# C42e calibration-product compensation（归档）

- 日期：2026-08-29
- 版本：v035 / C42e
- 父版本：v034 / C41b
- 唯一机制：在已有 Linear 静态 Weight 量化后，使用校准 `A@W` 对少量 64-channel block 做阻尼条件修正，并以独立 calibration fold 选择 `Q(W)`。
- 合规边界：该实验只修改 `weight_params`，未把产品输出或残差写入 `activation_state`；Attention 路径未改动。
- 根文件 SHA256：`65EBDAE677F9AF669235845A35C73C987F3BFDB828078A490D30C0501D62DC8F`
- 归档文件 SHA256：`65EBDAE677F9AF669235845A35C73C987F3BFDB828078A490D30C0501D62DC8F`

## 评测

- 命令：

  ```powershell
  .\.venv\Scripts\python -u evaluator\real_model_suite.py --models gpt2-small --solution solution.py --candidate-name c42e --device cpu --algorithm-device cuda --cache-mode read --seq 128 --calib 2 --test 4 --output artifacts\real_model_suite\c42e-gpt2small.json --report logs\evaluations\2026-08-29-c42e-gpt2small.md
  ```

- GPT-2 small local proxy：Linear sum `130.183032` / Attention sum `21.120464` / Total `151.303496`。
- API time：`35.181s`。
- 数据与协议：real-model-suite scoring protocol v2，cache revision 记录在对应 JSON 中；本地结果不冒充官方分数。

## 结论

该实现只在 GPT-2 small 的局部代理上有小幅正向信号，且具有高维校准过拟合风险，未作为后续父版本。后续实现从 C41b 恢复，保留本快照供诊断对照。
