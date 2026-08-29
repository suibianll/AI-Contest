# C61 CAT WᵀW 1024 行采样（归档，拒绝）

- 日期：2026-08-29
- 版本：v061 / C61
- 父版本：v056 / C56
- 唯一机制：CAT-64 候选的 operand-local `WᵀW` 统计改用最多 1024 个确定性输出行；候选指标仍使用原 256 行预算，在线状态、A@W 静态 Q(W) 选择和门限不变。
- 根文件 SHA256：`F1019B1F3046C3E72DC9223A9F63AB7BFFC2DDE97A51A6121A3A0EE5082665FE`
- 归档文件 SHA256：`F1019B1F3046C3E72DC9223A9F63AB7BFFC2DDE97A51A6121A3A0EE5082665FE`

## 验证

```powershell
\.venv\Scripts\python.exe -m pytest -q tests/test_release_candidate.py tests/test_linear_compliance_guard.py tests/test_weight_full64.py tests/test_reference_hif4.py
\.venv\Scripts\python.exe -u evaluator\real_model_suite.py --models gpt2-small --solution solution.py --candidate-name c61-cat-weightgram1024-gpt2small --device cpu --algorithm-device cuda --cache-mode read --seq 128 --calib 2 --test 4 --output artifacts\real_model_suite\c61-gpt2small.json --report logs\evaluations\2026-08-29-c61-gpt2small.md
\.venv\Scripts\python.exe -u evaluator\real_model_suite.py --models opt-125m --solution solution.py --candidate-name c61-cat-weightgram1024-opt --device cpu --algorithm-device cuda --cache-mode read --seq 128 --calib 2 --test 4 --output artifacts\real_model_suite\c61-opt.json --report logs\evaluations\2026-08-29-c61-opt.md
\.venv\Scripts\python.exe -u evaluator\real_model_suite.py --models qwen2.5-0.5b --solution solution.py --candidate-name c61-cat-weightgram1024-qwen --device cpu --algorithm-device cuda --cache-mode read --seq 128 --calib 2 --test 4 --output artifacts\real_model_suite\c61-qwen.json --report logs\evaluations\2026-08-29-c61-qwen.md
```

- 回归/合规测试：55 passed。
- GPT-2 small：Linear `133.668200`，Attention `21.120464`，Total `154.788664`，API `58.39s`；较 v056 `+0.441270`。
- OPT-125M：Linear `50.271056`，Attention `19.581565`，Total `69.852621`，API `54.45s`；较 v056 `−0.036477`。
- Qwen2.5-0.5B：Linear `266.266543`，Attention `62.862350`，Total `329.128893`，API `151.55s`；较 v056 `−20.215449`，其中 `proj` 宽层发生显著回退。
- 官方得分/时间：`NA`；本地 official-flow 代理只用于相对排序。

## 结论

C61 的更密集 `WᵀW` 统计在窄模型上并不稳定，Qwen 的 4864 输入宽度 `proj` 层出现结构性回退，不能作为跨模型提交版本。恢复 v056，并对超宽输入采用原 256 行统计。
