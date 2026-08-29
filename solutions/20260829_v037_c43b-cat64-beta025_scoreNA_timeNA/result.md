# C43b CAT-64 β=0.25

- 日期：2026-08-29
- 版本：v037 / C43b
- 父版本：v036 / C43
- 唯一变化：将 CAT-64 强度候选收缩为最低强度 `β=0.25`；其余 CAT 数学、状态和 C41b Attention 路径不变。
- A@W：未使用；C43b 仍为 operand-local 选择。
- 根文件 SHA256：`46D33909C11939495D531EDF030D6F72BCEFE293DD66655396AB589723AABF30`
- 归档文件 SHA256：`46D33909C11939495D531EDF030D6F72BCEFE293DD66655396AB589723AABF30`

## 评测

- 命令：

  ```powershell
  .\.venv\Scripts\python -u evaluator\real_model_suite.py --models gpt2-small --solution solution.py --candidate-name c43b --device cpu --algorithm-device cuda --cache-mode read --seq 128 --calib 2 --test 4 --output artifacts\real_model_suite\c43b-gpt2small.json --report logs\evaluations\2026-08-29-c43b-gpt2small.md
  ```

- GPT-2 small local proxy：Linear `130.939221` / Attention `21.120464` / Total `152.059685`。
- API time：`39.740s`。
- 相对 C41b：Linear 约 `+1.595912`，Attention 逐位相同。

## 结论

CAT 的过强强度会损害动态量化；β=0.25 在 GPT-2 small 上取得稳定正向，保留为当前候选并进入多模型验证。
