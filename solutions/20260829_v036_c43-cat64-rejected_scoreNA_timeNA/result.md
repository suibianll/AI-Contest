# C43 analytic CAT-64（归档）

- 日期：2026-08-29
- 版本：v036 / C43
- 父版本：v034 / C41b
- 唯一机制：在既有 Linear 对角平滑、置换和小块 Hadamard 之后，加入逐 64-channel block 的 SPD 几何均值 CAT 变换；Activation 保存 CAT 矩阵，Weight 使用逆变换。
- A@W：本候选未使用；C43 只做 operand-local 选择。
- Attention：代码路径未修改。
- 根文件 SHA256：`4D7F7196E31BCD65785EFC636B00BEFAFFA4F2DEC337505ADC14269B16ACC7A8`
- 归档文件 SHA256：`4D7F7196E31BCD65785EFC636B00BEFAFFA4F2DEC337505ADC14269B16ACC7A8`

## 评测

- 命令：

  ```powershell
  .\.venv\Scripts\python -u evaluator\real_model_suite.py --models gpt2-small --solution solution.py --candidate-name c43 --device cpu --algorithm-device cuda --cache-mode read --seq 128 --calib 2 --test 4 --output artifacts\real_model_suite\c43-gpt2small.json --report logs\evaluations\2026-08-29-c43-gpt2small.md
  ```

- GPT-2 small local proxy：Linear sum `128.441940` / Attention sum `21.120464` / Total `149.562404`。
- API time：`67.115s`。
- 与 C41b 同面板比较：Linear 下降约 `0.901369`，Attention 逐位相同。

## 逐层诊断

CAT 触发后，部分 O/Proj 层正向，但 Q/K 及少数 FFN 层出现部署误差回归。原因是
operand-local 标准 HiF4 proxy 与带动态 offset/refinement 的实际部署路径错位，不能把
该版本作为父版本。后续 C43b 保留 CAT 数学实现，改为低强度、部署一致的选择代理。
