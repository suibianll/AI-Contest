# C49 CAT block-Hessian selector（归档，拒绝）

- 日期：2026-08-29
- 版本：v049 / C49
- 父版本：v048 / C45i
- 唯一机制：CAT 候选的 weight operand loss 改用 smooth/permutation/Hadamard 坐标下的 64×64 calibration-Hessian block；CAT 变换、静态产品选择、在线 activation_state 均不变。
- 合规边界：只使用 `A^T A`/`W^T W` 及其 block Hessian 做离线 CAT 评价，不生成或保存 `A@W`，不影响 Q(A)。
- 根文件 SHA256：`0721C799D916D2AFF926DD509F4813BD31F951F5431CAC2ED7D4D48354293EDB`
- 归档文件 SHA256：`0721C799D916D2AFF926DD509F4813BD31F951F5431CAC2ED7D4D48354293EDB`

## 评测

```powershell
\.venv\Scripts\python.exe -u evaluator\real_model_suite.py --models gpt2-small --solution solution.py --candidate-name c49-cat-block-hessian-gpt2small --device cpu --algorithm-device cuda --cache-mode read --seq 128 --calib 2 --test 4 --output artifacts\real_model_suite\c49-gpt2small.json --report logs\evaluations\2026-08-29-c49-gpt2small.md
\.venv\Scripts\python.exe -u evaluator\real_model_suite.py --models qwen2.5-0.5b gpt2-medium --solution solution.py --candidate-name c49-cat-block-hessian-highrisk --device cpu --algorithm-device cuda --cache-mode read --seq 128 --calib 2 --test 4 --output artifacts\real_model_suite\c49-highrisk.json --report logs\evaluations\2026-08-29-c49-highrisk.md
```

| 模型 | Linear | Attention | Total | API |
| --- | ---: | ---: | ---: | ---: |
| GPT-2 small | 133.226930 | 21.120464 | 154.347394 | 50.35s |
| GPT-2 medium | 229.019937 | 43.767156 | 272.787093 | 112.13s |
| Qwen2.5-0.5B | 286.266123 | 62.862350 | 349.128473 | 128.59s |

- 三个已运行模型均与 v048 逐项相同；没有精度增量，且校准时间增加约 3–4 秒。
- 官方得分/时间：`NA`；本地 official-flow 代理只用于相对排序。

## 结论

block-local Hessian 评价在当前单一 CAT β=0.25 候选下没有改变选择结果，收益不足以承担额外成本；恢复轻量 diagonal CAT 评价作为 v048 父版本，避免无效计算进入提交路径。
