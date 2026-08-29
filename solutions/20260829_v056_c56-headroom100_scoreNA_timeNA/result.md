# C56 Weight headroom 100%（归档，本地接受）

- 日期：2026-08-29
- 版本：v056 / C56
- 父版本：v055 / C55
- 唯一机制：将 post-state FULL64 headroom 候选覆盖率由 `75%` 放宽到 `100%`；多折 A@W 仍只在 parent/headroom 静态 Q(W) 之间软选择，CAT/grouping/Q(A) 不变。
- 合规边界：A@W 只用于 activation_state 冻结后的静态 `weight_params` 选择，不影响 Q(A) 或在线 state。
- 根文件 SHA256：`06EDAF8BD9E82DFC480071458E297EB124BCEFF23877E3E2DDC83B76E3517402`
- 归档文件 SHA256：`06EDAF8BD9E82DFC480071458E297EB124BCEFF23877E3E2DDC83B76E3517402`

## 评测

```powershell
\.venv\Scripts\python.exe -u evaluator\real_model_suite.py --models opt-125m --solution solution.py --candidate-name c56-headroom100-opt --device cpu --algorithm-device cuda --cache-mode read --seq 128 --calib 2 --test 4 --output artifacts\real_model_suite\c56-opt.json --report logs\evaluations\2026-08-29-c56-opt.md
\.venv\Scripts\python.exe -u evaluator\real_model_suite.py --models qwen2.5-0.5b gpt2-medium --solution solution.py --candidate-name c56-headroom100-highrisk --device cpu --algorithm-device cuda --cache-mode read --seq 128 --calib 2 --test 4 --output artifacts\real_model_suite\c56-highrisk.json --report logs\evaluations\2026-08-29-c56-highrisk.md
\.venv\Scripts\python.exe -u evaluator\real_model_suite.py --models gpt2-small pythia-160m --solution solution.py --candidate-name c56-headroom100-extra --device cpu --algorithm-device cuda --cache-mode read --seq 128 --calib 2 --test 4 --output artifacts\real_model_suite\c56-extra.json --report logs\evaluations\2026-08-29-c56-extra.md
```

| 模型 | Linear | Attention | Total | API |
| --- | ---: | ---: | ---: | ---: |
| GPT-2 small | 133.226930 | 21.120464 | 154.347394 | 58.84s |
| GPT-2 medium | 229.148763 | 43.767156 | 272.915919 | 139.91s |
| OPT-125M | 50.307533 | 19.581565 | 69.889098 | 56.36s |
| Pythia-160M | 138.329016 | 40.614368 | 178.943384 | 57.87s |
| Qwen2.5-0.5B | 286.481992 | 62.862350 | 349.344342 | 156.16s |

- 五模型代理合计：`1025.440137`，较 v055 `1024.848464` 增加 `+0.591673`；较 v051 增加 `+7.135795`。
- 所有 API 均小于 300 秒；官方得分/时间：`NA`，本地 official-flow 代理只用于相对排序。

## 结论与下一步

全覆盖 headroom 在 A@W layer-level 软裁判下继续释放 OPT 与 GPT-2 medium 的有效块，未造成其他模型回退。v056 是当前本地父版本；后续应优化候选生成/预算而非再扩大 coverage。
