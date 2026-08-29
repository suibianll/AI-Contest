# C44 MR-GPTQ data-driven 97% coverage（归档）

- 日期：2026-08-29
- 版本：v041 / C44
- 父版本：v040 / C45c（评测时关闭 C45 产品选择器以隔离机制）
- 唯一机制：在 CAT-64 坐标下，将 FULL64 静态 act-order/GPTQ 的固定 25% block cap 改为按父级 full-H loss 覆盖 97% 的数据驱动 block 集合。
- 根文件 SHA256：`BAC09F49CE5C15E2AE9FFAD6943B2F8D857B8F3C2D73D0781C9B9AC5B9B05A2B`
- 归档文件 SHA256：`BAC09F49CE5C15E2AE9FFAD6943B2F8D857B8F3C2D73D0781C9B9AC5B9B05A2B`

## 评测

- 命令：

  ```powershell
  .\.venv\Scripts\python.exe -u evaluator\real_model_suite.py --models gpt2-small opt-125m qwen2.5-0.5b --solution solution.py --candidate-name c44-mr-gptq-coverage97 --device cpu --algorithm-device cuda --cache-mode read --seq 128 --calib 2 --test 4 --output artifacts\real_model_suite\c44-3model.json --report logs\evaluations\2026-08-29-c44-3model.md
  ```

- GPT-2 small（首个完成模型）：Linear `126.219696` / Attention `21.120464` / Total `147.340160` / API `39.72s`。
- OPT/Qwen 评测因 GPT-2 已明确回退而中止，保留 partial JSON 供诊断。
- 数据与协议：real-model-suite scoring protocol v2；本地结果只用于配对比较，不冒充官方分数。

## 结论

97% 覆盖把低贡献 block 也纳入全-H 误差反馈，GPT-2 small 比 C43b 下降 `-4.719525`。大范围 coverage 不是有效方向；后续恢复 C43b/C45c 的 25% cap，只研究在固定预算内的 act-order/scale 候选。
