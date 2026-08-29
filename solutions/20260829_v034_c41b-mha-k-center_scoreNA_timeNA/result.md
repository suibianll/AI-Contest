# v034 — C41b scale-aware K 公共平移（仅 MHA）

- Date: 2026-08-29
- Parent: v033 / C41 `8c6f4f4d331f23cd801fd55b526316db9e56596b36b64c9b002db7d9dc3659c9`（祖父母 v031 / C39-FW `b8c9f2a4…`）
- Change: 在 C41 基础上，量化感知 K 中心（`center_mode=4`）**仅在 MHA（`q_num_heads == kv_num_heads`）启用**；GQA 保持父版本行为
- Hypothesis: softmax 不变量的数学对 GQA 同样成立，C41 在 qwen 上的负向来自估计方差而非不变量本身——GQA 的 KV head 极少（qwen 为 2 个），居中后二阶矩 `k_sac_second_moment` 的估计方差大，会带偏与其耦合的 Smooth-QK 平滑系数。按结构差异禁用即保留 MHA 收益、消除 GQA 损失。
- Test command: `.\.venv\Scripts\python -u evaluator\real_model_suite.py --candidates c39 --solution solution.py --candidate-name c41b --device cpu --algorithm-device cuda --cache-mode read --seq 128 --calib 2 --test 4 --output artifacts\real_model_suite\c41b-full-20260829.json --report logs\evaluations\2026-08-29-c41b-full.md`
- Test config: cache_mode=read，seq=128，calib=2，test=4，全层，device=cpu / algorithm-device=cuda，协议 v2；报告 `logs/evaluations/2026-08-29-c41b-full.md`
- Source SHA256: `c1e68a5ba9ed798a582618758e45261ccd7c1426ce8f0b8b02c235664ed859c6`
- Cache: `artifacts/real_model_suite/cache/*__seq128__calib2__test4__layersall__schema1.pt`，schema 1；数据集 `wikitext-2-raw-v1` revision `b08601e04326c79dfdd32d625aee71d232d685c3`
- Revised official score: **21864**（250 Linear + 200 Attention cases）
- Revised official runtime: **159.4s**
- Revised panel time limit: **420s**（7 分钟）
- Legacy local status: `local-accepted`（晋级；五模型无一负向）
- Revised official status: `official-compliant-anchor`（与 v031/C39-FW 同分，快 1.9s）

## 结果（official_flow_total，逐 case 求和）

| 模型 | 结构 | c39 attention | c41b attention | Δ attention |
|---|---|---:|---:|---:|
| gpt2-small | MHA | 20.969992 | 21.120464 | **+0.150472** |
| gpt2-medium | MHA | 43.441214 | 43.767156 | **+0.325942** |
| opt-125m | MHA（未选中 mode 4） | 19.581565 | 19.581565 | 0.000000 |
| pythia-160m | MHA（未选中 mode 4） | 40.614368 | 40.614368 | 0.000000 |
| qwen2.5-0.5b | GQA 14Q/2KV | 62.862350 | 62.862350 | **0.000000**（C41 为 −0.550010，已消除） |

总分：c39 `996.745557` → c41b `997.221971`，Δ **+0.476414**。

- **Linear 五模型逐位不变**（129.343309 / 227.420890 / 49.205223 / 136.837139 / 266.469507），机制严格限于 Attention。
- **五模型无一负向**，满足 C40 诊断确立的"多模型方向一致"晋级门。
- 最慢模型 API 时间：c39 `70.98s` → c41b `70.71s`（无增加；GQA 下直接跳过求解反而略省）。

## 与 v033（C41）的差异

仅 18 行 diff，全部围绕 GQA 分支：

- 新增 `_ATTN_SCALE_AWARE_CENTER_GQA = False`；
- 求解 `sac_center` 的前置条件加入 `q_num_heads == kv_num_heads`；
- 候选循环对 `center_mode == 4` 增加运行时 GQA 跳过。

## 修订版官方结果

官方在新版 250/200 样例集上确认 C41b 为 `21864 / 159.4s`。该结果覆盖
旧文档中的 `Official score: NA`；本地五模型 proxy 仍只用于机制排序。

## 结论与下一步

C41b 是旧本地 proxy 下的改进，但新版官方分数与 C39-FW 持平；当前本地
归档官方冠军为 v051/C47b（`22451 / 234s`）。

增量规模评估：+0.476 / 996.7 ≈ **+0.048%**。这属于"小而稳定的正增量"，远不足以承担 14613 → 20000 的主增量（需 Linear mean 约 0.5357 → 0.68）。后续主攻方向：

1. **C42 Q/K 可逆平衡**（每 head 4×4）：同为精确 `QK^T` 不变量，零数学风险，Attention 仍有空间（opt/pythia 的门控未选中 mode 4，说明其 K 分布与 midrange/scale-aware 中心都不匹配）。
2. **Linear alignment（CAT，arXiv 2603.04359）**：Linear 占总分约 81%，是唯一可能承担主增量的方向；现有 rotation/Hadamard 只改善 concentration，alignment 维度工程尚未覆盖。
3. 官方提交验证：候选需在新版 `420s`（7 分钟）内并通过合规检查，再提交以
   校准本地→官方兑换率。
