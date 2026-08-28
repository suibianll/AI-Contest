# v033 — C41 scale-aware K 公共平移

- Date: 2026-08-29
- Parent: v031 / C39-FW `b8c9f2a4eb6553367dd17e73d30836ac8911dbef33759fa8cf95e8c629317a71`
- Change: Attention 的 K 公共平移中心由固定 midrange 扩展为量化感知中心（`center_mode=4`），用不动点迭代求解
- Hypothesis: `K' = K - 1c^T` 是精确 softmax 不变量，因此中心可以只针对 HiF4 重构误差优化；centering 后 K 的动态范围更贴合 E2M1 网格，降低 K 量化误差而不被 softmax 放大
- Test command:
  - `.\.venv\Scripts\python -u evaluator\real_model_suite.py --candidates c39 --solution solution.py --candidate-name c41 --device cpu --algorithm-device cuda --cache-mode read --seq 128 --calib 2 --test 4 --output artifacts\real_model_suite\c41-full-20260829.json --report logs\evaluations\2026-08-29-c41-full.md`
  - 单模型预筛：`--models gpt2-small`（同参数）
- Test config: cache_mode=read，seq=128，calib=2，test=4，全层，device=cpu / algorithm-device=cuda，协议 v2
- Source SHA256: `8c6f4f4d331f23cd801fd55b526316db9e56596b36b64c9b002db7d9dc3659c9`
- Cache: `artifacts/real_model_suite/cache/*__seq128__calib2__test4__layersall__schema1.pt`，schema 1；数据集 `wikitext-2-raw-v1` revision `b08601e04326c79dfdd32d625aee71d232d685c3`
- Official score: NA
- Official runtime: NA
- Status: `local-rejected`（未达晋级门：五模型方向不一致）

## 结果（official_flow_total，逐 case 求和）

| 模型 | c39 linear | c41 linear | c39 attention | c41 attention | Δ attention |
|---|---:|---:|---:|---:|---:|
| gpt2-small | 129.343309 | 129.343309 | 20.969992 | 21.120464 | **+0.150472** |
| gpt2-medium | 227.420890 | 227.420890 | 43.441214 | 43.767156 | **+0.325942** |
| opt-125m | 49.205223 | 49.205223 | 19.581565 | 19.581565 | 0.000000 |
| pythia-160m | 136.837139 | 136.837139 | 40.614368 | 40.614368 | 0.000000 |
| qwen2.5-0.5b | 266.469507 | 266.469507 | 62.862350 | 62.312340 | **−0.550010** |

总分：c39 `996.745558`，c41 `996.671962`，Δ **−0.073596**。

API 时间（每模型代理，均 <300s）：c39 最慢 `72.73s`，c41 最慢 `74.48s`（+1.75s，+2.4%）。

## 结论

- **Linear 在全部五模型上逐位不变**，证明该机制严格位于 Attention 路径，无副作用。
- **MHA 模型全部正向或中性**：gpt2-small +0.72%、gpt2-medium +0.75%；opt-125m 与 pythia-160m 的门控未选中 mode 4，结果与父版本完全相同。
- **唯一 GQA 模型（qwen2.5-0.5b，14Q/2KV + RoPE）显著负向 −0.550**，抵消了 MHA 的收益并导致总分小幅下降。

按 C40 诊断确立的规则（候选必须在多模型上方向一致），本候选**不晋级**。

## 失败归因（下一步的实验假设）

softmax 不变量的数学对 GQA 同样成立，因此负向不是来自不变量本身，更可能来自以下两点的交互：

1. **Smooth-QK 的统计量不匹配**：mode 4 使用居中后的 `k_sac_second_moment` 参与构造平滑系数 `d`。GQA 下 KV head 只有 2 个，居中后二阶矩的估计方差大，可能让平滑系数偏离，从而选中次优组合。
2. **门控代理在 GQA 上失效**：`_candidate_is_safe` 的门限（mean 改善 1%、worst 容差 2%）是按 MHA 校准的；GQA 的 K head 样本少，代理指标与真实 Attention 误差的相关性更弱。

据此，C41b 的候选方向（按优先级）：

- 仅在 `kv_num_heads == q_num_heads`（MHA）时启用 mode 4，GQA 保持父版本行为；预期保留 +0.476 的 MHA 收益、qwen 归零 → 总分约 `997.22` > c39 `996.75`。
- 或改用每 Q head 分组估计 center，再按 GQA 组平均，降低估计方差。
- 或为 mode 4 单独构造 Smooth-QK 统计量，避免与居中后的二阶矩耦合。

## 实现备注

- `_solve_k_center_scale_aware`：固定量化码后 MSE 最优中心为 `c = mean_tokens(K - dequant(Q(K - c)))`，从 `c = 0` 出发做不动点迭代（3 轮，1e-6 早停），因此 identity 候选始终可选，门控后不可能劣化。
- 新增 `center_mode = 4`；state 仅在该模式被选中时携带 `center_value` 键，flag 关闭时字段集与父版本一致。
- 踩坑记录：`hif4_calibration_attention` 形参名为 `calib_qkv_list`；`_build_qk_states` 内的 `k_transform` 也需透传中心向量；模块级 `_ATTN_CENTER_MODES` 需在候选循环内再运行时判断一次 flag 才能真正回退。
- 测试同步：`test_release_candidate.py` 的 feature-off 用例加入 `_ATTN_SCALE_AWARE_CENTER = False`；`test_weight_cross64.py` 的 cross64 用例改为按能力跳过（该函数属 C40 系，C39-FW 父版本不存在）。
