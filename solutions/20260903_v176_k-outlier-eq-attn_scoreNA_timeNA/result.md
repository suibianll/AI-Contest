# v176 候选：K 侧 static outlier-channel 等化（计划 C1）

> 状态：**CANDIDATE — 从 P\_A = v168 构造，加入官方批测队列（对 v164 13945 / 官方父侧 v168 14005）**
>
> 机制：KVQuant/ChannelQuant 配方——校准期在最终部署坐标检测跨 fold 稳定高幅稀疏 K
> 通道（peak/median ratio > rho=4.0），构造 per-channel equalizer `k_eq` 折进 K
> multiplier 压缩 outlier 幅值；Q 侧以连续域 `QK^T` 不变为约束取精确倒数补偿
> `q_eq = 1/k_eq`（GQA 组内按 KV head 布局展开）。零动态新增算子。
>
> 候选 SHA256：`DFA69838D8B0CC50411ADDDBF764ACC8B3D304D51C73F59FD0E61809CE5925CC2`
> （修正 GQA q\_eq 维度展开后，与 `artifacts/official_eval/archive.json` 最新记录一致）
>
> 官方结果：`unregistered / NA`

## 1. 构造（预注册，计划 2026-09-04 C1 固定数学规则）

- 父版本：P\_A = v168（standard Linear + A1 logits gain，官方 `14005 / 210s`）；
  Linear 侧未改动。

- 校准期（final 部署坐标，复用 a1\_k 前 128 tokens）按每 KV head：

  1. `peak_j = median_f(amax_t |K_f,t,j|)`、`med_j = median_f(median_t |K_f,t,j|)`；
  2. outlier 检测：`peak_j / med_j > rho`（rho=4.0 固定）且跨折符号一致
     （通过 cross-fold median 聚合实现）；
  3. `k_eq_j = (target / peak_j)`（outlier 通道），`target = median(peak)`；
     其余通道 1.0；平滑收缩 `k_eq = 1 + beta·(k_eq−1)`，beta=0.25 固定；
  4. `k_multiplier *= k_eq`；`q_multiplier *= q_eq`，其中
     `q_eq = (1/k_eq)` 从 per-KV-head 布局 `(kv_heads, head_dim)` 经
     `repeat_interleave(group)` 展开到 Q 通道布局 `(q_heads·head_dim,)`
     ——连续域 `QK^T` 内积不变，仅重分配两侧量化动态范围；
  5. 不搜索 rho/beta/通道数；不做 head/layer 路由；无 per-call 精化。

## 2. 本地验证（描述性；官方裁决）

| 项目                             | 结果                                                                                                 |
| ------------------------------ | -------------------------------------------------------------------------------------------------- |
| 隔离导入 + 六 API                   | OK（通过 `_check_attention_state` 合法 state 检查）                                                        |
| 机制 reachability                | outlier 注入压力测试：outlier 通道得到 `k_eq < 1` 压缩，`q_eq = 1/k_eq` 放大；state 含 `k_outlier_eq`/`q_outlier_eq` |
| attention compact 4（配对 v168）   | **mean Δgain +0.002444**、median +0.003312、3+/1−/0=；QK-only +0.0029、QK interaction +0.358           |
| attention default 120（配对 v168） | **mean Δgain −0.004450**、median −0.001634、56+/64−/0=（win 0.4667）；QK interaction +50.77 强正          |
| gpt2 attn compact 4（配对 v160 父） | **mean Δgain −0.002753**、median −0.015484、1+/3−/0=；QK-only +0.002406 为正                 |
| opt-125m attn 60（配对 v160 父）     | **mean Δgain −0.021851**、median −0.001365、30+/30−/0=（win 0.5）；QK-only −0.0246；Linear 侧差异来自父结构不同不可归因 |
| control                        | V 侧 `v_only_gain = 0.0` 未改动；Linear 未执行                                                             |
| API 时间                         | attention default：校准 60.15s（v168 基线 68.40s）、动态 Q/K/V 3.36s；零新增在线算子，无时间风险                           |

**分层/长度分解（default 配对）**：L16 consistent\_improvement（+0.0727，win 1.0），
L3/L11/L14/L20 小幅正向，L4 consistent\_regression（−0.0039）；len10 最负
（−0.0184，win 0.33），len128 −0.0026、len512 −0.0019、len1024 +0.0003
（长度越长越中性/正）；worst 集中在 len10 的 validation 窗口（layer 19/17/23
−0.147/−0.110/−0.106）。

## 3. 判读（计划 §2 排序冻结 / v165 约束）

- 轻微本地负向不取消首次官方测量：当前计划规则只以接口/合法 state/有限输出/
  机制 reachability/control 为提交硬门禁，全部通过；算法方向由相对 v168 的官方
  分数裁决。
- 跨模型（GPT-2 compact）：Attention mean Δgain −0.002753（1+/3−），标记
  `model-specific-risk`——Qwen 与 GPT-2 在该机制上存在轻微方向差异；按计划
  第 7 步只作封存 holdout，不据此调参数/路由，仍由 v176 首次官方结果裁决。
- 跨模型（opt-125m attn 60）：mean Δgain −0.021851（30+/30−），方向一致性
  弱且整体轻微负；两架构（GPT-2/opt-125m）对 C1 均无正向跨模型信号，风险标记
  维持，仍由首次官方结果裁决。
- 若官方负：C1 家族关闭，切换 C2（A1 细粒度化），不调 rho/beta/通道数重扫。
- 若官方正：C1 晋级；组合条件维持 `S_pred = 4590 + S_c1 − 1001`，仍按计划
  §3.3 登记。

## 4. 复现

```powershell
.venv\Scripts\python.exe -u evaluator\official_eval.py --solution solutions\20260903_v176_k-outlier-eq-attn_scoreNA_timeNA\solution.py --attention-only --compact-panel --cache-mode read --nvfp4-cache-mode auto --capture-device cuda --algorithm-device cuda --baseline-json artifacts\official_eval\v168-compact-attn.json --output artifacts\official_eval\v176-compact-attn.json --report logs\official_eval\v176-compact-attn.md

.venv\Scripts\python.exe -u evaluator\official_eval.py --solution solutions\20260903_v176_k-outlier-eq-attn_scoreNA_timeNA\solution.py --attention-only --cache-mode read --nvfp4-cache-mode auto --capture-device cuda --algorithm-device cuda --baseline-json artifacts\official_eval\v168-attn-default.json

.venv\Scripts\python.exe -u evaluator\cross_model_eval.py --model gpt2 --solution solutions\20260903_v176_k-outlier-eq-attn_scoreNA_timeNA\solution.py --name v176 --attention-only --compact-panel --cache-mode read --capture-device cuda --algorithm-device cuda --baseline-json artifacts\official_eval\s1-parent-v160-gpt2-attn-compact.json --output artifacts\official_eval\v176-gpt2-attn-compact.json --report logs\official_eval\v176-gpt2-attn-compact.md

.venv\Scripts\python.exe -u evaluator\cross_model_eval.py --model opt-125m --solution solutions\20260903_v176_k-outlier-eq-attn_scoreNA_timeNA\solution.py --name v176 --cache-mode read --capture-device cuda --algorithm-device cuda --baseline-json artifacts\official_eval\v160-opt-parent.json --output artifacts\official_eval\v176-opt-integration.json --report logs\official_eval\v176-opt-integration.md
```

