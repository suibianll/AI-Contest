# Qwen3.5-4B 结构代理面板（工作包）

> 日期：2026-09-07。用户批准：「后续测试完全使用4B模型的数据，不需要那么多OOD门禁等测试了」。
> 背景：官方评测模型经用户确认为 **Qwen3.5-35B-A3B**（FP8 权重已在 `models/qwen3.5-35b-a3b-fp8/`，
> config.json 实读）。本地 35B 不可行（32GB RAM / 消费级 GPU），按同架构族小模型方案执行。
> **状态（2026-09-07）：R0/R1/R2 全部通过**，执行记录见
> `logs/execution/2026-09-07-qwen35-4b-panel-r0-r1-r2.md`。基线读数：
> Linear 336c mean 0.524265 / Attention 72c mean 0.526517（v189，overall 0.524663）。
> **操作指引**：日常测试怎么跑见 [`docs/4b-panel-testing-guide.md`](../../4b-panel-testing-guide.md)。

## 1. 假设与证据边界

- 官方模型 = Qwen3.5-35B-A3B（用户确认）。结构：多模态 MoE（256 选 8 + shared，专家
  intermediate 512）、40 层 hybrid（3 linear_attention + 1 full_attention 循环）、hidden 2048、
  full-attn 16Q/2KV head_dim 256、partial RoPE 25%、attn_output_gate、vocab 248320。
- 本地面板改用 **Qwen3.5-4B**（`models/qwen3.5-4b/`，modelscope 2026-09-07 下载）：同族 hybrid
  布局 3:1、DeltaNet 32V/16QK×128 与官方一致、full-attn head_dim 256 / RoPE 64（25%）一致、
  同 tokenizer。差异：dense（FFN 9216，无 MoE）、hidden 2560、KV 4 头（GQA 4:1 vs 8:1）。
- **4B 面板不提供官方分数预测能力**：MAE≈1108 的定量拟合禁令（2026-09-04 修订）与模型选择
  无关；4B 本地正向只说明机制在同族结构上不失效。禁止把 4B 面板结果当晋级旁证
  （跨模型探针"同号旁证"教训，2026-09-04 修订 §1）。
- P3 官方分桶（W2/W3 全部增益、W0/W1 精确零）与官方结构自洽（无方形权重、k/v rows=512>256）；
  4B 面板保持同一桶拓扑（无方形、无 rows≤256），DeltaNet in_proj 切片另引入近方形 [2048,2560]
  形状族——0.5B 面板完全没有的覆盖。

## 2. 面板设计（qwen35-4b-panel-v1）

- **24/32 层**：blocks {0,1,2,4,5,7}（丢 blocks 3、6）→ 18 DeltaNet + 6 full-attention。
- **分片平衡排序**：FA 位于 panel 位置 {0,1,5,8,15,22}，使 proxy-v3 strided 分片
  （offset 6）每片恰 1 FA + 3 DN。
- **7 role 统一 schema**：FA 层 q/k/v/o = self_attn 投影；DN 层 q/k/v = in_proj_qkv 的连续
  q/k/v 行切片（[2048,2560]×2 + [4096,2560]），o = out_proj（[2560,4096]）；mlp 三件套稠密
  [9216,2560]×2 + [2560,9216]。in_proj_z/b/a 与 conv1d 非 Linear role，排除。
- **Attention 场景仅 FA 层**：官方 Q/K/V API 合同（单一 q_heads/kv_heads/head_dim + softmax
  attention 评分）无法表达 DN 层（q/k 16×128 vs v 32×128）。FA Q/K operand = post q_norm/k_norm
  + partial RoPE（与进注意力的一致）；v = v_proj 输出按 head-major 展平。
  Linear role 权重 = 原始模块权重（FA q 取双宽 q_proj 的 q 半行 [4096,2560]，sigmoid gate 半
  不作 role），Linear case 仍评 X@W^T，与 0.5B 面板语义一致。
- **fp16 存储**：forward 输出本为 fp16/bf16，`nvfp4_encode` 首步 `.to(float32)`，fp16 存储
  与 0.5B 的 fp32 升宽逐位等价，体积减半。
- **窗口裁剪**：校准激活仅存窗口 {0,1}（Linear 两折，`proxy_v3_eval.py:84` 与
  `official_eval.py:966` 均取前两窗）；**校准 QKV 存全部 5 窗**（attention 校准用
  `tuple(range(len(calibration_windows)))`，`official_eval.py:2559`）；测试激活仅存
  {1,2,6,7}（proxy-v3 compact pair，`COMPACT_WINDOW_INDICES`）；**测试 QKV 存全部
  12 窗**（`test_qkv_windows`，2026-09-07 构成修正——attention 侧从 12 例扩到 72 例，
  见执行日志 §6）。其余激活槽位 None，v3 shard 流程不会触碰；12 窗 Linear
  default/`--full-cases`/v2-flow `--compact-panel`（其 Linear 校准取 {1,2}）在此 cache
  上会显式报错（超出范围）。
- **两侧构成（修正后）**：Linear 336 例（24 层×7 role×2 窗）: Attention 72 例
  （6 FA×12 窗）。DeltaNet 不进 attention 场景是官方 Q/K/V API 合同的结构约束；
  DeltaNet in_proj/out_proj 计入 Linear 与官方权重族一致。
- 产物：`artifacts/official_eval/cache/qwen3.5-4b-proxy-v2.pt`（约 6–9GB）+
  sidecar `*.panel.json`（面板几何，供 `--reuse-existing` 快速取数）。

## 3. 代码入口与改动

- 新增 `evaluator/capture_qwen35.py`：加载 Qwen3.5-4B（显式 `Qwen3_5ForConditionalGeneration`
  优先）、hook 抓取、面板选择、`save_pack` + sidecar。
- **捕获实现记录（2026-09-07，两次失败后修正）**：
  1. **hook 覆盖 bug（致命，已修）**：初版所有 panel 层共用一组 hook key（`hidden_in`/`qn`
     等），单次全模型前向中逐层互相覆盖——循环里 24 层读到的都是最后一个 panel 层的激活
     （第二次运行若不 OOM 跑完，缓存会是垃圾数据；OOM 反而挡住了落盘）。修正：per-index
     key（`f"{panel_index}:..."`），hook 内即时 `_cpu16` 搬 CPU。
  2. **CUDA OOM 根因（已修）**：本机 GPU 仅 8GB（7GB 空闲）< 4B fp16 模型 ~8GB，前 15 窗
     能跑是 Windows WDDM sysmem fallback 撑着，必然崩（报 raw `CUDA error: out of memory`，
     连 `empty_cache` 都失败）。nvidia-smi 期间还报 NVML 错。修正：**默认 `--device cpu`**
     （bf16 前向，bf16→fp16 转换在值域内精确），hook 内即时落 CPU 使 GPU 不驻留激活；
     序列化前先释放模型（RAM 峰值 = pack 而非 pack+model）。
  3. RoPE 修正（第一次失败的修复）：cos/sin 保持 fp32 搬 CPU，RoPE 在 CPU 重放，
     Q/K 存 fp16。非存储测试窗口跳过抓取循环但仍跑前向（深层需要上下文）。
  4. **2-D 形状修正（第三次运行前修复）**：hook 内 `_cpu16` 保留 `[1, seq, width]` 3-D，
     `run_window` 重构时漏掉 hidden_in/o_in/fc_gate_in/proj_in/v 的 `_flat16`——首次 R0
     load_pack 校验即拦下（`calibration_activations[q][0][0] is not a 2-D tensor`）。
     校验器按预期工作了；补 `_flat16` 后重跑捕获。
- `evaluator/official_eval.py::load_pack`：新增 `panel_schema == "qwen35-heterogeneous-v1"`
  分支——权重按 metadata `weight_shapes` 逐层校验（替代 0.5B 方形假设）、激活槽按声明窗口
  校验（其余必须 None）、QKV 仅 FA 层校验。**无该 key 的 pack 走原路径，逐字节行为不变。**
- `evaluator/proxy_v3_eval.py::prepare_shard`：attention 层按 `metadata["attention_layers"]`
  过滤（legacy 无 key → 全层，行为不变）；metadata 记录 `attention_layers_pool`。
- `evaluator/eval_system.py`：`_expected_cases_for_shards` 参数化（legacy 默认 4/4 不变）；
  `_panel_geometry`（pack metadata 或 sidecar）+ `_expected_from_geometry` 接入 coverage 检查。

## 4. 工作流变更（用户批准，2026-09-07 生效）

1. **机制测试主面板 = 4B**。paired Δmean / L1 / 负向 case / control / 分组诊断全部在 4B
   面板跑；基线须用同面板重建（v189/R3 父在 4B 面板的 paired 基线是第一项标定任务）。
2. **OOD 精简**：不再逐候选强制 OOD；降级为可选诊断（重大机制变更或怀疑分布拟合时手动
   跑一次）。与 2026-09-07 OOD 政策修订（不作一票否决）方向一致并更进一步。
   **注意：4B 面板未抓 OOD 窗口**（capture 只抓 in-dist 12 窗）——4B OOD 诊断启用前
   须先扩展 capture_qwen35.py（未来项，登记后实施）。
3. **4B 面板数值不与 0.5B 面板数值混排**（协议断层纪律同 reeval5 教训）。历史 0.5B 面板
   结论（P3 分桶、R2 门禁数据、官方归因）全部保持有效，不受影响。
4. **时间门口径（2026-09-07 20:16 用户指令后更新：全面 4B 化）**：0.5B 面板退役，
   时间门改 **4B paired Δ**——候选与父同机同面板 fresh 全量 run 的六 API 实测差 ΣΔ
   加到父官方时间上作预测，<280s 才提交。0.5B 系数不可代入 4B 秒数（v189 4B 实测
   代入得 ~465s vs 官方 275s，自证失效）；Δ 1:1 传递是当前假设，每次官方回传登记
   残差，配对积累后评估 4B 重拟合。v189 4B 时间锚点 992.1s（六 API 明细见
   [`docs/4b-panel-testing-guide.md`](../../4b-panel-testing-guide.md) §1）。
5. 符号门（Δmean>0 且 L1<0.02）、机制类型先验（解析等价变换优先）、单侧隔离纪律、
   官方晋级须分数/时间/SHA 确认——全部不变。

## 5. 验收（本工作包）

- R0 抓取完成：pack + sidecar 落盘，`load_pack` 通过 panel-schema 校验。
- R1 冒烟：根 `solution.py`（v189 栈）在 4B cache shard0 `--scenario both`：
  56 Linear + 2 Attention case，六 API 全部执行、合法 state、有限 gain、coverage 检查 true、
  无形状崩溃（含 DeltaNet 近方形权重与 FA 双宽 q_proj 路径）。
- R2 标定：R3 父机制（或其 4B 兼容版）在 4B 面板全六 shard paired 基线，产出首个 4B 面板
  attention/linear 基线读数；结果写执行日志。
- 失败处理：抓取 OOM → 关闭其他 GPU/RAM 会话重试；transformers 兼容问题 → 记录版本并按
  最小 patch 处理；冒烟失败 → 修 capture/evaluator，不绕过校验。

## 6. 明确不做

- 不在本地跑 35B（权重仅作结构证据与未来对照）。
- 不重建 12 窗全量 default 面板（除非用户要求）。
- 不做 4B→官方分数换算或回归。
- 不改根 `solution.py`、不注册算法候选。
