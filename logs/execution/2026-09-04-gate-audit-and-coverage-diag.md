# 方向 3 门控审计 + 方向 1 截断率统计（2026-09-04，零配额本地诊断）

目的：针对榜首差距 4167 的机制假设审计。方向 3 = 跨模型门控 firing 率
（Qwen vs GPT-2 vs OPT-125m）；方向 1 = 校准侧覆盖率 caps 的本地余量测量
（COV-A 变体）。全部为本地诊断，不占官方配额，不产生版本号。

## 工具

- `logs/execution/diag_gate_wrapper.py`：加载 v182 归档（不动原文件），
  monkey-patch `_candidate_is_safe`（按 `min_mean_improvement` 分标签）、
  `_a1_gate_passes`、`_fit_attention_pair_matrix_smooth`，profile
  `_dense_to_hif4` 调用（按 refine ratio/blocks），atexit 打印。
- `logs/execution/diag_covA.py`：v182 副本，仅提高 12 个校准侧 cap
  （diff 已核验 caps-only；在线 Q/K/V/activation refine ratios 未动）。

## 方向 3 结果：跨模型门控 firing 率

| 门 | Qwen | GPT-2 | OPT-125m |
| --- | ---: | ---: | ---: |
| `_candidate_is_safe` 总体 | 93.8% (1119) | 80.6% (583) | 87.8% (596) |
| **weight block smooth (mmi=0.005)** | **92.9% (425)** | **68.4% (196)** | **70.1% (184)** |
| smooth 候选 (mmi=0.01) | 97.4% | 86.9% | 97.0% |
| mmi=0.02 | 88.5% | 86.3% | 90.5% |
| attention block (mmi=0.001) | 98.1% (54) | 88.5% (26) | 100% (14) |
| A1 终验门 | 24.4% (336) | 45.8% (24) | 37.5% (24) |
| pair-matrix smooth 接受 | 24/24 | 12/12 | 12/12 |

证据：`logs/execution/diag-qwen-linear.out`、`diag-gpt2.out`、`diag-opt.out`
（Qwen attention 数字来自 diag-v182-attn-default 运行 stdout）。

**判读**：

1. **weight block-smooth 门是最强的 Qwen-locked 嫌疑**：Qwen 接受 92.9%，
   GPT-2/OPT 仅 68-70%。非 Qwen 模型上约 30% 候选达不到 mmi=0.005
   （≥0.5% 平均改进）被拒。A1 本地负/官方正已证明官方模型 ≠ Qwen；
   若官方模型行为靠近 GPT-2/OPT，该门在官方模型上系统性更严。
2. A1 终验门方向相反（Qwen 更严 24.4%），是多候选筛选门，非 Qwen-locked。
3. pair-matrix 三模型 100% 接受，无 Qwen-lock。
4. gate 放宽（mmi 降低）方向不确定：Qwen 上已 92.9% 接受、放宽对 Qwen 无效，
   对官方模型的净影响未知（可能引入有害接受）。列为 v183 之后视结果的方向。

## 方向 1 结果：COV-A 覆盖率放大变体（Qwen default 配对 v182/v180）

| 侧 | 改动 caps | 结果 |
| --- | --- | --- |
| Linear 168 | weight quad8/16 ratio 0.05/0.02→0.30/0.12、groups 4×、activation quad8/16 0.08/0.10→0.40/0.50、C75 0.25→1.0 | **0/0/168 bit-identical no_effect**（API 337.9s，+5.5s） |
| Attention 120 | `_ATTN_BLOCK_SMOOTH_REFINE_RATIO 0.50→1.00`、`REFINE_BLOCKS 24_576→131_072`（attention-only 运行天然隔离 linear caps） | **mean +0.000511、11+/14−/95=、median 0、时间中性**（wall 69.1s / API 63.9s） |

证据：`artifacts/official_eval/diag-covA-{attn,linear}-default.json`、
对应 `logs/official_eval/diag-covA-*.md`。

**判读**：

1. **Linear 侧校准覆盖无余量**：weight/activation quadratic 与 C75 caps 在 Qwen
   上完全不 binding（168 case 全部位不变）。激活 adaptive refine ratio 实测
   0.98-0.99（`_LINEAR_RATIO_CAPTURE_TARGET=0.999` 已近满）。
2. **唯一有正余量的 cap 是 attention block-smooth search 的 refine 覆盖**
   （ratio 0.5→1.0）：改变 25/120 case 输出，净 +0.000511，校准时间中性。
   量级与 D1 的 local +0.000356（官方 +3）同阶。
3. 在线 Q/K/V refine（0.60/0.70/0.60）与 activation refine 0.70 属在线路径，
   未测不改（官方 timeout 家族禁区）。

## 对候选清单的结论

- **v183 候选（单机制、零在线新增）**：v182 + attention block-smooth refine
  覆盖 0.50→1.00 / blocks 131072。本地弱正混合（11/14/95）按现行规则不阻止
  首次官方测量；预期小幅正（D1 同量级），校准时间中性。官方配额 4/10。
- 方向 3 的 mmi 放宽列为 v183 之后的条件方向（跨模型证据支持但不定号）。

## 运行清单

```powershell
# gate 审计（wrapper，三模型）
.venv\Scripts\python.exe -u evaluator\official_eval.py --solution logs\execution\diag_gate_wrapper.py --attention-only ... --output artifacts\official_eval\diag-v182-attn-default.json ...
.venv\Scripts\python.exe -u evaluator\official_eval.py --solution logs\execution\diag_gate_wrapper.py --linear-only ... --output artifacts\official_eval\diag-v182-linear-default.json ...
.venv\Scripts\python.exe -u evaluator\cross_model_eval.py --model gpt2 --solution logs\execution\diag_gate_wrapper.py ...
.venv\Scripts\python.exe -u evaluator\cross_model_eval.py --model opt-125m --solution logs\execution\diag_gate_wrapper.py ...

# COV-A 覆盖变体
.venv\Scripts\python.exe -u evaluator\official_eval.py --solution logs\execution\diag_covA.py --attention-only --baseline-json artifacts\official_eval\v180-attn-default.json ...
.venv\Scripts\python.exe -u evaluator\official_eval.py --solution logs\execution\diag_covA.py --linear-only --baseline-json artifacts\official_eval\v182-linear-default.json ...
```
