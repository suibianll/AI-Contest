# 官方用例样本 vs 本地评测系统审计（2026-09-02）

> 数据来源：用户提供的官方 mini 样本 `linear.pt` / `attn.pt` 结构信息。两者各含 1 个
> case（1 Linear + 1 Attention），按官方说明仅用于接口/格式验证，不用于算法调优。
> 本记录只做契约与结构比对，不改写任何既有 JSON / report / 归档版本。

## 1. 官方用例结构摘要

### 1.1 Linear 场景（linear\[0]）

```
key = "linear"
weight = [quant, scale]                    # NVFP4 对
calib_activation_list = [fold]×5           # 校准激活 5 折
test_activation_list = [fold]×5            # 测试激活 5 折
```

- weight.quant `[8192, 2048]` bf16，值域 \[-6, 6]

- weight.scale `[8192, 128]` bf16，值域 \[0.002, 0.059]；block\_size = 2048/128 = 16

- rows=8192 > cols=2048，4:1 expansive 矩阵

- calib / test 形状一致，均为 NVFP4 对 \[quant, scale]，channels = 2048 与 weight cols 匹配

- 每折序列长度：`[10, 128, 512, 1024, 1024]`

### 1.2 Attention 场景（attn\[0]）

```
key = "attn"
q_num_heads = 16, kv_num_heads = 2, head_dim = 256, attn_type = "gqa"
calib = [{q,k,v}]×5
test  = [{q,k,v}]×5
```

- Q `[seq, 4096]` + scale `[seq, 256]`（16×256=4096，块 16）；

- K/V `[seq, 512]` + scale `[seq, 32]`（2×256=512，块 16）；

- group\_size = 16/2 = 8；

- 每折序列长度 `[10, 128, 512, 1024, 1024]`；

- Q scale 值域 \[0.078, 1.625]、K \[0.141, 1.625]、V \[0.047, 0.625]。

## 2. 官方样本与本地实现的位置对应

| 官方组件       | 本地实现                                                                                      |
| ---------- | ----------------------------------------------------------------------------------------- |
| NVFP4 编码配方 | `evaluator/nvfp4_sim.py`（`nvfp4_encode`、`e2m1_round_to_even`、`e4m3_round_up`）             |
| NVFP4 解码   | `evaluator/reference_hif4.py`（`dequantize_nvfp4`）                                         |
| 评测器调用图     | `evaluator/official_eval.py`（`proxy-v2`）                                                  |
| 本地结构假设     | `models/qwen2.5-0.5b/config.json`；`official_eval.py` 中 `MODEL_NAME`、`CALIBRATION_LENGTHS` |
| 候选 API     | 根 `solution.py` 六个公共 API                                                                  |

## 3. 契约层一致性（结论：与官方用例一致）

| 官方用例                         | 本地实现                                                                                                | 判定                                 |
| ---------------------------- | --------------------------------------------------------------------------------------------------- | ---------------------------------- |
| quant 值域 \[-6, 6]            | `nvfp4_sim.py` E2M1 = {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}，mode=`amax6`                              | 一致                                 |
| block\_size = 16             | `nvfp4_encode` 每 16 元素一块；`dequantize_nvfp4` blk\_size=16，scale 形状 `[..., channels/16]` 校验           | 一致                                 |
| scale 为 E4M3 含小值             | `e4m3_round_up` 从 subnormal 2^-9 起，未夹到正规数 2^-6                                                      | 一致                                 |
| 折叠长度 \[10,128,512,1024,1024] | `CALIBRATION_LENGTHS` 完全相同；测试窗口也含这些长度                                                               | 一致                                 |
| 变长结构                         | 官方注释确认其导致旧 `_build_pawv_metric` 崩溃；本地已弃用 PAWV（v128–v131 路线关闭）                                       | 一致（无副作用）                           |
| GQA 计算                       | `official_eval.py` `_attention`：`group = q_heads // kv_heads`，KV `repeat_interleave`，scaled softmax | 一致；heads/head\_dim 作为参数传入 API，结构无关 |
| Attention 校准折数               | `evaluate_solution` 中 `attention_calibration_indices = range(5)`，传全部 5 折                            | 一致                                 |
| API 维度通用性                    | 根 `solution.py` 无 Qwen 形状硬编码（896/4864/151936 等字面量不存在；14/64 均为 HiF4 block/算法常数）                      | 官方维度差异不破坏 API                      |

## 4. 差异与风险点（本地代理假设，非官方等价物）

1. **模型结构形状不匹配（最实质）**：

   - 本地 Qwen2.5-0.5B：hidden 896、q14/kv2/head\_dim 64、fc\_up \[4864, 896]、down \[896, 4864]；

   - 官方 Linear sample weight \[8192, 2048]（cols 2048），与 Qwen 任何层不重合；

   - 官方 Attention sample Q width 4096（16×256），非 Qwen 的 896。

   - ⇒ 官方隐藏结构 ≠ Qwen2.5-0.5B。本地 168+120 case 构成与官方 case 构成无对应关系；
     proxy 只做同机趋势/机制诊断，不能用于官方排序（与既有结论一致，此处由官方样本直接佐证）。
2. **Linear 校准折数**：本地默认只用 2 折（linear\_calibration\_indices=\[0,1]，compact 为 \[1,2]）；
   官方每 case 提供 5 折 calib。若官方把全部 5 折传入 API，本地候选校准数据量少于官方，
   校准质量与 runtime 估计会偏离。
3. **Linear 动态视窗数**：本地每 layer/role 仅 1 个 test window（168 个动态 case）；
   官方每 case 提供 5 折 test（同长度分布）。官方可能按多个 test fold 计分。
4. **dtype 微差**：本地 NVFP4 对以 float32 存储，官方为 bf16；
   E2M1/E4M3 值均可被 bf16 精确表示，仅反量化乘法舍入位置差约 1 ulp，可忽略。

## 5. 结论

- 本地评测系统在 API 数据契约层（NVFP4 编码、块布局、折叠长度、GQA 参数化、NVFP4 对布局）
  与官方用例一致，架设方向合理；

- 结构形状、Linear 校准折数（2 vs 5）、test fold 数（1 vs 5）是本地代理假设；
  任何本地分数只做同机诊断，官方排序与官方 `<300s` 必须由官方回传确认；

- 官方样本的 expansive 4:1 形状支持「fc 系为重点」的方向性推断，但通道数不同，
  逐层推断仅是方向证据。

## 6. 后续（未执行，仅记录选项）

- 若要把 Linear calibration 折数从 2 对齐 5，会改变全部既有 baseline 的调用图与 API 时间，
  应作为一次正式评测器变更（走配对重跑流程），不是小修补；本审计不发起该变更。

## 7. 5 折校准实验与回退（2026-09-02）

- 曾把 Linear 校准改为全部 5 折（对齐官方 sample `calib_activation_list`），并跑完全部 10 个
  有官方成绩版本的 default panel（JSON：`artifacts/official_eval/reeval5-*.json`，report：
  `logs/official_eval/reeval5-*.md`，汇总：`2026-09-02-reeval5-10versions-default-trend.md`）。

- 用户判定 5 折对分数规律无增益且总时间更长，已**全部回退到 2 折**（默认 `[0,1]`，
  compact `[1,2]`）；`git diff HEAD` 对评测器与测试为空，代码已与 HEAD 一致。

- 5 折批量 JSON 保留为审计证据，其 `data_metadata.linear_calibration_indices=[0,1,2,3,4]`，
  与当前 2 折协议（`[0,1]`）不可直接比较；其规律结论（官方分数主要跟随 Attention、
  本地 Linear 与官方负相关）不依赖折数选择。

- 风险点 2 仍作为「本地代理假设」保留，不再推进修复。

