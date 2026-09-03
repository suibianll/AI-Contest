# v170 候选：A3 动态 scale 搜索的静态策略编译 + standard Linear（低复杂度扩展计划第三包）

> 状态：**REJECTED（明确负优化；2026-09-04 按用户最终指示 '明确负优化的需要拒绝提交'）**
>
> 2026-09-04 终判：明确负优化——Qwen default `−0.0506`（`9+/111−`）与 GPT-2
> `−0.0551`（`1+/3−`），双模型一致系统性负向（动态 refine 为承重组件，静态编译无法
> 替代），按用户最终指示拒绝提交。不消耗官方提交。
>
> Attention 官方父侧：v168（A1 晋级，`14005 / 210s`，同日修正：初报 17248/237s 有误）；Linear = v164 standard tail 未动
>
> 候选 SHA256：`2CF06B0A5EAFF8FD9AE8543809282934DC7460A713A359130D1FE2DD370BBBDB`
>
> 官方结果：`unregistered / NA`（未提交）

## 1. 唯一算法机制（预注册，低复杂度扩展计划 §6）

`_dense_to_hif4`/`_nvfp4_to_hif4` 新增默认关闭参数 `fixed_scale_offset`：固定 E6M2
scale code 偏移 + `_solve_exact_hierarchy` 精确层级求解，不进入 search_offsets、edge
extension 与 coordinate refine。校准期 `_compile_attention_fixed_offsets` 按
Q → K → V 贪心选择 layer-global offset（候选 `(-1,0,1,2,3)`，冻结前步 winner、其余
operand 保持父编码；每候选用偶/奇两折、每样本前 128 tokens 的真实 causal attention
输出 MSE 评分 `mean + 0.25·max`，平局取绝对值小再取数值小）；三个动态 API 传
`fixed_scale_offset`，state 记 `fixed_scale_offset/offsets=空/max_refine_ratio=0`。
A1 增益完整保留（multiplier 与 v168 逐位一致）。

## 2. 机制诊断（§6.4 记录）

| 项目 | 结果 |
| --- | --- |
| Q/K/V winner（compact 4 层） | **11/12 为 0**（标准 scale 即输出最优），仅 layer 23 的 V = 1——输出感知选择确认标准 scale |
| 可达性 | Q mant 相对 v168 变化 2110–3176/8960（~25–35%）：来自精确 hierarchy 替代阈值启发式 + 去除 refine |
| A1 控制 | multiplier 与 v168 逐位一致 ✓；offsets 空、ratio=0 ✓ |
| 校准成本 | 每层 +0.17~0.66s 本地（候选评估 95 次动态调用/层，其中 85 次走低成本 fixed 分支） |
| API 时间 | default 120：69.1s vs v168 72.1s（本地 −3s：动态去 refine 的节省略超校准增量） |

## 3. 否决证据（跨模型结构性反向）

| 检查 | 结果 |
| --- | --- |
| attention compact 4（配对 v168） | mean **0.779773** vs 父 0.797753（−0.018；L23 −0.053 最差） |
| attention default 120（配对 v168） | mean **0.690913** vs 父 0.741474（**−0.0506**、median −0.0345、`9+/111−`、worst −0.31）；四个长度组全负（−0.043~−0.055） |
| GPT-2 compact 4（配对 v168） | mean delta **−0.055061**、`1+/3−`、worst −0.164；k_only −0.093、logit_mse +0.149（恶化） |

**结论**：去除动态 refine 的代价在两个架构上一致且系统性（远强于 v169 时的证据：
Qwen −0.0093/GPT-2 −0.017），触发 §12 step 9 阻止条款。refine 机制是 v160 官方
12944 Attention 贡献的承重组件；offset 搜索本身无增益空间（winner 几乎全 0）。

## 4. 纪律与后续

- 按预注册实现并仅运行一次；不调候选集/权重/token 数/折定义（邻域禁令）；
- 不提交官方；**A3 关闭，下一包 A4（矩匹配 mantissa 阈值）**；
- 战略注记：A3 原含降复杂度目标（为组合候选省时间）——否决后组合候选的动态路径
  保留完整 refine，组合时间风险（计划 §16：朴素 290s / 折扣后 ~262s）维持原判。

## 5. 证据

`v170-compact-attn.json`、`v170-attn-default.json`、`v170-gpt2-attn-compact.json`
（`artifacts/official_eval/`，对应 `logs/official_eval/` report）；winner/可达性/计时
诊断在会话记录与本文件 §2。

## 6. 复现

```powershell
.venv\Scripts\python.exe -u evaluator\official_eval.py --solution solutions\20260903_v170_standard-linear_fixed-offset-attn_rejected\solution.py --attention-only --cache-mode read --nvfp4-cache-mode auto --capture-device cuda --algorithm-device cuda --baseline-json artifacts\official_eval\v168-attn-default.json --output artifacts\official_eval\v170-attn-default.json --report logs\official_eval\v170-attn-default.md

.venv\Scripts\python.exe -u evaluator\cross_model_eval.py --model gpt2 --solution solutions\20260903_v170_standard-linear_fixed-offset-attn_rejected\solution.py --attention-only --compact-panel --cache-mode read --capture-device cuda --algorithm-device cuda --baseline-json artifacts\official_eval\v168-gpt2-attn-compact.json --output artifacts\official_eval\v170-gpt2-attn-compact.json --report logs\official_eval\v170-gpt2-attn-compact.md
```
