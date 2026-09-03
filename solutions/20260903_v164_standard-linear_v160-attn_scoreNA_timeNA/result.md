# v164 候选：标准 Linear + v160 Attention（两侧比重校准实验）

> 状态：**CANDIDATE — 本地验证通过，等待一次官方提交**
>
> 父版本：v160 归档，SHA `33B1D061CE6BFCD92659C597BE4830BB9B910E646FF518433DA67B925AE8680D`
> （官方 `17532 / 232s`）
>
> 候选 SHA256：`896B4ACA9F9F0C55D91C439E628B59D0B04D3BD77E23AA6F17144B0D665793D7`
>
> 官方结果：`unregistered / NA`

## 1. 构造方式

v160 归档原文件（10297 行，零改动）+ 末尾追加标准 encode 辅助
（`_ref_solve_standard_hierarchy` / `_ref_encode_standard_hif4`，照抄
`evaluator/reference_hif4.py`，复用 v160 已有的 `_standard_e6m2_scale`）与两个
Linear API 的模块级重定义（Python 后定义覆盖前定义）：

- `hif4_calibration_and_quantize_weight` → 忽略校准样本，
  `weight_params = 标准编码(NVFP4 解码 weight)`，`activation_state = {}`；
- `hif4_dynamic_quantize_activation` → 标准 codec。

v160 的 Attention 代码路径完全未动，输出保持与 v160 归档逐位一致。

## 2. 实验角色

测量 **Attention 优化的官方贡献**：Δ_A = `S(v164) − S(v162)`。与 v163（反向）、v162
（全标准）共同解出官方两侧比重并检验可加性 `S(v163)+S(v164)−S(v162) ≈ 17532`。

## 3. 本地验证

| 项目 | 结果 |
| --- | --- |
| 隔离导入 + 六 API | OK |
| 标准 codec vs reference（CPU+CUDA） | 逐位一致 |
| attention-only default 120 | attention_mean **0.742353635**，120 case 与 v160 逐位一致（max Δgain/Δmse = 0.0，sum `89.082436179` 相等） |
| linear-only default 168 | linear_mean **0.0**（全部 case gain = 0） |
| API total | attention-only 67.7s / linear-only **1.682s** |

证据：`sidecal-v164-attn-default.json`、`sidecal-v164-linear-default.json`（`artifacts/official_eval/`）。

## 4. 时间预算

完整调用图 = 标准 Linear（≈2s）+ v160 Attention（本地 67.7s）≈ 70s 本地；官方预计
**~70s < 300s**，时间风险可忽略。

## 5. 官方提交与解释（预注册）

一次官方提交。Δ_A = `S(v164) − S(v162)` 为 Attention 侧官方边际；判读表见
[`活动计划`](../../docs/superpowers/plans/2026-09-03-official-side-weight-calibration-plan.md) §3。

## 6. 复现

```powershell
.venv\Scripts\python.exe -u evaluator\official_eval.py --solution solutions\20260903_v164_standard-linear_v160-attn_scoreNA_timeNA\solution.py --attention-only --cache-mode read --nvfp4-cache-mode auto --capture-device cuda --algorithm-device cuda --output artifacts\official_eval\sidecal-v164-attn-default.json --report logs\official_eval\sidecal-v164-attn-default.md
```
