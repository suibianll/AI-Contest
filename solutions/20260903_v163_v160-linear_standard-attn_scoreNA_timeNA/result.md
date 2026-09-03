# v163 候选：v160 Linear + 标准 Attention（两侧比重校准实验）

> 状态：**MEASURED — 官方 Linear 边际** **`3586`（2026-09-03 用户回传，实验角色完成）**
>
> 父版本：v160 归档，SHA `33B1D061CE6BFCD92659C597BE4830BB9B910E646FF518433DA67B925AE8680D`
> （官方 `17532 / 232s`）
>
> 候选 SHA256：`3352BDEC1E858B9B71637123B8E799059C5AEB79DDD2CF6E0B14F8272C3EB612`
>
> 官方结果：**4587 / 202s**

## 1. 构造方式

v160 归档原文件（10297 行，零改动）+ 末尾追加标准 encode 辅助
（`_ref_solve_standard_hierarchy` / `_ref_encode_standard_hif4`，照抄
`evaluator/reference_hif4.py`，复用 v160 已有的 `_standard_e6m2_scale`）与四个
Attention API 的模块级重定义（Python 后定义覆盖前定义）：

- `hif4_calibration_attention` → 忽略校准，返回三个空 state；

- `hif4_dynamic_quantize_q/k/v` → 标准 codec（NVFP4 → BF16 中间解码 → 标准编码）。

v160 的 Linear 代码路径完全未动，输出保持与 v160 归档逐位一致。

## 2. 实验角色

测量 **Linear 优化的官方贡献**：Δ\_L = `S(v163) − S(v162)`。与 v164（反向）、v162
（全标准）共同解出官方两侧比重并检验可加性 `S(v163)+S(v164)−S(v162) ≈ 17532`。

## 3. 本地验证

| 项目                              | 结果                                                                                                  |
| ------------------------------- | --------------------------------------------------------------------------------------------------- |
| 隔离导入 + 六 API                    | OK                                                                                                  |
| 标准 codec vs reference（CPU+CUDA） | 逐位一致                                                                                                |
| linear-only default 168         | linear\_mean **0.633526215**，168 case 与 v160 逐位一致（max Δgain/Δmse = 0.0，sum `106.43240411465689` 相等） |
| attention-only default 120      | attention\_mean **0.0**（全部 case gain = 0）                                                           |
| API total                       | linear-only 227.4s / attention-only **0.849s**                                                      |

证据：`sidecal-v163-linear-default.json`、`sidecal-v163-attn-default.json`（`artifacts/official_eval/`）。

## 4. 时间预算

完整调用图 = v160 Linear（本地 calib\_w 166.6s + dyn\_a 60.7s）+ 标准 Attention
（空校准 + 标准编码 ≈ 1s）≈ 228s 本地。官方按 v160 官方 232s 减去 v160 Attention
份额（本地 63.4s，官方比例约 46s），预计 **\~186s < 300s**。实测官方 202s，
预测误差 16s，远小于时间余量。

## 5. 官方提交与解释（预注册）

一次官方提交。Δ\_L = `S(v163) − S(v162) = 4587 − 1001 = 3586` 为 Linear 侧官方边际；
已回传写入结果。后续判读待 v164 回传后汇总。

## 6. 复现

```powershell
.venv\Scripts\python.exe -u evaluator\official_eval.py --solution solutions\20260903_v163_v160-linear_standard-attn_scoreNA_timeNA\solution.py --linear-only --cache-mode read --nvfp4-cache-mode auto --capture-device cuda --algorithm-device cuda --output artifacts\official_eval\sidecal-v163-linear-default.json --report logs\official_eval\sidecal-v163-linear-default.md
```

