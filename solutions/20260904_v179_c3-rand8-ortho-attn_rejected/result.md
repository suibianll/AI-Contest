# v179 预研：C3 固定 8×8 随机正交旋转（QuaRot/TurboQuant 对照）（REJECTED）

> 状态：**REJECTED（本地预研明确大负优化，不提交官方、不占配额）**
>
> 性质：C3 计划 §2 条件对照（C1 官方负时考虑一次）。本地预研提前裁决——
> Longhorn（Qwen2 GQA 实测）已示 rotation delocalize error，本预研证实该
> 证据在 HiF4 本地结构同样成立；条件触发时不再耗配额提交。
>
> 构造：基于 P\_A = v168（standard Linear + A1 logits gain）+ C3 固定 8×8
> 随机正交旋转（校准期生成固定 seed 的 per-KV-head 正交矩阵，Q/K 编码前
> 一次 8×8 乘，QK^T 内积精确不变）。
>
> 官方结果：`not submitted（本地 REJECTED）`

## 1. 机制（计划 2026-09-04 C3）

- 固定 seed（20260904）QR 正交化每 KV head 一个 8×8 随机矩阵（R^T R = I，
  正交误差 \~5e-8）；Q 与同组 K 共享同一旋转，连续域 QK^T 内积精确不变，
  仅重分配编码后的幅值分布（delocalize channel 幅值）。

- 动态零新增状态：只增加编码前一次 8×8 乘（O(TD) 线性），无 per-call
  精化/循环；v165 约束满足。

- 不搜索 seed/块大小；C3 为对照，Longhorn 证据预计 Qwen 上负。

## 2. 本地实测（描述性；官方不参与——已 REJECTED）

| 项目                           | 结果                                                                        |
| ---------------------------- | ------------------------------------------------------------------------- |
| 隔离导入 + 六 API                 | OK（校准 + 动态 Q/K smoke 通过）                                                  |
| 正交性                          | per-head R^T R − I 最大误差 4.99e-8 / 3.73e-8（QR 浮点精度）                        |
| attention compact 4（配对 v168） | **mean Δgain −0.229273**、median −0.062230、1+/3−/0=、median MSE ratio 1.512 |

**判读**：compact 4 上 `−0.229` 是已测机制中量级最大的负向之一——旋转把
稀疏高幅 K 通道的幅值打散到 8 通道组内，NVFP4/标准 HiF4 编码对该组的
scale 均值化后，原 outlier 通道不再被保留其动态范围；per-token 保护丢失，
与 Longhorn 在 Qwen2 GQA 的实测结论一致。C3 对照本地确认负优化。

## 3. 对候选清单的影响

- C3 条件触发（C1 官方负）时直接用本预研结论 **REJECTED，不提交**，省配额。

- 计划候选清单 C1/C2/C3 至此全部裁决完毕（C1 官方批测中；C2/C3 本地关闭）。

## 4. 复现

```powershell
.venv\Scripts\python.exe -u evaluator\official_eval.py --solution solutions\20260904_v179_c3-rand8-ortho-attn_rejected\solution.py --attention-only --compact-panel --cache-mode read --nvfp4-cache-mode auto --capture-device cuda --algorithm-device cuda --baseline-json artifacts\official_eval\v168-compact-attn.json --output artifacts\official_eval\v179-compact-attn.json --report logs\official_eval\v179-compact-attn.md
```

