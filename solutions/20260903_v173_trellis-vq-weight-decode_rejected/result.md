# v173 候选：L3 固定宽度 HiF4 Trellis/VQ 解码（低复杂度扩展计划第六包）

> 状态：**REJECTED（明确负优化；2026-09-04 按用户最终指示 '明确负优化的需要拒绝提交'）**
>
> 父版本：v166（官方 `4590 / 226s`，Linear 父侧）；Attention = v162 standard tail 逐位一致
>
> 候选 SHA256：`1CC1D0B9D011F8E51BC68C89BC1CA0B08034E4EF906025A7C4D931242FD6F05B`
>
> 官方结果：`unregistered / NA`

## 1. 唯一算法机制（预注册，计划 §10）

保持父 scale/lv2/lv3，每 64-block 做固定 16-stage × 8-beam × 3-branch 的 Trellis 搜索
（每 stage 打包连续 4 坐标；beam 子代 `8*3^4` 用精确 quadratic partial loss + 未定坐标
对角下界排序，`topk` 保留 8 条、同 loss 按 code 字典序确定性；第 16 stage 后与 parent
完整 Hessian loss 逐 block 比较取较小者）。**只写回 sign/mant**（scale/lv2/lv3 保持父值，
与 L2 的层级 rescale 机制隔离）。Python 只允许固定 16 个 stage 循环，beam 内部 tensorized
`topk`；无 Python 候选循环。校准期调用一次，动态路径与 activation_state 逐位一致。

## 2. 本地验证（描述性；官方裁决）

| 项目 | 结果 |
| --- | --- |
| 隔离导入 + 六 API | OK |
| compact 56（配对 v166） | candidate_mean `0.682679` vs 父 `0.705628`（**−0.02295**、median −0.0199、`1+/47−`、worst −0.148 级）——**明确负：仅 1 个正 case，拒绝提交** |
| 机制可达 | 是（sign/mant 写回变化；只动 sign/mant，scale/lv2/lv3 保持父值） |
| API 时间 | 228.7s（父 46.1s，~5×：16×8 beam 的固有成本）——描述性，官方时间不可预测；若官方 timeout 按 §5 允许一次纯复杂度重构 |

## 3. 判读（§3.3）

```text
step_gain = S(v173) − 4590
```

`>0` → RETAINED；`≤0` → REJECTED 转 L4（Kronecker CAT）。L2 官方负向不取消 L3
（不同解码算法）。官方回传裁决。

## 4. 复现

```powershell
.venv\Scripts\python.exe -u evaluator\official_eval.py --solution solutions\20260903_v173_trellis-vq-weight-decode_standard-attn_scoreNA_timeNA\solution.py --linear-only --compact-panel --cache-mode read --nvfp4-cache-mode auto --capture-device cuda --algorithm-device cuda --baseline-json artifacts\official_eval\v166-compact-linear-smoke.json --output artifacts\official_eval\v173-compact-linear.json --report logs\official_eval\v173-compact-linear.md
```
