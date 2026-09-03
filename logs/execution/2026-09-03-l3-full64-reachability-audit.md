# L3 full64 reachability audit

日期：2026-09-03

## 结论

`artifacts/official_eval/l3-full64-qwen-compact.json` 与对应报告不能证明 full64 refine 无效。
候选 `workbench/l3_full64_on.py` 虽设置 `_WEIGHT_FULL64_APPLY=True`，但调用仍嵌套在
`if _WEIGHT_E2E_REFINE ...` 内，而该常量保持 `False`。因此 `_refine_weight_blocks64` 实际
不可达，56/56 zero delta 只是父候选行为相同。

原始 JSON/report 不覆盖，继续作为错误实验的审计证据。活动计划已要求把 full64 条件移到
`_WEIGHT_E2E_REFINE` 外，并以 attempted/accepted block 计数验证分支真实执行后再下结论。

## 影响

- 撤销“现有 Weight GPTQ 已在 full-H 64-block 下收敛”的结论；
- 撤销“L3 不是 17816−17532 差距来源”的结论；
- 不改变 v160 正式源码或 `17532/232s` 官方事实；根文件中的 gate 默认仍为 `False`。

## 正确重跑结果（同日，仅一次）

已将 full64 调用移出 `_WEIGHT_E2E_REFINE` 条件，只在现有
`workbench/l3_full64_on.py` 中启用；根 `solution.py` 未改。

```powershell
.venv\Scripts\python.exe -u evaluator/official_eval.py --solution workbench/l3_full64_on.py --name l3-full64-reachable --linear-only --compact-panel --cache-mode read --nvfp4-cache-mode auto --capture-device cuda --algorithm-device cuda --baseline-json artifacts/official_eval/v159-l1-batch-compact-parent.json --output artifacts/official_eval/l3-full64-reachable-qwen-compact.json --report logs/official_eval/l3-full64-reachable-qwen-compact.md
```

- source SHA256：`05DC0261000AD08C8685ADDA580BB5D1BBC64255B85C8C4D5569CA724DD58619`；
- 24 次真实 refine 调用；attempted `659456`、accepted `657540`（99.71%）、改变 code
  `15124875`；其余 4 个 `proj` state 无 `gram_full`，保持不变；
- parent/candidate Linear：`0.705507633 → 0.687587782`；
- paired delta：mean `-0.017919850`、median `-0.016153218`、`6+/42-/8=`；
- component delta：W-only `+0.107169`、A-only `-0.006271`、interaction `-0.118818`。

决定：**REJECTED**。当前 full64 块内目标与最终 `Q(A)Q(W)^T` 交互不一致；不再运行第二次，
不跑 default/跨模型/官方，不调整任何 full64 参数。
