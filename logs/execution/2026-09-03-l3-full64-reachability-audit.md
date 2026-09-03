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
