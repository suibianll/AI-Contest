# Attention source-scale proposal 优化执行记录

日期：2026-09-06  
研究父：v189 研究副本，根官方父仍为 v186，根 SHA256
`F8495DCA20334ACBDAD16FC18EE41A4970F31E1837FDEEDCEE9C70AEE54E7EB8`  
候选源码：`workbench/attention_source_scale_proposal_solution.py`  
候选 SHA256：`b365ff959f2848919a38eb5b77dd4c2314c0f4188a3e24ddc8ce2730266b2b86`  
输入：`artifacts/official_eval/cache/qwen2.5-0.5b-proxy-v2.pt`

## 结论

S0 单文件导入、六 API 存在和语法检查通过。S1 在 Attention shard 0、1 共 16 个
case 上候选与 v186 逐 case 逐位一致，`delta_mean=0`、`L1=0`、正/负/零为
`0/0/16`，eval-v3 按早停规则停止，状态为 `NOOP_REJECTED`。不进入 OOD/default
审计，不产生版本号，不提交官方。

## 阶段结果

| 阶段 | 结果 |
|---|---|
| S0 | `py_compile` PASS；绝对路径独立导入六个 API PASS；V/Linear 代码未改 |
| S1 shard 0 | Attention mean `0.777742183683`，候选与 baseline 完全一致，8/8 zero |
| S1 shard 1 | Attention mean `0.805886376364`，候选与 baseline 完全一致，8/8 zero |
| 直接 reachability | Q source codes shape `(128,14,3)`、K `(128,2,3)`，说明 proposal 生成可达；但在实际 Q/K hierarchy loss 中没有改变任何输出字段 |

候选调用只在 Q/K 传入 `source_scale_proposal=True`，V 与 Linear 保持父路径。直接对
同一 calibration 输入比较 source=True/False 的五字段参数，Q/K 的 scale、lv2、lv3、
sign、mant 全部差异为 0；因此不是 evaluator 缓存或 case identity 问题，而是 raw
source scale proposal 在已变换 Q/K 坐标下没有胜出。

## 复现

```powershell
.venv\Scripts\python.exe -m py_compile workbench\attention_source_scale_proposal_solution.py
.venv\Scripts\python.exe evaluator\eval.py --solution workbench\attention_source_scale_proposal_solution.py --baseline-solution solution.py --attention-only --shards 0,1,2,3,4,5 --cache artifacts\official_eval\cache\qwen2.5-0.5b-proxy-v2.pt --calibration-cache-mode auto --algorithm-device cuda --output-dir artifacts\proxy_v3\attention-source-scale-20260906\full-s1
```

产物：`artifacts/proxy_v3/attention-source-scale-20260906/full-s1/candidate/manifest.json`。

## 归档状态

状态：**CLOSED / NOOP_REJECTED**。不扫描 stats、threshold、offset 或其他邻域；下一
活动计划切换到
当时的后继计划现已归档为
`docs/superpowers/archive/plans/2026-09-06-attention-aligned-source-scale-plan-rejected.md`，测试把
NVFP4 source scale 按最终 Q/K multiplier/permutation 坐标对齐后再进入同一 canonical
候选池。
