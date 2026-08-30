# L1 full-hierarchy cross-block Weight-LRH 执行记录

日期：2026-08-30  
计划：[`2026-08-30-hif4-active-optimization-plan.md`](../../docs/superpowers/plans/2026-08-30-hif4-active-optimization-plan.md)  
候选归档：`solutions/20260830_v105_l1-full-hierarchy-lrh-rejected_screen523019_time266s/`

## 目的

验证 v092 的 cross-block Weight-LRH 是否因为 hierarchy 写回错误而被错误否定。
审计保存的 v092 源码后发现，它实际只把 parent denominator 传给 `_write_codes`，
并没有可复现的 scale/lv2/lv3 搜索。因此本次实现了真正的 full-hierarchy 原子写回：
scale、lv2、lv3、sign、mantissa 共同构成候选并共同解码。

## 实现与合成测试

- rank：8；最多 4 个高 leverage block；
- scale：parent E6M2 code 附近的合法 `_BASE_OFFSETS`；
- hierarchy：8 个合法 `(lv2, lv3_left, lv3_right)` 布局逐组比较；
- mantissa：15 个 signed levels 的精确坐标 admission；
- 目标：完整 residual 的二次型增量，而非独立 block MSE；
- 合成测试覆盖 round-trip、shape/dtype/finite、以及离散目标单调不增。

命令：

```text
python -m pytest -q tests/test_weight_lrh.py tests/test_linear_ceiling_dashboard.py tests/test_linear_error_decomposition.py tests/test_reference_hif4.py tests/test_linear_compliance_guard.py
```

结果：`29 passed in 5.72s`。

## Qwen 分层预筛

固定 cache：Qwen2.5-0.5B，`seq=128`、`calib=2`、`test=4`、`amax6`、CPU，层位
`0,5,11,17,23`，七个 Linear role。

```text
python evaluator/linear_candidate_screen.py --cache artifacts/real_model_suite/cache/qwen2.5-0.5b__seq128__calib2__test4__layersall__schema1.pt --solution solution.py --layers 0 5 11 17 23 --roles q k v o fc_gate fc_up proj --output artifacts/real_model_suite/l1-lrh-stratified-qwen.json --report logs/execution/2026-08-30-l1-lrh-stratified.md
```

| 指标 | L0 stable | L1 candidate | 差值 |
|---|---:|---:|---:|
| selected-layer `both_player` | 0.523019429222563 | 0.523019429222563 | 0 |
| cases | 35 | 35 | — |
| LRH candidates | — | 70 | — |
| cross-fold admitted | — | 1/70 | — |
| final field changes vs stable | — | 0/35 | — |

`both_player` 以及每个 layer/role 的四个 arm 均与 L0 JSON 逐条一致，故没有
正向预筛信号，不触发第 3.2 节的 24 层 full-layer 评测门禁。候选源码、测试和
结果已归档；根目录应恢复 stable parent。

## 诊断结论

LRH 在生成 fold 上常有明显下降，但在交换 fold 上通常恶化；唯一被 cross-fold
接受的 layer-23/proj 候选也没有胜过已有 HSDQ selector。这不是 hierarchy 写回
错误，而是当前 LRH 的跨 fold 泛化不足。依照 active plan 的失败处理，不再扩大
rank、block 数或 sweep，下一步进入 L2 expansive-FFN CAT/BOAT-2。
