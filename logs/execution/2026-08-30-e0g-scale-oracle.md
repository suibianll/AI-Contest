# E0-G：Qwen E6M2 scale-lattice oracle

日期：2026-08-30  
父版本：`solution.py` SHA `5d1128cc79fef58154da2f600ec4b472ff95030e1f1e61b96593d06fd9aac94f`  
脚本：`evaluator/e6m2_scale_lattice_oracle.py`  
缓存：Qwen2.5-0.5B，一层，`seq=128/calib=2/test=4`  
实验对象：当前 BOAT 变换后的 Qwen layer-0，前 32 行；每行 64-block 全部评估。  
本日志是 evaluator-side oracle，不修改 `solution.py`。

## 目的

比较当前 `_BASE_OFFSETS=(-3,...,3)` 与 evaluator 接受的全部 255 个有限无符号
E6M2 scale code，判断扩大顶层 scale 搜索是否值得进入 GALS 主实现。weight 侧使用
逐元素 block MSE；activation 侧使用当前静态 `gram64` 目标。全程不形成 `A@W`，
不写入或选择 `activation_state`。

## 运行

```text
python evaluator/e6m2_scale_lattice_oracle.py --cache \
  artifacts/real_model_suite/cache/qwen2.5-0.5b__seq128__calib2__test4__layers1__schema1.pt \
  --output artifacts/oracle_dashboard/e0g-qwen-layer1.json
```

脚本运行时间：约 11.4 秒（CPU）。完整原始 JSON 见
`artifacts/oracle_dashboard/e0g-qwen-layer1.json`。

## 结果

| role | 侧别/目标 | 改善 blocks | 总相对 gap | 最大单 block gap |
|---|---|---:|---:|---:|
| `fc_gate` | weight MSE | 5 / 448 | 0.0572% | 13.764% |
| `fc_gate` | activation Gram | 19 / 448 | 0.0836% | 17.326% |
| `fc_up` | weight MSE | 1 / 448 | 0.0709% | 22.199% |
| `fc_up` | activation Gram | 13 / 448 | 0.0470% | 12.815% |
| `v` | weight MSE | 4 / 448 | 0.0313% | 7.466% |
| `v` | activation Gram | 60 / 448 | **0.6302%** | **19.321%** |
| `proj` | weight MSE | 18 / 2432 | 0.0390% | 9.943% |
| `proj` | activation Gram | 50 / 2432 | 0.0475% | 20.534% |

`improved blocks` 表示全 255 oracle 严格优于当前局部 scale 候选的 block；
`total relative gap=(L_local-L_oracle)/L_local`。由于当前 `_encode_rows` 在
Gram 情况下仍按对角代理选择 hierarchy，以上不是完整 Gram-hierarchy oracle，
只是顶层 scale 扩张的保守诊断。

## 决策

1. 全局 GALS 不进入主代码：除 `v` activation Gram 外，各 role 总 gap 均低于 0.1%。
2. `v` 保留为 E1 的稀疏候选插件，只对高 gap/high-loss block 生成 GALS-C 候选。
3. 下一步转入 Progressive Cross-Fold Full-Hierarchy HSDQ；它比全局 scale 扩张更
   可能解决当前 Linear 的主要误差。
4. 若 E1 的 full-hierarchy legal oracle 仍无跨 fold 增益，立即进入 A2 FFN 扩张，
   不再增加 scale 候选数量。

## 验证

- `pytest -q tests/test_gals_hierarchy.py`：`2 passed`。
- `python -m py_compile evaluator/e6m2_scale_lattice_oracle.py tests/test_gals_hierarchy.py`：通过。
- `git diff --check`：通过。
