# E0/D0 多模型 E6M2 scale-lattice oracle dashboard

日期：2026-08-30  
状态：`completed-diagnostic`；只读 oracle，不改变主代码。

## 口径

对五个本地模型的第 1 层、前 32 行、全 role 运行完整 255 个合法 E6M2
top-scale code。每个 block 的 oracle 固定 BOAT 变换和现有 hierarchy solver，
分别以 weight plain-MSE 与 activation `gram64` 为目标。定义

\[
g_{scale}=\frac{L_{\pm3}-L_{all255}}{L_{\pm3}+\epsilon}.
\]

这是“只扩大顶层 scale/hierarchy 搜索”的理论上限，不是官方分数，也没有把
`A@W` 或 evaluator 输出传入算法。Qwen 使用 `layers1` 缓存；其他模型没有单层
缓存，脚本从 `layersall` 缓存取第 1 层，计算口径相同。

## 各模型 total gap（%）

格式为 `weight / activation-Gram`；`fc` 是非 Qwen 模型合并的 FFN role。

| 模型 | q | k | v | o | fc | proj |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gpt2-small | `0.0431 / 0.1183` | `0.1518 / 0.1772` | `0.0157 / 0.1920` | `0.1205 / 1.0794` | `0.0000 / 0.1083` | `0.1153 / 0.2212` |
| gpt2-medium | `0.0806 / 0.7304` | `0.0329 / 0.8594` | `0.1330 / 0.1014` | `0.0041 / 0.1942` | `0.0049 / 0.1738` | `0.0642 / 0.2878` |
| opt-125m | `0.0274 / 0.1675` | `0.0151 / 0.7494` | `0.0341 / 0.1211` | `0.0845 / 0.2005` | `0.0184 / 0.6351` | `0.0401 / 0.1168` |
| pythia-160m | `0.1526 / 0.2296` | `0.0668 / 0.3906` | `0.1224 / 0.2277` | `0.0866 / 0.1990` | `0.1005 / 0.1107` | `0.0313 / 0.0972` |
| qwen2.5-0.5b | `—` | `—` | `0.0313 / 0.6302` | `—` | `0.0572 / 0.0836`, `0.0709 / 0.0470` | `0.0390 / 0.0475` |

Qwen 的 `v` activation-Gram 是唯一明显的局部机会（`0.6302%` 总 gap、60/448
blocks、最大 block `19.321%`）；但非 Qwen 的高点换 role（GPT-2 small 的 `o`
为 `1.0794%`、GPT-2 medium 的 `k` 为 `0.8594%`、OPT 的 `k` 为 `0.7494%`），
不存在跨模型统一的 role 门控信号。

## 结论

1. 全 255-code oracle 的收益仍是亚百分比量级，不能承担 Linear mean 从 `0.5016`
   到 `0.9` 所需的 `0.3984` 绝对提升；GALS 不是 36,000 的主突破口。
2. E0-C 解析候选在 Qwen v role 的两折和四个 test 窗口召回率为 `1.0`，但其
   稀疏部署版已在 layer-1 回退并归档（见 `2026-08-30-e0c-gals-candidate.md`）。
3. 后续不做全局 scale 扩张；若继续研究，只能先建立 role/state 标识，再针对
   单一高 gap block 做严格跨 fold/跨模型 gate。

原始证据：

- `artifacts/oracle_dashboard/e0g-qwen-layer1.json`
- `artifacts/oracle_dashboard/e0g-gpt2-small-layer1.json`
- `artifacts/oracle_dashboard/e0g-gpt2-medium-layer1.json`
- `artifacts/oracle_dashboard/e0g-opt-125m-layer1.json`
- `artifacts/oracle_dashboard/e0g-pythia-160m-layer1.json`
