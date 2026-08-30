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
`A@W` 或 evaluator 输出传入算法。五个模型最终都使用 `layersall` 缓存按
`layer_index` 选择层；早期 Qwen 子集的 `layers1` 结果另行保留，计算口径相同。

## 第 1 层各模型 total gap（%）

格式为 `weight / activation-Gram`；`fc` 是非 Qwen 模型合并的 FFN role。

| 模型 | q | k | v | o | fc | proj |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gpt2-small | `0.0431 / 0.1183` | `0.1518 / 0.1772` | `0.0157 / 0.1920` | `0.1205 / 1.0794` | `0.0000 / 0.1083` | `0.1153 / 0.2212` |
| gpt2-medium | `0.0806 / 0.7304` | `0.0329 / 0.8594` | `0.1330 / 0.1014` | `0.0041 / 0.1942` | `0.0049 / 0.1738` | `0.0642 / 0.2878` |
| opt-125m | `0.0274 / 0.1675` | `0.0151 / 0.7494` | `0.0341 / 0.1211` | `0.0845 / 0.2005` | `0.0184 / 0.6351` | `0.0401 / 0.1168` |
| pythia-160m | `0.1526 / 0.2296` | `0.0668 / 0.3906` | `0.1224 / 0.2277` | `0.0866 / 0.1990` | `0.1005 / 0.1107` | `0.0313 / 0.0972` |
| qwen2.5-0.5b | `see complete JSON` | `see complete JSON` | `0.0313 / 0.6302` | `see complete JSON` | `0.0572 / 0.0836`, `0.0709 / 0.0470` | `0.0390 / 0.0475` |

Qwen 子集中的 `v` activation-Gram 是明显机会（`0.6302%` 总 gap、60/448
blocks、最大 block `19.321%`）；完整 role 运行还显示 q/k 在第 1 层有更大的局部
gap。非 Qwen 的高点也换 role（GPT-2 small 的 `o` 为 `1.0794%`、GPT-2 medium
的 `k` 为 `0.8594%`、OPT 的 `k` 为 `0.7494%`），不存在跨模型统一的 role 门控信号。

## 三层扩展 dashboard（layer 1/2/3）

为补齐 D0 计划，使用每个模型的 `layersall` 缓存，在第 1、2、3 层各跑一次全
role（每格为该层所有 role 的均值；weight / activation-Gram）：

| 模型 | layer 1 | layer 2 | layer 3 |
| --- | ---: | ---: | ---: |
| gpt2-small | `0.0744% / 0.3161%` | `0.0673% / 0.6096%` | `0.0667% / 2.7787%` |
| gpt2-medium | `0.0533% / 0.3912%` | `0.0642% / 0.8577%` | `0.0569% / 0.5605%` |
| opt-125m | `0.0366% / 0.3317%` | `0.0304% / 6.6520%` | `0.0596% / 1.6992%` |
| pythia-160m | `0.0934% / 0.2091%` | `0.0591% / 0.2362%` | `0.0840% / 0.1941%` |
| qwen2.5-0.5b | `0.0677% / 2.0133%` | `0.0544% / 0.3222%` | `0.0642% / 0.3667%` |

三层数据说明：scale oracle 的大 gap 是**层局部且模型/role 不稳定**（例如
Qwen layer-1 q/k、OPT layer-2 proj、GPT-2 small layer-3 proj），不能直接作为
全局 scale 算法收益；它支持的下一步是带 shape/role 状态的稀疏插件，而不是把
255-code 搜索写入所有调用。

## 结论

1. 多数 role 的全 255-code oracle 收益仍是亚百分比，但少数层/role 存在百分之几
   级的局部上限；即使全部兑现，也不足以单独承担 Linear mean 从 `0.5016` 到
   `0.9` 所需的 `0.3984` 绝对提升，GALS 仍不是 36,000 的主突破口。
2. E0-C 解析候选在 Qwen v role 的两折和四个 test 窗口召回率为 `1.0`，但其
   稀疏部署版已在 layer-1 回退并归档（见 `2026-08-30-e0c-gals-candidate.md`）。
3. 后续不做全局 scale 扩张；若继续研究，只能先建立 role/state 标识，再针对
   单一高 gap block 做严格跨 fold/跨模型 gate。E0-C 全局稀疏版已拒绝，role-aware
   版本仍需独立门禁。

原始证据：

- `artifacts/oracle_dashboard/e0g-qwen-layer1.json`
- `artifacts/oracle_dashboard/e0g-gpt2-small-layer1.json`
- `artifacts/oracle_dashboard/e0g-gpt2-medium-layer1.json`
- `artifacts/oracle_dashboard/e0g-opt-125m-layer1.json`
- `artifacts/oracle_dashboard/e0g-pythia-160m-layer1.json`
- `artifacts/oracle_dashboard/e0g-gpt2-small-layer{1,2,3}.json`
- `artifacts/oracle_dashboard/e0g-gpt2-medium-layer{1,2,3}.json`
- `artifacts/oracle_dashboard/e0g-opt-125m-layer{1,2,3}.json`
- `artifacts/oracle_dashboard/e0g-pythia-160m-layer{1,2,3}.json`
- `artifacts/oracle_dashboard/e0g-qwen2.5-0.5b-layer{1,2,3}.json`
