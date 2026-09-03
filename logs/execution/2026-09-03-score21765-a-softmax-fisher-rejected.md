# 21765-A 跨折收缩 Softmax-Fisher 执行记录

> 日期：2026-09-03
>
> 状态：**REJECTED**
>
> 父版本：v160，SHA256
> `33B1D061CE6BFCD92659C597BE4830BB9B910E646FF518433DA67B925AE8680D`
>
> 未编号 workbench：`workbench/score21765_crossfold_softmax_fisher.py`，SHA256
> `492F3D18EBA354E7E2FB5D922F362A28595A96A346AD0D7B6244EA5A07D3FBD2`

## 实现边界

- 从 v160 归档逐位复制，根 `solution.py` 未修改；
- 在最终 rotation、block smooth 和 Matrix-Smooth 坐标中计算五个 calibration sequence ×
  causal/non-causal 的对角 Softmax-Fisher；
- V 使用父版本动态量化后的实际 `Q(V)`；
- 只替换 Q/K state 的 `importance`；V、Linear 和四个动态 API 实现保持父版本；
- 使用固定 `clip=2`、解析 `rho` 和每 head 均值保持；无 blend、阈值或候选搜索。

## A0/A1

真实 compact 的四个深度校准 state：

| layer | Q changed | K changed | Q median rho | K median rho | calibration |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 114 | 0.0000 | 0.7532 | 2.807s |
| 8 | 448 | 64 | 0.0000 | 0.0000 | 2.572s |
| 15 | 832 | 128 | 0.4403 | 0.6554 | 2.533s |
| 23 | 576 | 128 | 0.4043 | 0.6495 | 2.664s |

K 在 4/4 层、Q 在 3/4 层发生解析更新，说明统计可达。数值检查确认六 API 可隔离导入、
state 合法、V state 逐位一致、除 Q/K importance 外的 Q/K state 逐字段一致、所有动态 API
及 Linear calibration API 源码与父版本一致，动态调用图未增加算子。

## A2 compact 配对

基线：`artifacts/official_eval/a1-parent-v160-attn-compact.json`，父 SHA 正确。

| 指标 | 结果 | 门禁 |
| --- | ---: | --- |
| parent attention mean | 0.797462121 | — |
| candidate attention mean | 0.789648797 | — |
| paired mean delta | **-0.007813325** | 失败（要求 >0） |
| paired median delta | **-0.004871463** | 失败（要求 >=0） |
| improved/regressed/equal | **1/3/0** | 失败 |
| worst delta | **-0.027699490** | 失败（要求 >=-0.005） |
| QK-only mean delta | **-0.008121125** | 失败 |
| probability MSE delta | **+4.9533e-6** | 失败（正数为恶化） |
| probability KL delta | **+1.4275e-5** | 失败 |
| V-only delta | **0.0** | control 通过 |
| candidate API total | 10.516s | 无动态成本回归 |

512-token 两个哨兵均回归，layer 23 最差 `-0.02769949`。主要病因为 Q/K 独立项与联合输出
不一致：overall K-only delta 为正，但 QK interaction delta 为 `-0.419898`，静态对角
importance 没有保留所需的 Q/K 误差耦合。

## 决策

A2 同时违反 mean、median、正负 case、worst、QK-only 和 probability 门禁，按预注册规则：

1. A 标记 **REJECTED**；
2. 不运行 Qwen default、GPT-2、OPT/Pythia 或官方提交；
3. 不调整 `clip`、`rho`、blend、阈值或长度路由；
4. 依赖 A 成功的低秩 Fisher 工作包 B 取消；
5. 活动计划转入 Linear C0：跨折 minimax 部署 A@W。

证据：

- `artifacts/official_eval/score21765-a0-softmax-fisher-audit.json`
- `artifacts/official_eval/score21765-a2-softmax-fisher-attn-compact.json`
- `logs/official_eval/score21765-a2-softmax-fisher-attn-compact.md`
- `workbench/score21765_crossfold_softmax_fisher_check.py`
