# E0-C GALS-C — 解析召回与稀疏部署门禁

日期：2026-08-30  
状态：`archived-rejected`；解析候选召回通过，但稀疏部署版未通过 layer-1 精度门禁。

## 1. 解析候选定义

对变换后 64-value block 的每个非零值 (x_i)，枚举

\[
s_{i,m,e}=\frac{|x_i|}{m2^e},
\quad m\in\{0.25,0.5,0.75,1,1.25,1.5,1.75\},
\quad e\in\{0,1,2\}.
\]

将 (s_{i,m,e}) 投影到最近合法 E6M2 code，并加入左右相邻 code、当前
`±3` offset 和 incumbent。对每个 row/block 使用完整 64 元素的 HiF4 hierarchy
solver 与静态 Gram 目标评分；不访问 evaluator 输出，也不写入 `activation_state`。

## 2. 召回实验

`evaluator/gals_candidate_recall.py` 在 Qwen2.5-0.5B layer-1、前 32 行、
`fc_gate/fc_up/v/proj` 的 weight-MSE 与 activation-Gram 两侧均达到全 255-code
oracle 的 `1.0` 召回。v activation 的跨窗口复核也达到 `1.0`：

| 窗口 | baseline→oracle gap | GALS-C recall | candidate/oracle improved blocks |
| --- | ---: | ---: | ---: |
| calibration-0 | `0.6302%` | `1.0` | `60/60` |
| calibration-1 | `0.6577%` | `1.0` | `60/60` |
| test-0 | `0.6725%` | `1.0` | `67/67` |
| test-1 | `0.6729%` | `1.0` | `58/58` |
| test-2 | `0.5690%` | `1.0` | `68/68` |
| test-3 | `0.6457%` | `1.0` | `73/73` |

召回原型每个 block 产生约 62–100 个不同 offset（全局 union 约 100 个），说明
解析公式确实覆盖 oracle，但直接部署会扩大每次动态量化的搜索成本。

## 3. 稀疏部署版 layer-1 门禁

部署实验 `v102` 只对每行静态 Gram loss 最高的 4 个 block 放宽 GALS-C，并要求
候选比 `±3` incumbent 严格下降。Qwen layer-1 结果：

| 指标 | e0c-gals-sparse | v100/B2 baseline | 变化 |
| --- | ---: | ---: | ---: |
| panel total | `335.988995` | `336.037091` | `−0.048096` |
| Linear mean | `0.602878` | `0.603071` | `−0.000192` |
| Attention mean | `0.926347` | `0.926347` | `0` |
| API time | `57.408s` | `16.038s` | `+41.370s` |

虽然静态 activation-Gram 目标逐块不增，但 layer-1 Linear panel 已回退，且成本
显著增加，因此不进入主线。根目录已恢复 v100（B2 PAWV diag-only + B1 GQRB）。

## 4. 结论与归档

- GALS-C 的**候选召回上限**已验证为可行；失败来自跨 role 的部署迁移与运行时，
  不是候选公式本身漏解。
- 不再全局扩大 scale 搜索；若未来重启，只应在明确 role/state 标识和更严格
  的跨 fold gate 下做 v-only 稀疏插件。
- 失败版本、layer-1 JSON、召回 oracle 与源码归档在
  `solutions/20260830_v102_e0c-gals-sparse-rejected_score335.988995_time57s/`。
