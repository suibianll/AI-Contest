# A2：四列跨结构对照——q/k/v 抬升是否跨代理一致高估（2026-09-02）

> 四列 = 官方分数（new-weight cohort）+ 本地 Qwen proxy（`reeval5-*.json`，default-panel）
>
> - GPT-2 cross-model probe（`artifacts/official_eval/gpt2-*-panel.json`）
>
> - 外部 hif4 real\_data\_eval（`logs/execution/2026-09-01-hif4-external-gpt2-v84-v86-v140-v147.md`）。
>   交集版本：v084 / v086 / v140 / v147（GPT-2 与外部 hif4 无 v158 单测）。

## 1. 各代理的 Linear QKV 均值（v140−v086 为变化主样本）

| ver  | official | Qwen QKV | GPT-2 QKV | 外部 hif4 QKV |
| ---- | -------: | -------: | --------: | ----------: |
| v084 |    16517 |   0.6211 |         — |      0.6232 |
| v086 |    16744 |   0.6211 |    0.5576 |      0.6232 |
| v140 |    15838 |   0.7450 |    0.6300 |      0.6696 |
| v147 |    16579 |   0.7450 |    0.6300 |      0.6696 |

## 2. v140−v086 逐 role delta（官方 = −906，回归）

| role       |    Qwen |   GPT-2 |     外部 hif4 | 方向判定              |
| ---------- | ------: | ------: | ----------: | ----------------- |
| q          | +0.0843 | +0.0761 |     +0.0409 | 全正                |
| k          | +0.1631 | +0.1092 |     +0.0900 | 全正                |
| v          | +0.1243 | +0.0317 |     +0.0085 | 全正                |
| o          | +0.0284 | +0.1053 |     −0.0018 | Qwen/GPT-2 正，外部≈0 |
| fc/ffn\_in | +0.0067 | +0.1258 | **−0.0451** | Qwen/GPT-2 正，外部负  |
| proj       | +0.0316 | +0.4206 | **−0.0152** | Qwen/GPT-2 正，外部负  |

## 3. 结论

1. **确认：q/k/v 抬升跨代理一致高估。**

   - 三个独立代理（Qwen / GPT-2 / 外部 hif4）对 v140−v086 的 q/k/v 全部给出**正的**抬升
     （q +0.041~~0.084、k +0.090~~0.163、v +0.009\~0.124），而官方这一差异为 **−906 分**；

   - Qwen QKV 与官方 4 点 Pearson = **−0.61**（大负）；

   - 因此"q/k/v 静态量化余量在本地被高估"不是 Qwen 特有结构伪影，而是**跨代理的系统性现象**；
     本地针对 q/k/v 的任何改进（激活 importance、QK Smooth、K 中心化、pair Matrix-Smooth 等
     以降低 q/k/v 编码误差为目标的手段）都不能作为官方方向证据。
2. **新增发现：fc/o/proj 上只有外部 hif4 与官方同向。**

   - 外部 hif4 的 fc（−0.045）与 proj（−0.015）为负，与官方 −906 一致；Qwen/GPT-2 均为正；

   - 即"问题在 fc/proj"的外部归因（v140 系本地的膨胀主要也来自 q/k/v + fc/proj 全正）中，
     只有外部 hif4 在 fc/proj 上给出了官方方向的证据 → 未来 Linear 侧若做机制反证，可把
     外部 hif4 的 fc/proj 负向作为与官方同向的独立校验列。
3. **v084 = v086（三代理 Linear 全等）+ 官方 +227**：三方一致复现"Linear no-change、增量纯
   Attention"，与 A1 的冻结复现互相印证。

## 4. 对策略的含义（更新 B 计划的依据）

- q/k/v 相关手段（QK Smooth、K-center、pair transform、q/k importance）**不再作为 Linear/Attention
  官方方向的依据**——它们在三个代理一致高估、官方不认；若 v158 的 Matrix-Smooth 已实现 q/k 平衡，
  不应再往 q/k 方向叠加同族机制；

- 外部 hif4 是唯一与官方在 fc/proj 同向的代理：任何 fc/proj 机制的离线排序应优先采信外部 hif4，
  并以官方单变量回传为最终裁决；

- Attention（非静态 q/k/v 编码）仍是与官方同序（Spearman 0.85）的有效面，保持不变。

