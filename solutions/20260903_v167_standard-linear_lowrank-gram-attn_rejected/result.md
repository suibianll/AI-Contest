# v167 候选：校准编译低秩 Gram 码本（§7.2，v165 timeout 的指定恢复路径）

> 状态：**REJECTED（本地机制否决，未提交官方；2026-09-03）**
>
> 官方共同基线：v162 `1001 / 146s`；Attention 官方父侧：v164（standard Linear + v160 Attention）
> `13945 / 204s`
>
> 父版本：v164 归档（Attention = v160 逐位），Linear 侧 standard 不变
>
> 候选 SHA256（最终版，gram-median + off-diag rank-2）：
> `268C39651B5FB1304C233DB8A5AC8FF53643C760228799BC15F52FBCB0B6AA57`
>
> 官方结果：`unregistered / NA`（未提交；机制在 compact 阶段被否决）

## 1. 机制（预注册 §7.2，Gram 目标与 v161 相同）

- **校准期**：逐校准窗口（fold）用 v161 的 `_qk_cross_gram64`（Q 块用同组 K 的
  `X^TX`，K 块用组内全部 Q 的）在最终部署坐标算交叉 Gram；折聚合后压 rank-2：
  `H = diag + Σ_k λ_k u_k u_k^T`，并按敏感度 `H_jj` 编译每四元素微块的两个 bump
  索引；
- **动态期**：5 候选（parent、j₁±1 码、j₂±1 码）一次批量向量化选择，精确二次型
  ΔL，无 sweep/无 topk/无逐坐标 Python 循环——完全移除 v161 超时元凶的算子类。

## 2. 实现正确性证明（消融变体 B）

λ=0（对角-only）变体在 compact 4 哨兵上与父版本**逐位一致**
（mean `0.797462`，四 case gain 逐位相同）。这不是巧合而是数学必然：父编码为
最近邻编码（`|e_j| ≤ step/2`），对角二次型 `ΔL = D(−2δe_j + δ²)` 恒 ≥ 0，
对角项永远无法证明 bump 合理——**机制的全部价值只能来自 rank-2 耦合项**。
计时：校准 ~0 增量（2.745 vs 2.769s），动态 +~5ms/call（v161 为 +90ms/call），
时间设计目标达成。

## 3. 否决证据（compact 4 哨兵）

| 候选 | 聚合 | L0 | L8 | L15 | L23 | mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| v164 父版本 | — | 0.9237 | 0.9246 | 0.7354 | 0.6062 | 0.797462 |
| v167 初版（折特征向量符号对齐+中位） | eigvec-median | 0.9126 | 0.9131 | **−0.7879** | −0.1501 | 0.221951 |
| v167 修正版（gram 逐元素中位 + off-diag rank-2） | gram-median | 0.9177 | 0.9200 | **−0.5424** | −0.0448 | 0.312658 |
| 变体 A（gram 均值聚合） | gram-mean | 0.9160 | 0.9185 | **−0.6005** | −0.0490 | 0.296229 |
| 变体 B（λ=0 消融） | gram-median | 0.9237 | 0.9246 | 0.7354 | 0.6062 | 0.797462（逐位=父） |

合成探针上初版相对真全 Gram 损失 +17.2%（被接受 bump 增加真实目标），修正版
与 λ=0 均为 −7% 量级；但真实 compact 数据上两种聚合的 rank-2 耦合都破坏深层
哨兵。Q/K/V 分解确认 q_only/k_only 灾难与 logit_mse 43917 是父版本 pair-transform
坐标的固有结构（父版本数值相同），v167 的破坏体现在 qk 联合 MSE
0.00896 vs 父 0.00343。

## 4. 根因结论

真实部署坐标下的 QK 交叉耦合 Gram 是**高秩**的：top-2 off-diagonal 仅占 ~7%
特征质量（合成探针实测）。rank-2 耦合估计不可信，被接受的"耦合动机" bump
（唯一可能非零的收益来源）实际增加真实 logits 误差与输出 MSE。v161 的本地
`+0.0525` 依赖完整 64×64 耦合——**该信号无法压缩到 rank-2 码本**。

## 5. 纪律与后续

- 按 §5 不做 rank 邻域扫描（rank-4/8 同受 7% 特征质量结构性约束，且属被禁
  邻域）；不扩大候选数；不试聚合方式第三种读法。
- v165 timeout 的"一次保持 Gram 目标的低复杂度重构"额度已用完：per-call
  全 Gram 官方超预算、rank-2 编译本地破坏、对角-only 数学上恒 no-op。
  **Attention 侧内部机制耗尽**；Gram 信号确证存在但不可在预算内部署。
- 下一 Attention 方向需新数学机制或外部 SOTA 搜索（侧向隔离计划 §5：机制
  失败后更换数学结构）。Linear 侧 v166 rank-1 候选不受影响，仍等待官方提交。

## 6. 证据

`v167-compact-attn-smoke.json`（初版）、`v167-compact-attn-smoke2.json`（修正版）、
`v167-variantA-compact.json`、`v167-variantB-compact.json`、
`v164-compact-attn-parent-ref.json`（父版本对照，`artifacts/official_eval/`，
对应 `logs/official_eval/` report）。合成探针诊断在会话记录中（真 Gram 损失
+17.2% / −7.6% / −7.1% 三变体对照）。
