# continuous_attention AC0：Correctness Hardening（正确性加固基线）

> 日期：2026-09-08。父：`v162_attention_r3-rotation-center_allgates`（官方 14405 / 238s，
> SHA256 `A5C679D7A2B349A879B2019B4A613244F5FEA7050E407B6475B9E29BD1C146DC`）。
> AC0 solution.py SHA256：
> `F817E4C24CAAA8D1325A5A0045F8A0F057EE0DC5EB4C0B7167298FE67BB4F5A2`。

## 变更内容

AC0 = R3 数学行为 + implementation hardening（只修正确性，不改算法/参数）：

1. 删除 `_nvfp4_to_hif4` 中 Q/K coupled transform 的 `except Exception: pass`
   静默回退，改为原子 Q/K-pair 回退 + `fallback_reason` 事件记录 + strict raise 模式。
2. 训练 hard forward 改用真实部署路径（`hif4_dynamic_quantize_q/k/v` 候选 state），
   V 走真实部署 V；rotation 施加位置与部署一致（栈末）；训练/部署五字段逐位一致。
3. 统一 dense-reference `_attention_transform_dense_reference`（部署顺序唯一标准），
   校准/测试共用。
4. 强制 shape 校验：GQA group、rotation `(kv_heads, head_dim, head_dim)`、
   center `(kv_heads, head_dim)`、K 侧 groups==kv_heads。
5. 校准期 FP64 QK-invariance audit（`L=QKᵀ−mean_key`，rel<1e-6）+ pair 校验，
   失败 → proposal 整体回退 parent（identity arm），理由写入 state。
6. 校准失败回退记录 `a2_fallback_reason`，不再静默。

详细证据见 [`audit.md`](audit.md)；测试见 [`correctness_battery.py`](correctness_battery.py)（30/30 PASS）。

## 本地评测（4B paired，attention-only，AC0 vs R3）

命令（六 shard，`--baseline-solution` R3，校准缓存 auto）：

```powershell
.venv\Scripts\python.exe evaluator/eval.py --solution solutions\continuous_attention_ac0-correctness-hardened\solution.py --baseline-solution solutions\v162_attention_r3-rotation-center_allgates\solution.py --attention-only --shards 0,1,2,3,4,5 --stop-after-nonpositive 7 --cache artifacts\official_eval\cache\qwen3.5-4b-proxy-v2.pt --calibration-cache-mode auto --algorithm-device cuda --reuse-existing --output-dir artifacts\proxy_v3\ac0-vs-r3\full
```

| shard | layer | Δmean | Δmedian | L1 | pos | neg | zero | min | max |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | L0 | +0.001931 | +0.000004 | 0.005254 | 6 | 6 | 0 | −0.0082 | +0.0187 |
| 1 | L1 | +0.011122 | +0.006597 | 0.027957 | 7 | 5 | 0 | −0.0350 | +0.1080 |
| 2 | L8 | +0.000000 | +0.000000 | 0.000000 | 0 | 0 | 12 | 0.0000 | 0.0000 |
| 3 | L15 | +0.012077 | +0.013504 | 0.017403 | 8 | 4 | 0 | −0.0128 | +0.0458 |
| 4 | L22 | −0.000383 | +0.000391 | 0.004059 | 6 | 6 | 0 | −0.0151 | +0.0158 |
| 5 | L5 | −0.003743 | +0.001045 | 0.012912 | 7 | 5 | 0 | −0.0588 | +0.0209 |
| 合计 | 72 case | **+0.003501** | — | **0.011264** | 34 | 26 | 12 | — | — |

- 12/72 与 R3 逐位相同（L8 identity 层）；差异集中在 rotation 层（训练路径修正后
  收敛点不同）与 L15 边际 gate 翻转（R3 identity 1.0396 → AC0 rotation 0.9973，
  见 audit.md F-注）。
- 通用符号门视角：Δmean>0 且 L1<0.02（聚合口径）。L1 仅记录；官方以 300s 硬限与
  官方分数裁决。
- api_total（诊断记录，不设门禁）：candidate 校准约 10s/层（shard 校准 api
  9.6~11.4s），scoring ~0.6s/shard。

## Status

- **本地**：`RETAINED`（本地基线，AC0 ≈ R3，无系统性退化证据；correctness battery 30/30）。
- **官方**：**14395 / 258s**（2026-09-08 回传，计分 SHA `F817E4C24CAAA8D1325A5A0045F8A0F057EE0DC5EB4C0B7167298FE67BB4F5A2`
  ——A29 骨架与 AC0 同 SHA，该结果即 AC0 官方结果）。相对 R3（14405/238s）：**−10 / +20s**。
  未触发「Δscore < −20 停止进入 A29」门；258s < 300s 官方硬限，但超出 245s 时间目标。
- **差异解释（−10/+20s）**：
  - 分数 −10：① 训练 hard forward 修正为部署五字段路径后，各层 rotation/center 收敛点
    与 R3 不同（本地 paired Δmean +0.0035 为正，官方 −10——再次确认本地 proxy 不换算官方，
    符号门不保证官方非负）；② L15 边际 gate 翻转（R3 identity → AC0 rotation，gate 0.9973），
    该层 rotation 在官方窗口未获净收益；③ mse_std 分母改部署 parent 语义。
  - 时间 +20s：§9/§10 强制 parity 的成本——训练期每步部署编码（32 步×4 窗×Q/K ×6 层）、
    每层 FP64 audit（≤256 行）与 pair 校验、V 部署编码。258s < 300s 硬限（余量 42s）。
- 结论：AC0 保持为正确性参考，不是官方最佳；官方 Attention 最佳仍为 A2 `14440/274s`，
  时间父仍为 R3 `14405/238s`。AC0 的官方 `−10/+20s` 按真实负结果记录，不称为正向收益。
- A29 状态：本目录对应的 `continuous_attention_a29-boundary-output/` 只是 AC0 逐位骨架；
  真正 A29 实现 `v163_attention_a29-final-residual-s` 已官方 TIMEOUT。当前下一卡为 A30；
  A29 只有另立、去重后的降时实现卡才可回访。
