# v030 — C38 beam2 + narrow FULL64 full-coverage

- Date: 2026-08-28
- Candidate ID: `C38`（FULL64 精修的最终配置组合）
- Parent 链：C21-C / v025（合规基线）→ C23 FULL64（机制开启）→
  C35 分层覆盖率 → C36 去第二坐标下降 → **C38（beam 4→2 + 窄层
  覆盖率 1.0）**
- Unique mechanism（相对 C23 归档）：
  1. `_WEIGHT_FULL64_MAX_RATIO` 全局 0.30，窄层（≤1024 通道）1.0
     全覆盖 / 宽层 0.25（C35 分层）；
  2. `_WEIGHT_FULL64_SECOND_COORDINATE = False`：toggle refine 后
     去掉尾随坐标下降（省 ~1/3 块时间，质量无损，C36）；
  3. `_WEIGHT_FULL64_BEAM_KEEP = 2`：beam 4→2（省 pair 求解），
     省下的时间全部投给窄层全覆盖（C38）。
- 其余（继承）：act QUAD8 gate off / sweeps 2 / ratio 0.60、act
  refine ratio 1.0、QUAD16 0.02、FULL64 beam offsets 6 档。
- Source SHA256: `648A27B3560EF7F5D939CD409301E445E5065047CBD5438C1A73A013730E467F`
- Local status: `local-champion`（offset0 本地位 0.5695，固定矩阵
  0.5695 / 0.5629 / 0.5766 全正向）
- Official status: **已提交，14092 分 / 170.57s**（2026-08-28，与
  本地锚点倒挂，详见下）

## 官方提交结果（2026-08-28）：14092 / 170.57s

| 版本 | Linear mean | Attention | 官方分数 | 官方时间 |
|---|---:|---:|---:|---:|
| C21-C / v025（合规锚点） | 0.5311 | 0.4497 | 14437 | 166.6s |
| **C38 / v030（当前）** | **0.5695** | **0.4497** | **14092** | **170.57s** |
| Delta | +3.84pp | 0 | **−345** | +4.0s |

- **分数倒挂（重大矛盾）**：官方 14092 < C21-C 的 14437；按锚点
  公式（658 + 25945×L）反推官方视角 Linear ≈ 0.5176，比本地
  0.5695 低 5.2pp、比 C21-C 锚点 0.5311 还低 1.4pp。
- **本地大数据对照排除过拟合**：calib4/test4/seq256 下 C21-C=
  0.5265、C38=0.5980（C38 优势 +7.15pp，反而扩大）；calib2/test2
  = 0.5311 vs 0.5695（+3.84pp）。C38 本地稳健优于 C21-C。
- **时间反证**：C38 本地 CPU 比 C21-C 重 54%（64→99s），官方仅
  +2.4% → 官方时间由动态+推理主导、校准在官方 GPU 近似免费 ⇒
  300s 上限余 43%（170.57/300），时间不是瓶颈。
- **定位**：C21-C 本地-官方一致（0.5311→14437 为校验锚点），C38
  不一致 ⇒ 官方评测对 C38 的路径行为 ≠ 本地评测（候选原因：官方
  传入校准数据/参数不同 → FULL64 在官方数据下退化/不稳定，或官方
  分布与本地 offset0 分布差异；待 A/B 提交 v025 对照二分定位）。
- 处置：官方结果已回填档案；**未晋级为官方新锚点**，本地保持
  local-champion。等待 A/B 提交（v025 对照）后决定继续优化还是
  先修官方兼容性。

## 开发结果（offset 0, amax6, CUDA）

| Component | v025 C21-C | v030 C38 | Delta |
|---:|---:|---:|---:|
| q | 0.6008 | 0.6408 | +4.00pp |
| k | 0.5936 | 0.6481 | +5.45pp |
| v | 0.5940 | 0.6218 | +2.78pp |
| o | 0.5178 | 0.5674 | +4.96pp |
| fc | 0.4749 | 0.5228 | +4.79pp |
| proj | 0.4058 | 0.4161 | +1.03pp |
| Linear mean | 0.5311 | **0.5695** | +3.84pp |

- Attention 不变（0.4497 causal / 0.4944 non-causal，与 v025 相同）。
- CUDA algorithm-stage ~30s（v025 同环境 ~24s）；官方时间 170.57s
  远低于 CPU×2.6 推算（257s）——时间预算实际按 CUDA×~5.7 模型。

## 固定回归矩阵（同日同评测器）

| Case | Linear mean | vs v025 |
|---|---:|---:|
| amax6 offset 0 | 0.5695 | +3.84pp |
| amax6 offset 97 | 0.5629 | +4.76pp（基线 0.5153） |
| amax6 offset 193 | 0.5766 | 历史最高 |
| 大数据（calib4/test4/seq256） | 0.5980 | C21-C 0.5265（+7.15pp） |

## 合规与测试

- `evaluator/linear_compliance_guard.py`：静态 + 运行时 `violations=[]`
  （A@W 红线全程未触碰；所有机制 operand-local）。
- pytest 60/60（flag 锁定测试已更新：BEAM_KEEP=2、NARROW=1.0、
  WIDE=0.25、SECOND_COORDINATE=False；chunk 测试改为单块位精确
  语义——beam2 跨子 chunk 存在浮点平局翻转但均为等价解）。
- holdout 预算：3/3 未动。

## Decision

`official-submitted, local-champion, not promoted`。官方 14092 与
本地 +3.84pp 矛盾，原因待 A/B（v025 官方对照）二分。在此之前
C38 只作为本地最优配置保留，不作为官方新锚点；优化方向等待官方
口径确认（若官方路径对 FULL64 退化，可能需要为官方校准输入做
鲁棒化：更大 calib、多窗口训练、或 FULL64 保守回退）。

Next: A/B 官方提交 v025（C21-C 原版）→ 判明 14092 是"C38 退化"
还是"官方环境漂移"；随后决定优化策略。