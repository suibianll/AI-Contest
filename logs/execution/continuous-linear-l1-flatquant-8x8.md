# L 侧持续优化执行日志：l1-flatquant-8x8

> 日期：2026-09-07。侧：Linear。run_id：l1-flatquant-8x8。
> 状态：CLOSED / REJECTED（LOCAL_NEGATIVE）＋ COST/DESIGN_HOLD。

## 背景

父：L4 侧包 `ACB16F76...F5263`（官方 4607/247s，v189 Linear 侧逐位复现）。

## 步骤与结果

1. **L0 部署坐标核对**：L4 = v160 Linear 校准栈 + rank-2 残差 + 静态 Hdiag
   activation-GPTQ 64-block 块序 + `_DYNAMIC_OFFSETS` +4 码窗。部署链：
   `dense → smooth_inv → permutation → block_hadamard → rank (A + (A U) V^T)
   → GPTQ activation`；权重在 `weight_smooth`（smooth/perm/hadamard 后）上
   编码。与 L1 T 的插入位置（最终连续坐标）兼容。

2. **成本探针**（`probe_l1_cost.py`，真实 Qwen 权重 + 本机 CUDA）：
   完整硬前向（weight GPTQ + activation GPTQ）单步 ≈ 0.6s（窄 768）/1.5s
   （proj）/1.7s（fc）。32 步 × 168 states ≈ 6451s 本地 → 官方 +742s
   （0.115 系数），远超 `<280s` 提交门。**完整硬前向每步重建不可承受**。

3. **方向探针（快速合法编码代理）**：
   - `probe_l1_direction.py`（随机输入）：窄 0.011018→0.011224、宽入
     0.011023→0.011140，无改善。
   - `probe_l1_direction_real.py` / `sweep_l1_direction.py`（真实
     proxy-v2 输入，5 层 × 7 role = 35 state，32 步 STE Adam 固定配置）：
     **改善 1、退化 34**。唯一改善 L0-o −12.8%；退化集中在 proj/fc 与
     深层（+20% ~ +35%）。

## 裁决

- 完整硬前向不可承受 → 按工作包 L1 COST 分支记录。
- 快速合法编码代理 34/35 退化 → LOCAL_NEGATIVE，无一致正向信号。
- 不跑六 shard、不创建候选、不注册官方探索；根 v189 / 父 L4 不变。

## 待办

- L2（GPTAQ）去重：JDRQ 目标（min||Xh Wh^T−Y||²，冻结激活 + 合法编码
  候选，v189 J1 关闭）。若同目标同更新 → DUPLICATE_CLOSED，跳过转下一个
  独立机制假设。