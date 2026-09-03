# Step 0：v128 家族同协议消融 — 余量确认，进入 S1

日期：2026-09-03
状态：**DONE — 判读表命中第一行（v128 ≥ +0.03 且 v138 ≈ 0/负 → 进入 S1）**

## 1. 方法

零实现：v128/v129/v138 归档源码的六个 API 直接在 proxy-v2 attention compact 面板
（4 深度/长度哨兵，长度 128/512，NVFP4 cache 命中）运行，`--baseline-json` 配对
v160 归档 parent（`a1-parent-v160-attn-compact.json`，attention mean `0.797462`）。
命令与判读表见
[`活动计划 §2`](../../docs/superpowers/plans/2026-09-03-attention-per-call-refinement-plan.md)。

## 2. 结果

| 版本 | compact attn | paired mean Δ | 改善/回归 | median MSE ratio | API total | calib/call | dyn Q/K per call |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v160 parent | 0.797462 | — | — | — | 10.8s | — | — |
| v128 | **0.861039** | **+0.063577** | 3/1/0 | 0.6726 | 37.9s | 9.27s | 0.100s |
| v129 | **0.861039** | **+0.063577** | 3/1/0 | 0.6726 | 21.6s | 5.20s | 0.095s |
| v138 | 0.783370 | −0.014092 | 0/4/0 | 1.0823 | 6.9s | — | 0.010s |

产物：`artifacts/official_eval/s0-v{128,129,138}-attn-compact.json`、
`logs/official_eval/s0-v{128,129,138}-attn-compact.md`。

## 3. 判读

1. **余量真实且属 per-call 自适应族**：v128 +0.0636（3+/1−，MSE ratio 0.67）；
   v138（静态无自适应）比 v160 还差 −0.014——v128 家族的静态变换不如 v158/v160 的
   Matrix-Smooth，**全部余量来自校准搜索残余 + dyn refine**；
2. **v129 与 v128 逐位相同**（0.861039125、paired 3/1/0 全同）：校准搜索砍半对最终
   选择是 bit-exact 无损，时间 37.9→21.6s（−43%）；搜索仍有大量可砍冗余；
3. **时间核算修正**：dyn Q/K refine 实测 `0.095–0.100s/call`（哨兵长度 128/512），
   与 v128 legacy official-shape-v1 的 0.08s/call 一致；compact 下超时元凶确认为
   校准搜索（v129 calib 仍占 20.8s / 4 states = 5.2s/call）；
4. **S1 时间外推**：default 120 cases（长度 10/128/512/1024/1024）dyn 增量约
   30–50s（线性于 T），加 gram64 校准 ~7s，边缘超出 parent+40s 门禁——S1 实现必须
   做 block 级并行化（v128 的 16 blocks × 16 coords × 3 sweeps = 768 Python
   iterations/call，blocks 相互独立可批处理，预计 ~6× 加速 → dyn 增量 ~7–10s）。

## 4. 决定

- 按判读表进入 **S1**（交叉算子 Gram64 per-call 精化，从 v160 归档分支）；
- 实现要求追加：refine 循环按 block 并行批处理（机制不变，纯执行优化），以满足
  default attention API ≤ parent+40s 门禁；
- 1 个回归 case 需在 S1 default 阶段复核（compact 单 case 回归可能是哨兵特异）。
