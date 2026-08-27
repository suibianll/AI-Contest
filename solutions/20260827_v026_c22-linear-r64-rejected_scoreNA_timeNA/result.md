# v026 — C22 Linear R64 Incoherence Transform (REJECTED)

- Date: 2026-08-27
- Candidate ID: `C22`（26000 计划 §5）
- Parent: `C21-C` / v025，SHA256
  `83AB4864254F80D221BB491BDEF89F8C9AB8E83534FD62D4DD5E0C1C292FEA12`
- Source SHA256: `8BF16F042C0AD45A8726A4E855FBEBF4B9E95E4E42289009626F5AF03306BD97`
  （本归档保存 flag=True 的候选版本；根目录 solution.py 随后默认关闭
  `_LINEAR_R64`，行为回到 C21-C）
- Local status: `rejected`（§5.7 开发门未达）
- Official status: `unavailable`

## 唯一机制（如预注册实施）

- `_fwht_last_dim`：butterfly FWHT（动态 R64 无 dense [64,64] matmul，
  与 dense Sylvester H64 误差 <1e-5）；
- `_linear_r64_signs` / `_apply_linear_r64`：`R64 = diag(s) · H64`；
- 两阶段 seed 选择：Stage A（≤64 act rows + ≤128 weight rows，标准
  HiF4，32 seeds 排序 top-4）；Stage B（两折双向验证 + robust metric
  `max(ratio_A, ratio_W) + 0.10·max(0, tail−1)` + 全量
  `_candidate_is_safe(0.005, 0.002)`）；
- state 仅 `block_smooth_size=64` / `block_smooth_seed` 两个整数。

## 开发评测（cuda amax6 offset 0 both）

| Component | C21-C | C22 | Delta |
|---:|---:|---:|---:|
| q | 0.6008 | 0.6008 | 0.00pp |
| k | 0.5936 | 0.5936 | 0.00pp |
| v | 0.5940 | 0.5940 | 0.00pp |
| o | 0.5178 | 0.5178 | 0.00pp |
| fc | 0.4749 | 0.4749 | 0.00pp |
| proj | 0.4058 | 0.4058 | 0.00pp |
| Linear mean | 0.5311 | 0.5311 | **0.00pp** |

- Attention 逐位一致（causal `0.4497` / non-causal `0.4942`）✓。
- 72/72 组件的 seed 选择全部回退 parent（`block_smooth_size=0`），
  即 R64 在所有门控下均被拒绝，输出与 C21-C 完全相同。
- Timing：algorithm-stage `36.62s` vs C21-C `24.03s`（ratio `1.52`，
  远超 §5.7 的 ≤1.12；校准 33.35s，R64 搜索增加约 12s 纯开销）。

## 拒绝证据（诊断，evaluator-side 临时脚本，已删除）

1. 两折门 288/288 全拒（4 top seeds × 72 组件）。
2. layer 0 fc 逐折 ratio：`ratio_A≈1.17–1.18`、`ratio_W≈1.05–1.06`
   ——64 宽 Hadamard 混合同时劣化激活与权重的 HiF4 层级编码。
3. 连续 seed 哈希不敏感：sign 哈希 `i·1103515245 + seed·214013 + 12345`
   取 bit30，seed 步进 214013 极少进位到 bit 30，`range(32)` 的 32 个
   seed 实际只产生 ≈1 个符号模式（诊断显示 seeds 21–26 指标逐位相同）。
   为排除假阴性，用 `seed = s·100003`（s=0..31）分散 seed 复测。
4. 分散 seed 复测（9 个采样点 layer∈{0,5,11} × comp∈{q,fc,proj}，
   每 top-1 seed）：

   | 位置 | fold0 ratio_A / ratio_W | fold1 ratio_A / ratio_W |
   |---|---:|---:|
   | L0 q | 1.042 / 0.993 | 1.039 / 0.993 |
   | L0 fc | 1.156 / 1.065 | 1.184 / 1.052 |
   | L0 proj | 1.300 / 1.203 | 1.224 / 1.197 |
   | L5 q | 1.346 / 1.302 | 1.374 / 1.302 |
   | L5 fc | 1.153 / 1.160 | 1.179 / 1.161 |
   | L5 proj | 1.060 / 1.101 | 1.044 / 1.103 |
   | L11 q | 1.183 / 1.142 | 1.165 / 1.149 |
   | L11 fc | 1.124 / 1.084 | 1.114 / 1.090 |
   | L11 proj | 1.187 / 1.341 | 1.175 / 1.245 |

   9/9 采样点最优 seed 仍在两个 fold 上满足 `max(ratio_A, ratio_W) > 1`
   （唯一 weight 侧 <1 的 L0 q 其激活侧 1.04 仍劣化）。结论：拒绝是
   机制本身的问题，不是 seed 多样化不足。

## Decision

`rejected` per §5.7：开发门（Linear mean ≥ +0.5pp）未达（实际 0.0pp），
CPU 时间门也未达（ratio 1.52 > 1.12）。停止 seed 扩展，不实现双
Hadamard。根目录 solution.py 默认关闭 `_LINEAR_R64`（行为与 C21-C
逐位一致，由 `test_linear_r64_disabled_matches_c21c` 验证）；Champion
仍为 C21-C（v025）。C23（Full-64 Weight Schur/GPTQ）从 C21-C 构建。

## 遗留发现（供后续候选使用）

- sign 哈希对连续 seed 近似不变（进位极少到达 bit 30）：任何依赖该哈希
  的 seed 搜索必须使用大间距 seed（如 `s·100003`）才有效；
- GPT-2 真实数据上 64 宽 Hadamard 预混合对 HiF4 层级编码是净损伤
  （平滑/置换已处理离群通道，再混合只破坏块内幅度结构）。

Holdout 台账：未消耗（`0/3`），seed_hash `96dd4ed7…` 不变。
