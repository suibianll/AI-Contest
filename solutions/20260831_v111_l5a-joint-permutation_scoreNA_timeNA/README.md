# v111 L5a joint equivalent-frame permutation

## Status

`screen-accepted; full-layer pending`。这是从 v110 precision parent 分出的第一个
L5a 候选；官方接口不可用，目录名保留 `scoreNA_timeNA`。时间只作记录，不参与本轮
精度门禁。

## 唯一变化

根 `solution.py` 增加一个严格可回退的 block-local permutation frame。对每个 64
维 HiF4 hierarchy block，从校准激活/权重的独立 `amax/rms` pressure 统计构造三种
低自由度候选（identity、同压力量级排序、低/高交错和四分位交错），使用两折
operand-local HiF4 proxy 误差选择；只有 aggregate 改善且两折均不变差时，才写入一
个 `int32` permutation state。该排列与原有 diagonal balance、signed-Hadamard 同步
作用于 W/A，因此未量化乘积严格保持：

```text
X' = X D^-1 P R,   W' = W D P R,
X' W'^T = X W^T.
```

没有使用 evaluator 输出、test/holdout、role/model id 或在线监督。完整 hierarchy
仍由现有合法 codec 重算，最终 gate 仍使用部署侧 Gram。

## Screen

- 命令：
  `python evaluator/linear_candidate_screen.py --cache artifacts/real_model_suite/cache/qwen2.5-0.5b__seq128__calib2__test4__layersall__schema1.pt --solution solution.py --layers 0 5 11 17 23 --roles q k v o fc_gate fc_up proj --output artifacts/real_model_suite/l5a-joint-permutation-stratified-qwen.json --report logs/execution/2026-08-31-l5a-joint-permutation-stratified.md`
- Qwen2.5-0.5B，layers `[0,5,11,17,23]`，7 roles，35 cases，cache read。
- Linear mean：`0.5318869456762372`；v110 screen `0.52929209`；增量约
  `+0.00259486`。
- Weight-perfect mean：`0.7140714612323843`；Activation-perfect mean：
  `0.8188904985846687`；35/35 cases 保持合法 gate。
- Screen elapsed：`136.616s`；此时间不是官方时间，也不作为拒绝理由。
- 规范 LF SHA256：`6b229081121c4a7edd69575c93dc01488be8f8b5e1479007522421e93e1adc57`。
- 原始 screen JSON 与日志已随本目录复制；full-layer 结果待补充。

## Reproduction / acceptance

Full-layer 只有在 screen 高于 `0.52929209` 后运行。若 full-layer panel 高于 v110
`295.242779647671`，则更新根为新的 precision parent；否则保留 v110，并将本候选
标为 `rejected`。无论结果如何都保存完整 source、SHA、JSON、日志和测试记录。
