# Linear integrated carrier-energy act-order — REJECTED_TIME

## Source

- SHA256: `a30e05b7fb71a64121908bc10d102dd50133a23e32f6568ed7044dbc45321730`
- Parent: v189 `17616/275s`
- Official: `unregistered/NA`（未提交）

## Local evidence

- R2 eval-v3 Linear mean `0.637750315` vs v189 `0.636799489`，delta
  `+0.000950827`；六片均为正向 aggregate，输出有限，reachability 完整。
- OOD Linear mean `0.649639183` vs v189 `0.648734547`；相对父版本的
  `Δ(in−OOD)` 变化约 `+0.000046`，通过 `0.01` 门。
- fresh default：Linear `0.641778372`、Attention `0.752173407`、Overall
  `0.687776303`；与前一 carrier-energy 实现逐位相同，未超过本地最高。

候选只把原始 NVFP4 激活的 32 行样本在既有校准解码循环中保存并复用，carrier-energy
排序公式和在线 GPTQ 路径保持不变。

## Time gate

- `W_calib=274.721588s`
- `A_calib=61.435543s`
- `dyn_act=60.547017s`
- `dyn_qkv=2.961587s`
- predictor: `284.291453s`
- measured API total: `399.665734s`; wall: `426.211627s`

预测时间超过强制 `<280s` 门，且分数没有超过当前本地最高，因此候选记为
`REJECTED_TIME`，未生成官方提交包；根 `solution.py` 保持 v189。

证据：`proxy-r1/`、`proxy-r2/`、`proxy-ood/`、`fresh-default.json` 和
`fresh-default.md`。
