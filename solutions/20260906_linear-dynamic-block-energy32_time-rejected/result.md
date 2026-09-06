# Linear dynamic 32-row block-energy act-order — REJECTED_TIME

## Source

- Final R3 source: `solution.py`
- SHA256: `325f9bbe616ab6f308f190c9b4b9cce4f767f9ea5e9c94895177700f51a02e6e`
- Parent: v189 `17616/275s`
- Official: `unregistered/NA`（未提交）

The candidate changes only the Linear dynamic GPTQ 64-block visitation order. It estimates
current block energy from a deterministic 32-row sample, multiplies it by mean deployed-weight
importance, and reuses the parent GPTQ and legal HiF4 encoder. Attention and the six API
interfaces otherwise remain unchanged.

## Local evidence

- R2 eval-v3 Linear six-shard delta mean: `+0.002531235`; all outputs finite and dynamic
  order reachable.
- OOD delta mean: `+0.003227747`; in-dist minus OOD delta-gap change: `-0.000696513`,
  within the `0.01` gate.
- Fresh default Linear: `0.643280557360`.
- Fresh default Attention: `0.752173407020`（与 v189 不变）。
- Fresh default Overall: `0.688652578052`，低于已测本地最高 `0.688994940507`。

## Time gate

Fresh default API decomposition:

- `W_calib=272.077215s`
- `A_calib=62.651416s`
- `dyn_act=62.213140s`
- `dyn_qkv=3.295353s`
- predictor: `285.526750s`
- measured API total: `400.237124s`; wall: `427.532596s`

The predictor is above the mandatory `<280s` submission gate, and the local Overall is below
the current measured local high. The candidate remains `REJECTED_TIME`; no official submission
was made and the root `solution.py` remains v189.
