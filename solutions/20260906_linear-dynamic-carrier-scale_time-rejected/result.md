# Linear dynamic carrier-scale act-order — REJECTED_TIME

## Source

- Final R3 source: `solution.py`
- SHA256: `a9c8ce0eb2649c0b968b46a0f8fa41f88c13b214779d936f67da7a4f9db3368d`
- Parent: v189 `17616/275s`
- Official: `unregistered/NA`（未提交）

The candidate changes only the Linear dynamic GPTQ 64-block visitation order. For each
activation call it uses the incoming NVFP4 block-scale square as a cheap block-energy proxy,
multiplied by mean deployed-weight importance. GPTQ, the legal HiF4 encoder, Attention, and
the six API interfaces otherwise remain unchanged.

## Local evidence

- R2 eval-v3 Linear six-shard delta mean: `+0.001647837`; all outputs finite and dynamic
  order reachable.
- OOD delta mean: `+0.001721161`; in-dist minus OOD delta-gap change: `-0.000073324`,
  within the `0.01` gate.
- Fresh default Linear: `0.642028877051`.
- Fresh default Attention: `0.752173407020`（与 v189 不变）。
- Fresh default Overall: `0.687922431205`，低于已测本地最高 `0.688994940507`。

## Time gate

Fresh default API decomposition:

- `W_calib=277.485721s`
- `A_calib=62.427184s`
- `dyn_act=61.542205s`
- `dyn_qkv=2.948263s`
- predictor: `286.049047s`
- measured API total: `404.403374s`; wall: `431.138606s`

The predictor is above the mandatory `<280s` submission gate, and the local Overall is below
the current measured local high. The candidate remains `REJECTED_TIME`; no official submission
was made and the root `solution.py` remains v189.
