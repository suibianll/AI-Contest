# Linear dynamic block-energy act-order — REJECTED_TIME

## Source

- Final R3 source: `solution.py`
- SHA256: `fba242715fc731f658a2dae1bb8375a81e3862bfc538fb8f711162a6c03e7256`
- Parent: v189 `17616/275s`
- Official: `unregistered/NA`（未提交）

The candidate changes only the Linear dynamic GPTQ 64-block visitation order. For each
activation call it ranks blocks by the product of mean block activation energy and mean
deployed-weight importance, then reuses the parent GPTQ and legal HiF4 encoder. Linear,
Attention, and the six API interfaces otherwise remain unchanged.

## Local evidence

- R2 eval-v3 Linear six-shard delta mean: `+0.003011996`; all outputs finite and dynamic
  order reachable.
- OOD delta mean: `+0.003559695`; in-dist minus OOD delta-gap change: about `-0.000548`,
  within the `0.01` gate.
- Fresh default Linear: `0.643820278430`.
- Fresh default Attention: `0.752173407020`（与 v189 不变）。
- Fresh default Overall: `0.688967415343`, above the local high `0.687776303363` by
  `+0.001191111979`.

## Time gate

Fresh default API decomposition:

- `W_calib=275.920499s`
- `A_calib=61.814679s`
- `dyn_act=62.296941s`
- `dyn_qkv=2.932736s`
- predictor: `286.022476s`
- measured API total: `402.964855s`; wall: `430.540664s`

The predictor is above the mandatory `<280s` submission gate. The candidate therefore remains
`REJECTED_TIME` despite the higher local proxy score; no official submission was made and the
root `solution.py` remains v189.
