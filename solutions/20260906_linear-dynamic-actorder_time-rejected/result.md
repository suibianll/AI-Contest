# Linear dynamic sample-energy act-order — REJECTED_TIME

## Source

- Final R3 source: `solution.py`
- SHA256: `7b1494ce5467e1797319cf02103d232c055615dc7538025475a044250005f432`
- R2 original source: `r2-original-solution.py`
- Parent: v189 `static-actorder-hdiag-recovered`
- Official: `unregistered/NA`（未提交）

The candidate changes only the Linear activation GPTQ block visitation order. For each dynamic
activation call it ranks 64-channel blocks by final transformed sample energy multiplied by the
parent deployed-weight importance, then reuses the parent legal GPTQ compensation and HiF4
encoder. R3 only removes a redundant CPU permutation validation/round-trip; its R1 outputs match
the R2 candidate.

## Local evidence

- Linear eval-v3 six-shard delta: mean `+0.002981296`, median `+0.001424445`, L1
  `0.003258954`, positive/negative/zero `292/44/0`.
- OOD delta mean `+0.003518763`; in-dist minus OOD delta-gap change `-0.000537467`, within
  the `0.01` gate.
- Fresh default Linear: `0.643867464427`.
- Fresh default Attention: `0.752173407020`（与 v189 不变）。
- Fresh default Overall: `0.688994940507`, above the local high `0.687776303363` by
  `+0.001218637144`.
- All outputs finite; full default coverage and dynamic-order reachability passed.

## Time gate

Fresh default API decomposition:

- `W_calib=277.721119s`
- `A_calib=61.447755s`
- `dyn_act=60.802674s`
- `dyn_qkv=2.997517s`
- predictor: `284.775756s`
- measured API total: `402.969065s`; wall: `429.641292s`

The predictor is above the mandatory `<280s` submission gate. The candidate therefore remains
`REJECTED_TIME` despite the higher local proxy score; no official submission was made and the
root `solution.py` remains v189.
