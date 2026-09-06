# Linear carrier-energy act-order — REJECTED_TIME

## Source

- SHA256: `c382bd4fe4acdbb21b4aacd6b009051093d3fd92a264ae645956f96def26958c`
- Parent: v189 `static-actorder-hdiag-recovered`
- Official: `unregistered/NA`（未提交）

## Local eval-v3 evidence

- Linear six-shard: `0.637750` vs parent `0.636799`; delta `+0.001045`
- OOD Linear: `0.649639` vs parent `0.648735`; delta-gap change about `+0.000046`
- Fresh default Linear: `0.641778372`
- Fresh default Attention: `0.752173407`（与父版本不变）
- Fresh default Overall: `0.687776303` vs v189 `0.686889609`; delta `+0.000886695`
- All outputs finite; default panel coverage complete; state reachability `1`

## Time gate

Fresh default API decomposition:

- `W_calib=271.089500s`
- `A_calib=57.589949s`
- `dyn_act=59.702568s`
- `dyn_qkv=2.938077s`
- predictor: `280.622241s`
- measured API total: `391.320094s`; wall: `417.378135s`

The predictor is above the mandatory `<280s` submission gate. The candidate therefore remains
`REJECTED_TIME` despite the higher local proxy mean; no official submission was made.
