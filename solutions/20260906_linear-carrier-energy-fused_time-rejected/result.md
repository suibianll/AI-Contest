# Linear fused carrier-energy act-order — REJECTED_TIME

## Source

- SHA256: `5fda25e6771434ddce66616b0c78d00151de15066aa0b220404a44b3f2a2593f`
- Parent: v189 `static-actorder-hdiag-recovered`
- Official: `unregistered/NA`（未提交）

## Local evidence

- Linear six-shard: `0.637750` vs parent `0.636799`; delta `+0.001045`
- OOD Linear: `0.649639` vs parent `0.648735`; delta-gap change about `+0.000046`
- Fresh default Linear: `0.641778372`
- Fresh default Attention: `0.752173407`（逐位不变）
- Fresh default Overall: `0.687776303` vs v189 `0.686889609`; delta `+0.000886695`
- All outputs finite; default panel coverage complete; reachability `1`

## Time gate

- `W_calib=282.014884s`
- `A_calib=57.234458s`
- `dyn_act=59.431035s`
- `dyn_qkv=2.958341s`
- predictor: `281.400628s`
- measured API total: `401.638718s`; wall: `428.579657s`

The fixed fused statistic did not pass the mandatory `<280s` predictor gate. It is archived as
`REJECTED_TIME`; no official submission was made and the root `solution.py` remains v186.
