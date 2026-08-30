# v115 L6a rank-16 global LRH — accepted precision parent

- Parent: v111 L5a block-local permutation.
- Change: `_ACT_GLOBAL_LRH_RANK: 8 → 16`; all transform, mix, block budget and exact
  deployed-Gram row gate settings unchanged.
- Candidate LF SHA256: `043e5401c7d8cf68339e9faec3f60943c11821e3b51bb1563d2ecd8a812f22e5`
- Targeted regression: 30 passed (`global_activation_lrh`, L5a transform, compliance,
  error decomposition, expansive CAT).
- Compliance guard: static violations `0`, runtime violations `0`, state tensor count `4`.
- Screen: Qwen2.5-0.5B, layers `0,5,11,17,23`, seven roles, fixed cache, CPU;
  Linear mean `0.53284175` vs v111 `0.53188695` (`+0.00095480`).
- Full layer: Qwen2.5-0.5B, 24 layers, `seq=128`, `calib=2`, `test=4`, `amax6`,
  cache read; Linear mean `0.5090910148`, Attention mean `0.8420394885`, Qwen panel
  `295.6806514001`, native total `422.9449528591`, API `716.482861s`, wall `748.372825s`.
- Delta vs v111: Linear `+0.0007927147`, panel `+0.1981786718`, native `+0.5327042697`,
  Attention unchanged. Runtime remains exploratory and is not yet official-valid (`>420s`).
- Decision: accepted as the new precision parent; root `solution.py` intentionally remains
  this source. Next plan step is L6b wide-input rank-4 factor.

Full JSON and report are copied beside this README; screen evidence remains in
`artifacts/real_model_suite/l6a-rank16-stratified-qwen.json` and
`logs/execution/2026-08-31-l6a-rank16-stratified.md`.
