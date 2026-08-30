# L6d structured block-circulant factor synthetic validation

- Date: 2026-08-31
- Candidate source: root `solution.py` (normalized LF SHA256
  `ec44cf79abcd5170c1667ef7e50fb0a494753c3a96c1b6fcceca9f5030630251`; raw
  workspace SHA256 `04CA0242D3034ADC9838A3091400692233C140491B176C3827413253FFC856CD`).
- Target: validate the L6d proposal before the Qwen screen. The representation uses
  at most four `64x64` kernels and one coefficient vector per block distance; distance
  zero remains in the existing `gram64` state.

## Checks

1. **Product orientation**: a four-block random block-circulant operand was multiplied
   both by `_structured_gram_matmul` and by an explicit source/target block loop. The
   maximum absolute difference was `0.0`.
2. **Exact block-circulant reconstruction**: with four distances and four requested
   components, the extracted rank was `3` (`B-1`), the relative off-block reconstruction
   error was `2.68e-7`, and compressed state size was `49,200` bytes (`4*64*64 + 4*3`
   float32 values).
3. **Random low-rank sequence**: a 16-block sequence generated from four latent kernels
   reconstructed with relative error `4.48e-7`; returned state shapes were kernels
   `(4,64,64)` and coefficients `(16,4)`.
4. **PSD wide-shaped operand**: a random `W∈R^(32×512)` produced finite state shapes
   `(4,64,64)` and `(8,4)`.
5. **Targeted regression**: `tests/test_global_activation_lrh.py` passed `11` tests;
   `py_compile` passed; `guard_solution_file('solution.py')` returned
   `violations=[]`, `static_violations=[]`, `contraction_count=22`.

## Interpretation

The rolled source/target convention is correct and the state is bounded. The synthetic
factor is an operand-local proposal only; each activation row is still re-evaluated by
the exact deployed `G_q` gate. These checks establish numerical validity, not a claim of
accuracy gain; Qwen screen is the next gate.
