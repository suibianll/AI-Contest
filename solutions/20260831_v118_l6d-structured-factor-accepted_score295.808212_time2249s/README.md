# v118 L6d structured block-circulant factor — accepted precision parent

- Parent: v117 L6c full `G_64` hierarchy coordinate sweep.
- Change: for wide activation inputs (`1024 < d <= 8192`), aggregate deployed-Gram
  cross-block pairs by circular distance, fit at most four `64x64` kernels with an
  SVD factor, and use the compressed product only as a frozen activation proposal.
  The exact deployed `G_q` row gate remains the final decision; no public API field or
  dense block-pair state was added.
- Candidate normalized LF SHA256: `ec44cf79abcd5170c1667ef7e50fb0a494753c3a96c1b6fcceca9f5030630251`
  (evaluator raw source SHA: `04ca0242d3034adc9838a3091400692233c140491b176c3827413253ffc856cd`).
- Synthetic/targeted validation: `36 passed` across the L6 regression suite;
  `guard_solution_file` returned `violations=[]`, `static_violations=[]`,
  `contraction_count=22`. Structured products matched explicit block loops exactly;
  four-block reconstruction error was `2.68e-7` and state size `49,200` bytes.
- Screen: Qwen2.5-0.5B layers `0,5,11,17,23`, seven roles, fixed cache, CPU;
  Linear mean `0.53337532` vs v117 `0.53329460` (`+0.00008072`).
- Full layer: Qwen2.5-0.5B, 24 layers, `seq=128`, `calib=2`, `test=4`, `amax6`,
  cache read; Linear mean `0.5096012555`, Attention mean `0.8420394885`, Qwen panel
  `295.8082115559`, native total `423.2878345580`, API `2249.746436s`, wall
  `2282.625213s`.
- Role effect: q/k/v/o/fc_gate/fc_up unchanged; `proj` improved from
  `0.4215743858` to `0.4222010863` (`+0.0006267005`). Attention was unchanged.
- Decision: accepted as the new accuracy-first precision parent. Runtime is far above
  the official `<420s` limit and must be addressed by the later C1 compression route;
  L6e records the cross-block checkpoint before changing the parent.
- Evidence: paired screen/full JSON and reports plus the synthetic log are copied beside
  this README.
