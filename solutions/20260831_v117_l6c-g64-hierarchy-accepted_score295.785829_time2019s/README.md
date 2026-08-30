# v117 L6c complete `G_64` hierarchy sweep — accepted precision parent

- Parent: v116 L6b wide rank-4 factor.
- Change: activation-side, fixed-scale hierarchy coordinate sweep. For at most four
  high-loss 64-channel blocks per row, try legal `lv2/lv3={1,2}` coordinates one
  sweep at a time, regenerate mantissas atomically, and accept only the exact
  `Delta J = 2e.T G Delta_q + Delta_q.T G Delta_q < 0` proposal. No state field
  or public API was added.
- Candidate normalized LF SHA256: `8746b8026495cb56a3dc1d622e463f89226b23e3206e2202bd468f45530d952c`
  (evaluator raw source SHA: `f199182848be486e63e75056da3515080249b41eff4f42419dc9abf3e9b84a6d`).
- Synthetic/targeted validation: `33 passed`; an independent one-block brute-force
  coordinate reference matched reconstructed values and hierarchy fields. Static/runtime
  compliance guard violations were `0`.
- Screen: Qwen2.5-0.5B layers `0,5,11,17,23`, seven roles, fixed cache, CPU;
  Linear mean `0.5332946034` vs v116 `0.5330906465` (`+0.00020396`).
- Full layer: Qwen2.5-0.5B, 24 layers, `seq=128`, `calib=2`, `test=4`, `amax6`,
  cache read; Linear mean `0.5095117268`, Attention mean `0.8420394885`, Qwen panel
  `295.7858293956`, native total `423.2276713111`, API `2019.475204s`, wall `2051.884441s`.
- Role effect: all seven roles were non-decreasing (q `0.616758`, k `0.629137`,
  v `0.571384`, o `0.498290`, fc_gate `0.395579`, fc_up `0.433860`, proj `0.421574`).
  Delta vs v116: Linear `+0.0002071374`, panel `+0.0517843527`, native `+0.1391963400`;
  Attention unchanged.
- Decision: accepted as the new accuracy-first precision parent. The implementation is
  not official-valid on runtime (`>420s`); L6d/L6e must assess structured compression
  and the C1 time path before any submission.
- Evidence: paired screen/full JSON and reports plus the synthetic log are copied beside
  this README.
