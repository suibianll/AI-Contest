# v116 L6b wide-input rank-4 factor — accepted precision parent

- Parent: v115 L6a rank-16 global LRH.
- Change: for input width `d>1024` and `d<=8192`, generate a rank-4 randomized
  range factor for the off-block deployed-weight Gram; the established `d<=1024`
  rank-16 path is unchanged. Qwen's affected shape is `proj(d=4864)`.
- Candidate normalized LF SHA256: `8fa4db38ac96ca0957e1b1cee61d0c5bd248cf3a4df5d24fa04bedc9239b25f4`
  (evaluator raw source SHA: `8e3398303993ee142f8026342d476bc7a4dff1018a5db6ded9edba0b3222b0c0`).
- Synthetic validation: `32 passed`; `d=2048/4096/4864/8192` factors were finite and
  bounded at shape `d×4`; channel-cap fallback and default narrow-path skip passed.
- Compliance: static/runtime guard violations `0`; the wide calibration probe produced
  legal five-field weight parameters and one additional CPU static factor tensor.
- Screen: Qwen2.5-0.5B layers `0,5,11,17,23`, seven roles, fixed cache, CPU;
  Linear mean `0.5330906465` vs v115 `0.53284175` (`+0.00024890`).
- Full layer: Qwen2.5-0.5B, 24 layers, `seq=128`, `calib=2`, `test=4`, `amax6`,
  cache read; Linear mean `0.5093045894`, Attention mean `0.8420394885`, Qwen panel
  `295.7340450430`, native total `423.0884749711`, API `739.424609s`, wall `771.865345s`.
- Role effect: only `proj` improved (`0.4200260922→0.4215211142`); q/k/v/o/fc_gate/fc_up
  and Attention are unchanged. Delta vs v115: Linear `+0.0002135746`, panel
  `+0.0533936429`, native `+0.1435221120`, API `+22.941748s`.
- Decision: accepted as the new accuracy-first precision parent. Runtime remains exploratory
  and is not official-valid (`>420s`); C1 time compression stays deferred until L6e.
- Evidence: paired screen/full JSON and reports are copied beside this README.
