# v119 C1a structured proposal vectorization — accepted equivalent parent

- Parent: v118 L6d structured block-circulant factor.
- Change: batch independent selected row/block proposals and the 15 signed-level
  evaluations; preserve ascending coordinate order, tie break, `_write_codes`, and
  final exact deployed-Gram row gate. The v118 reference helper remains in source for
  regression comparison.
- Candidate normalized LF SHA256: `c9c45a7911594b4b378d0c5e2769187d76dc587d79b6da9fa5f5a487e4b7cb11`
  (raw evaluator source SHA: `514eb0ae1ba9151b3433d9c005cc998213c407275913133d08444e0422ea021b`).
- Validation: 37 targeted tests passed; reference/vectorized synthetic outputs match
  field-by-field at `atol=1e-6`; compliance `violations=[]`, `static_violations=[]`.
- Qwen screen: Linear mean `0.5333753185`, exactly equal to v118 screen; no role
  regression.
- Qwen full layer: Linear mean `0.5096012555`, Attention mean `0.8420394885`, panel
  `295.8082115559`, native total `423.2878345580` — all exact score fields equal v118.
- Timing: API `2040.5046895s` vs v118 `2249.7464359s` (`-209.2417464s`, `-9.30%`);
  dynamic stage `1633.3390318s` vs `1832.8779521s` (`-10.88%`); wall
  `2072.6976340s` vs `2282.6252131s` (`-9.19%`). Still above the official 420s limit.
- Decision: accepted as v119 precision-equivalent/time-improved parent. C1b remains the
  next active-plan direction; no rank or budget change was mixed into this candidate.

