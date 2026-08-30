# v110 — L4b final-Gram GALS with exact deployed gate

- **Status:** accepted precision parent; time is exploratory and currently outside the 420 s submission limit.
- **Parent:** v109 L4a final deployed-Gram row gate.
- **Source:** `solution.py` snapshot with `_L4_GALS_FINAL_ENABLED = True`.
- **Source LF SHA256:** `b6d69e17c1224bfaba1c28bf22ee71b63da0ad388a22d0fbbb4c016b290953af`.
- **Screen:** selected-layer Linear mean `0.52929209` (v109 screen `0.52926909`).
- **Full Qwen panel:** Linear mean `0.507339527821859`; Attention mean `0.842039488461032`; panel total `295.242779647671` (report rounded to `295.242780`); native total `421.767953588548` (report rounded to `421.767954`).
- **Runtime:** official API total `701.901s`; this is intentionally not a rejection in the accuracy-first phase.

## Algorithm

For the same expansive shapes as L4a, derive analytical E6M2 offset candidates around critical mantissa/exponent boundaries. Rank activation blocks using the deployed 64-channel Gram and evaluate candidates with atomically selected hierarchy fields. A candidate is then retained row-wise only when the complete deployed Gram `G_q` does not increase the product loss. Calibration enables the route only when both calibration folds show positive exact gains.

## Evidence

- `l4b-gals-final-gated-stratified-qwen.json/.md`: five-layer / seven-role screen.
- `v110-l4b-gals-final-gated-qwen-full.json/.md`: 24-layer full-layer gate.
- `test_global_activation_lrh.py`: finite-state and exact-gate regression checks.
