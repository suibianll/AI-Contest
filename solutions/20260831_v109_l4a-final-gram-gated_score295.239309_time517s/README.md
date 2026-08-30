# v109 — L4a final deployed-weight Gram with exact row gate

- **Status:** accepted precision parent; later L4b experiments may supersede it.
- **Source:** `solution.py` snapshot with `_L4_GALS_FINAL_ENABLED = False`.
- **Screen:** selected-layer Linear mean `0.529269091330005`.
- **Full Qwen panel:** Linear mean `0.5073256468048845`; Attention mean `0.8420394884610322`; panel total `295.23930939342756`; native total `421.75862554514146`.
- **Runtime:** official API total `517.2857728987001s` (precision exploration only; outside the 420s submission limit).

## Algorithm

For expansive shapes with `rows > channels` and `channels <= 1024`, calibration decodes the final deployed weight `W_q`, forms both the activation calibration Gram and `G_q = W_q.T @ W_q`, and builds two activation proposals. The established v107 proposal is the parent. A final-Gram proposal is retained row-by-row only when its exact full deployed quadratic loss

`(Q(A)-A) @ G_q @ (Q(A)-A).T`

does not increase. This removes regressions caused by using a block-diagonal surrogate while keeping the change shape/statistics driven and compliant.

## Evidence

- `l4a-final-gram-gated-stratified-qwen.json/.md`: 5-layer/7-role screen.
- `v109-l4a-final-gram-gated-qwen-full.json/.md`: 24-layer Qwen full-layer gate.
- The preceding un-gated final-Gram attempt was only `0.52894502` on screen and was rejected.
