# v024 — C21 Gated Exact Cross Selection

- Date: 2026-08-27
- Candidate ID: `C21`
- Parent: `C17`
- Unique mechanism: enable the activation 8×8 cross term, cross-gain selection and exact discrete selection from C18/C19/C20, but guard the resulting exact cross state with a new calibration gate (`_ACTIVATION_QUADRATIC8_CROSS_CALIBRATION_GATE`): the exact cross state is kept only when it improves final output MSE over C17's pure 8×8 fallback on calibration samples; the gate applies to both narrow and wide layers.
- Source SHA256: `40F4D17C12F976F83856B9641BE9A3951867BC8979992D773C60C0C1C3E8066A`
- Parent SHA256: `C29E71C332E41E262B94FF68454CEB1F1589EE932FB4E1D55C5F221CFD060766`
- Local status: `local-champion`
- Official status: `unavailable`

## Development result

Offset 0, amax6, CUDA:

| q | k | v | o | fc | proj | Linear mean delta |
|---:|---:|---:|---:|---:|---:|---:|
| +0.12pp | +0.17pp | +0.10pp | +0.05pp | +0.15pp | +0.32pp | +0.152pp |

- Linear mean `0.5890 → 0.5930`.
- Attention remains exactly C17 (`0.4497/0.4942` causal at offset 0).
- CUDA algorithm-stage `25.87s`, versus C17 `24.63s`, ratio `1.050` (within the 1.15 limit).
- Release tests pass (exit 0).

## Fixed local matrix

| Case | C17 Linear mean | C21 Linear mean | Delta | Safety result |
|---|---:|---:|---:|---|
| amax6 offset 0 | 0.5890 | 0.5930 | +0.40pp | pass |
| amax6 offset 97 | 0.5696 | 0.5747 | +0.51pp | pass; proj min -0.14 inherited from C17 |
| amax6 offset 193 | 0.5888 | 0.5928 | +0.40pp | pass |
| amax6 offset 389 | 0.5867 | 0.5912 | +0.45pp | pass |
| amax4 offset 0 | 0.4927 | 0.4973 | +0.46pp | pass |
| pow2 offset 0 | 0.5524 | 0.5575 | +0.51pp | pass; C20's -5.87pp proj failure fixed |

- 6/6 configurations improve; the pow2 wide-projection regression that rejected C20 is closed by the calibration gate (pow2 proj `0.4890 → 0.4942`).
- GQA offset 193 A/B against the C17 archive under identical evaluator settings: Attention causal `0.3433`, non-causal `0.4044`, bit-identical to parent. (The historical `0.4169/0.4928` ledger numbers were recorded under a different evaluator configuration; they are not comparable to this run and do not indicate a regression.)
- offset 97 `proj min=-0.1404` is byte-identical in the C17 parent (verified by direct A/B) and is therefore inherited, not introduced.

## Decision

`accepted as local Champion`. C21 delivers 6/6 improvement across the fixed matrix, keeps the C20 exact-selection upside while eliminating its pow2 safety failure through the calibration gate, preserves Attention exactly, and stays within the 1.15 timing limit. C17 lineage update: v024 becomes the local mainline head.

Next direction: the calibration-gated exact cross selection is the new mainline; the next single-mechanism experiment should build on C21, with the pow2-safe gate retained as a mandatory parent mechanism. Official submission should be prioritized for C21 given the official-champion (v013, 15799) is already 0.5811-equivalent locally.
