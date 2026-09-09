# v227 — A-G1 Q/K 联合仿射 Gauge（rotation/center/reciprocal log-scale 联合训练）

Status: `REJECTED` locally; official status `unregistered/NA`. No official submission was made.

- Parent: retained v202 Linear + v195 Attention complete root, SHA256 `56dc805d6e5a3aef896db8021045740292735725d688b48e3d4393e55efcb2bd`.
- Candidate SHA256: `165e1a6bc50abd417a2945778741a5839a5636e7c93008bc34a2af47b3de674f`.
- Mechanism: inside the parent A2 `_a2_train_rotation` training loop, add one zero-mean log-scale vector
  `s[head_dim]` per KV group (clamped to `±log 2`), jointly updated by the same Adam as rotation/K-center
  (same 32 steps, no extra loop, no extra configuration). At deployment, after `learned_rotation` /
  `learned_center` and before `_dense_to_hif4`, Q is multiplied by `exp(+s)` and K by `exp(−s)`; V is
  untouched. Mathematically `Q'K'ᵀ = QKᵀ + constant-column`, so the pre-quantization softmax is unchanged.
  A true deployment-path MSE gate on the last calibration fold writes `(R,c,s)` only on strict improvement,
  otherwise all three revert to the parent state.
- Fit: parent A2 calibration folds (first N−1 fit, last fold gates), no parameter sweep, no official
  result awaited.

## Local result

`eval-v3`, Qwen3.5-4B proxy-v2, attention-only target side, all six shards (72 cases), early stopping
disabled and run to completion (`artifacts/proxy_v3/attention-ag1-sixshard-full-20260910/candidate/`):

| shard | delta gain mean | positive / negative / zero |
|---:|---:|---|
| 0 | -0.002287 | 6 / 6 / 0 |
| 1 | -0.021423 | 6 / 6 / 0 |
| 2 | +0.000000 | 0 / 0 / 12 |
| 3 | -0.003990 | 7 / 5 / 0 |
| 4 | +0.000814 | 4 / 8 / 0 |
| 5 | -0.004880 | 5 / 7 / 0 |

Equal-shard mean delta gain `-0.005294`; manifest candidate overall `+0.528703` vs baseline `+0.533998`;
pooled `28 / 32 / 12` of 72 cases. `all_outputs_finite=true`, `expected_case_coverage=true`; API total
(diagnostic only, 2 calibration cache hits) 29.136s. shard2 is bit-identical to the parent; shard0 error
concentrates at length=10 (mean `-0.016953`) and shard1 in the test split and mid/long sequences
(128/512/1024 all negative).

Controls (all PASS, `workbench/full_solution/attention-ag1-joint-affine-gauge/`): `s=0` restores the
parent five fields and outputs bit for bit; synthetic non-zero `s` keeps dense softmax output within
max|Δ|=3.7e-9 while hard five fields change significantly (reachable, non-no-op); six APIs import
standalone outside the repo; `validate_state` passes; a 12-seed gate probe accepts 6 / reverts 6
correctly; V and both Linear APIs are bit-identical to the root.

## Interpretation

The mechanism is reachable — 60/72 cases have changed hard output — but the net effect is negative
(equal-shard mean `-0.005294`), with error concentrated in short sequences and the test split. The
mechanism is closed as `REJECTED`; per plan §7 it is not retried with reduced steps, narrower scale
ranges, or finer per-head/per-block granularity, and no reciprocal-parameter neighborhood is appended.
The root remains v202 Linear + v195 Attention at `18053/281s`.
