# v228 — A-QB1 Q 侧加性 logit 偏置（per-Q-head `b_q[head_dim]`，全 folds 训练 + 全 folds gate）

Status: `REJECTED` locally; official status `unregistered/NA`. No official submission was made.

- Parent: retained v202 Linear + v195 Attention complete root, SHA256 `56dc805d6e5a3aef896db8021045740292735725d688b48e3d4393e55efcb2bd`.
- Candidate SHA256: `886a8b17736aa51a56d9cb595208975949fbfb815c597adc1ea1c7a7706c824a`.
- Mechanism: freeze all existing root state (including the rotation/center accepted by the root's own gate),
  and learn one additive logit bias `b_q[head_dim]` per Q head per layer. Training runs Adam for 32 steps
  (lr/β/clip reused from the root A2 constants) on the true deployment-path MSE, equal-weighted over all
  5 calibration fold windows; a per-layer gate on the case-equal-weighted true MSE over all folds writes
  `q_state["learned_q_bias"]` only on strict improvement. At deployment, after `learned_rotation` /
  `learned_center` and before `_dense_to_hif4`, `b_q` is added to Q elementwise; K/V/Linear paths are
  untouched.
- Fit: all 5 calibration folds for both training and gating (per the Linear-style no fit/select split
  directive); no parameter sweep, no official result awaited.

## Local result

`eval-v3`, Qwen3.5-4B proxy-v2, attention-only target side, all six shards (72 cases), early stopping
disabled and run to completion (`artifacts/proxy_v3/attention-aqb1-sixshard-full-20260910/candidate/`;
shard0 additionally has a standalone run at `artifacts/proxy_v3/attention-aqb1-shard0-20260910`):

| shard | delta gain mean | positive / negative / zero |
|---:|---:|---|
| 0 | -0.041709 | 1 / 11 / 0 |
| 1 | -0.063230 | 0 / 12 / 0 |
| 2 | -0.063832 | 0 / 12 / 0 |
| 3 | -0.105430 | 0 / 12 / 0 |
| 4 | -0.013893 | 1 / 11 / 0 |
| 5 | -0.030970 | 1 / 11 / 0 |

Equal-shard mean delta gain `-0.053177`; manifest candidate overall `+0.480820` vs baseline `+0.533998`;
pooled `3 / 69 / 0` of 72 cases. `all_outputs_finite=true`, `reasonableness_issues: 0`; shard0 was run
first to clear interface errors. API total (diagnostic only, 1 calibration cache hit) 51.815s. shard3
(layer 15) is the worst at `-0.105430`; all six layers are negative.

Controls (all PASS, `workbench/full_solution/attention-aqb1-q-bias/`, raw output in
`control_results.txt`): `b_q=0` restores the parent Q/K five fields and outputs bit for bit; synthetic
non-zero `b_q` significantly changes the Q five fields while K/V and both Linear APIs stay bit-identical
to the root; six APIs import standalone outside the repo; `validate_state` passes; the gate is verified on
both paths (8/8 synthetic seeds accepted with 1.9%–2.7% improvement; an injected harmful bias is reverted
and never written); `root_rotation_frozen=True`.

## Interpretation

The mechanism is reachable and the gate genuinely accepts (control 8/8 seeds accepted, improvement
1.9%–2.7%), yet all six layers are strongly negative on the eval window — training on all folds plus
gating on all folds still fails. This shows that the "systematic logit bias" the Q bias fits is likewise
calibration-window-specific rather than an intrinsic quantizer property. Together with v227 (gauge) and
A-RB1 (rounding boundary), per-channel/per-element calibration-fit degrees of freedom on the Attention
side have now been rejected for the third time. The mechanism is closed as `REJECTED` per A-QB1 plan §6;
it is not retried with different step counts, learning rates, fold schemes, or head-level granularity.
The root remains v202 Linear + v195 Attention at `18053/281s`.
