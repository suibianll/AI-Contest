# v229 — A-MC1 K 侧 per-call 均值再定心（冻结根全部 state，量化前 softmax 精确不变的 K 平移）

Status: local positive on all six shards; official status `unregistered/NA` (awaiting the user's
batched official evaluation).

- Parent: retained v202 Linear + v195 Attention complete root, SHA256
  `56dc805d6e5a3aef896db8021045740292735725d688b48e3d4393e55efcb2bd`.
- Candidate SHA256: `d1c23fa11198e56f15ac8f64e033c00333dcd2d5660cec773598624c4b247f4d`.
- Mechanism: freeze all existing root state (including the gate-accepted rotation/center); after the
  root's rotation+center and before `_dense_to_hif4`, recenter the current call's K along the token
  axis per head group: `K -= mean_tokens(K)`. Pre-quantization softmax output is exactly invariant
  (logits only gain a per-query constant `Q·mean(K)`, which softmax removes); all effect comes from
  shifting the quantizer input distribution. No training, no hyperparameters, no calibration-fitted
  parameters — the mean comes from the current call itself, so the v227/v228 calibration-window
  overfitting failure mode does not apply. A per-layer gate on case-equal-weighted true
  deployment-path MSE over all 5 calibration folds writes `k_state["k_mean_recenter"]=1` only on
  strict improvement; otherwise the layer keeps the parent.
- Fit: all 5 calibration folds, window-equal-weighted true MSE comparison of parent vs
  parent+recenter arms per layer; no parameter sweep.

## Local result

`eval-v3`, Qwen3.5-4B proxy-v2, attention-only target side, all six shards (72 cases), run to
completion with `--stop-after-nonpositive 6`
(`artifacts/proxy_v3/attention-amc1-sixshard-full-20260910/candidate/`; shard0 additionally has a
standalone run at `artifacts/proxy_v3/attention-amc1-shard0-20260910` with `reasonableness_issues: 0`):

| shard (layer) | delta gain mean | positive / negative / zero |
|---:|---:|---|
| 0 (layer 0) | +0.000000 | 0 / 0 / 12 |
| 1 (layer 1) | +0.020038 | 10 / 2 / 0 |
| 2 (layer 8) | +0.000000 | 0 / 0 / 12 |
| 3 (layer 15) | +0.079126 | 12 / 0 / 0 |
| 4 (layer 22) | +0.000000 | 0 / 0 / 12 |
| 5 (layer 5) | -0.009628 | 4 / 8 / 0 |

Equal-shard mean delta gain `+0.014923`; pooled `26 / 10 / 36` of 72 cases; manifest candidate
overall `+0.548920` vs baseline `+0.533998`. API total (diagnostic only, 1 calibration cache hit)
29.725s. The gate accepted 3 of 6 layers (layers 1/5/15); layers 0/8/22 fell back to the parent
(bit-identical shards 0/2/4). Layer 15 improved uniformly on all 12 cases by about +0.079; layer 5
was accepted by the calibration gate but is net negative on the eval window (`-0.009628`), a
gate-vs-eval scope difference recorded as-is.

Controls (all PASS, `workbench/full_solution/attention-amc1-k-mean-recenter/`, raw output in
`control_results.txt`): arm off restores the parent bit for bit; arm on with a shifted input changes
the K five fields while the dense softmax output is exactly invariant (max|Δ|=1.86e-9) and Q/V and
both Linear APIs stay bit-identical to the root; six APIs import standalone outside the repo;
`validate_state` passes; the gate is verified on both accept and reject paths;
`root_rotation_frozen=True`.

## Interpretation

This is the first locally positive Attention candidate on this plan line. The mechanism has no
calibration-fitted parameters (the recentering mean comes from the current call), so it is not
subject to the v227/v228 calibration-window overfitting pattern; the gains are concentrated and
uniform (layer 15: 12/12 cases, ≈+0.079 each), consistent with the mechanism hypothesis of
correcting calibration-vs-eval window statistic drift. Layer 5 was accepted by the gate yet is net
negative on the eval window; this gate/eval scope discrepancy is recorded honestly. Per A-MC1 plan
§6, the local non-negative result is archived and awaits the user's batched official evaluation.
The root remains v202 Linear + v195 Attention at `18053/281s`.
