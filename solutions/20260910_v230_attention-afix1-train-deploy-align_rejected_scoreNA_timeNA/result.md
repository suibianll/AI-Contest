# v230（Attention A-FIX1）— A2 训练/部署前向对齐（训练前向换完整部署编码，STE 不变）

Status: REJECTED / official TIMEOUT (>300s), reported by the user on 2026-09-10. Exact elapsed
time and score are unavailable. Official scored SHA and archive SHA:
`c2ff4ea0d6a3823e29351b616c330fa9358b588934e73019c382183130dfcd6f` (bound to the unique
registered v230 Attention candidate; the same-numbered v230 Linear L-EM2 is a different
candidate). Local six-shard was net negative and is kept as a diagnostic only. See
`logs/execution/2026-09-10-v230-attention-afix1-official-timeout.md`.

- Parent: retained v202 Linear + v195 Attention complete root, SHA256
  `56dc805d6e5a3aef896db8021045740292735725d688b48e3d4393e55efcb2bd`.
- Candidate SHA256: `c2ff4ea0d6a3823e29351b616c330fa9358b588934e73019c382183130dfcd6f`.
- Mechanism: train/deploy forward alignment — inside `_a2_train_rotation`, the Q/K quantization in
  the training forward is switched from bare `_dense_to_hif4` to the full deployment encoding (the
  same `hif4_dynamic_quantize_q/k` path the gate uses, with the player state carrying the current
  rotation/center); the STE backward (`_m_attention_backward`) is unchanged. Parameterization,
  step count, lr, windows and the gate are all identical to the root. This changes the fidelity of
  the training objective, not the searched (R, c) manifold.
- Plan: `docs/superpowers/plans/parallel/2026-09-10-attention-train-deploy-align-plan.md`.

## Local result

`eval-v3`, Qwen3.5-4B proxy-v2, attention-only target side, all six shards (72 cases), run to
completion with `--stop-after-nonpositive 6`
(`artifacts/proxy_v3/attention-afix1-sixshard-full-20260910/candidate/`; shard0 additionally has a
standalone interface-check run at `artifacts/proxy_v3/attention-afix1-shard0-20260910`):

| shard (layer) | delta gain mean | positive / negative / zero |
|---:|---:|---|
| 0 (layer 0) | -0.000133 | 7 / 5 / 0 |
| 1 (layer 1) | -0.004687 | 8 / 4 / 0 |
| 2 (layer 8) | +0.000000 | 0 / 0 / 12 |
| 3 (layer 15) | -0.013820 | 4 / 8 / 0 |
| 4 (layer 22) | +0.001307 | 7 / 5 / 0 |
| 5 (layer 5) | -0.011968 | 3 / 9 / 0 |

Equal-shard mean delta gain `-0.004884`; pooled `29 / 31 / 12` of 72 cases; manifest candidate
overall `+0.529114` vs baseline `+0.533998`. API total (diagnostic only, 1 calibration cache hit)
34.475s; the shard0 standalone run's calibration API was 11.017s (root ≈8s — the aligned forward
costs about 1.4× in calibration, the known fixed cost of the deployment-encoding forward).

Controls (all PASS, `workbench/full_solution/attention-afix1-train-deploy-align/`, raw output in
`control_results.txt`): zero training steps restores the parent bit for bit; the training forward
matches direct calls to the deployment APIs bit for bit (8 Q + 8 K recorded calls checked, rotation
updates between calls verified); six APIs import standalone outside the repo; `validate_state`
passes; V and both Linear APIs stay bit-identical to the root.

## Interpretation

The mechanism is reachable and non-equivalent: the seed probe
(`seed_probe.py` / `seed_probe_results.txt`) shows the aligned forward pushes training to a
different point on the (R, c) manifold. Locally it is net negative (`-0.004884`). Layer 15
regressed markedly both times a card retrained the rotation (v227 and this card, here
`-0.013820`), while layer 15 accepts its rotation in the root (gate +2.43%) — the contrast between
"freeze the parent rotation and add an increment" (A-MC1 style) and "retrain the rotation" (this
card) once again indicates that the root's rotation arm should not be retrained. The official
verdict is TIMEOUT (>300s): the ~1.4× calibration cost of the aligned forward is infeasible on the
official machine (same cost class as v229's calibration-time gate forwards), so the local accuracy
question was never priced. This closes this implementation; the train/deploy alignment route may
only be retried after its calibration-time cost is eliminated, not by shrinking steps or windows.
The root is now v230 Linear (L-EM2) + v195 Attention at `18428/292s`.
