# Linear compiled calibration sample-energy block order

Status: `CLOSED / R3_REJECTED_SCORE_TIE`

## Hypothesis

The recent dynamic sample-energy block order improves Linear output error, but
its online order calculation pushes the predicted official time above the
`<280s` submission gate. The same fixed rule can be compiled during weight
calibration: average the final transformed calibration-window block energies,
multiply by the parent deployed-weight importance, and sort the legal 64-channel
blocks once. Online activation quantization then uses only the compiled order
and the existing v189 GPTQ path.

This is one mechanism and one fixed aggregation. It does not change the HiF4
codec, the weight transform, residual rank, Attention, or GPTQ search settings.

## Parent and code entry

- Parent: root v189, source SHA
  `261202248a0146a2ee45f3df60bd1979bb8171b7c162921013b0024c848617af`.
- Candidate source: `workbench/linear_compiled_sample_energy_solution.py`.
- The candidate wraps the v189 static activation-GPTQ implementation. During
  calibration it reconstructs each supplied calibration pair after the parent
  final transform, averages its per-channel energy, applies parent
  `activation_state["importance"]`, and stores one legal block permutation.

## R1 result

Run the fixed eval-v3 compact Linear screen with the shared proxy-v2 cache and
CUDA. Record reachability, finite/legal state, focus median/worst quartile,
negative cases, validation/test agreement, and unchanged Attention control.
R1 passed. Both compact shards continued with positive paired means and no
reasonableness blockers; the compiled order was reachable (`14/76` legal
blocks). The unchanged Attention control remained finite.

- shard 0: mean delta `+0.001906528`, median `+0.0010857057`, `44+/12-`
- shard 1: mean delta `+0.001949470`, median `+0.0008583293`, `44+/12-`

## R2 result

The full six-shard Linear run passed the interface and reachability checks:
336 paired cases had delta mean `+0.002981296`, with `292+/44-` cases and
weighted L1 `0.0000357554`. OOD also passed: delta mean `+0.003518763`, and
the in-distribution minus OOD delta-gap change was approximately `-0.000537467`,
inside the `0.01` diagnostic gate.

## R2 and R3

If R1 passes, run all six Linear shards plus OOD and then one fresh default
both-scenario audit. Use the standard local safety signal `L1 < 0.02` and
`|delta gap| <= 0.01` only as risk diagnostics; local score is never converted
to an official score. Apply the official-time predictor and submit only if
the fresh prediction is `<280s` and the candidate exceeds the current local
default high `0.688994940507`.

R3 used three implementations of the same fixed rule. The initial wrapper and
fused implementation preserved the local score but predicted `280.227124s`
and `281.100919s`, respectively. The final direct-core implementation removed
the redundant wrapper and passed the time predictor at `279.445203s`:

- source SHA: `d66128a62e7e068edc50c91f4d8e212f586a6edcaee5bea7d3564166e258b0f6`
- Linear: `0.643867464427`
- Attention control: `0.752173407020`
- Overall: `0.688994940507`
- decomposition: `W=266.429245s`, `A=56.609574s`, `dyn_act=59.774616s`,
  `dyn_qkv=2.946691s`

The Overall is exactly tied with the measured local high
`0.688994940507429`; it is not strictly higher, so the candidate was not
submitted. The complete source variants, raw proxy evidence, fresh audits,
hashes, and execution record are archived under
`solutions/20260906_linear-compiled-sample-energy_score-tie/`. Root
`solution.py` remains v189.
