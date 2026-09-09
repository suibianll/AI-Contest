# v199 — GQA × 64-block hard reciprocal

Status: `REJECTED` (local proxy diagnostic only)

- Parent: v195 root `solution.py`
- Parent SHA256: `839ADB1E617C3115C6B55071A34B281C5DB0FF2AA070ADBBC71FD1549E761D7F`
- Candidate SHA256: `e07c7c74e15414e2f8a01e5b2f19e49a852f33a303558b136dd2089974dc4bb7`
- Mechanism: one reciprocal `u[g,b]` per GQA group/64-channel block; first real HiF4 boundary, hard Q/K output selection, one block per group, V frozen.
- Verification: CUDA focused check passed; 32 boundaries were attempted per real 4B Attention layer, with accepted states in 4/6 calibration layers.

Command: `.venv\Scripts\python.exe evaluator\eval.py --solution workbench\full_solution\attn-gqa-hard-reciprocal\candidate\solution.py --baseline-solution solution.py --attention-only --shards 0,1,2,3,4,5 --stop-after-nonpositive 99 --cache artifacts\official_eval\cache\qwen3.5-4b-proxy-v2.pt --calibration-cache-mode write --algorithm-device cuda --output-dir artifacts\proxy_v3\full_solution\attn-gqa-hard-reciprocal-six-shards-fixed`
Scope: `eval-v3` / `proxy-v3`, Qwen3.5-4B, six target-side shards, no OOD. [Manifest](../../artifacts/proxy_v3/full_solution/attn-gqa-hard-reciprocal-six-shards-fixed/candidate/manifest.json)

Real 4B `eval-v3`, Attention-only, six shards, paired against the v195 root:

| candidate mean | parent mean | delta | candidate API total | candidate calibration |
|---:|---:|---:|---:|---:|
| 0.5339246336849365 | 0.5339975851981205 | -0.0000729515 | 84.079834 s | 81.909471 s |

Shard hard-output deltas: `0`, `+0.0000842961`, `-0.0003923108`, `0`, `-0.0000476398`, `-0.0000820546`.

Calibration wall / scoring wall: `82.020024 s / 10.414066 s`.

The candidate changed real hard outputs but was negative in aggregate, so it is not the root.

## Official result

- **`TIMEOUT / >300s / score NA` (user-reported 2026-09-09).** No score is inferred;
  the candidate is not retained.
