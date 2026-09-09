# v203 — joint legal hierarchy-neighbor selection

Status: `REJECTED` (local proxy diagnostic only)

- Parent: v195 root `solution.py`
- Parent SHA256: `839ADB1E617C3115C6B55071A34B281C5DB0FF2AA070ADBBC71FD1549E761D7F`
- Candidate SHA256: `4268bae94e58ba86ca8d926437c42fac9c2b19e4bdc6675c0c24af4e5460b1ec`
- Mechanism: freeze the parent continuous Q/K transform; compare fixed ±1 adjacent E6M2 scale-factor-code neighbors for q-only, k-only, and joint qk 64-block proposals, then legally re-encode lv2/lv3/mantissa and select by real hard Attention output. V frozen.
- Verification: CUDA focused check passed; zero-offset re-encoding is bitwise equal to the parent and all selected outputs validate as legal HiF4 state.
- A first uncounted run exposed a CPU/CUDA importance-index bug; the minimal fix produced the candidate SHA above and was rerun completely.

Command: `.venv\Scripts\python.exe evaluator\eval.py --solution workbench\full_solution\attn-legal-hierarchy-selection\candidate\solution.py --baseline-solution solution.py --attention-only --shards 0,1,2,3,4,5 --stop-after-nonpositive 99 --cache artifacts\official_eval\cache\qwen3.5-4b-proxy-v2.pt --calibration-cache-mode write --algorithm-device cuda --output-dir artifacts\proxy_v3\full_solution\attn-legal-hierarchy-selection-six-shards-fixed`
Scope: `eval-v3` / `proxy-v3`, Qwen3.5-4B, six target-side shards, no OOD. [Manifest](../../artifacts/proxy_v3/full_solution/attn-legal-hierarchy-selection-six-shards-fixed/candidate/manifest.json)

Real 4B `eval-v3`, Attention-only, six shards, paired against the v195 root:

| candidate mean | parent mean | delta | candidate API total | parent API total |
|---:|---:|---:|---:|---:|
| 0.5329010969635250 | 0.5339975851981200 | -0.0010964882 | 40.585086 s | 30.997707 s |

Shard hard-output deltas: `0`, `0`, `0`, `0`, `0`, `-0.0065789294`. The legal neighbor was accepted on one calibration layer but regressed the aggregate proxy and added calibration cost, so it is not the root. Official status remains `unregistered/NA`; no official score or time is inferred.

Calibration wall / scoring wall: `38.170922 s / 10.924981 s`.

## Official result

- **`TIMEOUT / >300s / score NA` (user-reported 2026-09-09).** No score is inferred;
  the candidate is not retained.
