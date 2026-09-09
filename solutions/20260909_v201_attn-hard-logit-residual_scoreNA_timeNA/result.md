# v201 — hard-logit residual weighted reciprocal

Status: `REJECTED` (local proxy diagnostic only)

- Parent: v195 root `solution.py`
- Parent SHA256: `839ADB1E617C3115C6B55071A34B281C5DB0FF2AA070ADBBC71FD1549E761D7F`
- Candidate SHA256: `85ed5bb799c2079f9bc604b9551407cec160f824a221f37fb75fd5c47bf51846`
- Mechanism: retain v199 hard-boundary candidates, rank the top two per GQA group with softmax-Jacobian/V-weighted logit residual, then use real hard Attention output selection; V frozen.
- Verification: CUDA focused check passed; 32 candidates were attempted and residual-ranked per real 4B Attention layer, with accepted states in 4/6 calibration layers.

Command: `.venv\Scripts\python.exe evaluator\eval.py --solution workbench\full_solution\attn-hard-logit-residual\candidate\solution.py --baseline-solution solution.py --attention-only --shards 0,1,2,3,4,5 --stop-after-nonpositive 99 --cache artifacts\official_eval\cache\qwen3.5-4b-proxy-v2.pt --calibration-cache-mode write --algorithm-device cuda --output-dir artifacts\proxy_v3\full_solution\attn-hard-logit-residual-six-shards`
Scope: `eval-v3` / `proxy-v3`, Qwen3.5-4B, six target-side shards, no OOD. [Manifest](../../artifacts/proxy_v3/full_solution/attn-hard-logit-residual-six-shards/candidate/manifest.json)

Real 4B `eval-v3`, Attention-only, six shards, paired against the v195 root:

| candidate mean | parent mean | delta | candidate API total | candidate calibration |
|---:|---:|---:|---:|---:|
| 0.5338769734778293 | 0.5339975851981205 | -0.0001206117 | 90.238993 s | 88.204849 s |

Shard hard-output deltas: `0`, `-0.0000084818`, `-0.0004152939`, `-0.0003885254`, `0`, `+0.0000886307`.

Calibration wall / scoring wall: `88.315808 s / 10.214718 s`.

The residual ranking was reachable but did not improve the aggregate hard output, so it is not the root.

## Official result

- **`TIMEOUT / >300s / score NA` (user-reported 2026-09-09).** No score is inferred;
  the candidate is not retained.
