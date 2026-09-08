# v202 — Linear sample-energy calibration fusion

Status: `REJECTED` (no material local speed change)

- Parent: v195 root `solution.py`
- Parent SHA256: `839ADB1E617C3115C6B55071A34B281C5DB0FF2AA070ADBBC71FD1549E761D7F`
- Candidate SHA256: `56dc805d6e5a3aef896db8021045740292735725d688b48e3d4393e55efcb2bd`
- Mechanism: keep the first decoded full calibration activation tensors and compile the existing final sample-energy block order without the post-calibration NVFP4 decode/rebuild.
- Verification: CUDA focused check passed; weight params, activation state, compiled order, and dynamic activation output were bitwise equal on the controlled comparison. The post-calibration rebuild was not called.

Command: `.venv\Scripts\python.exe evaluator\eval.py --solution workbench\full_solution\linear-sample-energy-fusion\candidate\solution.py --baseline-solution solution.py --linear-only --shards 0,1,2,3,4,5 --stop-after-nonpositive 99 --cache artifacts\official_eval\cache\qwen3.5-4b-proxy-v2.pt --calibration-cache-mode write --algorithm-device cuda --output-dir artifacts\proxy_v3\full_solution\linear-sample-energy-fusion-six-shards`
Scope: `eval-v3` / `proxy-v3`, Qwen3.5-4B, six target-side shards, no OOD. [Manifest](../../artifacts/proxy_v3/full_solution/linear-sample-energy-fusion-six-shards/candidate/manifest.json)

Real 4B `eval-v3`, Linear-only, six shards, 336 cases, paired against the v195 root:

| candidate mean | parent mean | delta | candidate API total | parent API total |
|---:|---:|---:|---:|---:|
| 0.5292658476834800 | 0.5292658476834800 | 0 | 1021.600851 s | 1022.134326 s |

All six shard hard-output deltas were exactly `0`. Calibration API time was `772.582748 s` versus `772.430846 s` for the parent; the total difference is diagnostic-level and does not establish a material speed gain. The current root remains v195. Official status remains `unregistered/NA`.

Calibration wall / scoring wall: `816.080198 s / 388.320623 s`.
