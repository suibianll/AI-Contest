# 2026-09-09 hard-output optimization batch

This batch followed the active single-solution plan from the v195 root. No official result was awaited or inferred.
All local runs used `eval-v3` / `proxy-v3`, the Qwen3.5-4B dense cache
`artifacts/official_eval/cache/qwen3.5-4b-proxy-v2.pt`, CUDA, six target-side shards, and no OOD.

Parent SHA256: `839ADB1E617C3115C6B55071A34B281C5DB0FF2AA070ADBBC71FD1549E761D7F`.

| Version | Mechanism | Candidate SHA256 | Six-shard result | Decision |
|---|---|---|---|---|
| [v199](../../solutions/20260909_v199_attn-gqa-hard-reciprocal_scoreNA_timeNA/result.md) | GQA × 64-block hard reciprocal; real hard-output selection | `E07C7C74E15414E2F8A01E5B2F19E49A852F33A303558B136DD2089974DC4BB7` | `0.5339246336849365` vs parent `0.5339975851981205`, delta `−0.0000729515` | `REJECTED`, official `unregistered/NA` |
| [v201](../../solutions/20260909_v201_attn-hard-logit-residual_scoreNA_timeNA/result.md) | Hard-logit residual weighted reciprocal ranking | `85ED5BB799C2079F9BC604B9551407CEC160F824A221F37FB75FD5C47BF51846` | `0.5338769734778293` vs parent `0.5339975851981205`, delta `−0.0001206117` | `REJECTED`, official `unregistered/NA` |
| [v202](../../solutions/20260909_v202_linear-sample-energy-fusion_scoreNA_timeNA/result.md) | Linear sample-energy compilation fused with first decode | `56DC805D6E5A3AEF896DB8021045740292735725D688B48E3D4393E55EFCB2BD` | 336 cases and all six shard outputs exactly equal to parent; total API `1021.600851s` vs `1022.134326s` | `REJECTED`, no material speed change |
| [v203](../../solutions/20260909_v203_attn-legal-hierarchy-selection_scoreNA_timeNA/result.md) | Joint legal Q/K hierarchy-neighbor hard-output selection | `4268BAE94E58BA86CA8D926437C42FAC9C2B19E4BDC6675C0C24AF4E5460B1EC` | `0.5329010969635250` vs parent `0.5339975851981200`, delta `−0.0010964882` | `REJECTED`, official `unregistered/NA` |

Focused CUDA verifiers passed for all four candidates. The first v203 run exposed and was excluded for a CPU/CUDA
importance-index implementation error; the fixed source above was rebuilt and rerun across all six shards.
The official root remains v195 at `18053/289s`.
