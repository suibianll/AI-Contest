# v005 — C2a Query-Segment CVaR

- Date: 2026-08-27
- Candidate ID: `C2a`
- Parent: `C1`
- Unique mechanism: preserve complete causal/non-causal Attention and segment only output query rows when computing the robust calibration objective.
- Source SHA256: `CD2AD6B42DAB90619F6EA2DED0DDE988EE6F3EEC57175503AEF292C92216B162`
- Parent SHA256: `310570B265C705D6F09E3863CD56B1931EA9E971BCEE7E6D8E2DDC029A184B88`
- Local status: `local-rejected`
- Official status: `unavailable`
- Official score/runtime: `NA / NA`

## Development result

Configuration: GPT-2 12 layers, seq/calib/test 128/2/2, offset 0, amax6, CUDA, both masks.

| Topology | Metric | C1 | C2a | Delta |
|---|---|---:|---:|---:|
| MHA 12/12 | causal | 0.4497 | 0.4444 | -0.53pp |
| MHA 12/12 | non-causal | 0.4942 | 0.4890 | -0.52pp |
| GQA 12/6 | causal | 0.4066 | 0.4066 | 0.00pp |
| GQA 12/6 | non-causal | 0.4949 | 0.4891 | -0.58pp |

- MHA CUDA algorithm-stage `19.65s`; GQA `19.96s`. Dynamic call counts returned to C1 scale, confirming that complete-context query segmentation fixed C2's engineering overhead.
- Linear unchanged; 8 tests passed.

## Decision

`local-rejected`. The implementation preserved causal context and bounded cost, but the robust query-segment objective did not improve any aggregate development metric over C1. Per the preregistered gate, fixed regression offsets and CPU timing were not run.

C1 remains the local Champion. Next candidate: `C3 top-K 8×8 Linear second-order`, built from the exact C1 archive and isolated from all C2/C2a changes.
