# v004 — C2 Independent-Segment CVaR

- Date: 2026-08-27
- Candidate ID: `C2`
- Parent: `C1`
- Unique mechanism: split each calibration prefix into four independent contiguous Attention segments and rank candidates with `mean + 0.50×worst-quartile + 0.25×cross-segment-std`.
- Source SHA256: `7A65B6369EF5734990ACD373402950C8954ED46B1137C4695A44E7412B03D13A`
- Parent SHA256: `310570B265C705D6F09E3863CD56B1931EA9E971BCEE7E6D8E2DDC029A184B88`
- Local status: `local-rejected`
- Official status: `unavailable`
- Official score/runtime: `NA / NA`

## Development result

Fixed development configuration: GPT-2 12 layers, seq/calib/test 128/2/2, offset 0, amax6, CUDA, both masks.

| Topology | Metric | C1 | C2 | Delta |
|---|---|---:|---:|---:|
| MHA 12/12 | causal | 0.4497 | 0.4155 | -3.42pp |
| MHA 12/12 | non-causal | 0.4942 | 0.5250 | +3.08pp |
| GQA 12/6 | causal | 0.4066 | 0.4053 | -0.13pp |
| GQA 12/6 | non-causal | 0.4949 | 0.5008 | +0.59pp |

- Main-metric win rate: `50%`, below the preregistered 70% gate.
- MHA CUDA algorithm-stage: C1 `20.57s`, C2 `26.68s`, ratio `1.297`, above the 1.15 time gate.
- Linear is unchanged.
- Tests: 7 passed; source and archive SHA256 matched.

## Diagnosis and decision

The independent segment construction resets causal history at every segment. It therefore changes the calibration task rather than only measuring regional robustness. It also repeats deployed q/k/v quantization per segment, producing 480 nested calls instead of approximately 100–120.

`local-rejected`. Per the preregistered protocol, offsets 97/193/389 and CPU timing were not run after the development gate failed. C1 remains the local Champion; this branch does not roll the project back to B0.

Next candidate: `C2a Query-Segment CVaR`. It will quantize and evaluate the complete Attention sequence once, preserve full causal context, and compute robustness only by segmenting output query rows.
