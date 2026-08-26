# v003 — C1 A1 Real-Attention Local Champion

- Date: 2026-08-27
- Candidate ID: `C1`
- Parent: `B0`
- Unique mechanism: use real deployed Attention output error to select Q/K calibration transforms; A2 H64, A3 V candidates and L1 scale candidates are disabled.
- Source SHA256: `310570B265C705D6F09E3863CD56B1931EA9E971BCEE7E6D8E2DDC029A184B88`
- Parent B0 SHA256: `C3EC6101BF30BD42983D97664A27DEDFF13274234BE5B4C97B07C6276ACBB534`
- Local status: `local-champion`
- Official status: `unavailable`
- Official score/runtime: `NA / NA`

## Fixed configuration

- Model: GPT-2, 12 layers, hidden 768, MHA 12×64; GQA smoke topology 12/6×64.
- Sequence/calibration/test: 128 / 2 / 2.
- Masks: causal and non-causal.
- NVFP4 modes: amax6, amax4, pow2.
- Token offsets: 0, 97, 193, 389.
- Accuracy device: CUDA; final paired time also measured on CPU.

## Paired Attention results

| Case | B0 causal | C1 causal | Delta | B0 non-causal | C1 non-causal | Delta |
|---|---:|---:|---:|---:|---:|---:|
| MHA amax6 offset 0 | 0.3785 | 0.4497 | +7.12pp | 0.4116 | 0.4942 | +8.26pp |
| MHA amax6 offset 97 | 0.3808 | 0.4374 | +5.66pp | 0.3984 | 0.4635 | +6.51pp |
| MHA amax6 offset 193 | 0.4052 | 0.4555 | +5.03pp | 0.3589 | 0.4504 | +9.15pp |
| MHA amax6 offset 389 | 0.3831 | 0.4487 | +6.56pp | 0.3928 | 0.4679 | +7.51pp |
| MHA amax4 offset 0 | 0.3446 | 0.4037 | +5.91pp | 0.3341 | 0.4086 | +7.45pp |
| MHA pow2 offset 0 | 0.4302 | 0.4701 | +3.99pp | 0.3866 | 0.4623 | +7.57pp |
| GQA amax6 offset 0 | 0.3214 | 0.4066 | +8.52pp | 0.3667 | 0.4949 | +12.82pp |
| GQA amax6 offset 193 | 0.3333 | 0.4169 | +8.36pp | 0.4154 | 0.4928 | +7.74pp |

Aggregate observations:

- Six MHA cases average causal delta `+5.71pp`, non-causal delta `+7.74pp`.
- Two GQA cases average causal delta `+8.44pp`, non-causal delta `+10.28pp`.
- Linear q/k/v/o/fc/proj is unchanged from B0 because C1 only changes Attention calibration selection.

## Tail and time

- Known tail debt: GQA amax6 offset 193, non-causal layer index 10 changed from `0.3569` to `0.2880`, delta `-6.89pp`. This is retained as the explicit C2 optimization target; it does not erase the cross-case aggregate gain.
- CUDA algorithm-stage: B0 `18.57s`, C1 `20.57s`, ratio `1.1077`.
- CPU algorithm-stage: B0 `52.26s`, C1 `54.72s`, ratio `1.0471`.
- Release tests at archive time: six tests passed; static no-I/O/debug scan, feature-off B0 equality, GQA/head_dim 128 state legality, rotation invariant and nested timing accounting are covered.

## Decision

`accepted as local Champion`. C1 produces a large, repeated Attention mean improvement across token windows, masks, scale modes and MHA/GQA while preserving Linear and staying within the local time budget. The single observed GQA tail regression is recorded rather than hidden or used to revert the whole mechanism.

Next candidate: `C2 Segment-CVaR Attention selector`, built directly on C1. It will optimize cross-segment robustness and GQA non-causal tail behavior without enabling H64, V-bias candidates or new scale candidates.

If an official result becomes available later, append its date, submitted SHA, score and runtime here without replacing the local tables or conclusion.
