# v159 Linear GPTQ + v158 Attention

## Scope

- Base: v158 (`20260902_v158_v86-attention-matrix-smooth_retained`).
- Attention is retained unchanged from v158; only the Linear calibration and dynamic paths were
  replaced with the user-provided 17816 Linear implementation and its required GPTQ/AdaRound
  dependencies.
- `linear.txt` SHA256: `A9561B51DDF568AACA4C388762AEFF38A007C24C506C308B4B714915E7B49E9E`.
- `linear_dep.txt` SHA256: `0898A6554F4B12BFF46CEB9573181D24F04E71002A275D205E0A4689BDB758D`.
- Official-scored source SHA256: `0508045A0DDD0F17679DCA827C265CFC7588E76081D3AECEFF555D257DD4242`.
- Current archived source SHA256 after the GPU fix and exact intermediate reuse:
  `13C9CF0BFCF2277F0828D8CC1A18A8F7414DB183F3E27DD898D52597ACC5EC79`.

## Local proxy-v2 evidence

Run: Linear-only compact panel, existing dense/NVFP4 cache reused, 28 shared calibration states and
56 validation/test holdout cases. This is mechanism/generalization evidence, not an official score or
official runtime.

| Metric | v159 |
|---|---:|
| Linear mean gain | 0.705515 |
| Median / q25 / worst-quartile mean | 0.685400 / 0.591631 / 0.540141 |
| Min case gain | 0.420782 |
| Positive / negative / zero cases | 56 / 0 / 0 |
| Validation/test same-sign | 28 / 28 |
| API total / wall time | 167.570 s / 174.228 s |

Against the paired v158 compact run on the same cases, mean gain delta is `+0.149191` (56
improvements, 0 regressions). The smallest paired improvement is still positive; `proj` improves by
`+0.124209` on its eight cases. This is compact-panel evidence only and does not establish official
generalization beyond the captured Qwen proxy.

Exact evidence files are [`compact-final JSON`](../../artifacts/official_eval/v159-linear-gptq-compact-final.json),
[`compact-final report`](../../logs/official_eval/v159-linear-gptq-compact-final.md), and the
[`paired replay`](../../logs/official_eval/v159-v158-compact-paired-final.md).

## CUDA audit after device fix

| Scope | Linear | Median / q25 / worst quartile | Positive / negative | API / wall |
|---|---:|---:|---:|---:|
| compact 56 | 0.705508 | 0.685426 / 0.591841 / 0.540094 | 56 / 0 | 52.321 / 56.952 s |
| default 168 | 0.633526 | 0.626581 / 0.536043 / 0.434968 | 167 / 1 | 269.435 / 291.145 s |

The only negative default case is layer 22 `o`, validation length 10. Weight calibration consumes
`208.971s` of the default API time and is the next complexity target. Evidence: [`GPU compact`](../../logs/official_eval/v159-p0-gpu-compact.md), [`paired v158 comparison`](../../logs/official_eval/v159-v158-p1-gpu-compact-paired.md), and [`GPU Linear default`](../../logs/official_eval/v159-p1-gpu-linear-default.md).

P2 exact reuse keeps all 56 compact cases bit-for-bit equivalent in proxy output while changing API
time from `52.321s` to `51.055s`. It reuses transformed calibration samples and the deployed Weight
Gram; no candidate, threshold or state field changed. Evidence: [`P2 compact`](../../logs/official_eval/v159-p2-reuse-gram-gpu-compact.md) and [`zero-delta replay`](../../logs/official_eval/v159-p2-reuse-gram-gpu-compact-paired.md).

## Status

`official score 17532 / timeNA / runtime status unknown`

The pre-fix source scored **17532** officially. No official runtime was provided, so `<300s` remains
unknown. The current archived source changes only temporary tensor placement and has not been
officially re-evaluated; it remains v159 rather than creating a new archive version.
