# v159 Linear GPTQ + v158 Attention

## Scope

- Base: v158 (`20260902_v158_v86-attention-matrix-smooth_retained`).
- Attention is retained unchanged from v158; only the Linear calibration and dynamic paths were
  replaced with the user-provided 17816 Linear implementation and its required GPTQ/AdaRound
  dependencies.
- `linear.txt` SHA256: `A9561B51DDF568AACA4C388762AEFF38A007C24C506C308B4B714915E7B49E9E`.
- `linear_dep.txt` SHA256: `0898A6554F4B12BFF46CEB9573181D24F04E71002A275D205E0A4689BDB758D`.
- Candidate `solution.py` SHA256: `0508045A0DDD0F17679DCA827C265CFC7588E76081D3AECEFF555D257DD4242`.

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

## Status

`official score 17532 / timeNA / runtime status unknown`

The user reports that this exact merged version scored **17532** officially. No official runtime was
provided, so `<300s` remains unknown. The attempted CUDA default audit failed before scoring because
CPU state tensors were used in CUDA calibration arithmetic; this is an environment/code-path error,
not a local score. The next audit is GPU Linear-only after a math-preserving device fix.
