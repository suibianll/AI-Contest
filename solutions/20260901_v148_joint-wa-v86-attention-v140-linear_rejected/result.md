# v148 result

Status: **REJECTED / LOCAL TIME OVER BUDGET**.

v148 implements the planned A3 Weight--Activation alternating residual step on top of v140 Linear
and keeps v86 Attention fixed. It improves the public Linear panel, but the extra calibration pass
violates the 300-second budget, so it is not a submission candidate. The next implementation must
reuse v140's calibration products instead of repeating the full block oracle.

- Parent: v147 local attribution control (v140 Linear + v86 Attention).
- Change: one fixed A3 alternating round; no alpha/offset/seed sweep.
- Protocol: `official-shape-v1`, Qwen2.5-0.5B cache, 250 Linear + 200 Attention cases, CUDA,
  read-only cache.
- Source SHA256: `B3960A9AB0478CFDB143182055B9EFC21A29A7A6A766F88ED233267A0A16F928`.

| Metric | v147 | v148 | Delta |
|---|---:|---:|---:|
| Linear mean | 0.5073546371 | **0.5097287173** | **+0.0023740801** |
| Attention mean | 0.7196960689 | 0.7196960689 | 0 |
| Weight calibration | 143.691s | 291.582s | +147.891s |
| Dynamic activation | 16.234s | 16.041s | −0.194s |
| Attention calibration | 56.058s | 55.435s | −0.623s |
| API total | 222.227s | **369.038s** | **+146.811s** |
| Wall | 245.038s | **391.615s** | **+146.578s** |

The precision gain is real on the public panel, but both local time indicators exceed 300 seconds.
The official score/time is unregistered; this version is rejected on the local complexity evidence
and is not a future parent. Exact JSON/report:
[`v148 JSON`](../../artifacts/official_eval/v148-joint-wa-v86-attention-v140-linear-official-shape-v1.json),
[`v148 report`](../../logs/official_eval/v148-joint-wa-v86-attention-v140-linear-official-shape-v1.md).
