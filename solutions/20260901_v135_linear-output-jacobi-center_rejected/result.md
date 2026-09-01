# v135 result

Status: **REJECTED** (Linear regressed versus v134).

- Parent: v134
- Change: L3 diagonal Newton/Jacobi warm start applied during both Linear
  calibration and online activation refinement, using the cached block output
  gradient before the legal HiF4 coordinate sweep.
- Source SHA256: `796f7c304e5119b4d3f55855988bb0d3cee120962c005090c2a5af5f0bae999a`
- Protocol: `official-shape-v1`, same read-only Qwen cache, CUDA

| Linear mean | Attention mean | API total | Wall |
|---:|---:|---:|---:|
| 0.5008120504 | 0.8342564884 | 290.8230418 s | 313.3651503 s |

The warm start regressed Linear by `-0.0065075` versus v134, so it is rejected
and is not copied to the root.  Official score/time remain unregistered.
