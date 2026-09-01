# v136 result

- Parent: v134
- Change: the same L3 diagonal Newton/Jacobi warm start as v135, but enabled
  only in online activation calls; calibration and output-supervised weight
  selection remain byte-for-byte identical to v134.
- Source SHA256: `5bffc499084a961ee6a85a120dfb79b92b4b426c351f43780a378f95529bc7f1`
- Protocol: `official-shape-v1`, same read-only Qwen cache, CUDA

| Linear mean | Attention mean | API total | Wall |
|---:|---:|---:|---:|
| 0.5001323562 | 0.8342564884 | 287.8158247 s | 310.4720871 s |

The online-only warm start regressed Linear by `-0.0071871` versus v134, so it
is rejected and is not copied to the root.  Official score/time remain
unregistered.
