# v137 result

Status: **REJECTED** (additional calibration sweep regressed Linear versus v134).

- Parent: v134
- Change: one additional output-supervised activation coordinate sweep during
  calibration only; online activation remains at the v134 two-sweep budget.
- Source SHA256: `2c6c553e973a34885378c7b862fa22dcfc9aaf347e2922c693b89dc647c03319`
- Protocol: `official-shape-v1`, same read-only Qwen cache, CUDA

| Linear mean | Attention mean | API total | Wall |
|---:|---:|---:|---:|
| 0.5071628713 | 0.8342564884 | 296.7546561 s | 319.3055379 s |

The extra sweep regressed Linear by `-0.0001566` and raised API time by about
`6.9s` versus v134, so it is rejected and is not copied to the root.  Official
score/time remain unregistered.
