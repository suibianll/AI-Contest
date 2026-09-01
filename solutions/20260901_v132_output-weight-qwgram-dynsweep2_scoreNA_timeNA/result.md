# v132 result

Status: **RETAINED / LOCAL HISTORICAL PARENT (official untested)**.

- Parent: v131 algorithm family after reducing the Attention dynamic sweep budget.
- Source SHA256: `7C4884A710F17F44904E8C1C8EA1AC89667711A5B0162497AEAC4D5DAD389F3E`
- Protocol: `official-shape-v1`, read-only Qwen2.5-0.5B cache, CUDA.

| Linear mean | Attention mean | API total | Wall |
|---:|---:|---:|---:|
| 0.473131 | 0.834256 | 290.936 s | 314.251 s |
| 0.473131 | 0.834256 | 289.318 s | 311.974 s |

The two idle reruns are precision-identical. v132 is retained only to preserve the local lineage;
it has no official result and shares the Attention family that timed out in v129–v131.
