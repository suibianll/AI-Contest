# v133 result

Status: **RETAINED / LOCAL HISTORICAL PARENT (official untested)**.

- Parent: v132.
- Change: add the output gain used by the later v134 Linear path.
- Source SHA256: `59EB07683D12ECA26A4CA1892E7A03C477C5871F5C1C76A822A209202EA6CF05`
- Protocol: `official-shape-v1`, read-only Qwen2.5-0.5B cache, CUDA.

| Linear mean | Attention mean | API total | Wall |
|---:|---:|---:|---:|
| 0.483610 | 0.834256 | 287.941 s | 310.621 s |

An equivalent archived-source rerun returned API `291.275s`, wall `314.005s`, with identical
precision. v133 is retained only as the parent of v134; no official result exists.
