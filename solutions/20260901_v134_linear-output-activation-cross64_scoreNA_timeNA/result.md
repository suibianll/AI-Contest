# v134 result

Status: **RETAINED / LOCAL RESEARCH PARENT (official untested)**.

- Parent: v133 (`output-weight-qwgram-gain`)
- Change: L2 block output-supervised activation refinement.  The calibration path
  stores the contiguous 64x64 blocks of `Q(W)^T W`; online `Q(A)` uses the
  corresponding `Hq - D a` gradient while retaining the legal HiF4 code search.
  The block cross product is computed directly with batched matmul instead of
  materializing a full channel-by-channel matrix.
- Protocol: `official-shape-v1`, cache mode `read`, Qwen2.5-0.5B, CUDA
- Source SHA256: `5837e765e478b1a16a5e3170ace40fbadb670871e47c5ee2c8c748102a30478d`

## Local full-panel results

| Run | Linear mean | Attention mean | API total | Wall |
|---|---:|---:|---:|---:|
| first | 0.5073195049 | 0.8342564884 | 289.042407 s | 312.315192 s |
| idle rerun 2 | 0.5073195049 | 0.8342564884 | 289.832117 s | 313.181455 s |

The two runs are numerically identical in both score means.  API time remains
below the local 300-second proxy in both runs; wall time is recorded only as a
diagnostic.  Official score/time are not registered (`scoreNA_timeNA`).

Decision: promote this snapshot to the repository root as the current precision
parent, then continue with the L3 continuous output-compensation experiment.
