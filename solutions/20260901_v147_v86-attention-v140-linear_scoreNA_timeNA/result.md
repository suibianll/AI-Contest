# v147 result (direct single-file replacement)

Status: **RETAINED / DIRECT UPLOAD CANDIDATE / OFFICIAL UNREGISTERED**.

The old v147 wrapper was replaced in place. `solution.py` is now a readable, self-contained source
file: v140 Linear (including the one-pass A3 update) and the v86 Attention implementation are
copied directly into one module. It does not import an archive, load another solution, decode
Base64, or define duplicate public APIs.

- Current source SHA256: `44E37709A02B962CDAEDFC57E3AD999B2C9A2C0606B8B9DB7E4E81DC4DC92672`.
- Protocol: `official-shape-v1`, Qwen2.5-0.5B cache, 250 Linear + 200 Attention cases, CUDA,
  read-only cache.

Latest complete local run of this same algorithm (the subsequent source-only dead-code cleanup
does not alter reachable code):

| Metric | Value |
|---|---:|
| Linear mean | 0.5100503237 |
| Attention mean | 0.7196960689 |
| Equal-weight scale | 27145.179 |
| API total | 300.351s |
| Wall | 325.313s |

The local timer is diagnostic only; official score and official time are still **unregistered** and
remain the sole promotion criteria.

Complete local JSON/report: [`direct merge JSON`](../../artifacts/official_eval/active-v140-linear-v86-attention-a3-official-shape-v1.json),
[`direct merge report`](../../logs/official_eval/active-v140-linear-v86-attention-a3-official-shape-v1.md).
