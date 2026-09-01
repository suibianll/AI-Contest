# v147 result

Status: **RETAINED / LOCAL ATTRIBUTION CONTROL / OFFICIAL UNREGISTERED**.

This is the requested single-variable composition: the v140 ROAB-P2 Linear implementation is kept
unchanged, while all three Attention APIs are delegated to the immutable v86 C86 archive. It is a
local attribution candidate, not an official submission; no official score or official time is
inferred from this run.

- Parent Linear: v140 ROAB-P2 (`52521F1B996BF67641C22A90132ED7A7BCA477976D8A05BEC411CC9E04AA7C90`).
- Parent Attention: v86 C86 (`E7A16D6991DBB70A593FBE87D0C5D1D8FD38F801665354A01FFAF2F0A96F03CD`).
- Protocol: `official-shape-v1`, Qwen2.5-0.5B cache, 250 Linear + 200 Attention cases, CUDA,
  read-only cache.
- Composition source SHA256: `9B3EA5CB7871C1FB2F63E096C5DF08137778C3ED760A013214246D77E480B656`.

| Metric | v140 | v147 | Delta |
|---|---:|---:|---:|
| Linear mean | 0.5073546371 | 0.5073546371 | 0 |
| Attention mean | 0.7159419612 | 0.7196960689 | **+0.0037541077** |
| Local equal-weight scale | 27002.705 | 27077.787 | **+75.082** |
| API total | 205.365s | 222.227s | +16.862s |
| Wall | 229.337s | 245.038s | +15.701s |

The Linear score is bit-identical to v140, while Attention matches the clean v86 idle rerun exactly.
The combination therefore isolates the effect of restoring v86 Attention under the v140 Linear path.
The local time indicators are below 300s, but the official result remains **unregistered** and must
be obtained from the official evaluator before promotion.

Exact JSON/report: [`v147 JSON`](../../artifacts/official_eval/v147-v86-attention-v140-linear-official-shape-v1.json),
[`v147 report`](../../logs/official_eval/v147-v86-attention-v140-linear-official-shape-v1.md).
