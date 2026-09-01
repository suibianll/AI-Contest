# v147 result

Status: **REJECTED / OFFICIAL TIME PASS / SCORE BELOW v86**.

The user confirmed the official v147 result on 2026-09-01:

| Official field | Value |
|---|---:|
| Score | **16579** |
| Runtime | **211s** |
| Runtime status | pass (`<300s`) |
| Promotion decision | **REJECTED** |

The runtime passed, but the score is 165 points below the verified v86 baseline (`16744 / 222.7s`)
and 1237 points below the user-confirmed high score 17816. v147 is therefore not a future parent.

## Source attribution warning

The exact SHA256 of the officially submitted v147 file is **unconfirmed**. The archive directory was
previously modified in place, so the repository contains three distinct pieces of evidence:

| Evidence | Source SHA256 | Linear | Attention | API |
|---|---|---:|---:|---:|
| Original pre-A3 v147 local JSON | `9B3EA5CB7871C1FB2F63E096C5DF08137778C3ED760A013214246D77E480B656` | 0.5073546371 | 0.7196960689 | 222.227s |
| Later direct-merge A3 local JSON | `25C245DAFA6DD8D98AFBD6967F5024B6B57DD2E1EE7484A9913B782D5D999C1B` | 0.5100503237 | 0.7196960689 | 300.351s |
| Current archived source after cleanup | `44E37709A02B962CDAEDFC57E3AD999B2C9A2C0606B8B9DB7E4E81DC4DC92672` | not rerun after cleanup | not rerun after cleanup | NA |

The current archived source is a readable, self-contained single file containing v140 Linear,
v86 Attention, and one additional A3 pass. It does not import another solution. This source fact
does not prove that the same SHA was submitted officially.

## Local protocol

- Protocol: `official-shape-v1`
- Model: Qwen2.5-0.5B
- Cases: 250 Linear + 200 Attention
- Attention calibration lengths: `[10,128,512,1024,1024]`
- Cache mode: read-only
- Algorithm device: CUDA

Raw local evidence remains unchanged:

- [`pre-A3 JSON`](../../artifacts/official_eval/v147-v86-attention-v140-linear-official-shape-v1.json)
- [`pre-A3 report`](../../logs/official_eval/v147-v86-attention-v140-linear-official-shape-v1.md)
- [`later direct-merge JSON`](../../artifacts/official_eval/active-v140-linear-v86-attention-a3-official-shape-v1.json)
- [`later direct-merge report`](../../logs/official_eval/active-v140-linear-v86-attention-a3-official-shape-v1.md)
- [`official correction log`](../../logs/execution/2026-09-01-v147-official-result.md)

Decision: reject v147 and restore a reproducible pre-A3 control in an unnumbered workbench before
implementing the Activation-first Decoupled HiF4 Encoder from the active plan.
