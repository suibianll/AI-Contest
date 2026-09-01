# v086 / C86 attention block-smooth final-lattice candidate

- Date: 2026-08-30
- Parent: v084 / C84 full Gram-64 coverage + five coordinate sweeps
- Commit: `90844fe` (with GQA sign alignment in `31b99d6`)
- Source SHA256: `E7A16D6991DBB70A593FBE87D0C5D1D8FD38F801665354A01FFAF2F0A96F03CD`
- Official score/time: **`16744` / `222.7 s`**（新评分权重下官方通过）

> **2026-09-01 官方结果追加**：用户回传 v86 官方评测通过，**`16744 / 222.7s`
> （< 300s）**。相对父版本 v84（`16517 / 252.563s`）分数 `+227`、时间
> `−29.863s`，是目前**新权重下分数最高且最快的官方通过点**。
> 意义：C86 是本归档唯一的 Attention 侧改动（Q/K 共享 block-Hadamard），在
> 增加 Attention 工作的同时官方时间反而下降，说明"Attention 改动"本身不是
> 超时根因；真正触发超时的是 v098/v100/v121 的 B1 GQRB / B2 PAWV 路径
> （per-seq_len 分组 + Python 循环，官方端无向量化奖励）。详见
> [`v100 超时根因分析`](../../logs/execution/2026-08-31-v100-official-timeout-analysis.md)
> 与 [`v86 官方结果`](../../logs/execution/2026-09-01-v86-official-result.md)。

## 2026-09-01 local official-shape-v1 idle rerun

The candidate was rerun on an idle machine with the pinned read-only cache and the current
`official-shape-v1` protocol (250 Linear + 200 Attention cases):

| Linear mean | Attention mean | API total | Wall | Local API<300 |
|---:|---:|---:|---:|---|
| `0.4066682145` | `0.7196960689` | `299.3015726s` | `321.9955866s` | **True** |

The complete JSON and report are [`v086-idle-rerun-20260901-official-shape-v1.json`](../../artifacts/official_eval/v086-idle-rerun-20260901-official-shape-v1.json)
and [`v086-idle-rerun-20260901-official-shape-v1.md`](../../logs/official_eval/v086-idle-rerun-20260901-official-shape-v1.md).
The earlier local `462.239s / 501.257s` observation remains in its original report as a
concurrent-load/drift upper bound; it is not overwritten or used as the clean rerun result.

## Mechanism

C86 adds a shared head-local Hadamard candidate to attention Q/K calibration.
The same block size and deterministic sign pattern are used for each Q head
group and its corresponding KV head, so the continuous QK dot product is
unchanged. Candidates use block sizes 4, 8 and 16 (seed 0), are ranked with
the final offset/refinement lattice on the calibration output scorer, and only
the winning integer pair plus static signs enter Q/K state. No Linear
`activation_state`, A@W product or test output is used by this path.

## Local real-model evaluation

Configuration: `amax6 / seq128 / calib2 / test4 / cache_mode=read /
algorithm-device=cuda`; scores are local relative diagnostics, not official
score conversions.

| model | native total | panel total | Linear | Attention | API time |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B | **392.064774** | **267.307909** | 321.095451 | **70.969323** | 313.58s |
| GPT-2 small | 169.829549 | 226.872764 | 145.743266 | **24.086283** | 121.01s |
| OPT-125M | 92.579685 | 144.286224 | 73.201252 | 19.378433 | 122.22s |
| Pythia-160M | 190.239876 | 299.094679 | 149.630088 | 40.609788 | 123.18s |

Relative to v084, Qwen panel improves by `+0.018342` and remains below the
420-second primary limit. GPT-2 Attention improves substantially; OPT has a
small Attention regression and Pythia is nearly unchanged, so the mechanism is
promoted as a Qwen-primary candidate with soft guardrail caveats.

## Verification

```text
python -m py_compile solution.py
python -m pytest -q tests/test_release_candidate.py tests/test_reference_hif4.py \
    tests/test_linear_compliance_guard.py tests/test_jdrq.py \
    -k "not local_holdout_offsets"       # 48 passed, 1 deselected
```
