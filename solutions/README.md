# HiF4 Solution Archive

Root `solution.py` is the only active submission. Archived source files are immutable.

顺序实验索引见 [progressive candidate ledger](../docs/superpowers/logs/2026-08-27-progressive-candidate-ledger.md)。

| Version | Date | Topic | Local Linear | Local Attention | Local Time | Official Score | Official Time | Delta | Status | Directory |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| v000 | 2026-08-25 | v9 baseline | NA | NA | NA | ~9000+ | NA | NA | accepted | [archive](20260825_v000_v9-baseline_score9000plus_timeNA/) |
| v001 | 2026-08-26 | current baseline | NA | NA | NA | 10250 | 127s | NA | accepted | [archive](20260826_v001_current-baseline_score10250_time127s/) |
| v002 | 2026-08-26 | youxilee/hif4 v2.0 | 0.5668* | 0.3786* | NA | ~15000 | ~140s | NA | champion | [archive](20260826_v002_youxilee-hif4_score15000plus_timeNA/) |
| v003 | 2026-08-27 | C1 A1 real Attention selector | 0.5668 | 0.4497 | CPU stage 54.72s | NA | NA | local +0.0712 Attention | local-champion | [archive](20260827_v003_a1-real-attention-local_scoreNA_timeNA/) |
| v004 | 2026-08-27 | C2 independent-segment CVaR | 0.5668 | 0.4155 | CUDA stage 26.68s | NA | NA | local -0.0342 causal | local-rejected | [archive](20260827_v004_c2-segment-cvar-local_scoreNA_timeNA/) |
| v005 | 2026-08-27 | C2a query-segment CVaR | 0.5668 | 0.4444 | CUDA stage 19.65s | NA | NA | local -0.0053 causal | local-rejected | [archive](20260827_v005_c2a-query-segment-cvar-local_scoreNA_timeNA/) |
| v006 | 2026-08-27 | C3 top-K 8×8 Linear quadratic | 0.5779 | 0.4497 | CPU stage 54.29s | NA | NA | local +0.0110 Linear | local-champion | [archive](20260827_v006_c3-topk-8x8-quadratic-local_scoreNA_timeNA/) |
| v007 | 2026-08-27 | C4 8×8 coverage 10% | 0.5788 | 0.4497 | CUDA stage 20.02s | NA | NA | local +0.0009 Linear | local-accepted-not-promoted | [archive](20260827_v007_c4-8x8-coverage10-local_scoreNA_timeNA/) |
| v008 | 2026-08-27 | C5 top-K 16×16 Linear quadratic | 0.5802 | 0.4497 | CPU stage 55.92s | NA | NA | local +0.0023 Linear | local-champion | [archive](20260827_v008_c5-topk-16x16-quadratic-local_scoreNA_timeNA/) |
| v009 | 2026-08-27 | C6 16×16 coverage 4% | 0.5808 | 0.4497 | CUDA stage 20.64s | NA | NA | local +0.0006 Linear | local-accepted-not-promoted | [archive](20260827_v009_c6-16x16-coverage4-local_scoreNA_timeNA/) |
| v010 | 2026-08-27 | C7 top-K 32×32 Linear quadratic | 0.5814 | 0.4497 | CUDA stage 21.99s | NA | NA | local +0.0012 Linear | local-accepted-not-promoted | [archive](20260827_v010_c7-topk-32x32-quadratic-local_scoreNA_timeNA/) |
| v011 | 2026-08-27 | C8 bounded 64×64 Linear quadratic | 0.5811 | 0.4497 | CUDA stage 23.55s | NA | NA | local +0.0009 Linear | local-accepted-not-promoted | [archive](20260827_v011_c8-topk-64x64-quadratic-local_scoreNA_timeNA/) |
| v012 | 2026-08-27 | C9 16×16 second sweep | 0.5804 | 0.4497 | CUDA stage 22.35s | NA | NA | local +0.0003 Linear | local-accepted-not-promoted | [archive](20260827_v012_c9-16x16-second-sweep-local_scoreNA_timeNA/) |
| v013 | 2026-08-27 | C10 wide activation quadratic | 0.5811 | 0.4497 | CPU stage 50.99s | NA | NA | local +0.0054 proj | local-champion | [archive](20260827_v013_c10-wide-activation-quadratic-local_scoreNA_timeNA/) |
| v014 | 2026-08-27 | C11 wide activation 8×8 residual | 0.5816 | 0.4497 | CPU stage 60.02s | NA | NA | local +0.0031 proj | local-champion | [archive](20260827_v014_c11-wide-activation-8x8-local_scoreNA_timeNA/) |
| v015 | 2026-08-27 | C12 wide activation 16×16 residual | 0.5817 | 0.4497 | CUDA stage 22.80s | NA | NA | local +0.0007 proj | local-accepted-not-promoted | [archive](20260827_v015_c12-wide-activation-16x16-local_scoreNA_timeNA/) |
| v016 | 2026-08-27 | C13 all-width activation 8×8 | 0.5862 | 0.4497 | CUDA stage 23.34s | NA | NA | local +0.0046 Linear; amax4 o -0.0091 | local-accepted-not-promoted | [archive](20260827_v016_c13-all-width-activation-8x8-local_scoreNA_timeNA/) |
| v017 | 2026-08-27 | C14 gated all-width activation 8×8 | 0.5861 | 0.4497 | CPU stage 58.05s | NA | NA | local +0.0045 Linear | local-champion | [archive](20260827_v017_c14-gated-all-width-activation-8x8-local_scoreNA_timeNA/) |
| v018 | 2026-08-27 | C15 quantized-weight activation Gram | 0.5861 | 0.4497 | CUDA stage 25.62s | NA | NA | local ~0.0000 Linear | local-accepted-not-promoted | [archive](20260827_v018_c15-quantized-weight-activation-gram-local_scoreNA_timeNA/) |
| v019 | 2026-08-27 | C16 gated activation 8×8 coverage 4% | 0.5876 | 0.4497 | CUDA stage 24.78s | NA | NA | local +0.0015 Linear | local-accepted-not-promoted | [archive](20260827_v019_c16-gated-activation-8x8-coverage4-local_scoreNA_timeNA/) |

`*` v002 的 Linear/Attention 数值最初来自远程仓库 `CHANGELOG.md` 的 GPT-2
12 层、2 calib + 2 test 报告，之后已由 GPU-compatible B0 derivative 在本地
复现。该 derivative 与 v002 归档行为等价但 SHA256 不同，因此表中 Local Time
仍为 `NA`；官方总分和耗时仍是用户提供的近似值，尚未完成精确 SHA 绑定。

v003 起允许只有本地结果时立即归档。官方列保持 `NA`，未来结果返回时追加
提交 SHA、分数和时间，不覆盖既有本地配对表或实验结论。

## Local-first workflow

1. Modify only the root `solution.py`, with one primary mechanism change per iteration.
2. Run the real-GPT evaluator:

   ```powershell
   .\.venv\Scripts\python evaluator\real_data_eval.py --solution solution.py --model gpt2
   ```

3. Record the six Linear components, full Attention matrix, tails, evaluator parameters and paired local runtime.
4. Immediately create the next immutable `vNNN` directory, even when only local results exist or the candidate regresses.
5. Use `scoreNA_timeNA` while official evaluation is unavailable; copy the exact evaluated source and verify SHA256 equality.
6. Add `result.md` with parent/candidate IDs, local breakdown, single change, hypothesis, conclusion and next direction.
7. Append one row to this table and mark `local-champion`, `local-accepted` or `local-rejected`.
8. Build the next single-mechanism candidate from the latest local Champion; a rejected branch does not roll the Champion back to an older baseline.
9. If official results later become available, append the submitted SHA, score, runtime and date to the same archive record and update this table.
10. Never overwrite the local evidence when recording later official feedback.

Use `NA`, `score9000plus`, or `time300plus` when a historical value is unavailable, approximate, or timed out. Never replace an unknown official value with a local estimate.
