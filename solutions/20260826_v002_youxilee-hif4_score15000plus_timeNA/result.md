# v002 — youxilee/hif4 v2.0

- Date: 2026-08-26
- Source repository: `git@github.com:youxilee/hif4.git`
- Source commit: `6abbf36e1208ac7afffd2ba3e2e4a8aa9a1f3757`
- Source SHA256: `E126B23A7992E28FBB8E5973521B49AE40A930B76522265A6F36F641EB133A4B`
- Change: archive the remote v2.0 solution, including calibration-gated block-diagonal SmoothQuant and bounded refinement.
- Local evaluator method check: the remote and current evaluator use the same NVFP4 simulator, real GPT-2 hooks, standard HiF4 baseline, and relative-MSE score formula. The current evaluator only adds configurable `--solution`/`--model` loading and keeps one active configuration.
- Reported remote component scores (GPT-2 12 layers, 2 calib + 2 test): q `0.6272`, k `0.6824`, v `0.5930`, o `0.5266`, fc `0.4893`, proj `0.4822`, Linear mean `0.5668`, Attention `0.3786`.
- Local runtime: `NA` (not rerun locally; the changelog reports calibration/dynamic timings for separate sample-size settings).
- Official score: `15000+` (user-provided).
- Official runtime: `NA`.
- Status: champion archive and current active root `solution.py`.
- Next direction: port or compare the remote block-SmoothQuant and refinement mechanisms one at a time, then rerun the local evaluator before official submission.
