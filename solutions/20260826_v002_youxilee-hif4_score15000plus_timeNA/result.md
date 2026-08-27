# v002 — youxilee/hif4 v2.0

- Date: 2026-08-26
- Source repository: `git@github.com:youxilee/hif4.git`
- Source commit: `6abbf36e1208ac7afffd2ba3e2e4a8aa9a1f3757`
- Source SHA256: `E126B23A7992E28FBB8E5973521B49AE40A930B76522265A6F36F641EB133A4B`
- Change: archive the remote v2.0 solution, including calibration-gated block-diagonal SmoothQuant and bounded refinement.
- Local evaluator method check: the remote and current evaluator use the same NVFP4 simulator, real GPT-2 hooks, standard HiF4 baseline, and relative-MSE score formula. The current evaluator adds configurable solution/model/device/mask loading and evaluator-side timing/telemetry.
- Component scores (GPT-2 12 layers, 2 calib + 2 test): q `0.6272`, k `0.6824`, v `0.5930`, o `0.5266`, fc `0.4893`, proj `0.4822`, Linear mean `0.5668`, causal Attention `0.3786`. These remote values were reproduced locally with the GPU-compatible B0 derivative `solution_b0_tmp.py` (SHA256 `C3EC6101...`), not with this archive's exact SHA256.
- Local runtime: exact archived source `NA`. The behavior-equivalent B0 derivative measured CPU algorithm-stage `52.26s` and process wall `57.91s` on 2026-08-27; these timings are not hash-bound to this archived source and are recorded only as engineering references.
- Official score: `15313` (user-confirmed, 2026-08-27).
- Official runtime: `137s` (user-confirmed, 2026-08-27).
- Official binding: the user explicitly confirmed that this result closes the v002/B0 submission. The uploaded file itself was not re-downloaded, so the archive source SHA256 above remains the reproducible source identity; the GPU-compatible local derivative SHA256 is retained separately and is not presented as the uploaded-file hash.
- Status: official B0 closed; historical champion archive. Root `solution.py` is a later locally validated candidate and has no official score yet.
- Next direction: use `15313 / 137s` only as the official B0 anchor. Continue candidate ranking with paired local evidence until a later candidate receives its own official result.
