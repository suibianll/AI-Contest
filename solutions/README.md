# HiF4 Solution Archive

Root `solution.py` is the only active submission. Archived source files are immutable.

| Version | Date | Topic | Local Linear | Local Attention | Local Time | Official Score | Official Time | Delta | Status | Directory |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| v000 | 2026-08-25 | v9 baseline | NA | NA | NA | ~9000+ | NA | NA | accepted | [archive](20260825_v000_v9-baseline_score9000plus_timeNA/) |
| v001 | 2026-08-26 | current baseline | NA | NA | NA | 10250 | 127s | NA | champion | [archive](20260826_v001_current-baseline_score10250_time127s/) |

## Manual workflow

1. Modify only the root `solution.py`, with one primary mechanism change per iteration.
2. Run the real-GPT evaluator:

   ```powershell
   .\.venv\Scripts\python evaluator\real_data_eval.py --solution solution.py --model gpt2
   ```

3. Record the six Linear component scores, Attention score, evaluator parameters, and local runtime.
4. Submit the exact same root `solution.py` to the official evaluator.
5. After the official score and runtime return, create the next immutable `vNNN` directory even when the result regresses or times out.
6. Name the directory `YYYYMMDD_vNNN_topic_scoreSCORE_timeTIMEs`, keeping score and time at the end.
7. Copy the submitted source to the archive as `solution.py` and verify that both SHA256 hashes match.
8. Add `result.md` with the local breakdown, official result, single change, hypothesis, conclusion, and next direction.
9. Append one row to this table. Mark the version `champion`, `accepted`, or `rejected`.
10. Choose the next single-mechanism experiment by comparing whether local component changes and the official score move in the same direction.

Use `NA`, `score9000plus`, or `time300plus` when a historical value is unavailable, approximate, or timed out. Never replace an unknown official value with a local estimate.
