# v001 — current baseline

- Date: 2026-08-26
- Source SHA256: `4acc1b8a4e751a75a68158efa19b2ee44f9d5a9e8821a05659d7425723f90ed5`
- Change: Improved calibration and dynamic HiF4 quantization relative to v9.
- Hypothesis: More accurate scale search, smoothing, permutation, and output-aware calibration improve official score within the time limit.
- Local evaluator: NA — this official result predates adoption of the current real-GPT archive workflow.
- Local Linear q/k/v/o/fc/proj: NA
- Local Attention: NA
- Local runtime: NA
- Official score: 10250
- Official runtime: 127 seconds
- Official score delta: NA because v000 has only an approximate score.
- Status: `champion`
- Conclusion: This is the active official baseline for all subsequent single-mechanism experiments.
- Next direction: Run the real-GPT evaluator before every new official submission and compare local component changes against this baseline.
