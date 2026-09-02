# v155/v156 official result correction

User-reported official results received on 2026-09-02:

| version | official score | official time | versus v86 score | decision |
|---|---:|---:|---:|---|
| v155 | 16581 | 208.5s | -163 | REJECTED |
| v156 | 16580 | 204.3s | -164 | REJECTED |

Both runs pass the strict runtime requirement but fail the accuracy baseline. v156 is also `-1`
point versus v155 while being `4.2s` faster. These results confirm that the Qwen proxy movements
(`+0.000116536` for v155 and `+0.000107624` for v156) did not predict official improvement.

Both archive directories were renamed with `_rejected`; root `solution.py` remains unchanged.
The next single-variable candidate at that point was v157, branched from exact v86. Its later
official result `16729 / 218.96s` also fell below v86 and closed the ROAB route.
