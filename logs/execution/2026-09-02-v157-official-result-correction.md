# v157 official result correction

User-reported official result received on 2026-09-02:

- Version: v157 exact-v86 + ROAB-only
- Source SHA256: `984BF752156187B8892894060A99FE52027E2457F37FC23C11657041B29B86E1`
- Official score: `16729`
- Official time: `218.96s`
- Judge status: pass
- Delta versus v86: `-15` score, `-3.74s`
- Archive decision: `REJECTED`

The exact-v86 control makes this an actionable negative result. Although ROAB produced `+123` in
the fixed reduced-Attention v138-to-v140 comparison, it loses 15 points when added alone to exact
v86. Therefore the earlier gain was context-dependent and cannot be treated as a portable ROAB
main effect. The archive directory is renamed with `_rejected`; root `solution.py` is unchanged.

Next planned direction: one single-pass block-Schur HiF4-GPTQ experiment from exact v86 with v86
Attention frozen. No ROAB parameter, threshold, pair-size or role-gate sweep is authorized by this
result.
