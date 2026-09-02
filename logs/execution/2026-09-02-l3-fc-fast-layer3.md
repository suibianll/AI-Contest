# L3-D0 fc legal-code oracle — proxy-v2

- parent SHA256: `800ca10ec3414e4fe886b93ca62bd4a350d26bba015287df7e8df2dd871ac23d`
- layers: `[3]`; roles: `['fc_gate', 'fc_up']`
- calibration folds: `[10, 128]`
- device: `cuda`
- teacher wall: `10.349s` (research cost; not candidate API time)

## Class summary

| edit class | cases | mean margin | median margin | positive | negative | conclusion |
|---|---:|---:|---:|---:|---:|---|
| joint | 4 | -0.045015 | -0.039849 | 2 | 2 | mixed |

## Per layer / role / fold

| layer | role | fold length | class | exact parent MSE | exact teacher MSE | joint margin | quadratic+ / total | exact-single median | exact-single +/- |
|---:|---|---:|---|---:|---:|---:|---:|---:|---:|
| 3 | fc_gate | 10 | joint | 1.080944e-02 | 1.079945e-02 | 0.000924 | 14/14 | -0.000028 | 6/8 |
| 3 | fc_gate | 128 | joint | 2.323040e-03 | 2.510326e-03 | -0.080621 | 14/14 | -0.005711 | 0/14 |
| 3 | fc_up | 10 | joint | 5.473500e-03 | 5.450265e-03 | 0.004245 | 12/14 | 0.000000 | 6/6 |
| 3 | fc_up | 128 | joint | 1.248503e-03 | 1.379104e-03 | -0.104606 | 14/14 | -0.008169 | 0/14 |

Block quadratic uses the parent deployed Q(W) Gram/cross metric for local move acceptance; exact full output MSE is recomputed once per class after all block edits. Positive/negative quadratic and exact-single-block counts plus cheap feature summaries are localization diagnostics, not a deployable candidate or a ranking score.

Decision: `margin_exists_but_not_compile_safe`. The joint teacher has positive same-fold margin in most records, but class signs are mixed and layer 3 / fold 128 has a large exact output regression for both fc roles. No student or v155 is created from this oracle; the next experiment must test cross-fold feature/decision stability or switch to L2.


Screening scope: layer 3 only, joint class only, one batched Jacobi pass per component. This is not a deployable candidate and does not replace canonical D0.
