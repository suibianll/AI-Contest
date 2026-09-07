# Attention error ledger 2026-09-08

- parent: A23 (8714ac2a..., official 14437/276s); comparison A25 14057/254s
- target: F4 <= 0.1 (final gain >= 0.9)

| grid | mean | share of MSE_STD | note |
|---|---|---|---|
| F4 final output | 0.468035 | = itself | 72/72 zero API; prev run A25 0.465398 |
| F1 Q/K total | 0.001329 | 0.1896 | 72/72 cache recomputation |
| F2 code/scale | 0.001344 | 0.1920 | pure quantization cost |
| F1-F2 transform residual | -0.000016 | -0.0024 | center/GQA/inverse alignment |
| F3a logits (abs) | 2.1496 | logits space | softmax absorbs most |
| F3b probability (abs) | 0.014679 | prob space | |
| F5 residual | 0.466706 | | F4 - F1 (softmax/V/distribution) |

## per-layer share (of mse_standard)

| layer | F1 share | F2 share | F1-F2 share | F4 |
|---|---|---|---|---|
| L0 | 0.1621 | 0.1660 | -0.0039 | 0.4298 |
| L1 | 0.2533 | 0.2548 | -0.0015 | 0.4532 |
| L5 | 0.1512 | 0.1526 | -0.0013 | 0.2851 |
| L8 | 0.2250 | 0.2268 | -0.0018 | 0.3624 |
| L15 | 0.3128 | 0.3169 | -0.0040 | 0.5846 |
| L22 | 0.0334 | 0.0351 | -0.0017 | 0.6932 |
