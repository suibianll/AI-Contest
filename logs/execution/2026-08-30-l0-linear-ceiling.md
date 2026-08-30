# L0 Linear ceiling / error decomposition

> evaluator-side diagnostic only; no deployment code or activation state was changed.

- Cache: `D:\工作内容\AI竞赛\artifacts\real_model_suite\cache\qwen2.5-0.5b__seq128__calib2__test4__layersall__schema1.pt`
- Solution: `D:\工作内容\AI竞赛\solution.py`
- Layers: `[0, 5, 11, 17, 23]`; roles: `q, k, v, o, fc_gate, fc_up, proj`
- Oracle rows per sample: `32`; scale candidates: `255`
- Solution LF SHA256: `617482cee04ff9514a8d41226b651336e4b8b86692673308e835de1091693eba`
- Dashboard LF SHA256: `c5e20e8f0ae144a9e7593a923123ca64c5ba27c6a18f55c2f3b51f4aef4d63ad`
- Elapsed: `103.002s`

## Overall deployment arms

| arm | mean gain |
|---|---:|
| both_player | `0.52301943` |
| weight_perfect | `0.70417026` |
| activation_perfect | `0.82035698` |
| both_perfect | `1.00000000` |

Weight-side headroom: `0.18115083`; activation-side headroom: `0.29733755`; relaxed both-perfect headroom: `0.47698057`.

Diagnostic classification: **activation-dominant**.

## Layer summary

| layer | both player | weight perfect | activation perfect | both perfect | W headroom | A headroom | class |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 0.60107612 | 0.72877052 | 0.87917994 | 1.00000000 | 0.12769439 | 0.27810381 | activation-dominant |
| 5 | 0.48433279 | 0.65235677 | 0.83182894 | 1.00000000 | 0.16802398 | 0.34749615 | activation-dominant |
| 11 | 0.52291341 | 0.70979337 | 0.81224927 | 1.00000000 | 0.18687997 | 0.28933586 | activation-dominant |
| 17 | 0.50591228 | 0.72212043 | 0.78383676 | 1.00000000 | 0.21620815 | 0.27792448 | activation-dominant |
| 23 | 0.50086255 | 0.70781020 | 0.79468999 | 1.00000000 | 0.20694765 | 0.29382744 | activation-dominant |

## Role summary

| role | both player | weight perfect | activation perfect | both perfect | W headroom | A headroom | class |
|---|---:|---:|---:|---:|---:|---:|---|
| q | 0.65295906 | 0.85721337 | 0.79565847 | 1.00000000 | 0.20425432 | 0.14269942 | weight-dominant |
| k | 0.66434283 | 0.90017072 | 0.76796906 | 1.00000000 | 0.23582789 | 0.10362623 | weight-dominant |
| v | 0.58427030 | 0.78581581 | 0.79881652 | 1.00000000 | 0.20154551 | 0.21454622 | transform-coupled |
| o | 0.52487909 | 0.73215238 | 0.79741085 | 1.00000000 | 0.20727329 | 0.27253177 | activation-dominant |
| fc_gate | 0.38180399 | 0.62352360 | 0.75985387 | 1.00000000 | 0.24171961 | 0.37804989 | activation-dominant |
| fc_up | 0.43581101 | 0.49233485 | 0.94470375 | 1.00000000 | 0.05652384 | 0.50889274 | activation-dominant |
| proj | 0.41706973 | 0.53798107 | 0.87808632 | 1.00000000 | 0.12091134 | 0.46101659 | activation-dominant |

## Legal scale oracle summary

The oracle searches all finite E6M2 scale codes while retaining the legal HiF4 hierarchy. It is a sampled operand-local ceiling diagnostic, not a deployment candidate.

| layer | role | weight plain gap | weight Gram gap | activation Gram gap |
|---:|---|---:|---:|---:|
| 0 | q | 0.00022586 | 0.05662518 | 0.07980678 |
| 0 | k | 0.00031945 | 0.05980576 | 0.05119468 |
| 0 | v | 0.00000000 | 0.00313072 | 0.00197814 |
| 0 | o | 0.00000000 | 0.00187704 | 0.00271455 |
| 0 | fc_gate | 0.00000000 | 0.00000000 | 0.00024461 |
| 0 | fc_up | 0.00000000 | 0.00000000 | 0.00007258 |
| 0 | proj | 0.00000000 | 0.00000000 | 0.00003728 |
| 5 | q | 0.00039299 | 0.00192473 | 0.00099261 |
| 5 | k | 0.00153744 | 0.00693326 | 0.00853690 |
| 5 | v | 0.00000000 | 0.00000000 | 0.00109730 |
| 5 | o | 0.00000000 | 0.00000000 | 0.00125966 |
| 5 | fc_gate | 0.00060240 | 0.00181132 | 0.00638192 |
| 5 | fc_up | 0.00000000 | 0.00000000 | 0.00003879 |
| 5 | proj | 0.00000000 | 0.00000000 | 0.00008262 |
| 11 | q | 0.00001885 | 0.00515559 | 0.00371145 |
| 11 | k | 0.00086990 | 0.01298970 | 0.01412779 |
| 11 | v | 0.00000000 | 0.00466674 | 0.00150453 |
| 11 | o | 0.00065018 | 0.00378361 | 0.00550732 |
| 11 | fc_gate | 0.00000000 | 0.00000000 | 0.00009102 |
| 11 | fc_up | 0.00000000 | 0.00000000 | 0.00005787 |
| 11 | proj | 0.00000000 | 0.00005667 | 0.00016223 |
| 17 | q | 0.00066376 | 0.00475000 | 0.00312977 |
| 17 | k | 0.00035957 | 0.00857977 | 0.01194070 |
| 17 | v | 0.00014806 | 0.00382757 | 0.00306356 |
| 17 | o | 0.00106402 | 0.00577478 | 0.00635166 |
| 17 | fc_gate | 0.00092322 | 0.00140568 | 0.00124911 |
| 17 | fc_up | 0.00000000 | 0.00000000 | 0.00004945 |
| 17 | proj | 0.00000000 | 0.00000000 | 0.00002377 |
| 23 | q | 0.00016373 | 0.00382162 | 0.00132828 |
| 23 | k | 0.00000000 | 0.01144014 | 0.00735060 |
| 23 | v | 0.00008061 | 0.01071800 | 0.00796783 |
| 23 | o | 0.00000000 | 0.00304505 | 0.00187035 |
| 23 | fc_gate | 0.00000000 | 0.00000000 | 0.00012145 |
| 23 | fc_up | 0.00000000 | 0.00000000 | 0.00009004 |
| 23 | proj | 0.00000000 | 0.00015333 | 0.00022407 |

## Interpretation boundary

1. `weight_perfect` and `activation_perfect` are evaluator-side one-sided arms; they do not claim that a legal algorithm can reach those values.
2. The 255-code oracle uses calibration tensors and static Gram only. It never writes a state or selects a test-time candidate.
3. A small scale-oracle gap rules out scale search as the main source of a large gain, but does not rule out coordinate transforms or cross-block solvers.
4. A large one-sided arm is headroom evidence, not a guarantee of cross-layer transfer. L1/L2 still require the stratified and full-layer gates in the active plan.
