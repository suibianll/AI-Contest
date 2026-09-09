## Config flags

```json
{
 "_WEIGHT_QUADRATIC": true,
 "_WEIGHT_QUADRATIC_MAX_FEATURES": 3072,
 "_WIDE_LAYER_MIN_DIM": 2048,
 "_BLOCK_SMOOTH_SIZES": [
  4,
  8
 ],
 "_BLOCK_SMOOTH_SEEDS": [
  0,
  1,
  2,
  3
 ],
 "_LINEAR_SMOOTH_END_TO_END": "hybrid",
 "_WEIGHT_GPTQ": true,
 "_ACTIVATION_GPTQ": true,
 "_ADAPTIVE_OFFSETS": false,
 "_ADAPTIVE_OFFSET_CANDIDATES": [
  [
   -1,
   1,
   2,
   3
  ]
 ],
 "_ADAPTIVE_ACT_GPTQ_REG": false,
 "_ADAPTIVE_ACT_GPTQ_REG_CANDIDATES": [
  0.05,
  0.2
 ],
 "_WEIGHT_RESIDUAL_RANK": 2,
 "_ATTN_OUTPUT_SELECTOR": true,
 "_ATTN_FISHER_IMPORTANCE": false,
 "_ATTN_PAIR_MATRIX_SMOOTH": true,
 "_ATTN_LOGIT_GAIN": true,
 "_A2_TRAIN_STEPS": 32,
 "_ATTN_SCALE_AWARE_CENTER": true,
 "_LINEAR_SMOOTH_BLOCK_JOINT": true,
 "_WEIGHT_SMOOTH_ALPHAS": [
  0.25,
  0.5,
  0.75
 ],
 "_WEIGHT_SMOOTH_ALPHAS_WIDE": [
  0.25,
  0.5,
  0.75
 ],
 "_WEIGHT_E2E_REFINE": false,
 "_ACTIVATION_QUADRATIC": true,
 "_DATA_DRIVEN_RATIO": true,
 "_BLOCK_SWAP_ROUNDS": 0,
 "_PERMUTATION_BASES": true,
 "_V_IMPORTANCE_CANDIDATES": false
}
```

## Linear layer=0 role=proj shape=[2560, 9216] wall=8.810s

### linear_base (7802 body, per-statement buckets)
| component | seconds | % of wall |
|---|---:|---:|
| misc_setup/arg-checks | 0.054 | 0.6% |
| calib_loop (NVFP4 decode + stats + cov + sampling) | 0.055 | 0.6% |
| smooth candidate gen + weight sampling | 0.068 | 0.8% |
| baseline metrics eval | 0.202 | 2.3% |
| smooth x perm candidate search | 0.469 | 5.3% |
| permutation bases search | 0.090 | 1.0% |
| block-swap optimize | 0.000 | 0.0% |
| post-perm smooth re-search | 0.000 | 0.0% |
| block-Hadamard combos search | 0.061 | 0.7% |
| joint smooth/perm/block search (small layers) | 0.000 | 0.0% |
| E2E verify of selected transform | 0.000 | 0.0% |
| final pair transform + 2nd moment | 0.009 | 0.1% |
| quadratic gram transform | 0.000 | 0.0% |
| residual probes (base codec, folds) | 0.030 | 0.3% |
| rank1/rank2 nested defs (no-op) | 0.000 | 0.0% |
| v166 rank-1 residual fit (power iters) | 1.710 | 19.4% |
| L-R2 rank-2 residual fit + fused update | 4.613 | 52.4% |
| weight encode (GPTQ / dense_to_hif4) | 0.137 | 1.6% |
| weight E2E refine | 0.000 | 0.0% |
| importance + state prep | 0.010 | 0.1% |
| v202 sample-energy block order | 0.137 | 1.6% |
| activation ratio capture | 0.014 | 0.2% |
| activation gram | 0.092 | 1.0% |
| activation h_inv + adaptive-reg cholesky loop | 1.055 | 12.0% |
| adaptive-offsets search | 0.000 | 0.0% |
| state build + return | 0.003 | 0.0% |
| (sum of buckets) | 8.807 | 100.0% |
| (traced function total) | 8.807 | 100.0% |

### linear_wrapper (11645)
| component | seconds | % of wall |
|---|---:|---:|
| base calibration call | 8.808 | 100.0% |
| sample-energy order (fallback path) | 0.000 | 0.0% |
| (sum of buckets) | 8.808 | 100.0% |
| (traced function total) | 8.808 | 100.0% |

### nested linear frames (drill-down)
| frame | seconds | % of wall |
|---|---:|---:|
| linear_base._eval_smooth_candidate | 0.686 | 7.8% |
| linear_base._rank1_fix_sign | 0.007 | 0.1% |
| linear_base._rank1_top2 | 1.693 | 19.2% |
| linear_base._rank1_top2._mv | 1.122 | 12.7% |
| linear_base._rank2_residual_complement | 4.606 | 52.3% |
| linear_base._rank2_residual_complement._rank2_top2 | 4.585 | 52.0% |
| linear_base._rank2_residual_complement._rank2_top2._mv | 1.781 | 20.2% |
| linear_base._rank2_residual_complement._rank2_top2._proj | 1.799 | 20.4% |

### wrapped function totals (cross-callsite, overlapping)
| function | seconds | calls |
|---|---:|---:|
| _linear_smooth_hybrid_metrics | 0.684 | 18 |
| _dense_to_hif4 | 0.546 | 117 |
| _linear_candidate_metrics | 0.366 | 18 |
| _linear_output_candidate_metrics | 0.317 | 18 |
| _v202_sample_energy_block_order_from_calibration | 0.137 | 1 |
| _linear_pair_transform | 0.126 | 136 |
| _dequantize_hif4 | 0.068 | 117 |
| _linear_output_candidate_metrics_combos | 0.059 | 1 |
| _dequantize_nvfp4_float32 | 0.050 | 3 |
| _loss_capture_ratio | 0.005 | 1 |

## Linear layer=0 role=o shape=[2560, 4096] wall=5.092s

### linear_base (7802 body, per-statement buckets)
| component | seconds | % of wall |
|---|---:|---:|
| misc_setup/arg-checks | 0.002 | 0.0% |
| calib_loop (NVFP4 decode + stats + cov + sampling) | 0.008 | 0.2% |
| smooth candidate gen + weight sampling | 0.017 | 0.3% |
| baseline metrics eval | 0.024 | 0.5% |
| smooth x perm candidate search | 0.469 | 9.2% |
| permutation bases search | 0.135 | 2.6% |
| block-swap optimize | 0.000 | 0.0% |
| post-perm smooth re-search | 0.000 | 0.0% |
| block-Hadamard combos search | 0.074 | 1.5% |
| joint smooth/perm/block search (small layers) | 0.000 | 0.0% |
| E2E verify of selected transform | 0.000 | 0.0% |
| final pair transform + 2nd moment | 0.004 | 0.1% |
| quadratic gram transform | 0.000 | 0.0% |
| residual probes (base codec, folds) | 0.040 | 0.8% |
| rank1/rank2 nested defs (no-op) | 0.000 | 0.0% |
| v166 rank-1 residual fit (power iters) | 1.613 | 31.7% |
| L-R2 rank-2 residual fit + fused update | 2.448 | 48.1% |
| weight encode (GPTQ / dense_to_hif4) | 0.052 | 1.0% |
| weight E2E refine | 0.000 | 0.0% |
| importance + state prep | 0.003 | 0.1% |
| v202 sample-energy block order | 0.002 | 0.0% |
| activation ratio capture | 0.005 | 0.1% |
| activation gram | 0.011 | 0.2% |
| activation h_inv + adaptive-reg cholesky loop | 0.182 | 3.6% |
| adaptive-offsets search | 0.000 | 0.0% |
| state build + return | 0.002 | 0.0% |
| (sum of buckets) | 5.091 | 100.0% |
| (traced function total) | 5.091 | 100.0% |

### linear_wrapper (11645)
| component | seconds | % of wall |
|---|---:|---:|
| base calibration call | 5.092 | 100.0% |
| sample-energy order (fallback path) | 0.000 | 0.0% |
| (sum of buckets) | 5.092 | 100.0% |
| (traced function total) | 5.092 | 100.0% |

### nested linear frames (drill-down)
| frame | seconds | % of wall |
|---|---:|---:|
| linear_base._eval_smooth_candidate | 0.586 | 11.5% |
| linear_base._rank1_fix_sign | 0.006 | 0.1% |
| linear_base._rank1_top2 | 1.604 | 31.5% |
| linear_base._rank1_top2._mv | 1.030 | 20.2% |
| linear_base._rank2_residual_complement | 2.434 | 47.8% |
| linear_base._rank2_residual_complement._rank2_top2 | 2.422 | 47.6% |
| linear_base._rank2_residual_complement._rank2_top2._mv | 0.935 | 18.4% |
| linear_base._rank2_residual_complement._rank2_top2._proj | 0.950 | 18.7% |

### wrapped function totals (cross-callsite, overlapping)
| function | seconds | calls |
|---|---:|---:|
| _linear_smooth_hybrid_metrics | 0.585 | 18 |
| _dense_to_hif4 | 0.462 | 117 |
| _linear_candidate_metrics | 0.314 | 18 |
| _linear_output_candidate_metrics | 0.270 | 18 |
| _linear_pair_transform | 0.091 | 136 |
| _linear_output_candidate_metrics_combos | 0.071 | 1 |
| _dequantize_hif4 | 0.069 | 117 |
| _dequantize_nvfp4_float32 | 0.002 | 3 |
| _v202_sample_energy_block_order_from_calibration | 0.002 | 1 |
| _loss_capture_ratio | 0.002 | 1 |

## Linear layer=0 role=q shape=[4096, 2560] wall=18.678s

### linear_base (7802 body, per-statement buckets)
| component | seconds | % of wall |
|---|---:|---:|
| misc_setup/arg-checks | 0.003 | 0.0% |
| calib_loop (NVFP4 decode + stats + cov + sampling) | 0.004 | 0.0% |
| smooth candidate gen + weight sampling | 0.005 | 0.0% |
| baseline metrics eval | 0.015 | 0.1% |
| smooth x perm candidate search | 0.219 | 1.2% |
| permutation bases search | 0.073 | 0.4% |
| block-swap optimize | 0.000 | 0.0% |
| post-perm smooth re-search | 0.000 | 0.0% |
| block-Hadamard combos search | 0.033 | 0.2% |
| joint smooth/perm/block search (small layers) | 0.000 | 0.0% |
| E2E verify of selected transform | 0.000 | 0.0% |
| final pair transform + 2nd moment | 0.004 | 0.0% |
| quadratic gram transform | 0.006 | 0.0% |
| residual probes (base codec, folds) | 0.022 | 0.1% |
| rank1/rank2 nested defs (no-op) | 0.000 | 0.0% |
| v166 rank-1 residual fit (power iters) | 1.873 | 10.0% |
| L-R2 rank-2 residual fit + fused update | 5.394 | 28.9% |
| weight encode (GPTQ / dense_to_hif4) | 10.886 | 58.3% |
| weight E2E refine | 0.000 | 0.0% |
| importance + state prep | 0.006 | 0.0% |
| v202 sample-energy block order | 0.054 | 0.3% |
| activation ratio capture | 0.013 | 0.1% |
| activation gram | 0.012 | 0.1% |
| activation h_inv + adaptive-reg cholesky loop | 0.053 | 0.3% |
| adaptive-offsets search | 0.000 | 0.0% |
| state build + return | 0.002 | 0.0% |
| (sum of buckets) | 18.677 | 100.0% |
| (traced function total) | 18.677 | 100.0% |

### linear_wrapper (11645)
| component | seconds | % of wall |
|---|---:|---:|
| base calibration call | 18.677 | 100.0% |
| sample-energy order (fallback path) | 0.000 | 0.0% |
| (sum of buckets) | 18.677 | 100.0% |
| (traced function total) | 18.677 | 100.0% |

### nested linear frames (drill-down)
| frame | seconds | % of wall |
|---|---:|---:|
| linear_base._eval_smooth_candidate | 0.288 | 1.5% |
| linear_base._rank1_fix_sign | 0.009 | 0.1% |
| linear_base._rank1_top2 | 1.863 | 10.0% |
| linear_base._rank1_top2._mv | 1.180 | 6.3% |
| linear_base._rank2_residual_complement | 5.380 | 28.8% |
| linear_base._rank2_residual_complement._rank2_top2 | 5.347 | 28.6% |
| linear_base._rank2_residual_complement._rank2_top2._mv | 1.889 | 10.1% |
| linear_base._rank2_residual_complement._rank2_top2._proj | 2.225 | 11.9% |

### wrapped function totals (cross-callsite, overlapping)
| function | seconds | calls |
|---|---:|---:|
| _dense_to_hif4 | 10.899 | 156 |
| _gptq_quantize_weight | 10.886 | 1 |
| _linear_smooth_hybrid_metrics | 0.286 | 18 |
| _linear_candidate_metrics | 0.154 | 18 |
| _linear_output_candidate_metrics | 0.131 | 18 |
| _v202_sample_energy_block_order_from_calibration | 0.053 | 1 |
| _dequantize_hif4 | 0.051 | 156 |
| _linear_pair_transform | 0.038 | 136 |
| _linear_output_candidate_metrics_combos | 0.032 | 1 |
| _transformed_covariance | 0.005 | 1 |
| _loss_capture_ratio | 0.002 | 1 |
| _dequantize_nvfp4_float32 | 0.001 | 3 |

## Linear layer=0 role=k shape=[1024, 2560] wall=9.157s

### linear_base (7802 body, per-statement buckets)
| component | seconds | % of wall |
|---|---:|---:|
| misc_setup/arg-checks | 0.002 | 0.0% |
| calib_loop (NVFP4 decode + stats + cov + sampling) | 0.010 | 0.1% |
| smooth candidate gen + weight sampling | 0.012 | 0.1% |
| baseline metrics eval | 0.038 | 0.4% |
| smooth x perm candidate search | 0.478 | 5.2% |
| permutation bases search | 0.162 | 1.8% |
| block-swap optimize | 0.000 | 0.0% |
| post-perm smooth re-search | 0.000 | 0.0% |
| block-Hadamard combos search | 0.078 | 0.9% |
| joint smooth/perm/block search (small layers) | 0.000 | 0.0% |
| E2E verify of selected transform | 0.000 | 0.0% |
| final pair transform + 2nd moment | 0.002 | 0.0% |
| quadratic gram transform | 0.010 | 0.1% |
| residual probes (base codec, folds) | 0.033 | 0.4% |
| rank1/rank2 nested defs (no-op) | 0.000 | 0.0% |
| v166 rank-1 residual fit (power iters) | 1.388 | 15.2% |
| L-R2 rank-2 residual fit + fused update | 3.052 | 33.3% |
| weight encode (GPTQ / dense_to_hif4) | 3.794 | 41.4% |
| weight E2E refine | 0.000 | 0.0% |
| importance + state prep | 0.004 | 0.0% |
| v202 sample-energy block order | 0.005 | 0.1% |
| activation ratio capture | 0.011 | 0.1% |
| activation gram | 0.005 | 0.1% |
| activation h_inv + adaptive-reg cholesky loop | 0.070 | 0.8% |
| adaptive-offsets search | 0.000 | 0.0% |
| state build + return | 0.002 | 0.0% |
| (sum of buckets) | 9.156 | 100.0% |
| (traced function total) | 9.156 | 100.0% |

### linear_wrapper (11645)
| component | seconds | % of wall |
|---|---:|---:|
| base calibration call | 9.156 | 100.0% |
| sample-energy order (fallback path) | 0.000 | 0.0% |
| (sum of buckets) | 9.156 | 100.0% |
| (traced function total) | 9.156 | 100.0% |

### nested linear frames (drill-down)
| frame | seconds | % of wall |
|---|---:|---:|
| linear_base._eval_smooth_candidate | 0.638 | 7.0% |
| linear_base._rank1_fix_sign | 0.007 | 0.1% |
| linear_base._rank1_top2 | 1.380 | 15.1% |
| linear_base._rank1_top2._mv | 0.848 | 9.3% |
| linear_base._rank2_residual_complement | 3.039 | 33.2% |
| linear_base._rank2_residual_complement._rank2_top2 | 3.008 | 32.9% |
| linear_base._rank2_residual_complement._rank2_top2._mv | 1.029 | 11.2% |
| linear_base._rank2_residual_complement._rank2_top2._proj | 1.255 | 13.7% |

### wrapped function totals (cross-callsite, overlapping)
| function | seconds | calls |
|---|---:|---:|
| _dense_to_hif4 | 4.063 | 156 |
| _gptq_quantize_weight | 3.794 | 1 |
| _linear_smooth_hybrid_metrics | 0.637 | 18 |
| _linear_candidate_metrics | 0.357 | 18 |
| _linear_output_candidate_metrics | 0.278 | 18 |
| _linear_pair_transform | 0.096 | 136 |
| _dequantize_hif4 | 0.088 | 156 |
| _linear_output_candidate_metrics_combos | 0.077 | 1 |
| _transformed_covariance | 0.008 | 1 |
| _v202_sample_energy_block_order_from_calibration | 0.005 | 1 |
| _loss_capture_ratio | 0.002 | 1 |
| _dequantize_nvfp4_float32 | 0.001 | 3 |

## Linear layer=0 role=v shape=[1024, 2560] wall=11.837s

### linear_base (7802 body, per-statement buckets)
| component | seconds | % of wall |
|---|---:|---:|
| misc_setup/arg-checks | 0.002 | 0.0% |
| calib_loop (NVFP4 decode + stats + cov + sampling) | 0.011 | 0.1% |
| smooth candidate gen + weight sampling | 0.010 | 0.1% |
| baseline metrics eval | 0.037 | 0.3% |
| smooth x perm candidate search | 0.498 | 4.2% |
| permutation bases search | 0.147 | 1.2% |
| block-swap optimize | 0.000 | 0.0% |
| post-perm smooth re-search | 0.000 | 0.0% |
| block-Hadamard combos search | 0.077 | 0.6% |
| joint smooth/perm/block search (small layers) | 0.000 | 0.0% |
| E2E verify of selected transform | 0.000 | 0.0% |
| final pair transform + 2nd moment | 0.002 | 0.0% |
| quadratic gram transform | 0.010 | 0.1% |
| residual probes (base codec, folds) | 0.034 | 0.3% |
| rank1/rank2 nested defs (no-op) | 0.000 | 0.0% |
| v166 rank-1 residual fit (power iters) | 3.221 | 27.2% |
| L-R2 rank-2 residual fit + fused update | 5.539 | 46.8% |
| weight encode (GPTQ / dense_to_hif4) | 2.171 | 18.3% |
| weight E2E refine | 0.000 | 0.0% |
| importance + state prep | 0.003 | 0.0% |
| v202 sample-energy block order | 0.003 | 0.0% |
| activation ratio capture | 0.006 | 0.0% |
| activation gram | 0.003 | 0.0% |
| activation h_inv + adaptive-reg cholesky loop | 0.060 | 0.5% |
| adaptive-offsets search | 0.000 | 0.0% |
| state build + return | 0.002 | 0.0% |
| (sum of buckets) | 11.836 | 100.0% |
| (traced function total) | 11.836 | 100.0% |

### linear_wrapper (11645)
| component | seconds | % of wall |
|---|---:|---:|
| base calibration call | 11.837 | 100.0% |
| sample-energy order (fallback path) | 0.000 | 0.0% |
| (sum of buckets) | 11.837 | 100.0% |
| (traced function total) | 11.837 | 100.0% |

### nested linear frames (drill-down)
| frame | seconds | % of wall |
|---|---:|---:|
| linear_base._eval_smooth_candidate | 0.641 | 5.4% |
| linear_base._rank1_fix_sign | 0.010 | 0.1% |
| linear_base._rank1_top2 | 3.203 | 27.1% |
| linear_base._rank1_top2._mv | 1.913 | 16.2% |
| linear_base._rank2_residual_complement | 5.531 | 46.7% |
| linear_base._rank2_residual_complement._rank2_top2 | 5.518 | 46.6% |
| linear_base._rank2_residual_complement._rank2_top2._mv | 1.845 | 15.6% |
| linear_base._rank2_residual_complement._rank2_top2._proj | 2.402 | 20.3% |

### wrapped function totals (cross-callsite, overlapping)
| function | seconds | calls |
|---|---:|---:|
| _dense_to_hif4 | 2.532 | 156 |
| _gptq_quantize_weight | 2.170 | 1 |
| _linear_smooth_hybrid_metrics | 0.639 | 18 |
| _linear_candidate_metrics | 0.341 | 18 |
| _linear_output_candidate_metrics | 0.297 | 18 |
| _linear_pair_transform | 0.099 | 136 |
| _linear_output_candidate_metrics_combos | 0.076 | 1 |
| _dequantize_hif4 | 0.075 | 156 |
| _transformed_covariance | 0.008 | 1 |
| _dequantize_nvfp4_float32 | 0.003 | 3 |
| _v202_sample_energy_block_order_from_calibration | 0.003 | 1 |
| _loss_capture_ratio | 0.001 | 1 |

## Attention layer=0 wall=5.910s

### attn_wrapper (11529, R3)
| component | seconds | % of wall |
|---|---:|---:|
| v189 stack call | 3.541 | 59.9% |
| window prep (NVFP4 decode) | 0.004 | 0.1% |
| R3 _a2_train_rotation (32 steps) | 2.175 | 36.8% |
| R3 gate losses + state update | 0.190 | 3.2% |
| fallback handler | 0.000 | 0.0% |
| (sum of buckets) | 5.910 | 100.0% |
| (traced function total) | 5.910 | 100.0% |

### attn_v189 (9250 body)
| component | seconds | % of wall |
|---|---:|---:|
| arg checks | 0.000 | 0.0% |
| SAC K-center solve (C41) | 0.017 | 0.3% |
| stats loop (decode + sampling + moments) | 0.054 | 0.9% |
| A1 identity reference outputs | 0.036 | 0.6% |
| V importance (+A3 candidates) | 0.000 | 0.0% |
| moment finalize + identity perms | 0.001 | 0.0% |
| dual-track selection (whole Q/K sweep, x2 tracks) | 0.994 | 16.8% |
| _build_v_state def+call | 0.007 | 0.1% |
| _build_qk_states def | 0.000 | 0.0% |
| build winner Q/K states | 0.029 | 0.5% |
| A1 final gate (deployed MSE) | 0.263 | 4.5% |
| Fisher importance (C76.2) | 0.000 | 0.0% |
| A2 fixed H64 rotation | 0.000 | 0.0% |
| C76.4 variable H16/H32 rotations | 1.742 | 29.5% |
| A3 V importance candidates | 0.000 | 0.0% |
| v158 pair-matrix smooth | 0.150 | 2.5% |
| logit gain fit | 0.244 | 4.1% |
| return | 0.000 | 0.0% |
| (sum of buckets) | 3.540 | 59.9% |
| (traced function total) | 3.540 | 59.9% |

### nested attention frames (drill-down)
| frame | seconds | % of wall |
|---|---:|---:|
| attn_v189._build_qk_states | 0.394 | 6.7% |
| attn_v189._build_qk_states.k_transform | 0.078 | 1.3% |
| attn_v189._build_qk_states.q_transform | 0.077 | 1.3% |
| attn_v189._build_v_state | 0.007 | 0.1% |
| attn_v189._run_selection | 0.993 | 16.8% |
| attn_v189._run_selection._block_signs | 0.001 | 0.0% |
| attn_v189._run_selection._centered_k | 0.003 | 0.0% |

### wrapped function totals (cross-callsite, overlapping)
| function | seconds | calls |
|---|---:|---:|
| _dense_to_hif4 | 2.885 | 887 |
| _a2_train_rotation | 2.175 | 1 |
| _attention_deployed_mse | 1.679 | 16 |
| _attention_candidate_metrics | 0.969 | 41 |
| _fit_attention_logit_gain | 0.243 | 1 |
| _a2_true_path_gate_loss | 0.190 | 2 |
| _attention_forward | 0.137 | 368 |
| _dequantize_hif4 | 0.101 | 887 |
| _dequantize_nvfp4_float32 | 0.042 | 254 |
| _fit_attention_pair_matrix_smooth | 0.040 | 1 |
| _solve_k_center_scale_aware | 0.014 | 1 |
| _loss_capture_ratio | 0.011 | 29 |
| _attention_rotation_signs | 0.004 | 27 |

## Attention layer=5 wall=5.729s

### attn_wrapper (11529, R3)
| component | seconds | % of wall |
|---|---:|---:|
| v189 stack call | 3.407 | 59.5% |
| window prep (NVFP4 decode) | 0.003 | 0.1% |
| R3 _a2_train_rotation (32 steps) | 2.147 | 37.5% |
| R3 gate losses + state update | 0.172 | 3.0% |
| fallback handler | 0.000 | 0.0% |
| (sum of buckets) | 5.729 | 100.0% |
| (traced function total) | 5.729 | 100.0% |

### attn_v189 (9250 body)
| component | seconds | % of wall |
|---|---:|---:|
| arg checks | 0.000 | 0.0% |
| SAC K-center solve (C41) | 0.017 | 0.3% |
| stats loop (decode + sampling + moments) | 0.027 | 0.5% |
| A1 identity reference outputs | 0.029 | 0.5% |
| V importance (+A3 candidates) | 0.000 | 0.0% |
| moment finalize + identity perms | 0.001 | 0.0% |
| dual-track selection (whole Q/K sweep, x2 tracks) | 1.015 | 17.7% |
| _build_v_state def+call | 0.006 | 0.1% |
| _build_qk_states def | 0.000 | 0.0% |
| build winner Q/K states | 0.020 | 0.4% |
| A1 final gate (deployed MSE) | 0.244 | 4.3% |
| Fisher importance (C76.2) | 0.000 | 0.0% |
| A2 fixed H64 rotation | 0.000 | 0.0% |
| C76.4 variable H16/H32 rotations | 1.694 | 29.6% |
| A3 V importance candidates | 0.000 | 0.0% |
| v158 pair-matrix smooth | 0.126 | 2.2% |
| logit gain fit | 0.227 | 4.0% |
| return | 0.000 | 0.0% |
| (sum of buckets) | 3.407 | 59.5% |
| (traced function total) | 3.407 | 59.5% |

### nested attention frames (drill-down)
| frame | seconds | % of wall |
|---|---:|---:|
| attn_v189._build_qk_states | 0.379 | 6.6% |
| attn_v189._build_qk_states.k_transform | 0.079 | 1.4% |
| attn_v189._build_qk_states.q_transform | 0.075 | 1.3% |
| attn_v189._build_v_state | 0.006 | 0.1% |
| attn_v189._run_selection | 1.015 | 17.7% |
| attn_v189._run_selection._block_signs | 0.000 | 0.0% |
| attn_v189._run_selection._centered_k | 0.005 | 0.1% |

### wrapped function totals (cross-callsite, overlapping)
| function | seconds | calls |
|---|---:|---:|
| _dense_to_hif4 | 2.831 | 887 |
| _a2_train_rotation | 2.147 | 1 |
| _attention_deployed_mse | 1.635 | 16 |
| _attention_candidate_metrics | 0.989 | 41 |
| _fit_attention_logit_gain | 0.225 | 1 |
| _a2_true_path_gate_loss | 0.171 | 2 |
| _attention_forward | 0.137 | 368 |
| _dequantize_hif4 | 0.103 | 887 |
| _dequantize_nvfp4_float32 | 0.040 | 254 |
| _solve_k_center_scale_aware | 0.014 | 1 |
| _loss_capture_ratio | 0.012 | 29 |
| _fit_attention_pair_matrix_smooth | 0.008 | 1 |
| _attention_rotation_signs | 0.003 | 27 |

## Attention layer=15 wall=5.549s

### attn_wrapper (11529, R3)
| component | seconds | % of wall |
|---|---:|---:|
| v189 stack call | 3.251 | 58.6% |
| window prep (NVFP4 decode) | 0.003 | 0.1% |
| R3 _a2_train_rotation (32 steps) | 2.127 | 38.3% |
| R3 gate losses + state update | 0.168 | 3.0% |
| fallback handler | 0.000 | 0.0% |
| (sum of buckets) | 5.549 | 100.0% |
| (traced function total) | 5.549 | 100.0% |

### attn_v189 (9250 body)
| component | seconds | % of wall |
|---|---:|---:|
| arg checks | 0.000 | 0.0% |
| SAC K-center solve (C41) | 0.013 | 0.2% |
| stats loop (decode + sampling + moments) | 0.028 | 0.5% |
| A1 identity reference outputs | 0.033 | 0.6% |
| V importance (+A3 candidates) | 0.000 | 0.0% |
| moment finalize + identity perms | 0.001 | 0.0% |
| dual-track selection (whole Q/K sweep, x2 tracks) | 1.039 | 18.7% |
| _build_v_state def+call | 0.008 | 0.1% |
| _build_qk_states def | 0.000 | 0.0% |
| build winner Q/K states | 0.025 | 0.4% |
| A1 final gate (deployed MSE) | 0.268 | 4.8% |
| Fisher importance (C76.2) | 0.000 | 0.0% |
| A2 fixed H64 rotation | 0.000 | 0.0% |
| C76.4 variable H16/H32 rotations | 1.539 | 27.7% |
| A3 V importance candidates | 0.000 | 0.0% |
| v158 pair-matrix smooth | 0.103 | 1.9% |
| logit gain fit | 0.193 | 3.5% |
| return | 0.000 | 0.0% |
| (sum of buckets) | 3.250 | 58.6% |
| (traced function total) | 3.250 | 58.6% |

### nested attention frames (drill-down)
| frame | seconds | % of wall |
|---|---:|---:|
| attn_v189._build_qk_states | 0.300 | 5.4% |
| attn_v189._build_qk_states.k_transform | 0.043 | 0.8% |
| attn_v189._build_qk_states.q_transform | 0.043 | 0.8% |
| attn_v189._build_v_state | 0.008 | 0.1% |
| attn_v189._run_selection | 1.039 | 18.7% |
| attn_v189._run_selection._block_signs | 0.000 | 0.0% |
| attn_v189._run_selection._centered_k | 0.003 | 0.1% |

### wrapped function totals (cross-callsite, overlapping)
| function | seconds | calls |
|---|---:|---:|
| _dense_to_hif4 | 2.806 | 897 |
| _a2_train_rotation | 2.127 | 1 |
| _attention_deployed_mse | 1.565 | 16 |
| _attention_candidate_metrics | 1.016 | 42 |
| _fit_attention_logit_gain | 0.192 | 1 |
| _a2_true_path_gate_loss | 0.168 | 2 |
| _attention_forward | 0.142 | 378 |
| _dequantize_hif4 | 0.105 | 897 |
| _dequantize_nvfp4_float32 | 0.037 | 254 |
| _loss_capture_ratio | 0.012 | 29 |
| _solve_k_center_scale_aware | 0.011 | 1 |
| _fit_attention_pair_matrix_smooth | 0.007 | 1 |
| _attention_rotation_signs | 0.002 | 15 |
