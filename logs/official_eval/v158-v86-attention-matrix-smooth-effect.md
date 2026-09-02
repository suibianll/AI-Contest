# v158-v86-attention-matrix-smooth — proxy-v2

本报告是本地 proxy；隐藏官方数据和鲲鹏硬件不可本地复制。

- evaluation scope: `effect-panel` / `paired-mechanism-diagnosis`
- proxy ranking comparable: `False`; official-score equivalent: `False`
- calibration lengths: `[10, 128, 512, 1024, 1024]`
- cases: `56 Linear + 5 Attention` (stratified real-W/A panel by default)
- calibration calls: `168 weight + 24 attention` (shared state)
- input codec: `e4m3-subnormal-ceil-v1` / mode `amax6`
- test splits: `['test', 'validation']`
- source SHA256: `18f9de037a29ad96ee06fb5c73095e9ad36d0d04da2953162181be3aea528277`
- data pack: `{'train': 'e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7', 'validation': '204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c', 'test': '5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91'}`

| 指标 | 值 |
|---|---:|
| Linear mean | 0.480787684 |
| Attention mean | 0.764627976 |
| Overall mean (all captured cases) | 0.504053282 |
| Linear role macro mean | 0.480787684 |
| Attention layer macro mean | 0.764627976 |
| Candidate wall | 307.247s |
| Candidate API total | 293.102s |

## 父版本配对效果

基线：`v086-proxy-v2-effect`；候选：`v158-v86-attention-matrix-smooth`。

| 范围 | cases | mean Δgain | median Δgain | 改善/回归/不变 | median MSE ratio | 结论 |
|---|---:|---:|---:|---:|---:|---|
| Linear overall | 56 | 0.000000 | 0.000000 | 0/0/56 | 1.000000 | no_effect |
| Attention overall | 5 | 0.007195 | 0.000000 | 1/0/4 | 1.000000 | consistent_improvement |

完整 role/family、W/A 来源、最坏 case、Attention 控制臂和 API 时间差分位于 JSON `paired_effect`。

## 误差源分解（evaluator-only）

控制臂只在评测器内重算，不增加六个候选 API 调用，也不改变主分数。gain 为相对标准输出误差的改善；interaction 为正表示超加性互补，负表示收益重叠或递减。

### Linear：W / A / 交互

| 分组 | cases | W-only gain | A-only gain | Both gain | interaction | W operand rel-MSE | A operand rel-MSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| overall | 56 | -236.484437 | -44.052136 | 0.480788 | 281.017361 | 4.957995e-01 | 5.166801e-01 |
| family:fc | 16 | -115.796223 | -34.626805 | 0.421910 | 150.844938 | 4.385676e-01 | 4.449300e-01 |
| family:o | 8 | -269.765499 | -60.945785 | 0.445098 | 331.156382 | 9.199748e-01 | 7.669781e-01 |
| family:proj | 8 | -811.995872 | -125.515202 | 0.185575 | 937.696649 | 1.132164e+00 | 1.073341e+00 |
| family:qkv | 24 | -114.012415 | -17.550117 | 0.630340 | 132.192872 | 1.804408e-01 | 2.955273e-01 |
| role:fc_gate | 8 | -152.019555 | -26.308497 | 0.420463 | 178.748516 | 7.081058e-02 | 1.560625e-01 |
| role:fc_up | 8 | -79.572891 | -42.945113 | 0.423357 | 122.941361 | 8.063246e-01 | 7.337975e-01 |
| role:k | 8 | -124.237100 | -24.376607 | 0.634145 | 149.247852 | 1.786938e-01 | 2.223790e-01 |
| role:o | 8 | -269.765499 | -60.945785 | 0.445098 | 331.156382 | 9.199748e-01 | 7.669781e-01 |
| role:proj | 8 | -811.995872 | -125.515202 | 0.185575 | 937.696649 | 1.132164e+00 | 1.073341e+00 |
| role:q | 8 | -146.265938 | -19.456678 | 0.616569 | 166.339186 | 2.057877e-01 | 3.069006e-01 |
| role:v | 8 | -71.534206 | -8.817067 | 0.640307 | 80.991580 | 1.568410e-01 | 3.573024e-01 |
| shape:hidden_to_hidden | 16 | -208.015719 | -40.201232 | 0.530834 | 248.747784 | 5.628813e-01 | 5.369394e-01 |
| shape:hidden_to_wide | 32 | -106.840938 | -25.611821 | 0.529568 | 132.982327 | 3.031675e-01 | 3.673853e-01 |
| shape:wide_to_hidden | 8 | -811.995872 | -125.515202 | 0.185575 | 937.696649 | 1.132164e+00 | 1.073341e+00 |

Linear overall interpretation: `paired_coordinate_coupling_likely`.
Linear 的完整 layer/role/window 结果位于 JSON 的 `decomposition.linear` 和 `case_scores.linear`；优先查看 role 与 shape 行，再定位对应 layer/case。

### Attention：Q / K / V / softmax

| 分组 | cases | Q-only | K-only | V-only | QK-only | Both | QK interaction | QKV interaction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 5 | -9.530735 | -58.037067 | -0.002313 | 0.758151 | 0.764628 | 68.325953 | 0.008790 |
| layer:0 | 1 | -0.092814 | -0.006203 | 0.008294 | 0.938974 | 0.942927 | 1.037991 | -0.004340 |
| layer:12 | 1 | -7.862716 | -52.715567 | -0.004866 | 0.705221 | 0.709622 | 61.283504 | 0.009267 |
| layer:17 | 1 | -11.341867 | -24.711127 | -0.006043 | 0.759176 | 0.763079 | 36.812170 | 0.009946 |
| layer:23 | 1 | -16.768763 | -195.429871 | 0.011059 | 0.576886 | 0.592705 | 212.775520 | 0.004760 |
| layer:6 | 1 | -11.587515 | -17.322569 | -0.020006 | 0.810497 | 0.814807 | 29.720581 | 0.024316 |
| length:10 | 1 | -0.092814 | -0.006203 | 0.008294 | 0.938974 | 0.942927 | 1.037991 | -0.004340 |
| length:1024 | 2 | -14.055315 | -110.070499 | 0.002508 | 0.668031 | 0.677892 | 124.793845 | 0.007353 |
| length:128 | 1 | -11.587515 | -17.322569 | -0.020006 | 0.810497 | 0.814807 | 29.720581 | 0.024316 |
| length:512 | 1 | -7.862716 | -52.715567 | -0.004866 | 0.705221 | 0.709622 | 61.283504 | 0.009267 |

| Attention 中间量（overall） | standard | player |
|---|---:|---:|
| logit MSE vs reference | 1.340842e+02 | 2.601623e+04 |
| probability MSE vs reference | 6.110683e-03 | 3.186568e-04 |
| probability KL(reference || estimate) | 1.893840e-02 | 6.394628e-04 |

Attention overall interpretation: `paired_qk_coupling_likely`.
Attention 的完整 layer/length/window 结果位于 JSON 的 `decomposition.attention` 和 `case_scores.attention`。

官方成绩只保留为独立历史字段，不参与本地 proxy 评分或时间换算。
