# L0 Linear ceiling / error decomposition

> evaluator-side diagnostic only; no deployment code or activation state was changed.

- Cache: `D:\工作内容\AI竞赛\artifacts\real_model_suite\cache\qwen2.5-0.5b__seq128__calib2__test4__layersall__schema1.pt`
- Solution: `D:\工作内容\AI竞赛\solution.py`
- Layers: `[0, 5, 11, 17, 23]`; roles: `q, k, v, o, fc_gate, fc_up, proj`
- Oracle rows per sample: `32`; scale candidates: `255`
- Solution LF SHA256: `6b229081121c4a7edd69575c93dc01488be8f8b5e1479007522421e93e1adc57`
- Dashboard LF SHA256: `c5e20e8f0ae144a9e7593a923123ca64c5ba27c6a18f55c2f3b51f4aef4d63ad`
- Elapsed: `160.146s`

## Overall deployment arms

| arm | mean gain |
|---|---:|
| both_player | `0.53188695` |
| weight_perfect | `0.71407146` |
| activation_perfect | `0.81889050` |
| both_perfect | `1.00000000` |

Weight-side headroom: `0.18218452`; activation-side headroom: `0.28700355`; relaxed both-perfect headroom: `0.46811305`.

Diagnostic classification: **activation-dominant**.

## Layer summary

| layer | both player | weight perfect | activation perfect | both perfect | W headroom | A headroom | class |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 0.61670768 | 0.75352817 | 0.86973172 | 1.00000000 | 0.13682050 | 0.25302404 | activation-dominant |
| 5 | 0.48554060 | 0.65570158 | 0.82985441 | 1.00000000 | 0.17016098 | 0.34431381 | activation-dominant |
| 11 | 0.52760337 | 0.71505739 | 0.81097893 | 1.00000000 | 0.18745402 | 0.28337556 | activation-dominant |
| 17 | 0.51050457 | 0.72576016 | 0.78421277 | 1.00000000 | 0.21525560 | 0.27370821 | activation-dominant |
| 23 | 0.51907851 | 0.72031000 | 0.79967465 | 1.00000000 | 0.20123149 | 0.28059614 | activation-dominant |

## Role summary

| role | both player | weight perfect | activation perfect | both perfect | W headroom | A headroom | class |
|---|---:|---:|---:|---:|---:|---:|---|
| q | 0.65573462 | 0.86005113 | 0.79565847 | 1.00000000 | 0.20431651 | 0.13992385 | weight-dominant |
| k | 0.67062548 | 0.90586272 | 0.76796906 | 1.00000000 | 0.23523724 | 0.09734358 | weight-dominant |
| v | 0.59124290 | 0.79297084 | 0.79881652 | 1.00000000 | 0.20172794 | 0.20757362 | transform-coupled |
| o | 0.54638810 | 0.74596088 | 0.80370841 | 1.00000000 | 0.19957278 | 0.25732031 | activation-dominant |
| fc_gate | 0.40618361 | 0.66113440 | 0.74617320 | 1.00000000 | 0.25495079 | 0.33998959 | activation-dominant |
| fc_up | 0.44192540 | 0.49708107 | 0.94544108 | 1.00000000 | 0.05515567 | 0.50351568 | activation-dominant |
| proj | 0.41110851 | 0.53543918 | 0.87446675 | 1.00000000 | 0.12433067 | 0.46335824 | activation-dominant |

## Legal scale oracle summary

The oracle searches all finite E6M2 scale codes while retaining the legal HiF4 hierarchy. It is a sampled operand-local ceiling diagnostic, not a deployment candidate.

| layer | role | weight plain gap | weight Gram gap | activation Gram gap |
|---:|---|---:|---:|---:|
| 0 | q | 0.00022586 | 0.05662518 | 0.07980678 |
| 0 | k | 0.00031945 | 0.05980576 | 0.05119468 |
| 0 | v | 0.00000000 | 0.00313072 | 0.00197814 |
| 0 | o | 0.00000000 | 0.00187704 | 0.00271455 |
| 0 | fc_gate | 0.00666768 | 0.00668396 | 0.00046221 |
| 0 | fc_up | 0.00000000 | 0.00000000 | 0.00007258 |
| 0 | proj | 0.00000000 | 0.00000000 | 0.00003728 |
| 5 | q | 0.00039299 | 0.00192473 | 0.00099261 |
| 5 | k | 0.00153744 | 0.00693326 | 0.00853690 |
| 5 | v | 0.00000000 | 0.00000000 | 0.00109730 |
| 5 | o | 0.00000000 | 0.00000000 | 0.00050692 |
| 5 | fc_gate | 0.00060240 | 0.00181132 | 0.00638192 |
| 5 | fc_up | 0.00000000 | 0.00000000 | 0.00003879 |
| 5 | proj | 0.00000000 | 0.00000000 | 0.00007249 |
| 11 | q | 0.00001885 | 0.00515559 | 0.00371145 |
| 11 | k | 0.00086990 | 0.01298970 | 0.01412779 |
| 11 | v | 0.00000000 | 0.00466674 | 0.00150453 |
| 11 | o | 0.00065018 | 0.00378361 | 0.00550732 |
| 11 | fc_gate | 0.00000000 | 0.00000000 | 0.00005259 |
| 11 | fc_up | 0.00000000 | 0.00000000 | 0.00005787 |
| 11 | proj | 0.00000000 | 0.00002325 | 0.00010941 |
| 17 | q | 0.00066376 | 0.00475000 | 0.00312977 |
| 17 | k | 0.00035957 | 0.00857977 | 0.01194070 |
| 17 | v | 0.00014806 | 0.00382757 | 0.00306356 |
| 17 | o | 0.00106402 | 0.00577478 | 0.00635166 |
| 17 | fc_gate | 0.00092322 | 0.00140568 | 0.00124911 |
| 17 | fc_up | 0.00000000 | 0.00000000 | 0.00004175 |
| 17 | proj | 0.00000000 | 0.00000000 | 0.00002377 |
| 23 | q | 0.00016373 | 0.00382162 | 0.00132828 |
| 23 | k | 0.00000000 | 0.01144014 | 0.00735060 |
| 23 | v | 0.00008061 | 0.01071800 | 0.00796783 |
| 23 | o | 0.00000000 | 0.00005835 | 0.00750048 |
| 23 | fc_gate | 0.00000000 | 0.00000000 | 0.00019491 |
| 23 | fc_up | 0.00000000 | 0.00000000 | 0.00010425 |
| 23 | proj | 0.00000000 | 0.00015333 | 0.00022407 |

## Interpretation boundary

1. `weight_perfect` and `activation_perfect` are evaluator-side one-sided arms; they do not claim that a legal algorithm can reach those values.
2. The 255-code oracle uses calibration tensors and static Gram only. It never writes a state or selects a test-time candidate.
3. A small scale-oracle gap rules out scale search as the main source of a large gain, but does not rule out coordinate transforms or cross-block solvers.
4. A large one-sided arm is headroom evidence, not a guarantee of cross-layer transfer. L1/L2 still require the stratified and full-layer gates in the active plan.

## L5e 可达性判定（追加）

当前 screen 的 `linear_mean=0.5318869457`，到 `0.9` 的绝对差为
`0.3681130543`。用残差比 `r=1-g` 表示，当前 `r=0.4681130543`，要达到 0.9
必须把剩余输出误差压到 `0.1`，即至少减少

\[
1-\frac{0.1}{0.4681130543}=0.7863763912
\]

（约 78.64%）。这不是把分数线性加 `0.368` 就能保证的，因为 score 的分母按
case 的 standard MSE 归一化。

固定当前等价 frame 后，单侧理想臂为：

- weight-perfect：`0.7140714612`，剩余误差 `0.2859285388`；
- activation-perfect：`0.8188904986`，剩余误差 `0.1811095014`；
- both-perfect：`1.0`，但这是 evaluator dense reference，不是合法部署算法。

所以任何只修一侧的方案都不足以到 0.9；必须同时显著改善两侧，且不能使用输出
残差作为在线 state。

### 合法 scale/hierarchy oracle 的量级

对 35 个 layer×role，完整 255-code E6M2 scale oracle 的 baseline→oracle 加权损失
下降为：

| 目标 | 加权下降 |
|---|---:|
| weight plain | `0.04746%` |
| weight 64×64 Gram | `4.56979%` |
| activation 64×64 Gram | `0.11279%` |

单 case 最大 gap 为 weight Gram `5.98058%`、activation Gram `7.98068%`。即便把最大
`7.98068%` 不现实地当作当前全部输出残差都可按比例消除，得到的乐观估计也只有

\[
g_{optimistic}=1-(1-0.5318869457)(1-0.0798068)\approx0.5691,
\]

仍远低于 0.9。该式是量级上界/筛选估计，不把局部 oracle 宣称为官方可部署分数；
它足以排除“继续扩大 offset/scale 搜索即可到 0.9”。

### 跨 block coupling

对当前 v111 frame 的 30 个 896 输入宽度组件，定义

\[
\rho_{off}=\frac{\|G-\operatorname{blockdiag}_{64}(G)\|_F}{\|G\|_F},
\qquad G=X^{\mathsf T}X\;\text{或}\;W^{\mathsf T}W.
\]

weight frame 的平均 `ρ_off=0.76125`，calibration activation frame 的平均
`ρ_off=0.88382`。这说明 64×64 block-diagonal state 丢失了大量跨 block 相关性；
外部 4×4 group solver 只改善局部目标而使完整 `J_64` 变差，正与此统计一致。
但把完整 dense Gram 写入在线 state 会违反现有 state/时间边界，因此下一阶段若继续
优化，必须研究压缩的合法跨 block 表达（低秩/结构化、离线生成、在线 exact gate），
而不是重复 offset 或局部 group hierarchy。

### 结论

在“当前 HiF4 表示 + 64-channel legal hierarchy + 现有 CPU static state 接口 +
不使用输出残差”的约束下，`linear_mean=0.9` 没有可信可达路径；固定 frame 的
scale/hierarchy 余量与单侧上界均低于要求，故记录为 **当前表示/接口不可达（证据性
结论，不是对所有未来合法等价变换的形式不可能证明）**。若要继续追求 0.9，只能开
新的表示级方向：压缩跨 block 结构、合法共享等价变换，或规则允许的新 state 预算；
不再投入已否决的 sampler、offset 扩窗、joint residual、H32/H64 和局部 group-only
solver。
