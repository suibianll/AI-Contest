# L2 expansive-FFN CAT balance 执行记录

日期：2026-08-30  
计划：[`2026-08-30-hif4-active-optimization-plan.md`](../../docs/superpowers/plans/2026-08-30-hif4-active-optimization-plan.md)  
候选：`v106-l2-cat`  
归档：`solutions/20260830_v106_l2-expansive-cat-active_score294.272633_time413s/`

## 假设与实现

v088/v089 证明在 expansive FFN 上直接增加逐行 HSDQ 会回退，但 L0 显示
`fc_gate/fc_up` 的 activation-side headroom 仍较大。因此只尝试一个低自由度
CAT balance，不恢复 v093 的全 block 搜索：

\[
b_j=\operatorname{normalize}\left((\operatorname{RMS}(A_j)/
\operatorname{RMS}(W_j))^{\alpha}\right),\qquad \alpha=0.25.
\]

仅当静态 `weight_rows > weight_channels` 时启用；不读取 role-id、模型名或调用
顺序，不增加 permutation/Householder state。与当前 BOAT balance 相乘后，保持
`A/b · R` 与 `W·b · R` 的等价产品，并用两侧 operand-local HiF4 重建误差决定
是否保留；proxy 不优于 parent 时返回 parent。

## 分层预筛

固定 Qwen cache，层位 `{0,5,11,17,23}`、七 role、`amax6`、两折 calibration。

| 指标 | L0 parent | v106 screen | 差值 |
|---|---:|---:|---:|
| selected-layer `both_player` | 0.523019429223 | 0.525228958652 | +0.002209529429 |
| layer 0 | 0.601076124891 | 0.612123772036 | +0.011047647145 |
| layer 5/11/17/23 | unchanged | unchanged | 0 |
| `fc_gate` role mean | 0.381803986554 | 0.397270692557 | +0.015466706003 |

分层 JSON：[`l2-cat-stratified-qwen.json`](../../artifacts/real_model_suite/l2-cat-stratified-qwen.json)。

## Full-layer 门禁

命令：

```text
python evaluator/real_model_suite.py --models qwen2.5-0.5b --primary-model qwen2.5-0.5b --panel-profile qwen-official --device cpu --algorithm-device cpu --cache-mode read --solution solution.py --candidate-name v106-l2-cat --output artifacts/real_model_suite/v106-l2-cat-qwen-full.json --report logs/execution/2026-08-30-v106-l2-cat-qwen-full.md
```

| 指标 | v100/v101 parent | v106 | 差值 |
|---|---:|---:|---:|
| Linear mean | 0.501557612498 | **0.503458942243** | **+0.001901329745** |
| Attention mean | 0.842039488461 | 0.842039488461 | 0 |
| Linear panel | 125.389403124 | **125.864735561** | +0.475332437 |
| Attention panel | 168.407897692 | 168.407897692 | 0 |
| Qwen panel total | 293.797300817 | **294.272633253** | **+0.475332436** |
| native total | 417.882506491 | **419.160200079** | +1.277693588 |
| API time | 392.423565s | **412.654599s** | +20.231034s |

API 时间余量为 `7.345401s`，仍严格小于 `420s`；wall time `446.069189s` 仅作
诊断。收益只来自 `fc_gate`：其 mean `0.375125974236 → 0.388435282450`；
其余六个 role 逐项不变。候选通过 L2 full-layer gate，提升为当前最高本地 parent。

## 复核与后续

合成/合规测试：`30 passed in 6.23s`。source LF SHA256：
`708081b5281e02da0c2a6e21881027b2e8d31eed423fd3c70e4572424667dd77`。
固定 cache raw SHA256：`ff40b5e0ce9568faae6582004f23f4f9b1f3f28a913671b9cd1fd40397f65aea`。

该正增益只证明一个固定 CAT balance 在 Qwen expansive shape 上有效，不证明
所有 FFN 或其他模型都有效；下一步按 active plan 进入 L3，并保留 v106 作为
parent。若后续候选无法保持 `linear_mean > 0.503458942243`，必须回退到 v106。
