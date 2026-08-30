# v106 L2 expansive-FFN CAT balance

## 裁决

L2 低自由度 CAT balance 通过固定 Qwen full-layer 门禁，提升为当前本地最高
精度 parent。根目录保留该实现；Attention 未改变。

| 指标 | v100/v101 parent | v106 | 增量 |
|---|---:|---:|---:|
| Linear mean | 0.501557612498 | 0.503458942243 | +0.001901329745 |
| Attention mean | 0.842039488461 | 0.842039488461 | 0 |
| Linear panel | 125.389403124 | 125.864735561 | +0.475332437 |
| Attention panel | 168.407897692 | 168.407897692 | 0 |
| Qwen panel total | 293.797300817 | **294.272633253** | **+0.475332436** |
| native total | 417.882506491 | **419.160200079** | +1.277693588 |
| API time | 392.423565s | **412.654599s** | +20.231034s |
| wall time | 424.693400s | 446.069189s | — |

API 时间仍小于官方 `420s` 限制（余量 `7.345401s`）；wall time 仅作诊断。

## 算法

仅对静态结构条件 `weight_rows > weight_channels` 的 expansive FFN 形状启用，
不使用显式 role 或模型名。先由校准激活/权重 RMS 构造固定

\[
b_j=\operatorname{normalize}\left((\operatorname{RMS}(A_j)/
\operatorname{RMS}(W_j))^{0.25}\right),
\]

再与已有 BOAT balance 相乘，并用 operand-local HiF4 重建误差评分；若 proxy
不优于当前 BOAT，返回 parent。旋转、hierarchy 和在线 state 格式保持不变，
因此等价关系仍为 `A/b · R` 与 `W·b · R`。

full-layer 的实际收益只来自 `fc_gate`：

- `fc_gate` mean：`0.375125974236 → 0.388435282450`；
- `q/k/v/o/fc_up/proj` 逐项不变；
- screen layer-0 的 expansive gain 为 `+0.011047647145`，其他选定层不变。

## 命令与证据

分层预筛：

```text
python evaluator/linear_candidate_screen.py --cache artifacts/real_model_suite/cache/qwen2.5-0.5b__seq128__calib2__test4__layersall__schema1.pt --solution solution.py --layers 0 5 11 17 23 --roles q k v o fc_gate fc_up proj --stage L2 --output artifacts/real_model_suite/l2-cat-stratified-qwen.json --report logs/execution/2026-08-30-l2-cat-stratified.md
```

full-layer：

```text
python evaluator/real_model_suite.py --models qwen2.5-0.5b --primary-model qwen2.5-0.5b --panel-profile qwen-official --device cpu --algorithm-device cpu --cache-mode read --solution solution.py --candidate-name v106-l2-cat --output artifacts/real_model_suite/v106-l2-cat-qwen-full.json --report logs/execution/2026-08-30-v106-l2-cat-qwen-full.md
```

source LF SHA256：`708081b5281e02da0c2a6e21881027b2e8d31eed423fd3c70e4572424667dd77`。
固定 cache raw SHA256：`ff40b5e0ce9568faae6582004f23f4f9b1f3f28a913671b9cd1fd40397f65aea`。
合成/合规测试：`30 passed in 6.23s`。
