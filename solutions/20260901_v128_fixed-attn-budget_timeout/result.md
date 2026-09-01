# v128 / fixed-attn-budget

Status: **TIMEOUT / OFFICIAL FAILURE (user-confirmed)**.

- Date: 2026-09-01
- Parent: v127 / root-v127-fixed-attn-budget
- Source SHA256: `0D4A0E91F6D076A9B694390DAE3D63A931D3D759AB609252AB3B54366F22F638`
- Local protocol: `official-shape-v1`, cached Qwen2.5-0.5B, 250 Linear + 200 Attention
- Local result: Linear `0.465655`, Attention `0.837789`, API `310.732s`, wall `332.557s`
- Official result: **score unavailable, time `>300s`, timeout (official; user confirmed)**

v128 的固定预算 Attention 校准提高了 Attention 代理精度，但没有满足官方端到端
时间限制。`310.732s` 是本地六 API 代理，不能替代官方裁决；本版本不再作为官方
提交候选。

该目录已经进入历史归档，遵守不可变规则，保留原有 `scoreNA_timeNA` 命名；官方
结果回传通过追加记录更新，不追溯重命名目录。
