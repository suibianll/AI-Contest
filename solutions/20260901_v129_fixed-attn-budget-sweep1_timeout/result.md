# v129 / fixed-attn-budget-sweep1

Status: **TIMEOUT / OFFICIAL FAILURE (user-confirmed)**.

- Date: 2026-09-01
- Parent: v128 / fixed-attn-budget
- Source SHA256: `7319F00E5259FE15E7C5ECA99E214A8F7482CF5CF066D6E3025E86C92D9095EC`
- Local protocol: `official-shape-v1`, cached Qwen2.5-0.5B, 250 Linear + 200 Attention
- Local result: Linear `0.465655`, Attention `0.836579`, API `248.363s`, wall `270.606s`
- Official result: **score unavailable, time `>300s`, timeout (official; user confirmed)**

v129 将 Attention shortlist 精修 sweep 降为 1，因而本地六 API 代理降到 248.363 秒，
但官方评测仍然超时。该结果确认本地 API `<300s` 不能作为官方通过保证；后续提交判断
必须以官方返回为准。

该目录已经进入历史归档，遵守不可变规则，保留原有 `scoreNA_timeNA` 命名；官方
结果回传通过追加记录更新，不追溯重命名目录。
