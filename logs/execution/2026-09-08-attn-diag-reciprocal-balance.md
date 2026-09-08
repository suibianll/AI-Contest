# 2026-09-08 `attn-diag-reciprocal-balance`

## 结论

候选从当前完整根构建，六 API、状态合法性、数学关系和实际可达性均通过。4B eval-v3
Attention shard0 没有实现异常；唯一测试层的闭式 Q/K 平衡被固定双窗口门控拒绝，最终回退到
父状态，根方案保持不变。候选已归档，官方状态为 `unregistered/NA`。

## 固定配置与源码

- 父 SHA256：`12352efdd4e23cc5e1e17953008664fbaa5ea5d693373635fdafc4d28ce4e24e`
- 候选 SHA256：`7ac5cd94a769c82299c0f88264cf94f51b451c69fcc1a1b8fd8ec624f7e08a36`
- fit windows：`0,1,2`；gate windows：`3,4`；最多 256 token；chunk 8。
- 公式：`d=0.25*log((b+1e-12)/(a+1e-12))`，GQA group 去均值后 clamp 到
  `+/-log(2)/2`，部署为 Q/K 互逆缩放并同步 K learned center。

## 检查结果

```text
check_math_and_import.py: PASS
verify.py: PASS
```

验证覆盖闭式解、真实输出 Jacobian 能量、六 API 隔离导入、合法状态、rotation/center
折叠、向量编译、父回退和强制可达性。

## 4B shard0

命令使用 eval-v3、Qwen3.5-4B proxy-v2 dense cache、CUDA、Attention-only、shard 0。
候选与当前根均为 12 cases、Attention mean `0.5702418426766733`，paired `0/0/12`，
delta `0`；`reasonableness_issues=0`。候选 API total `5.625367s`，父 API total
`0.748119s`，本地时间只作风险记录，不换算官方时间。

## 机制审计

唯一测试层 `diag_attempted=1`、`diag_accepted=0`、`diag_fit_windows=3`、
`diag_arm=parent`。提案的 `d_norm=2.76898599`、`d_max_abs=0.34657359`；两个门控窗口
实际改变 Q/K 编码，累计 Q `1,081,640`、K `267,086` 个码。window 3 因 non-causal
loss 回归而失败，window 4 的 mean/causal/non-causal 均失败，故保留父状态。

## 证据位置

- 归档：`solutions/attn-diag-reciprocal-balance/`
- 评测：`artifacts/proxy_v3/full_solution/attn-diag-reciprocal-balance-shard0-r2/candidate/`
- 工作脚本：`workbench/full_solution/attn-diag-reciprocal-balance/`
