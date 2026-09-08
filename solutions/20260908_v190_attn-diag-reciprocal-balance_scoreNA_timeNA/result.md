# v190 — attn-diag-reciprocal-balance

## 状态

- 机制：在当前根的 Q/K 已有 rotation、K-center 之后，按最终 Attention 输出传播能量做逐通道闭式 Q/K 互逆平衡。
- 父 SHA256：`12352efdd4e23cc5e1e17953008664fbaa5ea5d693373635fdafc4d28ce4e24e`
- 候选 SHA256：`7ac5cd94a769c82299c0f88264cf94f51b451c69fcc1a1b8fd8ec624f7e08a36`
- 固定配置：fit windows `0,1,2`；gate windows `3,4`；最多 256 token，chunk 8。
- 官方状态：`unregistered/NA`；根 `solution.py` 未替换。

公式为 `d = 1/4 * log((b+1e-12)/(a+1e-12))`，先按 GQA group 去均值，再限制到
`[-log(2)/2, log(2)/2]`；部署为 `Q_parent*exp(d)`、`K_parent*exp(-d)`，并同步折叠
K learned center。候选保留当前根的全部六个 API。

## 检查

- `check_math_and_import.py`：PASS（闭式、Jacobian 能量、公共契约、隔离导入）。
- `verify.py`：PASS（rotation/center 折叠、向量编译、父回退、强制可达性）。
- shard0：PASS，`reasonableness_issues=0`，12 个 Attention cases，candidate/base mean
  均为 `0.5702418426766733`，paired `0/0/12`，delta `0`。
- candidate API total `5.625367s`，baseline API total `0.748119s`；本地时间仅作诊断，
  不换算官方时间，也不作为本候选的否决门。

## 校准审计

唯一测试层执行了 `diag_attempted=1`、`diag_fit_windows=3`，闭式平衡得到
`d_norm=2.76898599`、`d_max_abs=0.34657359`。两个 gate window 均实际改变了 Q/K
编码（合计 Q `1,081,640`、K `267,086` 个码），证明机制可达；但 window 3 的
non-causal loss 变差，window 4 的 mean/causal/non-causal 均变差，因此按固定门控保留父状态
（`diag_accepted=0`、`diag_arm=parent`）。

评测记录保存在 `artifacts/proxy_v3/full_solution/attn-diag-reciprocal-balance-shard0-r2/candidate/manifest.json`；
按当前计划，候选已归档，等待官方提交与回传后再决定是否替换根。
