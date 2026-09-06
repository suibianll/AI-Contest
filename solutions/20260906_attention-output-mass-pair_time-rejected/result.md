# Attention output-mass pair — time rejected

日期：2026-09-06  
父版本：v189  
状态：**REJECTED_TIME**  
候选 SHA-256：`10154C7EF5FAC891433687BA34B88BCECF4E60C6C8BCE0B12A7753BB32D44EBF`

固定 eval-v3 六 shard 的 Attention mean 为 `0.752881086113999`，父版本为
`0.752772354840816`；fresh default Attention 为 `0.752399782610857`，Overall
为 `0.686983932005295`。Linear `0.640258324429894` 与父版本逐位一致。

六 API 时间分解代入已校准官方时间模型得到 `282.286164185503s`，超过 `<280s`
门禁；因此不分配 v190、不提交官方、不扫描邻域。官方状态为 `unregistered/NA`，root
仍保持 v186。

证据：

- eval-v3：`artifacts/proxy_v3/attention-output-mass-pair-20260906/m1-full`
- OOD：`artifacts/proxy_v3/attention-output-mass-pair-20260906/m2-ood`
- fresh default：`artifacts/official_eval/attention-output-mass-pair-fresh-default.json`
