# Linear calibration-energy act-order — time rejected

日期：2026-09-06  
父版本：v189  
状态：**REJECTED_TIME**  
候选 SHA-256：`2c37e66a01550a2b83818ae76a45daccafbd625d90d1652804bbbb1fbbe5a03b`

eval-v3 六 shard 的 Linear mean 为 `0.637795463532374`，父版本为
`0.636799488553279`；OOD mean 为 `0.649708907181630`，父为 `0.648734547234220`。
fresh default 为 Linear `0.641470702272374`、Attention `0.752173407020070`、
Overall `0.687596829250581`，Linear 顺序 reachability 为 `1`。

时间模型分解为 `W_calib=273.194363500457s`、`A_calib=58.889074399835s`、
`dyn_act=59.334178600693s`、`dyn_qkv=2.925786998589s`，预测
`281.514913071177s`，超过 `<280s`；不分配新版本、不提交官方、不扫描邻域。官方状态
为 `unregistered/NA`，root 仍保持 v186。

证据：

- eval-v3：`artifacts/proxy_v3/linear-calibration-energy-actorder-20260906/e1-full`
- OOD：`artifacts/proxy_v3/linear-calibration-energy-actorder-20260906/e2-ood`
- fresh default：`artifacts/official_eval/linear-calibration-energy-actorder-fresh-default.json`
