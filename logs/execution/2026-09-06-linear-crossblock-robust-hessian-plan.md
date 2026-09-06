# Linear 多折 cross-block Hessian 执行记录

日期：2026-09-06  
计划：[`2026-09-06-linear-crossblock-robust-hessian-plan.md`](../../docs/superpowers/archive/plans/2026-09-06-linear-crossblock-robust-hessian-plan-rejected.md)  
状态：`CLOSED / B1_REJECTED`

## B0

候选源码：`workbench/linear_crossblock_robust_hessian_solution.py`  
候选 SHA256：`96a6bacbdc9057b0dfaa282d969347fd732b62c92e0f543db501a4789d22e62e`。

`py_compile`、六 API 导入和 `_crossblock_fold_pair_covariances` 单元检查通过，返回
四折 `(4, 2, 128, 128)` 有限 covariance。候选快照保存于
`solutions/20260906_linear-crossblock-robust-hessian_rejected/solution.py`。

## B1

当前 `eval-v3` 命令使用固定 `proxy-v2` cache、CUDA、Linear 六 shard 请求，baseline
为 v189 `workbench/linear_static_actorder_hdiag_recovered_solution.py`，输出目录为
`artifacts/proxy_v3/linear-crossblock-robust-hessian-20260906/default-b1/`。

评测在 shard 0/1 后因负向停止，共 112 个配对 case：

| shard | cases | delta mean | delta median | L1 | 正/负/零 | worst-20% |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 56 | `-0.202975` | `-0.200357` | `0.202975` | `0/48/8` | `-0.400162` |
| 1 | 56 | `-0.185789` | `-0.179085` | `0.185789` | `0/48/8` | `-0.296210` |

`o` role 是主要回退来源；所有输出有限、case 唯一，但真实部署效果与 Hessian 目标
不一致。API 总计 `142.992s`，其中 calibration `108.998s`、dynamic activation
`33.994s`；这些秒数仅作诊断，不是官方时间。

## 裁决

B1 已超过 L1 门禁并出现系统性负向，关闭该机制；不运行 default/OOD/跨模型/官方，
不调整 ratio、damping、fold、solver 或码字邻域。根 v186 未改。

