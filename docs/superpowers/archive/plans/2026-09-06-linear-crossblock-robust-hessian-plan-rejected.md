# Linear 多折 cross-block Hessian 联合坐标计划

> 创建：2026-09-06  
> 状态：**CLOSED / B1_REJECTED**  
> 研究父：v189（Linear `0.640258324430`、Attention `0.752173407020`、Overall
> `0.686889608842`）  
> 根父：v186（官方 `17599/272s`，SHA
> `F8495DCA20334ACBDAD16FC18EE41A4970F31E1837FDEEDCEE9C70AEE54E7EB8`）

## 1. 机制

在最终部署坐标中，为相邻 64-channel block 编译四折 activation covariance，并对
当前静态 Weight 做一次 conditional block-coordinate 更新：固定另一块误差，用
64×64 对角块逆构造 shifted target，再以 pooled + 四折 mean/worst 完整二次型接受。
配置固定为 128-channel pair、一轮 pass、窄层全 pair、宽层 cross-term 前 25%、robust
mix `0.5`。

该机制不构造 `A@W`，不使用失败的 cross-fold minimax A@W 字典序聚合，不改变在线 state
或动态 API；但它与关闭的 cross-fold minimax 一样，最终仍需通过真实部署配对。

## 2. 执行结果

- B0：单文件导入、`py_compile`、四折 covariance 形状 `(4, 2, 128, 128)` 和有限性
  检查通过；候选 SHA256 为
  `96a6bacbdc9057b0dfaa282d969347fd732b62c92e0f543db501a4789d22e62e`。
- B1：当前 `eval-v3` 默认 Linear 六 shard 请求在前两 shard 后按失败停止，共 112
  个配对 case。shard 0 delta mean/median `-0.202975/-0.200357`，shard 1 为
  `-0.185789/-0.179085`；两 shard 均为 `0/48/8` 正/负/零，L1 分别
  `0.202975/0.185789`，worst-20% 分别 `-0.400162/-0.296210`。`o` role 为主要
  回退来源，所有输出有限但不具备可部署收益。
- API 诊断总计 `142.992s`（112 cases，非官方时间），未运行 default/OOD/官方。

## 3. 裁决

B1 的真实部署输出与 pooled/fold Hessian 目标严重错位，按 L1 和负 case 门禁关闭该机制。
不调整 pair ratio、阻尼、fold 聚合、solver 顺序或码字邻域。根 `solution.py` 仍为 v186；
v189 仍是本地最高控制。

## 4. 证据

- 候选快照：`solutions/20260906_linear-crossblock-robust-hessian_rejected/solution.py`
- B1：`artifacts/proxy_v3/linear-crossblock-robust-hessian-20260906/default-b1/`
- 执行记录：`logs/execution/2026-09-06-linear-crossblock-robust-hessian-plan.md`

