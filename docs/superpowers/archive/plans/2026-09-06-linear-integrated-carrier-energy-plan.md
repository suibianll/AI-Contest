# Linear 校准内 carrier-energy 块序执行计划

> 创建：2026-09-06  
> 状态：**CLOSED / R3_REJECTED_TIME**  
> 父版本：v189 `17616/275s`，根源码 SHA256
> `261202248a0146a2ee45f3df60bd1979bb8171b7c162921013b0024c848617af`

## 1. 唯一假设

上一版 carrier-energy act-order 在 eval-v3 与 default proxy 上高于 v189，但 fresh
default 的官方时间预测为 `280.622241s`，只超 `<280s` 门。新候选仍使用相同的
carrier-energy 块排序规则，唯一实现变化是把原始 NVFP4 激活的 32 行样本在既有
Linear 校准解码循环中保存并复用，避免 wrapper 再次解码输入；在线六 API、HiF4
状态、排序公式和候选路径不变。

这不是参数邻域扫描，也不恢复已关闭的排序/曲率机制。若仍超时则直接关闭该实现
路线，不再改样本数、排序权重或角色路由。

## 2. 固定门禁

1. R0：单文件导入、`py_compile`、六 API、合法 state、有限输出和排序 reachability。
2. R1：使用 v189 baseline，运行 eval-v3 Linear shard 0/1；若 no-op、系统性回退、
   非法状态或 control 异常，立即关闭。
3. R2：R1 通过后运行 Linear 六 shard 与 OOD，记录 median、L1、尾部、split 和
   control；OOD 只按 `|Δgap|<=0.01` 作补充门禁。
4. R3：fresh default proxy-v2 时间审计。只有 Overall 严格超过当前本地最高
   `0.687776303363468` 且预测时间 `<280s`，才归档、生成提交包并推送；否则记
   `REJECTED_TIME`，根 `solution.py` 保持 v189，不提交官方。

## 3. 执行记录

执行结果写入 `logs/execution/2026-09-06-linear-integrated-carrier-energy-plan.md`。

## 4. 执行结论

- R0 通过；候选源码 SHA256 为 `a30e05b7fb71a64121908bc10d102dd50133a23e32f6568ed7044dbc45321730`。
- R1 两片、R2 六片和 OOD 均保持正向；R2 Linear mean `0.637750315`，v189
  `0.636799489`，OOD mean `0.649639183`，相对 v189 的 in−OOD gap 变化约
  `+0.000046`。输出有限且 state reachability 完整。
- fresh default 与前一 carrier-energy 候选逐位相同：Linear `0.641778372`、
  Attention `0.752173407`、Overall `0.687776303`，没有超过当前本地最高。
- fresh default 时间分解为 `W=274.721588s`、`A=61.435543s`、
  `dyn_act=60.547017s`、`dyn_qkv=2.961587s`，预测 `284.291453s`，未通过
  `<280s` 门。候选归档为 `REJECTED_TIME`，不分配版本、不提交官方，根保持 v189。
