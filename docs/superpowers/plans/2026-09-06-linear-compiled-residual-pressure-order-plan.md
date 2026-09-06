# Linear 编译校准残差压力块序计划

状态：`ACTIVE / R0_PENDING`

## 假设与边界

父版本为根 v189（官方 `17616/275s`）。前一条窗口极值块序在 R1 shard0 的
mean/median 均为负，已关闭。本计划测试一个不同的输出误差近似：在最终部署坐标
中，用校准激活能量与最终权重量化残差的列能量形成每个 64-channel block 的固定
压力分数

```text
score(block) = sum(mean_window(A_window^2) * (W_smooth - W_hat)^2)[block]
```

其中 `W_hat` 是最终部署 HiF4 权重，`W_smooth - W_hat` 是最终权重量化残差。
该分数是 `trace(C_A · E_W^T E_W)` 的对角块近似，直接优先激活 GPTQ 中对最终
线性输出误差最敏感的 block；它不改变权重 codec、连续变换或量化状态，只编译
一个 block permutation。所有校准窗口使用固定 mean 聚合；不对该聚合做扫描。

禁止扫描残差/权重混合、mean/max/median、block 大小、fold、layer/role 路由、
阈值或候选邻域；没有样本或非 64 整除宽度时保留父静态路径。Attention 保持
父版本逐位不变。

## 执行顺序

### R0：单文件与合法 state

从根 v189 复制单文件候选，运行 `py_compile`、六 API 导入、合成合法调用和状态
检查。确认 residual-pressure order 有限且为完整合法 64-block permutation，
并且在线路径只读取校准编译的 order。

### R1：Linear 前两 shard

使用 `evaluator/eval.py`、固定 proxy-v2 cache、CUDA、`--linear-only
--shards 0,1`，baseline 为根 v189。记录 paired mean/median、L1、正负 case、
尾部、validation/test、order reachability 和未修改 Attention control。方向或
合法性失败即关闭，不扫描残差压力公式邻域。

### R2：六 shard、OOD、fresh default

R1 通过后运行 Linear 六 shard 与 OOD，再运行一次 fresh default。只把
`L1 < 0.02`、`|delta gap| <= 0.01` 作为风险诊断；本地分数不换算官方分数。
官方时间模型预测必须 `<280s` 才具备提交资格。

### R3：裁决

只有 fresh Overall 严格高于当前本地最高 `0.688994940507429` 且时间通过，才按
用户规则归档候选并提交官方；否则完整归档为 `REJECTED`，根保持 v189。官方结果
未确认前写 `unregistered/NA`。
