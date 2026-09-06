# Linear fused carrier-energy act-order 计划（REJECTED_TIME）

> 状态：`REJECTED_TIME / LINEAR-CARRIER-ENERGY-FUSED`
>
> 日期：2026-09-06
>
> 父版本：v189 `static-actorder-hdiag-recovered`。

## 1. 目标与边界

上一候选 `linear-carrier-energy-actorder` 的固定统计在 eval-v3 六 shard 和 OOD 上均为正，
但 fresh default 时间预测为 `280.622241s`，超过 `<280s` 门禁。本计划只优化该统计的
实现路径：保持 32-row、每个 calibration pair 等权、部署坐标中的 `smooth_inv` 与固定
`permutation`，将逐 pair 的 carrier 解码与逐元素变换改为一次 batched energy reduction。

这不是 sample 数、权重、阈值或 block-order 参数扫描，也不改变候选的量化状态或在线路径。
不改 Attention，不重开已关闭的 block-order、动态 per-call、码本、V-side 或 cross-fold 家族。

## 2. 单一机制与等价归约

对每个 pair 固定取 `_sample_rows(..., 32)`，保持上一候选的 pair 等权统计：

```text
E_j = mean_over_rows(carrier[i,j]^2)  # 每个 pair
score_b += sum_j(E_j * importance_j)  # pair 等权累加
```

实现上把各 pair 的 sampled carrier 拼接，并用每个 pair 的固定行数归一化后一次求和；再把
`smooth_inv^2` 和 permutation 作用在 channel energy 上，而不是对每一行创建 transformed
dense 矩阵。除浮点归约顺序外，该统计与已归档候选相同；候选仍复用 v189 的 GPTQ block
reorder 和全部其余状态。

## 3. 执行门禁

### F0：单文件与统计一致性

通过 CUDA venv 做编译、六 API 导入、最小合法 state smoke；在 CPU 合成输入上核对 fused
energy 与 reference pair-wise energy 的相对误差，确认 block order 是合法 permutation。

### F1：两 shard 配对

用同一 proxy-v2 cache、v189 baseline、eval-v3 Linear-only 运行 shard `0,1`。只检查接口、
有限输出、reachability、control、逐 case delta 和本地 API 时间；出现非法 state 或无
reachability 即关闭。

### F2：六 shard 与 OOD

F1 通过后运行 Linear 六 shard 和 OOD 六 shard。记录 mean/median、尾部、负 case、L1、
split/length/role 覆盖及 `Δ(gain_in - gain_ood)`；OOD 门禁为 `|Δgap| <= 0.01`。本地
proxy 不能换算官方分数。

### F3：fresh default 时间

运行一次 fresh default，确认 Attention 与 v189 逐位不变。时间模型预测必须 `<280s`；若
统计仍正但预测超门，按 `REJECTED_TIME` 归档，不再改该实现的局部参数。

## 4. 归档与提交

每次运行保存 source、SHA、manifest、JSON/report、OOD 证据、执行日志和父版本引用。只有
local default 高于 v189、OOD/合法性门禁通过且预测 `<280s` 才生成只含候选 `solution.py`
的 `solution.zip`，登记 `official: unregistered/NA`；根 `solution.py` 保持 v186。官方上传
入口未提供给当前工具时不宣称已上传。官方回传后才按总分/时间相对 v186 裁决是否更新根版本。

## 5. 最终裁决

fused reduction 保持前一候选的本地分数和 OOD 结果，但 fresh default 时间模型预测为
`281.400628s`，仍未满足 `<280s`。候选记为 `REJECTED_TIME`，未提交官方，根 `solution.py`
保持 v186。
