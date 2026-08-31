# 本地评测统一口径与官方锚点校准

日期：2026-08-31  
目的：修复本地评测中 native、panel、official 和异构时间混用的问题，建立一个可重复、较快、可横比的本地主指标。

> 本文件保留的是 sampled-means-v1（224L/32A）历史校准过程。当前活动口径已经
> 升级为 sampled-means-v2（112L/96A）；v84/v98/v100 的同配置复评和官方超时
> 差异以 [v84/v98/v100 运行时分析](2026-08-31-v84-v98-v100-runtime-analysis.md)
> 为准，本文中的 v1 时间不得与 v2 时间混排。

## 1. 现在唯一使用的本地主指标

对每一个测试 case，先计算相对标准 HiF4 的 gain：

```math
g_i=\frac{MSE_{STD,i}-MSE_{PLAYER,i}}{MSE_{STD,i}}.
```

新评测 profile 是 `sampled-means-v1`：

- 默认只评估 Qwen2.5-0.5B；
- 24 层中分层抽取 8 层，并强制包含第 0 层和最后一层；当前固定 seed
  `20260831` 得到 `[0,1,5,10,13,15,22,23]`；
- 保留全部 7 个 Linear role（`q/k/v/o/fc_gate/fc_up/proj`）；
- 保留全部 2 个 calibration window，并按位置保留 4 个 validation window；
- 因此每次运行固定为 `8×7×4=224` 个 Linear case 和 `8×4=32` 个
  Attention case。

本地只报告两个算术平均：

```math
\bar g_L=\frac{1}{|C_L|}\sum_{i\in C_L}g_i,
\qquad
\bar g_A=\frac{1}{|C_A|}\sum_{i\in C_A}g_i.
```

报告同时显示比例和百分比：`0.509408` 就是 `50.9408%`。候选比较只能在
相同 profile、seed、cache、device、mode 和数据 revision 下进行。

## 2. 官方分数、本地旧字段和新均值的区别

官方新版面板是 250 个 Linear case + 200 个 Attention case。官方分数为：

```math
S_{official}=100\left(\sum_{i\in L_{official}}g_i+
\sum_{i\in A_{official}}g_i\right)
=100\left[250(1-\bar r_L)+200(1-\bar r_A)\right],
```

因此满分是 `45,000`，官方平均 gain 是 `S_official/45,000`。官方隐藏 case
不在本地，因此本地均值不能直接变成官方分数。

历史 JSON 中仍保留以下兼容字段，但它们不再是新报告的主结果：

| 字段 | 定义 | 允许用途 |
|---|---|---|
| `official_flow_score` / native total | 当前本地所有层、role、window 的 `g_i` 直接求和（通常 Qwen 为 672/96 cases） | 追溯旧日志，不能跨样例数比较 |
| `panel_score` | `P=250·mean(L)+200·mean(A)`，单位是 0–450 的 gain-sum，不乘 100 | 旧版相对排序，不能当官方分数 |
| `mean_scores.linear_mean` | 新 profile 抽样 Linear case 的平均 `g_i` | 当前唯一 Linear 主指标 |
| `mean_scores.attention_mean` | 新 profile 抽样 Attention case 的平均 `g_i` | 当前唯一 Attention 主指标 |

## 3. 已确认官方结果与历史本地记录

下表前三列是用户确认的官方结果；“历史本地”仅在同一归档记录中给出，不能
把不同设备的时间拼成一条曲线。旧 v31/v34 的本地计时不是当前 Qwen
`seq128/calib2/test4` API 全量，时间列故意标为不可比。

| 版本 | 官方分数 | 官方时间(s) | 历史本地 Qwen native | 历史本地 Qwen panel P | 历史本地时间/设备 |
|---|---:|---:|---:|---:|---|
| v31 / C39-FW | 21864 | 161.3 | — | 230.096230 | 旧 CUDA stage，协议不可比 |
| v34 / C41b | 21864 | 159.4 | — | 230.096230 | 旧 API 记录，协议不可比 |
| v51 / C47b | 22451 | 234.0 | 349.344342 | 237.541351 | 149.00s，CUDA |
| v66 / C66 | 22557 | 217.2 | 350.152420 | 238.282409 | 151.91s，CUDA |
| v72 / C74 | 22662 | 226.0 | 356.605602 | 240.683147 | 163.41s，CUDA |
| v74 / C75 | 22750 | 239.387 | 361.503707 | 242.505358 | 179.27s，CUDA |

官方平均 gain 分别为 `0.485867, 0.485867, 0.498911, 0.501267,
0.503600, 0.505556`。v74 距目标 36,000 的官方分差是 `13,250`，但这不是
本地均值的线性差值。

## 4. 官方分数对本地旧 panel 的拟合（仅事后诊断）

对上面 6 个官方锚点，以旧 panel `P` 为自变量、官方分数 `S` 为因变量做
普通最小二乘，得到：

```math
\hat S=4758.8504+74.4043P,
\qquad R^2=0.9894,
\qquad RMSE=37.15\text{ 分}.
```

留一法 RMSE 为 `53.96` 分。锚点范围只有 `P∈[230.096,242.505]`；v127
全量旧 panel `294.261` 已超出该范围约 `4.17` 个跨度，代入公式得到的
`26653` 不是可信预测。把公式反解到 `36000` 得到 `P≈419.884`，同样是
严重外推，不能作为提交目标或分数承诺。

这组拟合只能说明在已知官方通过版本附近，panel 排序大致跟随官方排序；它
不能修复隐藏 case 分布差异，也不能替代官方提交。

## 5. 官方时间与本地时间的拟合

只有 v51/v66/v72/v74 的 Qwen CUDA `seq128/calib2/test4` 记录勉强属于同一
协议。四点为：

| 版本 | 本地 API(s) | 官方(s) | 官方/本地 |
|---|---:|---:|---:|
| v51 | 149.00 | 234.0 | 1.5705 |
| v66 | 151.91 | 217.2 | 1.4298 |
| v72 | 163.41 | 226.0 | 1.3830 |
| v74 | 179.27 | 239.387 | 1.3353 |

线性拟合为：

```math
\hat T_{official}=163.8250+0.405984T_{local,Cuda},
\qquad R^2=0.332,
\qquad RMSE=6.85\text{ 秒}.
```

相关性很弱，倍率还从 `1.57` 变到 `1.34`，所以“本地 300 秒就是官方
300 秒”是错误的。官方时间必须单独记录，不能从本地绝对秒数硬判。

## 6. v74 复测及当前候选的配对结果

v74 历史本地时间没有丢失：归档仍有 `179.27s CUDA`。为和当前根在同一
机器比较，另外执行了当前 CPU 全量复测：

- v74 CPU full：panel `242.500393`，API `658.877s`，wall `690.600s`；
- v127 CPU full：panel `294.260802`，API `453.102s`，wall `485.285s`。

v74 CPU 超过 420 秒但官方实际只有 239.387 秒，构成直接反例；这两次 CPU
数据不能用于宣称官方通过/超时，只能用于同机相对比较。

在新的统一采样 profile 下，两次运行共用完全相同的 `224/32` case：

| 版本 | Linear mean | Attention mean | Linear cases | Attention cases | API(s) | Wall(s) |
|---|---:|---:|---:|---:|---:|---:|
| v74 | 0.440305 | 0.671106 | 224 | 32 | 218.619 | 229.485 |
| v127 | 0.509408 | 0.828395 | 224 | 32 | 151.136 | 161.840 |

v127 在该抽样计划上 Linear 增加 `0.069103`、Attention 增加 `0.157289`，
API 快 `30.9%`。这只是当前本地 A/B 事实，v127 仍需官方 Attention/长序列
行为验证。

## 7. 长序列风险不能被 seq128 样本掩盖

官方公开的 Attention calibration mini pattern 为 `[10,128,512,1024,1024]`。
同一输入上单次 calibration：

| 版本 | 时间(s) |
|---|---:|
| v74 | 0.441 |
| v127（变长 PAWV 修复） | 10.873 |

v127 约慢 `24.6×`。原因是新 PAWV 路径对长 token 矩阵执行额外的
`O(HL^2)` probability/diagonal 工作；固定 `seq=128` 的 Qwen A/B 不能外推
官方长序列总时间。任何正式候选都必须同时通过变长 shape smoke，并单独记录
长序列校准耗时。

## 8. 新规则

1. 新实验默认只读固定 cache，使用 `sampled-means-v1`，日志主表只写
   `linear_mean`、`attention_mean`、case 数、Local API 和 Wall。
2. 候选横比必须完全匹配 profile、seed、layer/window index、model/data
   revision、device、algorithm-device、mode 和 cache；否则标记为不可比。
3. `official_flow_score`、`panel_score`、官方分数和官方时间分属三个独立
   证据层，禁止在同一列或同一“总分”中相加。
4. 本地 API=候选六个 API 的 calibration+dynamic 累计；Wall 包含调度和报告
   开销。两者都标注本地 device，不再命名为 official API 或 official time。
5. 本地 300 秒只作为可选的“本地参考阈值”，不改变结果有效性；官方 300
   秒只有在官方平台实测后才能填写。

原始证据：

- [v74 官方通过](v74-official-pass.md)
- [v74 当前 CPU 全量复测](v74-cpu-recheck-qwen-full.md)
- [v127 当前 CPU 全量复测](v127-v106-pawv-fixed-qwen-full.md)
- [v74 新采样结果](v74-sampled-means-qwen.md)
- [v127 新采样结果](v127-sampled-means-qwen.md)
