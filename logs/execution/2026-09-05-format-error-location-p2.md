# P2 固定小面板格式误差定位报告（v186）

> 计划 §6。运行：`evaluator/coordinate_diagnostics.py --relax`，六 shard（48 Attention +
> 336 Linear，全 24 层 × v3 windows），产物
> `artifacts/proxy_v3/coordinate-diagnostics-v186/run-relax/`。
> 面板构成与计划 §6.1 的差异：eval-v3 每层只提供 2 个 test window（非 default 五长度），
> 因此本报告使用 24 层 × 2 windows（Attention）与 24 层 × 7 role × 2 windows（Linear）；
> 该构成在读取误差结果前已由 P1 CLI 固定（不按最大误差挑块）。放宽臂全部为
> `legal_candidate=false` 的诊断数学，不导出任何候选。

## 1. 方法

以 P1 的同坐标系量化分解为参照：对每个操作数，在**固定父编码其余字段**的前提下单独放宽
一个约束，经注意/线性前向测量输出 MSE（对 O_t 或对 ref）：

| 臂 | 放宽内容 | 固定内容 |
|---|---|---|
| R1 `r1_mant` | mantissa 连续于 [0, 1.75]（去掉 0.25 舍入网格） | sign / scale / lv2 / lv3 |
| R2 `r2_scale` | 每 64 块一个非负最小二乘 scale（不要求 E6M2） | sign / mantissa(离散) / lv2 / lv3 |
| R3 `r3_lv2` / `r3_lv3` | lv2 / lv3 独立投影到 [1,2]（不要求 1/2） | 其余字段 |

## 2. 结果（输出 MSE 均值；baseline = P1 的单操作数量化误差）

**Attention（48 case）**

| 操作数 | baseline（仅该侧量化） | r1_mant | r2_scale | r3_lv2 | r3_lv3 |
|---|---:|---:|---:|---:|---:|
| Q | 1.109e-3 | **1.94e-4** | 1.092e-3 | 9.92e-4 | 8.85e-4 |
| K | 1.252e-3 | **2.25e-4** | 1.142e-3 | 1.103e-3 | 9.78e-4 |
| V | 4.497e-4 | **1.76e-4** | 4.15e-4 | 4.22e-4 | 4.23e-4 |

**Linear（336 case）**

| 操作数 | baseline | r1_mant | r2_scale | r3_lv2 | r3_lv3 |
|---|---:|---:|---:|---:|---:|
| X（激活） | 1.079e-3 | **5.34e-4** | 1.097e-3 | 1.235e-3 | 1.273e-3 |
| W（权重） | 2.032e-3 | **4.69e-4** | 1.884e-3 | 1.866e-3 | 1.834e-3 |

R1 去除 player 误差的比例：Attention Q/K **82%**、V 61%、Linear W **77%**、X 51%。

## 3. 稳健性

| 检查 | 结果 |
|---|---|
| R1 改善率（player > r1_mant×1.001） | Attention Q 24/24 层、Linear W 168/168 layer-role pairs（100%）、Linear X 94.6% |
| R1 跨 test/validation split 同号 | Attention 0/24 层出现混合；Linear W 0/168 pairs 混合 |
| R1 深层覆盖 | 逐层 100% 改善，深层（L9/L21/L23）同样成立，非浅层伪影 |
| R2/R3 vs player | Attention R2 仅 0–2% case 差于 player；R3 同——放宽 scale/lv2/lv3 **无有效余量** |

## 4. 机制判定表（§6.3 研究筛选）

| 臂 | 统计筛选（paired median 下降、≥3/4 深度组、≥3/4 val/test 同号） | 是否有合法实现路径 | 判定 |
|---|---|---|---|
| R1 mantissa | **通过**（全面积改善、100% 同号） | **无**：HiF4 mantissa 依法为 3-bit 0.25 网格，固定 | `DIAGNOSTIC_FINDING`：舍入网格是最大可识别误差来源，但不可编译 |
| R2 scale | 不通过（≈player） | — | `NO_MARGIN`：scale/E6M2 选择器已饱和 |
| R3 lv2/lv3 | 不通过（≈player，个别略差） | — | `NO_MARGIN`：层级求解器已饱和 |

## 5. P2 结论与 P4 建议

1. **唯一有系统余量的格式约束是 mantissa 0.25 舍入网格**（占 Q/K 单侧量化误差 ~82%、
   Linear 权重 ~77%）。它不可通过合法字段调整来利用——合法的 3-bit mantissa 网格、E6M2
   scale 与 lv2/lv3 ∈{1,2} 三者在本轮全部达到各自约束下的饱和（R2/R3 放宽无收益证明
   scale 与层级的离散约束不再是误差来源）。
2. 因此，在**合法 HiF4 表示内**，P1 已证连续坐标变换族零偏差、P2 已证字段级求解器饱和，
   本计划 P4 的注册条件不满足可编译新机制 → 记为 **`NO_SUPPORTED_MECHANISM`**（与计划
   §8 结束分支一致）。剩余与榜首的 4166 分差距主要来自 3-bit mantissa 表示能力与榜首
   实现之间的机制代际差，而非本机剩余合法自由度。
3. 方向性保留（不进注册卡，仅作为 P3 解释框架）：若要再突破，必须改变**进入编码器的值
   分布粒度**（更细的块内结构 / 更深的层级利用），而这正是已关闭家族（scale 窗口 +4、
   v183 coverage、Householder/full64、A4/L4/C1）所在区域；本报告不支持以任何新参数化
   重启这些家族。
4. 按计划 §"P2 无新机制信号也可继续 P3 获取官方证据"：继续 P3 官方贡献探针（配对 QK/V、
   A1 长度桶、Linear 形状桶）以获取官方层面的分桶证据，验证"哪些官方桶仍承载收益"，
   用于判断是否值得为跨桶收益设计代价更高的机制。
