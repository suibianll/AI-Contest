# R2 执行记录：L24-C/L24-A/L25 三卡关闭 + fit_gain 目标达成（2026-09-08）

执行者：Linear 侧执行代理。依据 [持续研究循环](../docs/superpowers/archive/plans/2026-09-08-continuous-research-loop.md)。

## 账本定位（R-2）

R1 账本 v0（shard0 28 states）：

| 格 | 数值 | 标签 |
|---|---|---|
| E1 连续低维拟合残差 | 0.0074 | OPEN（极小） |
| E2 合法量化投影损失 | 0.0320 | OPEN（小） |
| E3 最终部署误差 | 0.6537 | 面板 1−gain |
| E4 部署失配残差 | 0.6143 | **OPEN（主导，94% of E3）** |

最大 OPEN 格 = E4 → 先跑 L24-C（E4 靶点）；E2 次之 → L24-A、L25（E2 靶点）。

## 三张卡执行结果（均 REJECTED）

| 卡 | 靶点 | 机制 | shard0 证据 | 关闭原因 |
|---|---|---|---|---|
| **L24-C** | E4 | 动态 activation scale 分桶加权拟合（替换 fold 能量 omega 为 per-row log2-scale 等权） | fit_gain 0.925<0.948；delta −0.242<−0.218（0/56/0） | 双向负向：分桶破坏校准拟合且未改善部署 → 卡证伪 |
| **L24-A** | E2 | 跨块残差 carry-over（OBC/GPTQ 式输出坐标误差补偿，C+=Xb(Wd−W_legal)ᵀ 入下一块求解） | fit_gain 0.824<0.948；宽层 accepted 3/144；delta −0.112 | carry 残差破坏后续块求解方向，接受崩溃 → 卡证伪 |
| **L25** | E2 | 块级 lv2/lv3=1 细格点联合选择（一次固定组合，L_all 严格下降才写回 scale_lv2/lv3=1） | fit_gain 0.945<0.948；面板 delta 与 L23b 逐位相同 | 细格点从未改变部署输出（接受判定不占优），拟合略降 → 卡证伪 |

三卡均按 §3.1 卡模板证伪判据关闭，未扫邻域；E2/E4 格保持 OPEN。

## fit_gain 目标状态

- **Linear fit_gain（L23b，13639FB2）= 0.9486 ≥ 0.9 目标达成**（168/168 states、
  336 fold-rows，全胜 STD 与父；父 0.7219）。协议：校准折叠 [0,1]、真实动态激活 +
  最终五字段解码、同 SHA 校准产物复用、CPU。
- L23b 标记 **READY_FOR_OFFICIAL**（本环境无官方上传入口；交付包路径
  `solutions/continuous_linear_l23b-residual-subspace/solution.py` + SHA
  `13639FB2…10FE0`；官方 300s 为唯一时间裁决）。
- 停止条件 S1 的 Linear 部分（fit_gain ≥ 0.9 + 合法输出 + 完整同口径覆盖）已满足；
  官方 <300s 与 Attention 侧 0.9 分别依赖官方回传与另一代理。

## 下一步（R-3）

E4（0.6143）仍是最大 OPEN 格；E2（0.032）/E1（0.0074）极小。已关闭的具体实现：
分桶加权 / carry-over / lv2/lv3 细格点。下一张卡须为**有实质区别的 E4 机制**
（非上述三种的邻域），或按 §6 停止条件给出去重比对表。