# V 码分配类探针结论：NO_SUPPORTED_MECHANISM（2026-09-08）

用户决策：2026-09-08 解禁 F5-V（0.275，原 FAMILY_CLOSED）中的 **V 码分配类**（旧四类 per-head/per-channel/per-token/bias 关闭不变）。
本探针 = 解禁后的 R-1 账本定位步骤：在构建任何候选之前，先量化"码分配还有多少可捕获空间"。

## 结论

**V 重编码（NVFP4→HiF4）的码分配空间已 100% 饱和，无可执行假设。** 按 AGENTS §5：REJECTED_BEFORE_EVALUATION / NO_SUPPORTED_MECHANISM，不进入评测。

## 证据

`v_error_attribution.py`（零 API，4B proxy 校准缓存 5 折 × 6 FA 层，GPU）：

| 层 | standard（无 offset/refine） | 部署基线 | oracle（14 值 offset + ratio 1.0 + blocks 10⁹） | oracle_gain | 去掉 importance |
|---|---|---|---|---|---|
| L0 | 4.180e-3 | 3.515e-3 | 3.515e-3 | 0.00% | 3.515e-3 |
| L1 | 4.738e-3 | 4.071e-3 | 4.071e-3 | 0.00% | 4.071e-3 |
| L5 | 3.998e-3 | 3.464e-3 | 3.464e-3 | 0.00% | 3.464e-3 |
| L8 | 8.184e-3 | 7.041e-3 | 7.041e-3 | 0.00% | 7.041e-3 |
| L15 | 9.696e-3 | 8.651e-3 | 8.651e-3 | 0.00% | 8.651e-3 |
| L22 | 9.811e-2 | 8.419e-2 | 8.419e-2 | 0.00% | 8.419e-2 |

- **逐位确认**（L22 fold4，1024 token）：码参数 scale_factor/scale_lv2/scale_lv3/sign/mant 在 oracle 配置下与部署基线**逐位完全相同**。
- 部署 state 实测：`max_refine_ratio = 1.0`（动态捕获已达上限）、`max_refine_blocks = 24576`（每 fold ≤ 8192 块，上限从未绑定）→ 预算无稀缺，refine 已全量。
- **importance 实际无作用**：当前 head 级 E[A²]（repeat_interleave）去掉后码参数零变化——与 A3 候选（head 级统计量变体，GQA 持平）未晋级的深层原因一致。
- 与 v186 注释先验吻合：`_DYNAMIC_OFFSETS=(-1,1,2,3,4)` "oracle diag: captures 99.6% of exhaustive gain"——本次以 14 值网格实测达 100%。

## 覆盖的码分配维度（全部饱和/关闭）

| 维度 | 结论 |
|---|---|
| offset 搜索网格 | oracle 逐位饱和（本次实测）；扩展无增益 |
| refine 预算/排序 | ratio 已 1.0、上限未绑定；排序在全覆盖下无自由度 |
| importance（head 级/通道级） | head 级实测无作用；通道级 attention 内对称无信息（o_proj 不在校准签名内）；per-token 接口不支持且部署新 token 无意义（§7） |
| L1 scale/相邻码 | 历史已试未晋级（真实输出退化） |
| 折内数值优化 | A27-B 刚证伪（校准折不 transfer） |
| per-head/per-channel/per-token/bias | §7 关闭不变 |

## 剩余维度（超出本族范围，需用户决策）

唯一未试的是**改变码语义本身**（scale 结构 / lv2-lv3 判据 / E6M2 mantissa 码空间 / NVFP4→HiF4 格式映射），但它必须**同步修改编码端与解码端**（`_dequantize_hif4` 为 Q/K/V/Linear 共享路径），属于全侧码格式变更卡，风险与范围远超"V 码分配"。是否开设此类卡待用户决策。

## 结构性事实（长期有效）

> 2026-09-08 解释纠偏（[a28-interpretation-correction](../../../logs/execution/2026-09-08-a28-interpretation-correction.md)）：
> 本节"gain 0.9 目标的结构性含义"一段已作废——0.275 只对应固定 P,V̂ 干预条件，
> 忽略 QK 项与 V 项的负交叉补偿，不构成最终输出下界；A28 只关闭当前码语义下的
> V 码分配，F4 尚无已证明下界。以下原文保留不改写，以纠偏日志为准。

- V 量化对象 = **激活**（部署时对新 token 动态执行 `hif4_dynamic_quantize_v`），非权重；校准期 state 只能携带通道方向/网格参数，token 级信息部署时不存在。
- F5-V = 0.275 的残余为**当前码语义下**的格式级固有损失（NVFP4→HiF4 重编码，码分配自由度内不可再降）；L22 重编码误差（8.4e-2）为其余层 10-20 倍。
- ~~gain 0.9 目标的结构性含义不变：V 冻结上限 ≈ 0.725；突破需上述"码语义变更"类卡或官方侧 V 路径变化。~~（作废，见纠偏日志纠偏 2）
