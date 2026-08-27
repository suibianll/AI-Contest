# v029 — C29 HAES Rejected（Level-0 oracle 否决，零提交代码变更）

- Date: 2026-08-28
- Candidate ID: `C29`（HAES，Hierarchy-Aligned Equivalent Scaling）
- Parent: `C21-C` / v025，根目录 `solution.py` SHA256
  `E94CD30A52B8361E99536BB9DD98EB604912946D4336E600DD9246305F466C35`
  （flag 全关，行为与 v025 逐位等价）
- Unique mechanism: 在 P 之后、R 之前插入层级对齐正对角缩放 S
  （组大小绑定组件 block_smooth_size，0→4），互逆折叠进 D。
- Source SHA256: 未变更（快照 = 父版本原样复制；Level-0 探针不修改
  提交代码）
- Local status: `rejected`（Level-0 oracle 预注册否决门）
- Official status: 不适用（未产生可提交变体）

## Level-0 S 网格 oracle 结果（预注册否决门 <5% → 否决）

矩阵：真实 GPT-2，层 0/5/11 × q/k/v/o/fc/proj（18 组件），amax6，
seq=128，calib=2；oracle = per-4 组独立枚举 z ∈ {−4..4}\{0}
（s=2^(z/8)），每组取激活硬重构误差（parent 坐标系度量）最小 z，
码字经 deployed 动态量化路径（`_nvfp4_to_hif4`，仅 multiplier 折叠
替换）重适配；z 选择用样本 0（128 行），simultaneous 终测用全部
校准行；变体逐校准样本调用（refinement 排名为调用内全局）。

| 指标 | 数值 |
|---|---:|
| 全矩阵能量加权降幅（bound 臂，simultaneous） | **0.14%** |
| 否决门（预注册） | 5% |
| 单组件最好 | +1.08%（layer 5 k） |
| 单组件最差 | −0.00%（layer 5 v，组间耦合微负） |
| 组件 block_smooth_size 分布 | **18/18 = 0**（Hadamard 全关） |

18/18 组件 block=0 意味着 S 全部处于**最有利对齐情形**（无 Hadamard
打散、per-4 与 lv3 完全对齐）——即便在此上限条件下，per-4 离散
scale 网格经码字重适配也只拿到 0.14% 激活能量。结论对绑定规则中的
H8/H16 退化分析免疫：退化只会让结果更差。

探针脚本：`evaluator/hierarchy_scale_probe.py`；完整数据：
`probe_results.json`（本目录）/ `evaluator/hierarchy_scale_probe_results.json`。

## 否决的机制解释

HiF4 的 per-4 有效 scale = lv1（e6m2，块级连续）× lv2（{1,2}，per-8）
× lv3（{1,2}，per-4）。S 的 2^(z/8) 细网格（0.707~1.414）与现有
{1,2,4} 档位 × 块级连续适配的组合差集极小；且 4bit mantissa 量化
误差由尾数精度主导，scale 再细分不改变尾数舍入误差。0.14% 即
scale 微调在尾数主导误差下的全部残余收益。与 C28（固定坐标码字
穷举上限 8.1% 能量）互相印证：激活码字/scale 拟合类机制已全部封顶。

## 方法论教训（S 坐标度量陷阱）

探针首版按计划 §4.5 的 L_A 定义（S 坐标系绝对误差）度量，得
49.92%（fc）——其中几乎全部是**纯缩放作弊**：组缩放 α 使 S 坐标系
误差能量 ×α²，但逆映射回 parent 坐标后完全抵消（输出误差不变）。
修正为 parent 坐标签（重构经 R^-1 与 S 逆映射后与 parent pre-R
激活比较）后同一配置降至 0.27%。任何未来涉及坐标缩放的候选
（C30 及以后），operand-local 指标必须在**固定坐标系**测量，
计划 §4.5 的 L_A 定义需按此修订。

## 排除了什么 / 没排除什么

- 排除：层级对齐对角缩放 S（pre-R 放置、9 档离散网格、per-4 到
  per-16 绑定粒度）作为激活侧误差的显著来源；C29 不进入 Level-1
  可行性探针与坐标下降实现。
- 没排除：C30 Hessian 感知层级排列（下一候选，前置 guard 预裁定）、
  C31 C23-lite 贡献拆分；未测试 S 与 Hadamard 组合开启（block=0
  前提下无意义）；未测试非 2 的幂步长或更大 z 范围（oracle 已证
  明量级不可行，细化网格无意义）。

按计划 §13 转入 C30：先执行 edge(i,j) 模式的
`linear_compliance_guard` 预裁定。
