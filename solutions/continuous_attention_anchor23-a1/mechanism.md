# A23：父坐标下量化块 scale 乘积目标

登记于 2026-09-07；一个机制、一个配置，见 config.json。官方状态 NA。
隶属 [21071 机制证据驱动下一轮计划](../../../../docs/superpowers/plans/workpackages/21071-evidence-driven-research.md) §5。
父为 **A22-2**（`solutions/continuous_attention_anchor22-a2/solution.py`，
SHA `4686ad817128d60e1a1648e0e14793692097ee8171fac2bc5202466d5787c0b7`，
官方 14424/271s）；高分对照 A2（14440/274s）；低成本对照 R3（14405/238s）。

## 机制依据与目标

A22-2 已有官方 +19（vs R3）；其残余训练 loss 是 Q 与 K **各自**相对父 amax
的平方和相加，不等于两者对 QK 误差的联合作用：若变换使 Q 块幅度 ×c、
K 块幅度 ÷c，可加目标中一项降一项升，对冲方向不可见；而 QK 误差正比于
q·k 幅度乘积，危害由乘积决定。A23 **只改变训练目标**为联合乘积：

```text
a_Q(f,g,b) = mean over (tokens, 组 g 的全部 Q heads) of amax²(Q 的 head 内第 b 个 64 块)
a_K(f,g,b) = mean over tokens of amax²(K head g 的第 b 个 64 块)
loss = mean_{f,g,b} [ a_Q(f,s,g,b)·a_K(f,s,g,b) / max(a_Q(f,0,g,b)·a_K(f,0,g,b), 1e-12) ]
       + 1e-3 · mean(S²)
```

常量 Q×c、K/c 的乘积不变 → 不虚报收益。s 指残余变换 exp(±S) 后的坐标，
0 指 S=0 的完整父坐标（分母为预计算聚合常数，非逐 token amax）。

## 保留不变（与 A22-2 逐字节同结构）

- 基础流程：B 栈 → R3 原始旧训练 + R3 自身 gate（identity/rotation 臂）→
  完整父 P；残余提案在 **P 坐标**训练；gate 为 C' vs **完整 A22-2 流程父**
  的真实 readout MSE（原最后校准窗口），严格小于才接受，平局/不合法保留 P。
- K-center 同步编译 `c_new = c·exp(−S)`；部署 `Rq_new = Rq·exp(S)`、
  `Rk_new = Rk·exp(−S)`；S=0 逐位恢复 P。
- 优化器与预算：32 步 Adam/lr0.01/clip1/正则1e-3、谱投影（零迹 + ±log2/2、
  cond≤2）、手工矩阵指数梯度、epsilon/amax 并列极值次梯度。不另加训练阶段。
- V 与两个 Linear API 逐位冻结；不吞异常；动态 API 无新增候选循环。
- fit/select 隔离审读：训练窗口 = `calib_qkv_list[:-1]`（5 窗前 4），
  gate = 最后窗（index 4）；validation/test 为评测器独立测试窗，不参与
  训练/选择/gate。与 A22-2 划分相同，无重叠。

## 块-组映射（与真实布局一致）

- head 内 64 块索引 b = 0..dim/64−1；GQA 组 g 的 Q heads = qh/kh 个。
- 4B 面板（qh=16, kh=4, dim=256）：nb=4，每组 4 个 Q head；a_Q 跨
  (4 heads × tokens) 均值，a_K 跨 tokens 均值，乘积在 (g∈0..3, b∈0..3) 网格。
- dim 不能整除 64 的布局（仅合同 fuzz 的 nonpow2dim 80）：块跨 head，
  head 内 b 无定义——该布局退化为组级聚合乘积（块按起始 head 归组、
  组内块合并），保证六 API 合同存活；此布局不在任何评测面板中。

## 去重审读（R0，逐项）

| 对照 | 目标 | 参数化/插入点 | gate | 结论 |
|---|---|---|---|---|
| A21-1 | Q/K 各自 (amax/denom)² 均值相加 | R0·exp(±S) 替换式 | C vs B | 目标/参数化/gate 均不同 |
| A22-1 | 同 A21-1 目标 | 同左 + 完整 R3 回退 | C vs P | 目标不同 |
| A22-2 | 同 A21-1 目标 | 父坐标残余 exp(±S) | C' vs P | **仅目标不同**（可加 vs 乘积）——新自由度 |
| 旧 +4 scale 窗口 / block-smooth 覆盖率 | 单侧静态 scale 选择 | 无联合训练 | 静态 | 无 QK 联合训练目标 |
| 旧 Jacobian / 动态 Gram | 输出敏感度静态权重 | — | — | 无互逆乘积代理 |

数学不等价：可加与乘积目标在 Q×c/K÷c 方向梯度不同；非同名重试。

## 验证清单

- 手工乘积梯度 vs autograd（主布局 + 回退布局，ties/零块/正常块）。
- 互逆转置（tq@tkᵀ = rq@rkᵀ）、GQA 组映射（4 几何独立参考）、
  center 同步编译恒等式、S=0 强制回退 = A22-2 官方源码逐位（真实 4B 窗口）。
- `fuzz_official_contract.py` 合同（训练类必跑）、脱离仓库单文件六 API 导入、
  inference_mode 可达、V/Linear 冻结 control。
- attempted/accepted/changed 计数；Q/K 解码误差、QK、logits/probability、
  QKV 分解仅作解释，不按测试桶加路由。

## 评测与官方探索

4B 面板 paired：父 = A22-2 归档源码（`--baseline-solution`），同 cache
`qwen3.5-4b-proxy-v2.pt`、同协议/device；缺同口径 A22-2 4B 基线时由本卡
paired run 首跑补齐一次。顺序：shard0 冒烟 → 固定六 shard（Attention 72 例）。
判读门：Δmean>0、validation/test 各正、专项负向损失 mean(max(−Δgain,0))<0.02
（总 L1 记录）。**无本地时间门**；api_seconds 仅记录，官方 300s 唯一硬限；
不跑 0.5B/OOD/跨模型。

结果分支：乘积下降而真实输出不改善 → 只关闭此目标，下一卡才考虑实际
量化 QK 目标（先与旧关闭族去重）。不扫 loss 权重、窗口或 gate 阈值。
官方正向且 <300s 才更新侧父；同 SHA 不复测。

## 复现入口

CUDA venv：`build.py` → `verify.py` → `check_math_and_import.py` →
`fuzz_check.py`；`run.py screen` → `run.py full`。
结果目录 `artifacts/proxy_v3/continuous/attention/anchor23-a1/`。
