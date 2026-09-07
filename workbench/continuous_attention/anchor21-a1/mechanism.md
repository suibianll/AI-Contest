# A21-1：联合 64 块 scale 的互逆变换

登记于 2026-09-07；一个机制、一个配置，见 config.json。官方状态 NA。

父为 R3 14405/238s；完整 R3 只作独立评测对照。候选从 R3 编码栈出发，
删除其旋转/center 输出反传训练器，使用 T=R0 exp(S) 的联合 scale 训练器。
冻结 v162 standard Linear 与 R3 V；根 solution.py 不替换。

## 执行前澄清

原计划同时要求“替换旧训练”和“校准 gate 对比完整父”，两者不能兼得：
获得完整 R3 的 gate 回退状态仍必须执行其旧训练。本轮明确选择真正替换：
校准 gate 比较 R3 **训练前栈**与一个新候选；独立 holdout 始终对比完整 R3。
不得把训练前回退状态的成绩记作完整 R3，也不把删除旧训练的差异隐去。
gate 无逐组 oracle，整层接受或回退，平局回退。

## 固定实现

- R0 为旧训练初始化的 Hadamard，非 2 的幂维度为单位阵；S 每 KV 组对称零迹。
- Q/K 分别乘 R0 exp(S)、R0 exp(-S)，编译为不同的 learned_rotation 矩阵。
- 真实前处理只做一次；每个训练 fold 使用全部 Q/K tokens。
- 每 role 按实际展平后的连续 64 块计算 amax，基准为同输入 S=0 的 amax。
  每 fold 等权，Q/K 分别取块均值后相加；同宽 KV 组等权。
  head_dim 不整除64时，块跨 head，仍按实际64块切分，不能重置到每个head。
- epsilon=1e-12；amax 并列极值均分次梯度，零块梯度为0。
- 32步 Adam/lr0.01/clip1/正则0.001；谱投影同时满足零迹与±log2/2，cond≤2。
- 手工矩阵指数梯度；动态 API 只有已有前处理、固定矩阵乘法与合法编码。
- 最后校准窗口执行真实 QK-softmax-V 输出门；独立验证集不参与学习/选择。

## 验证与复现

先运行 `build.py` 和 `verify.py`，随后 `run.py screen`、`full`。
通过负向损失门后运行 `ood`、`timing`、`cross`。命令均使用仓库 CUDA venv。
R3 原始 JSON 按原字节复制为 baseline，验证源码 SHA、路径、协议、panel、cache；
原始产物不修改、不重复调用父 API。梯度参考使用 autograd，仅限测试。
独立 GQA 参考逐 head 矩阵乘法，不复用部署 applier；真实编码前坐标拦截检验。

记录 scale 比例、attempted/accepted、互逆误差、逐 case 最终输出及六 API 耗时。
scale 下降不等于输出提升。此卡失败只否定本目标/初始化/固定配置，后继按 A21-2/3，
不扫描参数或宣称互逆变换无效。
