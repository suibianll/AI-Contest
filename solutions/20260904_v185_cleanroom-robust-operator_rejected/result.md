# v185 clean-room 稳健算子量化

> 状态：**REJECTED（本地）— clean-room 实现与真实调用图合法，但相对 v182 精度明显不足**
>
> 该版本从空白文件实现，不继承任何历史 `solution.py` 源码。根 `solution.py` 仍为
> v182 官方父；v184 工作区未修改。

## 唯一方案边界

- 通用 HiF4：标准 E6M2 scale + 精确 lv2/lv3 hierarchy，top-loss block 才搜索固定邻域；
- Linear：一个 identity-shrunk RMS 对角等价变换候选，跨校准折 MatMul 输出 gate；
- Attention：K center、KV-head 共享 Q/K balance、收缩 logits gain、门控 `+4` scale code；
- 高维统计不直接进入在线 state；动态路径无 Hessian/Gram/矩阵逆；
- 所有步骤均有 identity/上一阶段回退。

## 本地证据

- 独立随机形状 smoke：六 API、HiF4 五字段、CPU state 全部通过
  `evaluator/reference_hif4.py` 校验；
- Linear compact：28 个 calibration state、56 个跨 holdout case 全部成功，mean
  `0.417230809`、median `0.412433`、`56+/0-/0=`、cross-holdout `28/28` 同号，API
  `3.860s`；稳定优于标准 HiF4，但明显低于 v182 compact 约 `0.7055`；
- Attention compact：4 个 state/case 全部成功，mean `0.686541396`，API `3.766s`；
- Attention default：24 个 state、120 case 全部成功，mean `0.403767322`，API
  `23.599s`；相对 v182 等价 Attention 父 mean delta `-0.338062`、median
  `-0.351476`、`4+/116-/0=`，触发明确本地拒绝；
- 不继续运行 GPT-2/OPT：Qwen default 已是大幅、广覆盖回归，继续运行不改变裁决。

全仓 `pytest -q` 为 `38 passed / 3 failed / 1 error`：3 个失败是根 v182 私有辅助函数
`_choose_boat/_encode_rows` 的历史测试，不导入 v185；1 个 error 是系统 pytest 临时目录
权限。候选专属 smoke 与真实 evaluator 调用图均通过。

源码 SHA256：`3EA046594FB18DD86FD8CCFD2364A391039B0112E29986C8F949F9AF526C136C`。

## 结论

低有效自由度方案非常快且稳定优于标准 HiF4，但仅靠对角平衡、K center、收缩 logits gain
和稀疏 scale 邻域无法恢复成熟父版本的块级离散优化收益。v185 按计划 `REJECTED`，不提交
官方、不扣配额、不围绕阈值/refine ratio 调参；源码保留为干净研究基线。

## 官方

- Score：NA
- Time：NA
- Status：not submitted / local REJECTED
