# v185 clean-room 稳健算子量化

> 状态：IMPLEMENTED / LOCAL VALIDATION PENDING
>
> 该版本从空白文件实现，不继承任何历史 `solution.py` 源码。根 `solution.py` 仍为
> v182 官方父；v184 工作区未修改。

## 唯一方案边界

- 通用 HiF4：标准 E6M2 scale + 精确 lv2/lv3 hierarchy，top-loss block 才搜索固定邻域；
- Linear：一个 identity-shrunk RMS 对角等价变换候选，跨校准折 MatMul 输出 gate；
- Attention：K center、KV-head 共享 Q/K balance、收缩 logits gain、门控 `+4` scale code；
- 高维统计不直接进入在线 state；动态路径无 Hessian/Gram/矩阵逆；
- 所有步骤均有 identity/上一阶段回退。

## 证据

待依次补充：独立导入、五字段/state 合法性、Linear/Attention smoke、Qwen compact/default、
GPT-2、OPT、API 时间和父子配对。任何本地数字均不写入官方字段。

## 官方

- Score：NA
- Time：NA
- Status：unregistered
