# Qwen 主模型本地评测报告

运行时间：2026-08-31 04:12:22（配置 mode=amax6，seq=128，calib=2，test=4，cache_mode=read）

主评测配置：`qwen-official`，主模型 `qwen2.5-0.5b`，参考形状为 250 Linear + 200 Attention。
本地 shaped panel 只把冻结语料上每个组件的平均 case gain 投影到官方样例数量，不复制 case、不拟合官方绝对分数。官方分数没有进入候选校准状态，也没有传给 `solution.py`。评估器内部的输出矩阵乘法只在候选返回量化结果之后，用作固定参考误差；候选离线校准可以自行用 `A@W` 优化 `Q(W)`，但不得将其用于 `Q(A)` 或写入 `activation_state`。
官方上下文：外部 `youxilee/hif4` 用户提供结果为 24153/239s，仅作不可导入的参考；新增 2 个用例呈 Qwen 30B-like 特征，但完整输入尚未公开。

## 数据与模型完整性

- 数据集：`Salesforce/wikitext` / `wikitext-2-raw-v1` / revision `b08601e04326c79dfdd32d625aee71d232d685c3`。
- 评分协议：v3；标准 codec SHA256 `7fb21539c0556d10c77859e3e9ffb3d50dd3a2d0240e1cbb0923455e21bd6d3f`。
- calibration 来自 train，test 来自 validation；每个窗口来自一个文档，禁止环形重复、窗口重叠和跨 split 文档复用。
- Qwen2.5-0.5B（GQA、RoPE、SwiGLU）承担主排序；其他模型只作为软 guardrail，缺失或轻微回退不会覆盖 Qwen 主分。
- 模型状态：

| 模型 | 状态 | 层数 | hidden | heads / kv-heads | 数据来源 | 说明 |
|---|---|---:|---:|---:|---|---|
| qwen2.5-0.5b | loaded | 24 | 896 | 14 / 2 | cache | qwen2 |

## 候选在各模型上的结果

每个 native case 先计算 `(MSE_STD-MSE_PLAYER)/MSE_STD`。`official_flow_total` 保留原始 case 求和；主排序使用 `panel_score.total = 250*Linear_mean + 200*Attention_mean`，因此不会因模型层数或本地窗口数不同而放大。global-MSE 和组件均值只保留为诊断。

| 模型 | 候选 | Native total | Panel total | Panel Linear | Panel Attention | Source cases (L/A) | API time(s) |
|---|---|---:|---:|---:|---:|---:|---:|
| qwen2.5-0.5b | v117-l6c-g64-hierarchy | 423.227671 | 295.785829 | 127.377932 | 168.407898 | 672/96 | 2019.475 |

## 与官方锚点的排序审计

本次运行评测的是自定义候选，没有把官方分数传入候选或当次拟合；本报告只输出官方流程代理总分，不执行官方绝对分数回归。

### 时间与有效性预筛

| 候选 | 已评模型 | 主模型 API 时间(s) | 主模型 <420s | 软 guardrail 完整 | 本地提交有效 |
|---|---:|---:|---|---|---|
| v117-l6c-g64-hierarchy | 1/1 | 2019.475 | False | True | False |

## 解释与使用边界

1. 默认候选晋级看 Qwen 主模型的 `primary_panel_score_total`；Linear/Attention 目标权重固定为 250/200。其他模型的 panel 均值只作软 guardrail 和回归诊断。
2. `official_flow_total` 仍完整保留，便于和旧报告逐位对比，但不再因模型层数或本地窗口数量差异直接主导排序。
3. 本地数据不是官方隐藏数据，因此 shaped panel 只能用于相对排序；官方锚点只用于事后审计排序一致率，不能把 panel 分数线性换算成 Official Score。
4. `synthetic_attention_eval.py` 不由本套件调用；它只能做接口/性质测试，不能用于候选排名。
5. `cache_mode=read` 时本次结果只来自已保存的模型前向快照，不加载 tokenizer/model，也不读取网络；`cache_mode=write` 才会刷新快照。
6. 本地时间按每个模型代理的六个正式 API 调用累计；主模型必须严格小于 420 秒，多模型代理时间不相加。
