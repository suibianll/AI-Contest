# A21-1 执行结果：可进入官方探索，尚无官方回传

2026-09-07。用户要求整理现状并执行 Attention 优化。本轮只运行一个预注册配置；
Linear 未优化，根 solution.py 仍为 v189。候选源码 SHA256：
`870d5848f95887307ad7faa6364b5d7f7480f5b7be6001c812c44ead02bdb48a`。

完整对照为 R3 14405/238s，源码 A5C679D7…；高分对照仍 A2 14440/274s。
外部21071/283s保持用户确认、源码未绑定，不继承为本候选成绩。

## 实现与隔离

用联合 Q/K 64块 scale 损失训练 T=R0 exp(S)，32步固定 Adam；
手工矩阵指数梯度，逐KV组对称零迹，cond≤2，动态只部署固定互逆矩阵。
旧R3输出旋转/center训练被移除，未叠加A2全长反传；V和标准Linear保持原路径。

执行前明确修正计划歧义：校准gate的回退为 **R3训练前栈**，独立评测父仍为完整R3。
否则生成完整R3回退仍需旧训练，与“替换旧训练”矛盾。该变化不是同父状态上的纯增量，
所以必须区分新变换收益与移除旧训练的损失。

## 同协议结果

以下均为本地误差指标，不是官方分数；48个case精确匹配身份、STD MSE及reference energy。

| 指标 | ID 六片 | OOD 六片 |
|---|---:|---:|
| R3 mean gain | 0.775557865 | 0.769095528 |
| 候选 mean gain | 0.776791845 | 0.766331460 |
| Δmean | +0.001233981 | −0.002764068 |
| 总 L1（记录） | 0.016404645 | 0.017170223 |
| 负向 L1 | 0.007585332 | 0.009967145 |
| 正 / 负 / 相同 case | 16 / 18 / 14 | 13 / 21 / 14 |
| test Δmean | +0.002411230 | −0.006991233 |
| validation Δmean | +0.000056731 | +0.001463097 |

ID最坏四分位Δmean −0.028346647，最差层5/18/13；验证/测试同号率（含零）75%。
OOD Δgap +0.003998049，不触发0.01风险提示，且OOD本来不作否决门。
validation提升很小，case仍有18负向，不能声称稳定大幅改善。
逐case、分组与源码/evaluator SHA见manifest.json。

## 机制是否有效到达

24/24层均执行32步；6/24层接受，分别为2、8、9、12、16、23。
训练各fold等权平均 Q/K scale 平方比为0.796117 / 0.772315，分母为同输入S=0，
这不是相对完整R3的scale比例，也不是原始scale下降百分比。
最大连续互逆误差2.563e-5；不是数学上逐位无损的承诺。

- 接受层的12个ID case：相对R3 Δmean **+0.022871972**。
- 回退层的36个ID case：相对R3 Δmean **−0.005978683**。
- 二者等权合成总Δmean +0.001233981。分组由校准gate决定，holdout不参与选择。
- 行去均值logits误差19改善/15退步，probability误差18/16，最终readout16/18，其余相同。

因此既有scale与输出失配，也有删除旧训练后回退到较弱栈的损失。
不能把整包的小增益写成互逆scale方向饱和，也不能把接受组的收益外推全部层或官方。
真实E6M2 scale、码变化、同坐标纯量化MSE、超过所选合法层级容量的元素比例，
以及logits/probability/readout数据见diagnostics.json；全部来自真实动态API与缓存state，未重新训练。

## 成本与检查

fresh default实际调用168次W校准、168次动态A、24次Attention校准、各120次动态Q/K/V。

| API | 秒 |
|---|---:|
| W calibration | 0.614950 |
| dynamic A | 1.514001 |
| Attention calibration | 75.714435 |
| dynamic Q | 1.305787 |
| dynamic K | 0.974316 |
| dynamic V | 0.722207 |

API total **80.845696s**，wall **107.767481s**；六API模型预测官方 **219.284164s <280s**。
两种本地时间都不是官方时间；模型仍有误差。fresh default Attention mean0.766631390，
此120case兼容面板不与48case eval-v3混排。
GPT-2 compact四case运行成功，mean0.434346011；仅记录，不提供晋级/否决依据。

验证通过：

- 手工指数梯度与完整scale目标梯度对照独立autograd；最大误差3.219e-6 / 1.014e-6。
- 四种GQA/MHA几何、不同Q/K长度、逐head独立参考、零块与amax并列极值。
- 真实L0/L23编码前坐标逐位一致；inference_mode训练32步可达。
- 全部eval合法state/五字段/有限输出；V在48个真实case上五字段逐位相同。
- 两个Linear API及共享依赖保持父源码前缀，fresh default168个Linear case gain严格为0。
- 脱离仓库工作目录、isolated Python单文件导入六API通过。

## 裁决与下一步

**READY_FOR_OFFICIAL_EXPLORATION / official unregistered/NA**。
独立归档至 `solutions/continuous_attention_anchor21-a1/solution.py`，不覆盖正式父。
原评测器某些分片报告仍使用总L1/单片正均值规则；最终裁决按本轮负向L1与完整六片规则，
见manifest，不能直接采用旧分片reject文本。

优先等待本候选官方回传；不得重复提交相同SHA。后继先解决回退损失：
设计保留R3已有有效变换的低成本结构，或按A21-2同预算联合学习正交与对称部分；
需要新的机制卡，不在本候选上扫步数/学习率/阈值。若保留旧变换后仍scale下降但readout不提升，
再按A21-3研究输出敏感度目标。两个问题分开验证，避免再次将替换损失归咎于新目标。

## 复现入口

CUDA venv运行本目录 `build.py` → `verify.py` → `check_math_and_import.py`；
`run.py screen` → `run.py full` → `run.py ood` → `run.py timing` → `run.py cross`；
最后 `diagnose.py` → `summarize.py`。已有同SHA完整结果直接复用，fresh timing脚本拒绝覆盖重跑。
原R3 JSON逐字节复制并校验身份，未调用旧父API重跑。
结果目录：`artifacts/proxy_v3/continuous/attention/anchor21-a1/`。

状态整理：更正A2的过期DOMINATED、R3仍最佳、OOD排序有效、274距硬限只有6s等错误；
删除当前入口的旧“未启动实验/下一步A0/A3”指令，原始归档结果与日志未覆盖。
两处pytest临时目录删除被自动审批策略拒绝，保留原处；未清理其他产物。
