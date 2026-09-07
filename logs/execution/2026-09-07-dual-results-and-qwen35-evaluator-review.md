# 双侧执行结果与Qwen3.5评测器迁移审查

本次为只读代码/产物审查、原始JSON零API重算和CPU面板元数据检查。
没有运行新模型捕获或候选GPU评测，没有修改用户正在编辑的评测器。
用户当前确认官方数据由Qwen3.5-35B生成；具体checkpoint/精度/抓取边界仍需官方说明确认。

## 1. 双侧计划完成情况

| 侧/卡 | 已有证据 | 本次结论 |
|---|---|---|
| Attention A22-1 | 官方14291/273s，SHA25310c6e… | 相对R3 −114/+35s；已按官方关闭 |
| Attention A22-2 | 官方14424/271s，SHA4686ad81… | 相对R3 +19/+33s，有官方增益；仍比A2低16分、快3s |
| Linear L21-1 | 数学条件求解PASS，区别于旧JDRQ；三个真实state探针负向 | 不是完整六片/官方验证完成，不能证明目标天然不可迁移 |
| Linear L21-2 | v166删除对照、三state拟合；另有低维rb8候选shard0 | 有进展，但实现/登记/状态未一致，且负向L1裁决错误 |

Attention两份归档源码SHA已与manifest核对。A2仍高分对照，R3仍低成本对照，
A22-2是残余scale研究线；19分收益不足以弥补当前巨大差距，不宜继续无条件堆校准训练。
时间状态里的“271距280预测门9s”仍错误：271为官方实测，只能直接算距300硬限29s。
近期A21-1/A22-1/A22-2预测分别低估约24.72/33.15/35.68秒，成本模型存在本路线系统偏差的迹象，
不是由这三个点就能推导可靠的新公式。

### Linear错误裁决的精确重算

候选原始JSON：`artifacts/proxy_v3/continuous/linear/anchor21-l1/smoke-check/candidate/candidate-linear-shard0.json`。
父JSON：`artifacts/proxy_v3/v162-independent/linear/l4-v189-linear-exact/id/v162-linear-l4-v189-linear-exact/candidate-linear-shard0.json`。
候选计分SHA `7cfc0a031a70bc8b6bedf98ef4881764349abb1a62d8c1469a38fc58682cf094`，父ACB16F76…。
按layer/role/window/split/length匹配56case，逐项校验STD MSE与reference energy一致：

| 指标 | 精确值 |
|---|---:|
| Δmean | −0.0178795174405598 |
| 平均负向L1 | **0.018546532828210376** |
| 总L1 | 0.019213548215860955 |
| 最坏case Δ | −0.05727867588413377 |

平均负向L1小于0.02，旧报告用最坏case大于0.02推断平均超门不成立。
这不意味着候选有效或已满足全部官方门，只表示“因负向L1超门停跑”的理由错误。
低维rb8候选也不是原始L21-1逐列卡，不能自动继承原卡的有限官方探索授权。
必须先对齐机制登记、候选SHA和完整验证状态，再决定后续；不因本次纠错直接补交官方。

另有三项证据缺陷：

- `probe_real_loop.py`实际将全部calibration folds拼接训练，再用fold1计算L_val；
  fold1不是独立验证。报告“只用fold0学习、fold1独立”的表述与实现冲突。
- L21-2 config声明删除rank1/rank2且保留actorder，但报告实测精简父是v166（rank1、无rank2/actorder）；
  删除了哪些组件需要按实际源码重新登记，不能称原配置完整复现。
- 三个代表state负向不能推出“拟合不可迁移是目标本身性质”；
  “官方与校准同分布、本地holdout是OOD”也不是21071分事实所能证明的。

## 2. 换模型方向合理，但4B只是代理

官方模型发布配置支持以下结构对照：

| 属性 | Qwen3.5-35B-A3B | Qwen3.5-4B |
|---|---|---|
| hidden | 2048 | 2560 |
| full attention | 16Q/2KV，head_dim256 | 16Q/4KV，head_dim256 |
| 层数 | 40 | 32 |
| FFN | MoE，256专家/每token选8，专家宽512 | dense，intermediate9216 |

来源：[35B配置](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/raw/main/config.json)、
[4B配置](https://huggingface.co/Qwen/Qwen3.5-4B/raw/main/config.json)。

4B更适合检查同族大head_dim/混合结构，不能代表35B的MoE专家权重和路由激活分布。
Linear尤其不能仅将dense FFN改名后视为专家场景；Attention也仍有GQA比差异。
仅知道数据来自该模型，不等于知道官方采样层、专家、QKV抓取点、mask、源精度或权重。
官方算子契约若仍为普通softmax(QKᵀ)V，就保持该计分，不因模型有gate/DeltaNet而擅自改评分函数。

## 3. 当前评测器实现的阻断问题

以下针对审查时正在编辑的代码快照，可能随后由用户修改；不是已经生成缓存必然受污染的断言。

### P0：跨层hook覆盖，可能产生形状正确但来源错误的数据

`evaluator/capture_qwen35.py`的`_pre/_post`写`captured[key]`，所有层注册相同
`hidden_in/o_in/fc_gate_in/proj_in/qn/kn/vn`键。整模型forward结束后再循环各层读取，
只能取得最后一次相应hook输出。权重来自各层，而输入/QKV可能统一来自最后捕获层。

修复：按`(真实model_layer_id, role)`隔离捕获，记录实际模块路径和行切片。
验收应由独立层的实际模块输入/输出对照`XWᵀ`，不能两份脚本共用同一错误捕获字典。
修复后旧快照生成的数据必须有不同捕获身份，不允许沿用旧候选state缓存。

### P0：裁剪test QKV与加载器契约冲突

抓取器只保存test窗口1/2/6/7的QKV，其余为None；
`official_eval.py::validate_qkv_bank`只跳过非Attention层，未跳过声明裁剪的test窗口。
因此按此契约生成的pack会在其余窗口的full-attention层被拒绝。
修复时区分calibration/test各自所需槽位；缺失必须明确标未捕获，不用零张量填充。

### P1：首次运行与零API重放的coverage不一致

`eval_system.py::_panel_geometry`要求metadata含`total_layers`；capture的主metadata没有该字段，
只有sidecar有。raw已加载时不会读sidecar，于是返回None并退回旧24个Attention层计数。
本地CPU函数检查复现：缺total_layers返回None；补24后新六shard预期为12个Attention case。
应从唯一面板描述/raw.layers得到结构，fresh与reuse两种路径使用同一规则。

### P1：缓存身份不足以隔离捕获器修复

calibration identity包含model_revision、语料hash和几何，但没有真实模型层映射、
权重/捕获版本、dtype与归一化/RoPE边界等身份；model_revision又是固定下载日期字符串。
同文件名下修复hook或调整层映射后仍可能命中旧state。
结果JSON复用主要检查路径、candidate SHA、shard等，也不足以判别同路径dense内容变化。

应统一使用capture fingerprint：checkpoint revision/权重身份、tokenizer和窗口token身份、
模块/层/专家映射、源与存储dtype、QKV边界、codec、捕获器与evaluator版本。
dense、NVFP4 carrier、calibration state、结果JSON都绑定该身份，生成时计算一次摘要。
不是只换cache文件名；读取应拒绝不匹配的旧身份。

## 4. 面板设计需要改进的部分

1. **固定24层×7role是旧模型布局**。4B compact可以做快速诊断，但6个FA层×2窗口仅12个Attention
   case，不能用48case或250case的名义描述。建议manifest逐case列出真实层/模块/长度/形状/来源；
   为FA增加深度和短/长窗口覆盖，优先覆盖已知官方实际形状，不按旧shard余数设计语义。
2. **门控投影并非“不是Linear”**。in_proj_z/b/a等在数学上属于线性映射；是否进入赛题应由
   官方提取范围与64块契约决定。q_proj截query半而不测gate半同样是面板取样假设，需明确记录。
3. **FP16/BF16不能声称与官方无损等价**。代码CUDA选FP16、CPU选BF16，metadata固定写float16；
   `_cpu16`实际保留输入dtype。应保存实际dtype与compute dtype，核对权重加载完整性、有限值、
   截断与溢出。将同一个FP16值扩大到FP32精确，不代表FP16前向与BF16前向等价。
4. **旧本地否定结论不自动迁移**。真实官方分数和算法合法性保持有效；旧0.5B上“某桶饱和/某目标
   不能泛化”等结论需降级。新4B分数与旧0.5B分数不能混排。
5. **时间模型不能输入4B秒数**。若保留旧default成本锚，必须fresh六API实测，不能用reuse缓存校准时间。
   若全面停用旧模型，旧时间公式也必须停用，先记录新面板成本与官方回传；不能移植系数。
6. 新4B工作包又写回通用总L1门，与持续计划的负向L1政策冲突。捕获协议迁移与门禁规则分开登记，
   不因换模型重新引入已经纠正的误拦。OOD可按最新明确授权精简，但不能把没有OOD数据写成风险已通过。

## 5. 建议执行顺序

1. **先修数据正确性**：分层hook、QKV槽位、coverage、缓存身份；做两层独立来源验证和新旧cache拒绝测试。
   不需要先跑整个4B模型才发现这些错误。
2. **固定官方算子边界**：确认具体checkpoint/BF16或FP8、Q/K的norm/RoPE位置、MoE专家覆盖、长度和mask。
   未知项在manifest写UNKNOWN，不由模型config反推评测协议。
3. **建立两级数据**：4B用于同族快速筛查；资源允许时一次抓取真实35B数据到独立高保真面板，
   本地只重放算子。可在有足够资源的机器捕获后转移数据，不能用4B权重切片冒充35B专家。
4. **新面板先测固定锚**：标准、L4、R3、A2、A22-2，各侧严格隔离；A22-1仅在需要验证排序时作
   已有负向对照，不把全部历史版本重跑一遍。所有父子必须同capture fingerprint。
5. **再续两侧算法**：Linear先纠正登记/负向L1/验证划分，再评估真实部署A@W求解；
   Attention先检查head_dim256/GQA2的变换开销与效果，保留父的残余变换只作已有有效起点。
   新模型正确性稳定前暂停用新proxy淘汰或晋级算法。

## 6. 快照与审查范围

审查时SHA256：

- capture_qwen35.py：393bd58e2add91758ef7c479444ec91457a582e1fd11851c6319a0d33b99315a
- eval_system.py：c37e514d93c6a903370e322e4918191f805ada025e771d779df58c7a8bbfe924
- official_eval.py：9fba6057c9506372115b5244c7dcf27fc1041e2ac4992f47f2d8f779ca25f8fe
- proxy_v3_eval.py：2f525f77d8bb49466f8b54cf4799a51698d6042483198395aea07d3183481bc5

评测器仍在并发编辑，RoPE先搬CPU造成device冲突的早期代码已在审查中修正，
因此不将它列为最终未修复问题。其他发现应按上述快照复核。
本次不覆盖当前state/计划或原始报告，避免与正在执行的模型迁移任务冲突；本审查作为修订证据。
