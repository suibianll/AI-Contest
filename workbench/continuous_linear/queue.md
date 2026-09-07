# Linear 当前机制队列

更新2026-09-07。当前执行入口：[Linear独立工作包](../../docs/superpowers/plans/workpackages/linear-output-followthrough.md)。
父L4官方4607/247s，冻结v162 standard Attention，尚无新候选待官方。

| 顺序 | 卡 | 状态 |
|---|---|---|
| 1 | L21-1 真实A@W固定层级逐列条件求解 | READY；一次state、64块单遍、合法码、真实输出验证 |
| 2 | L21-2 精简输出拟合主干 | 首卡state契约或成本受阻后登记；精简损失与拟合收益分开 |
| 3 | L21-3 合法层级联合输出求解 | 首卡有效或有直接机制证据后登记，不扫旧offset |

保留的具体历史结果：正交特征基5state退步、Kronecker4state两退两平、旧全栈训练成本过高、旧JDRQ实际负结果。
上述结果不提供以下结论：全部连续变换饱和、所有合法mantissa更新无路、仅fc/proj隐藏桶有效。
旧GPTAQ/JDRQ去重没有完成真正逐列条件约束求解对照，不能关闭当前首卡。

Attention同时按A22-1推进，GPU共用锁、候选与控制侧分离，分别官方确认后才组合。
官方探索沿用L21-1既有单代表规则，详细验收、公式、复现顺序和失败分支见独立工作包。
