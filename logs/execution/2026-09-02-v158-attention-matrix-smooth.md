# v158 Attention Matrix-Smooth

v158 从 exact v86 分支，Linear 与 V 冻结，只为 Q/K 增加解析的 GQA 组内 2×2 等价变换。
连续域 Q/K 点积不变，部署时只增加固定 O(TC) 2×2 乘法；校准使用交叉窗口拟合/门控。

本地 effect-panel 相对 v86：Linear `0/0/56` 严格 no-effect；Attention `1/0/4`，mean delta
`+0.007194699`。default-panel：Linear `0/0/168`；Attention `49/16/55`，mean delta
`+0.011017609`，属于 mixed。本地结果仅用于机制归因，不预测官方排序。用户明确要求先保留
并推送供官方评测，因此状态为 `RETAINED / official pending`，不按本地 mixed 拒绝。

源码 SHA256：`18F9DE037A29AD96EE06FB5C73095E9AD36D0D04DA2953162181BE3AEA528277`。
官方 score/time：`unregistered / NA`。

流程修正：后续 Linear 实验只执行 Linear 校准/计分，Attention 实验只执行 Attention
校准/计分；不再为单侧机制重复无关侧的完整调用图。本轮已产生的全 panel 仅作为一次性证据。
