# E1 Progressive Full-Hierarchy HSDQ — rejected

日期：2026-08-30  
状态：`archived-rejected`  通过 E0-G 后执行的一次性诊断实验；主代码已恢复到 parent。

## 目的

验证固定 hierarchy 是否是当前 Linear 短板，并检查把 E6M2 顶层 scale、lv2/lv3
和 mantissa 一起做 progressive product-loss 优化后，收益能否从一层迁移到完整模型。

## 可复现输入

- 模型：Qwen2.5-0.5B；CPU；`--cache-mode read`。
- parent source SHA256：
  `5D1128CC79FEF58154DA2F600EC4B472FF95030E1F1E61B96593D06FD9AAC94F`。
- E1 source SHA256：
  `499FAF5B390A27962B5466111BFF42A4145DABE47055B5C74D61D962CDBC002A`。
- 算法只在非 expansive、`channels >= 512` 的 Linear role 运行；每个 block 枚举
  `scale code ±3`、合法 lv2/lv3 层级，再做一轮坐标 mantissa polish；两折
  cross-fit admission 保留 parent。

## 结果

| 范围 | 方案 | Linear mean | Attention | panel | API time |
| --- | --- | ---: | ---: | ---: | ---: |
| layer-1 | parent | 0.603071 | 0.926339 | 336.035344 | — |
| layer-1 | E1 | 0.613438 | 0.926339 | 338.627176 | 28.25s |
| 24 layers | parent | 0.501558 | 0.841829 | 293.755106 | 382.15s |
| 24 layers | E1 | 0.490233 | 0.841829 | 290.923906 | 693.21s |

角色级全层 Linear：

| role | parent | E1 | Δ |
| --- | ---: | ---: | ---: |
| q | 0.616561 | 0.610400 | −0.006161 |
| k | 0.620526 | 0.614121 | −0.006405 |
| v | 0.563596 | 0.546157 | −0.017440 |
| o | 0.483463 | 0.509730 | +0.026267 |
| fc_gate | 0.375126 | 0.375126 | 0 |
| fc_up | 0.430255 | 0.430255 | 0 |
| proj | 0.421376 | 0.345841 | −0.075535 |

## 裁决

E1 拒绝进入主线：一层收益不能迁移到完整 24 层，`q/v/proj` 中存在明显回退，
panel 下降 `2.831200`，且 API 时间增加 `311.06s` 并违反 420s 门禁。渐进式
full-hierarchy 不再扩大；下一实验转 A2，只对 expansive `fc_gate/fc_up` 的少量
高 leverage 行复用已验证 fixed-hierarchy HSDQ。
