# v162_attention R2b：栈内旋转官方 WA 加固版

> 日期：2026-09-06。父：R2（`0B56CCA1...9C7C`，官方 wrong answer）。
> 机制与 R2 一致；OOD 门裁决维持 REJECTED，本版仅用于官方 WA 根因定位与加固验证。

## 加固内容（同 A2b + R2 专属）

1. v189 校准调用移入总 `try`，`except Exception` → 逐位回退 R1 行为（官方 14009 已验证合法）；
2. 训练气泡（inference-off/grad-on + normal 张量重建）；
3. `_nvfp4_to_hif4` 旋转注入 `except Exception` → 无旋转继续（合法输出）；
4. state 注入失败时回退：删除 `learned_rotation` 键 = 精确 R1 state。

## 验证

- 官方契约模糊测试 CLEAN；
- screen（shards 0,2,3,5）mean **+0.776965 = R2 逐位一致**（加固零精度损失）；
- 32 步完整训练确认（修复了中间版本缩进手术导致的 1 步早退 bug，未入库官方链）。
- **solution.py SHA256：`58B1214F076E9EF0D1F45DA63398EEB05A52FD7B9B3C1745F45B5784FCD57A8A`**

## 官方

- unregistered / NA（待重提交）。最坏情形 = R1 行为（不会 WA）。
- 注意：OOD 门未过（Δgap +0.0153 > 0.01）依然成立——本版用于 WA 根因确认；
  若官方给出分数即证明 WA 根因 = 未设防异常，R2 机制本身仍受 OOD 门约束不晋级。
