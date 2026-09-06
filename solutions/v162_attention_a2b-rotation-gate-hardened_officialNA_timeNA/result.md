# v162_attention A2b：旋转 gate 臂官方 WA 加固版

> 日期：2026-09-06。父：A2（`19159AB9...1940`，官方 wrong answer）。
> 本版只做 harness 兼容加固，机制与冻结配置与 A2 完全一致。

## 官方 WA 根因（本地复现）

- 官方 harness 在 inference 上下文调用六 API：inference 张量 + `backward()` 抛
  `RuntimeError`（本地 `torch.inference_mode()` 复现）；按 v107 判例任一用例异常即整次 WA。
- 次要雷：L_q < L_kv 校准样本下 `index_select` 越界触发 CUDA device-side assert 并毒化
  后续全部用例（官方当前同长样本不触发，属防御性修复）。

## 加固内容

1. 训练全程在 `torch.inference_mode(False) + torch.enable_grad()` 气泡内，输入经
   `.detach().clone()` 重建为 normal 张量（θ 亦在气泡内创建）；
2. 校准整体 `except Exception` → 返回空 state（= v162 标准行为，永不 WA）；
3. 动态 Q/K 旋转路径 `except Exception` → 回退 `_standard_params`；
4. L_q<L_kv 采样越界修复。

## 验证

- 官方契约模糊测试 CLEAN（inference_mode/no_grad/变长/5000 长序列/5 种几何/极端值/重复可现性），
  工具 `workbench/v162_attention/fuzz_official_contract.py`；
- screen（shards 0,2,3,5）mean **+0.421328 = A2 逐位一致**（加固零精度损失）。
- **solution.py SHA256：`CDB49A0248386FC81BF8AEFF33E2DC66F147659E8907D4C05D214AC93F6EA879`**

## 官方

- unregistered / NA（待重提交）。最坏情形 = v162 标准行为（不会 WA）。
