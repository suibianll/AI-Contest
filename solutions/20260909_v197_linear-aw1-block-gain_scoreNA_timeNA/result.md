# v197 — linear-aw1-block-gain

## 状态

- 机制：64-block 共享标量增益的低维 A@W 拟合（设计文档 §4.1 方案 A）。根 Linear 校准产出部署
  state 后，用充分统计量（`H = Σ ω_f ÂᵀÂ`、`C = Σ ω_f ÂᵀY_f`，`ω_f = 1/(D_f+eps)`，官方
  MSE_STD 归一化口径）闭式 ridge 解出每 block 增益 `g*`，按 block 重编码 `g_b·Ŵ_b` 为合法
  五字段 state；二次型门控严格改善才接受，否则回退父。配置：fit windows=1、tokens=256、
  ridge 1e-3·mean(diag(G))、clamp [0.5, 2.0]。
- 父（当前根 v195）SHA256：`839adb1e617c3115c6b55071a34b281c5db0ff2aa070adbbc71fd1549e761d7f`
- 候选 SHA256：`01aa73f60a49651c4410875d2bde7e48804bb5a36ec483eb45a9d5c2b0e72742`
- 官方状态：不提交（本地灾难性负向）。

## 检查

- `verify.py`（合成数据）全 PASS：闭式解执行、充分统计量与显式逐 fold 重算相对误差 ~2.7e-6、
  改善数据接受 / 退化数据拒绝、control 与父逐位一致、合法 state、脱离仓库六 API 导入通过。

## 4B shard0（REJECTED）

- eval-v3、Qwen3.5-4B proxy-v2、CUDA、Linear-only、shard 0、calibration-cache-mode write：56 cases。
- candidate mean `0.3096915995`，父 mean `0.5173485923`；paired delta mean **−0.2076570**、
  median −0.2179643、+/-/0 = **0/56/0**，mse ratio **1.4507**。
- calibration API：父 `135.100658s` → 候选 `150.864245s`（+15.8s / 28 层）。
- 判定：本地 REJECTED。门控在拟合坐标内通过但真实部署输出全部恶化 ⇒ 门控的二次型评估与部署
  路径不一致，最可能是 block 索引错位（根存在 GPTQ 块重排路径，拟合用 Â/Ŵ 的块序与重编码
  部署的块序未对齐；合成验证未覆盖该路径）。机制（官方口径闭式拟合）本身未被证伪，
  此归档只关闭当前实现。

## 证据位置

- 归档：`solutions/20260909_v197_linear-aw1-block-gain_scoreNA_timeNA/`
- 评测：`artifacts/proxy_v3/full_solution/linear-aw1-block-gain-shard0/candidate/manifest.json`
- 工作脚本：`workbench/full_solution/linear-aw1-block-gain/`
