# W0 probe（P3-C）

- 状态：**PROBE_COMPLETE（本地 control 通过，待官方回传）**
- 构造：Linear 形状桶 W0（rows<=256，本地 k/v 权重）保持 v160 Linear；其余桶 standard。Attention standard（v163 基）。
- 源码：`./20260905_p3c_w0_linear-bucket_probe/solution.py`，SHA256 `8408647A1F3CCA15...`
- 归档基：v164（官方 13945/204s）→ A10/A01；v163（官方 4587/202s）→ W0-W3。
- 端点（复用官方，不重提）：A00 = v162 (1001/146s)；A11 = v164 (13945/204s)。
- 本地 control：桶路由：目标 8 state 非空(v160)，std 20 state 空且 weight_params == v162。0 失败。
- 时间论证：探针只把部分 API 替换为更便宜的 standard codec（或对目标桶保持原路径），最坏工作量 <= 归档基（A: v164 官方 204s；W: v163 官方 202s），官方时间预测余量 > 20s，安全（`T_pred < 280s` 满足）。
- 目的（计划 §7）：P3-C 官方分块贡献辨识；正负均保存，不自动替换根。

## 官方回传（2026-09-05）

- 官方：**1001 / 149s**，正常完成、无超时；状态 **PROBE_COMPLETE（正收益保存，已记录）**。
- C_W0 = 0（rows≤256 桶无 v160 Linear 收益）
- 判定总账：P3-A interaction I = S11−A10−A01+S00 = 13945−12010−2974+1001 = **−38**（≈0，
  Q/K 与 V 路径近似可加）；P3-C 残差 R = 3586−ΣC_Wj = **+1**（形状桶近似可加且全覆盖）。
  全部时间 < 210s，均低于归档基官方时间（上界论证成立）。
