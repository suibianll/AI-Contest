# L26：跨折叠一致性接受判定（REJECTED）

研究循环卡 L26，靶点 E4（部署失配 0.6143）。2026-09-08 实现并本地验证，**关闭**。

## 机制

L23b 接受判定用全校准行 L_all 均值下降。L26 要求**每个校准 fold 的 L 都严格下降**
才接受（跨折叠一致性），连续解与合法投影不变。假设：fold0（10 行）omega 过大
主导接受判定，导致对 fold1（部署 scale 分布相近）有害的块被接受；跨折叠一致性
可让写回更贴近部署。

与已关闭卡去重：L24-C 改拟合权重、L24-A 改求解残差、L25 改合法格点；L26 只改
接受判定逻辑（变量/插入点均不同）。

## 证据（4B shard0 paired，父 L4）

| 指标 | L23b | L26 |
|---|---|---|
| shard0 fit_gain（校准折叠） | 0.9478 | **0.8172** |
| 宽层 accepted | 144/144 | **49/144** |
| shard0 delta_mean（vs L4 面板） | -0.2182 | -0.1197（改善但仍负） |

**关闭原因**：跨折叠一致性以牺牲大量校准拟合（0.817<0.948，破坏 fit_gain≥0.9
目标）换取轻微部署改善（delta −0.120 vs −0.218，candidate gain 仍低于父）。
按卡证伪判据（"E4 未降 → 关闭该一致性实现"）**关闭**。

## 关闭粒度

只关闭"跨折叠一致性接受判定这一实现"。E4 格仍 OPEN。L23b（13639FB2）保持
READY_FOR_OFFICIAL，fit_gain 0.9486 ≥ 0.9 仍是最佳候选。

## 产物

- 源码：`workbench/continuous_linear/l26-crossfold-accept/candidate/solution.py`
  （SHA `1B04B9904C200B0A…`）
- 归档：`solutions/continuous_linear_l26-crossfold-accept_rejected/`
- 评测：`artifacts/proxy_v3/continuous/linear/l26-crossfold-accept/shard0/`