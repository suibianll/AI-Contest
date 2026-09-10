# A-GR1（一般非对称互逆矩阵残差）执行日志 — 2026-09-10

计划卡：`docs/superpowers/plans/parallel/2026-09-10-attention-general-reciprocal-plan.md`。
工作目录：`workbench/full_solution/attention-agr1-general-reciprocal/`。
父：v230 Linear (L-EM2) + v195 Attention 完整根，官方 `18428/292s`，
SHA `0f1af6dbc207ff32b2c6be16987e9c4fe50f3f10747de26782ef52a6f2fab7bc`（现场核验）。

## 实现

- 候选 = 根逐位复制 + 末尾追加 A-GR1 段（`build.py` 字节级追加，前缀与根逐字节一致，
  `cmp` 验证）。机制块源码 `agr1_block.py`。
- 候选 SHA256：`4f27fb59252de95c7d8a65db9d742af1bc2ce16cd7ca513d49e8fbc7813a9267`。
- v192 单变量推广：对称零迹 S → 一般 N（M = I + N），K 侧 M^{-T}；fit 0,1,2 /
  gate 3,4、32 步 Adam lr 0.01 clip 1.0 reg 1e-3、amax 尺度比目标、逐层全窗口严格
  改善门、center 同步编译，全部与 v192 相同；每步 M 奇异值钳 [1/√2, √2]。
  唯一实现差异：训练设备用 cuda-if-available（对齐当前根 A2；v192 的 CPU 训练是其
  完整包 TIMEOUT 的因素之一），不改数学定义。
- **过程缺陷与修正**：初版 K 侧解析梯度误写为 `−M⁻¹G_kM⁻¹`（缺两个转置），
  合成种子上训练损失不降反升（2.0→2.16）被 control 4 抓到；光滑损失的有限差分
  验证定位后修正为 `grad_N = G_q − P·G_k^T·P`（P = M^{-T}），修正后 FD worst
  relerr 5.3e-8，损失 2.0→~1.21。control 2 初版 logits einsum 形状错误（测试侧
  bug，非机制 bug），改为逐 head `ighd,jgd->ighj` 后不变性 1.3e-8。

## Control（`control_results.txt`，CPU 跑——GPU 当时被 Linear 会话占用）

1. 0 训练步：Q/K/V 五字段与最终输出与根逐位一致（PASS）。
2. 互逆性：编译对 `tq@tk^T` vs `rq@rk^T` 最大偏差 2.34e-7（fp64）；量化前 dense
   logits 最大偏差 1.30e-8 <1e-6（PASS）。
3. 六 API 脱离仓库 importlib 独立导入 PASS；`validate_state`、
   `validate_hif4_params` PASS。
4. V/Linear 与根逐位不变 PASS；可达性：8/8 合成种子 attempted=1 且训练损失下降
   （2.0→~1.21）；合成同分布数据上 0/8 接受（随机数据无结构可挖，属预期）。

## 本地评测（GPU 串行，后台轮询 <2000 MiB 后启动）

- shard0（接口检查）：status ok，reasonableness_issues 0；层 0 全 12 case
  delta=0（gate 判 parent）。api delta +0.98s。
- 六 shard（`--stop-after-nonpositive 6`）：records 6，无异常。

| shard | 层 | delta_mean | +/-/0 |
|---|---|---|---|
| 0 | 0 | 0.0 | 0/0/12 |
| 1 | 1 | 0.0 | 0/0/12 |
| 2 | 8 | 0.0 | 0/0/12 |
| 3 | 15 | **+0.017449** | 10/2/0 |
| 4 | 22 | **+0.005618** | 11/1/0 |
| 5 | 5 | 0.0 | 0/0/12 |

等权 mean **+0.003845**（21/3/48，72 case），非逐位相同 → 不是 NO_EFFECT。
Attention 侧 api total delta 约 +5.5s/六 shard（缓存口径混合，仅诊断记录）。

真实 4B 逐层 gate 审计（校准缓存读出）：attempted 6/6，accepted 2/6（层 15、22）；
层 0 gate 均值改善（4.0530e-4 < 4.0923e-4）但 v192 全窗口规则未过 → parent。
inverse_error ~1e-6。所有层训练损失 2.0 → 0.87~1.12（真实下降）。

## 侧隔离探针

- `workbench/standard_linear_attention_probes/build.py` CANDIDATES 加
  `standard-linear_v234-attn`（attention_source = v234 归档），build + verify 全 PASS
  （探针 SHA `b7dbc8d727481b9ef4ec1c94c05fc62fbf2f036c09ef59b8d6e1bc9c1e11c160`）。
- 探针 vs 归档候选 attention-only 六 shard 逐位配对：
  `artifacts/proxy_v3/attention-agr1-probe-pair-20260910`（结果见下方登记段）。

## 版本号

用户指定 v233，但归档前检查发现 `solutions/20260910_v233_linear-tf1-gradient-reuse_…`
已被并行 Linear 会话占用 → 按预定规则改用 **v234** 并同步全部引用。

## 归档与登记

- 归档：`solutions/20260910_v234_attention-agr1-general-reciprocal_scoreNA_timeNA/`
  （solution.py 与评测候选逐位一致；config.json、result.md、official-result.json、
  verification.json）。官方 `unregistered/NA`，用户统一评测。
- 登记：solutions/README.md、plans/README.md、current-solution-status.md。

## 官方回传（2026-09-10）

侧隔离探针 `standard-linear_v234-attn` 官方 **14455/263.7s**（用户回传）：相对 v195 侧基准
`14426/243s` 为 **+29/+20.7s**（相对 R3 基线 14405 为 +50）。A-GR1 官方侧价值 +29，
继 C76.4（+84）、A1（+60）后 Attention 第三大官方正向机制；表达力梯度
（对角 0 < 三角 0 < 对称全矩阵 +22 < 一般矩阵 +29）获官方确认。本地 +0.003845（2/6 层
gate 接受）与官方 +29 再次确认本地 proxy 不预测官方（校准在官方隐藏数据上重跑）。
完整包形态未提交；侧隔离分不直接晋级。登记见
`logs/execution/2026-09-10-v234-agr1-side-official.md`。
