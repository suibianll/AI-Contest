# A3：transform 剥离对照——本地 Linear 增益来源（2026-09-02）

> 方法：从 v158（Linear 字段级 = v86）制作 `LOCAL ATTRIBUTION CONTROL` 变体
> `artifacts/official_eval/v158-transform-off-solution.py`：切断全部等价变换候选
> （diagonal SmoothQuant alpha、hierarchy/single-side permutation、\_PERMUTATION\_BASES、
> block-Hadamard 4/8/16、C75 尺寸扩展、CAT64 grouping、CAT64 transform），
> **保留全部量化 refine**（offsets/importance/gram8,16,64、group8/16、blocks64/cross64、
> C45 product selector、JDRQ）。
> 协议：`proxy-v2`，2 折 Linear 校准，`--linear-only` default-panel（168 cases），同一 cache/CUDA。
> 文件：`artifacts/official_eval/a3-*-lin.json`。

## 1. 结果（2 折口径）

| 版本                     |          linear\_mean |      q |      k |      v |      o | fc\_gate | fc\_up |   proj |
| ---------------------- | --------------------: | -----: | -----: | -----: | -----: | -------: | -----: | -----: |
| v158 parent（=v86）      |          **0.448180** | 0.5831 | 0.6020 | 0.6166 | 0.3614 |   0.4092 | 0.4098 | 0.1551 |
| v158 **transform-off** | **0.321107**（−0.1271） | 0.3990 | 0.5285 | 0.4644 | 0.2736 |   0.2998 | 0.2432 | 0.0393 |
| v86                    |              0.448180 | 0.5831 | 0.6020 | 0.6166 | 0.3614 |   0.4092 | 0.4098 | 0.1551 |

role 回退幅度：q **−0.184** > fc\_up −0.167 > v −0.152 > proj −0.116 > fc\_gate −0.109 > o −0.088 > k −0.074。

## 2. 判读

1. **v158 parent 与 v86 完全一致**（0.448180，逐 role 相同）——再次确认 v158 Linear 冻结；
   2 折口径下本地 Linear 基线是 0.448（5 折口径是 0.519，不可混用）。
2. **本地 Linear 的增益几乎全部来自等价变换族**：去掉变换后，即使保留全部量化 refine，
   linear\_mean 从 0.448 掉到 0.321（−28%），七大 role 全部显著回退、无一幸免。
3. **q/k/v 最依赖变换**（q −0.184 / v −0.152 / k −0.074，QKV 平均 −0.137 高于 fc−0.138、
   o −0.088、proj −0.116 的同类水平）：本地 q/k/v 高分的主体 = 变换族贡献。
4. **与 A1/A2 闭环**：

   - A1：本地 q/k/v 抬升最大（−0.63 负相关最强）→ A3：该抬升正来自变换族；

   - 官方 10 次提交已验证变换族增量（ROAB `+123` 不可移植、permu `+0.0001` 不迁移、
     stored-scale 不迁移、本地 0.58 系官方反而不认）→ **本地 Linear 高分的可迁移性极低，
     其主体正是“变换族在本地结构上的伪收益”**。
5. 纯量化 refine 的真实本地水平 ≈ **0.321**（变换全关）；它相对标准 HiF4 编码仍为正增益，
   但与官方无对照，不能由本地推断。

## 3. 策略含义（更新 A3 之后的方向）

- **本地 Linear 分数不再作为任何晋级依据**：其组成 = 变换族伪收益（大头）+ 量化 refine（小头），
  两者都未在官方取得一致证据；

- 量化 refine 族（offsets/importance/GPTQ/JDRQ）是无变换的、结构较无关改进，仍是官方验证的
  合理候选对象，但必须走官方单变量回传；

- 保持既有路线：Attention 为已验证同序面；Linear 侧只做结构+闭式算法（block-Schur GPTQ），
  且以官方裁决为准。

## 4. 资产

- 控制文件：`artifacts/official_eval/v158-transform-off-solution.py`（头部明确
  `LOCAL ATTRIBUTION CONTROL`，不作为正式版本/提交包）；

- JSON：`a3-v158-parent-lin.json`、`a3-v158-transform-off-lin.json`、`a3-v86-lin.json`；

- report：`logs/official_eval/a3-*-lin.md`。

