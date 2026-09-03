# v180 候选：A1 Q/K 非对称折叠分配（计划 D1）

> 状态：**RETAINED — 官方 `17597/242s`，相对 v175 `+3/−3s`，成为新完整官方父版本；
> GPT-2 轻微负的 model-specific-risk 保留为归因记录**
>
> 构造：基于 v175（新完整官方父，`17594/245s`）Attention 侧；只改 A1 的
> per-KV-head gain 折叠方式（对称 → 非对称 alpha=0.3），Linear 与 v175 逐位一致。
>
> 候选 SHA256：`2BA401228CACC49FADC7C78AC388616F490F3DC31CECA98F1DDE53C64EBF8AA3`
>
> 官方结果：`17597 / 242s`（用户回传，2026-09-04）

## 1. 机制（计划 2026-09-04 post-official A1-freedom D1）

- A1（官方正向 +60）把 per-KV-head 的 `gamma` 以 `sqrt(gamma)` 对称折叠进 Q/K
  multiplier。D1 改为非对称：`q_mult *= sqrt(g)^(1-alpha)`、`k_mult *=
  sqrt(g)^(1+alpha)`，指数和为 1 → 连续域 `QK^T` logits 缩放恒为 `gamma`（任意
  alpha 与 A1 等价），仅重分配 Q/K 各自量化动态范围。
- alpha=0 时与 v175 逐位一致（已实测 max |Δmultiplier|=0.0）；alpha=0.3 为本预注册
  单配置门禁（非邻域扫参）。
- 动态零新增：只改 state 中 Q/K multiplier 值；v165 约束满足。

## 2. 本地验证（描述性；官方裁决）

| 项目 | 结果 |
| --- | --- |
| 隔离导入 + 六 API | OK（py_compile + 评测器加载） |
| alpha=0 与 v175 逐位一致 | q/k multiplier max abs diff = 0.0（机制正确） |
| attention compact 4（配对 v175） | **mean Δgain +0.000088**、median +0.000249、3+/1−/0= |
| attention default 120（配对 v168 同口径） | **mean Δgain +0.000356**、median +0.000125、69+/51−/0=（win 0.575）；QK-only +0.00051、QK interaction +0.01106 正向；L16 +0.0078/L11 +0.0016/L12 +0.0016 最强 |
| control | V 侧 v_only_gain = 0.0 未改动；Linear 未执行 |
| gpt2 attn compact 4（配对 v168） | **mean Δgain −0.008984**、median −0.010140、1+/3−/0= → 标记 model-specific-risk |
| API 时间 | attention default 校准 ~60s、动态 Q/K/V 3.4s；零新增在线算子，无时间风险 |

**判读**：Qwen default 120 上 win 0.575、QK interaction +0.01106 正向——非对称折叠
重分配 Q/K 量化动态范围，使 QK 交互误差减小，收益来源与 A1 机制正交（A1 定 gamma，
D1 定 Q/K 分配）。跨模型 GPT-2 轻微负，按计划第 7 步不据此调参/路由，仍由首次官方
结果裁决。

## 3. 判读（计划 D1 预注册）

```text
step_gain = S(v180) − S(v175) = S(v180) − 17594（组合到 v175 的增量）
可加性    = interaction 已证 0，单侧机制组合无惩罚
```

官方结果为 `17597/242s`：`step_gain = 17597 − 17594 = +3`，D1 晋级，v180 成为新的
完整官方父版本。D1 没有新增在线算子，因此相对 v175 的 `−3s` 只登记为官方实测，
不宣称为稳定速度收益。不调 alpha 重扫（预注册单配置门禁）。

v180 是完整组合版本而非独立 Attention 单侧测量；按 v175 已验证的可加性可推导
`S_A(v180) = 17597 − 4590 + 1001 = 14008`，但该值仅用于归因，不登记为官方单侧结果。

## 4. 复现

```powershell
.venv\Scripts\python.exe -u evaluator\official_eval.py --solution solutions\20260904_v180_a1-asym-fold-attn_scoreNA_timeNA\solution.py --attention-only --compact-panel --cache-mode read --nvfp4-cache-mode auto --capture-device cuda --algorithm-device cuda --baseline-json artifacts\official_eval\v175-compact-attn.json --output artifacts\official_eval\v180-compact-attn.json --report logs\official_eval\v180-compact-attn.md

.venv\Scripts\python.exe -u evaluator\official_eval.py --solution solutions\20260904_v180_a1-asym-fold-attn_scoreNA_timeNA\solution.py --attention-only --cache-mode read --nvfp4-cache-mode auto --capture-device cuda --algorithm-device cuda --baseline-json artifacts\official_eval\v168-attn-default.json --output artifacts\official_eval\v180-attn-default.json --report logs\official_eval\v180-attn-default.md

.venv\Scripts\python.exe -u evaluator\cross_model_eval.py --model gpt2 --solution solutions\20260904_v180_a1-asym-fold-attn_scoreNA_timeNA\solution.py --name v180 --attention-only --compact-panel --cache-mode read --capture-device cuda --algorithm-device cuda --baseline-json artifacts\official_eval\v168-gpt2-attn-compact.json --output artifacts\official_eval\v180-gpt2-attn.json --report logs\official_eval\v180-gpt2-attn.md
```
