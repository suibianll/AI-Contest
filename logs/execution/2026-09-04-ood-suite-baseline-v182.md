# 2026-09-04 OOD 泛化套件基线（v182）

## 目的

落实 `docs/official-local-fitting-analysis-2026-09-04.md` §6/§13：本地校准与测试同源
WikiText-2，导致"本地 +0.06、官方 −1163"类失败（v140/v155/v156）本地不可检出。
新增 OOD 测试集后，用 `gain_in − gain_ood` 作为过拟合诊断指标。不消耗官方配额。

## 交付物

- 语料：`data/ood-suite-v1/`，三域各一份 parquet，单列 `text`：
  - `code`：site-packages（torch/transformers/numpy/pyarrow/sympy）60 个 ≥120 行 .py，
    哈希定序、截断 8000 字符，117,329 tokens；
  - `news`：ag_news 400 行，22,148 tokens；
  - `zh`：XLSum chinese_simplified 400 行（≥200 字符），273,165 tokens；
  - 生成器 `workbench/build_ood_corpus.py`（确定性；HF 走 hf-mirror）。
- 评测器：`evaluator/official_eval.py` 新增 `--ood` 模式：
  - 校准保持 WikiText（部署语义：算子校准在分布内、测试在分布外）；
  - 15 个 OOD 测试窗口（3 域 × 长度 10/128/512/1024/1024，逐域不重叠、哈希确定性）；
  - 面板 168 Linear（全层全 role，窗口轮换）+ 120 Attention（8 层深度铺开 × 15 窗口），
    case 数与 in-dist 默认面板一致；
  - `Window.split` 携带域名；`ood_summary` 按 overall/by_domain 汇总；
  - NVFP4 缓存 profile 仅在 OOD 模式加 `ood: True` 键 → 旧 in-dist 缓存 profile 不变、全部有效；
  - 报告：`artifacts/official_eval/ood.json` + `logs/official_eval/ood.md`。
- 测试：`tests/test_official_eval.py` 新增 5 个 OOD 用例，35/35 通过。

## 基线运行（根 solution.py = v182，SHA `F3E39E99...A438`）

命令：`.venv/Scripts/python.exe evaluator/official_eval.py --ood --solution solution.py --cache-mode auto`

- 数据：`data_source = model_forward_ood`；逐域 data_sha256 记录在 ood.json；
  OOD dense 缓存 `qwen2.5-0.5b-proxy-v2-ood.pt`、NVFP4 缓存
  `qwen2.5-0.5b-proxy-v2-ood-both-default-nvfp4.pt`（独立文件）。
- 首跑 8m38s（含捕获）；candidate API total 413.4s（本地 CUDA 墙钟，不与官方时间混用）。

| 侧 | 域 | cases | gain mean | median | worst-quartile | 正/负/零 |
|---|---|---:|---:|---:|---:|---|
| linear | code | 50 | 0.629836 | 0.681942 | 0.413607 | 49/1/0 |
| linear | news | 68 | 0.631807 | 0.622882 | 0.427753 | 68/0/0 |
| linear | zh | 50 | 0.596479 | 0.548550 | 0.425032 | 50/0/0 |
| linear | **overall** | 168 | **0.620706** | 0.620416 | 0.419926 | 167/1/0 |
| attention | code | 40 | 0.711659 | 0.733797 | 0.523692 | 40/0/0 |
| attention | news | 40 | 0.728499 | 0.744960 | 0.545776 | 40/0/0 |
| attention | zh | 40 | 0.723557 | 0.743679 | 0.549104 | 40/0/0 |
| attention | **overall** | 120 | **0.721239** | 0.741542 | 0.539414 | 120/0/0 |

## gain_in − gain_ood（对照 in-dist 基线 `v182-integration.json`，同 SHA、同机、同 panel）

| 侧 | gain_in | gain_ood | gap |
|---|---:|---:|---:|
| Linear | 0.636609 | 0.620706 | **+0.015903** |
| Attention | 0.741829 | 0.721239 | **+0.020590** |

- v182 父版本本身存在小幅分布内偏置（预期内，任何校准算子都有）；
- 候选判读必须用 **Δ(gain_in − gain_ood)**（候选两侧差值 − 父版本两侧差值，按候选配对），
  不直接比较绝对差值；
- zh 域 Linear 最低（0.596），符合语言迁移直觉；Attention 三域均匀（0.71~0.73）。

## 判读规则（登记）

1. OOD 面板只做过拟合诊断：`comparable_for_proxy_ranking = False`，不参与排名、不预测官方分；
2. in-dist 与 OOD 结果不得混排（不同 test 集合）；只有同 SHA 的两跑可以相减；
3. 候选若本地增益上升而 `gain_ood` 同步大幅下降 → 拟合型机制，按过拟合处理（对应
   v140/v155/v156 失败模式）；解析/结构等价机制预期 gap 与父版本相当；
4. 时间与配额：全流程本地 GPU，不消耗官方提交。

## 后续（未做，留待排期）

- §6 建议的标定重跑（v158/v160/v168/v171/v176 五版本的 OOD 差值）尚未执行；
  先由后续真实候选按规则 3 自然积累对照。
