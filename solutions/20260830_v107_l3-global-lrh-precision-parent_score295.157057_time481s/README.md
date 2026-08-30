# v107 L3 Global Activation-LRH（精度 parent）

日期：2026-08-31（实验在 2026-08-30 夜间启动）  
父版本：v106 L2 expansive-FFN CAT balance  
候选：修复 v095 的 Global Activation-LRH gate。低秩 Gram 残差只负责生成
候选；最终离散激活参数以部署量化权重 `G_q=W_q.T@W_q` 做逐行
`tr(E G_q E.T)` 门控。输入宽度超过 1024 的层不生成 global-LRH state。

## 结果

固定 Qwen2.5-0.5B cache、`qwen-official` panel、CPU、2 calibration / 4 test：

| 版本 | Linear mean | Attention mean | panel | native total | API time | 结论 |
|---|---:|---:|---:|---:|---:|---|
| v106 parent | 0.5034589422 | 0.8420394885 | 294.272633 | 419.160200 | 412.654599s | 精度/时间 parent |
| v107 4-block | **0.5069966356** | 0.8420394885 | **295.157057** | **421.537530** | 481.036527s | 当前精度最高；只看精度时接受 |
| v107b1 | 0.5043033601 | 0.8420394885 | 294.483738 | 419.727649 | 446.290123s | 1-block 对照，不作为 parent |

4-block 版本相对 v106：Linear `+0.0035376934`，panel `+0.8844233`。
按当前用户指令暂不以时间否决；`481.036527s` 仍须在最终冻结阶段压缩。

## 门禁证据

- 五层七角色 4-block screen：`both_player=0.52894931`；1-block 对照为
  `0.52610054`。
- calibration-only 4-block：6684 行提案、4900 行 Gram 接受，接受率
  `0.733094`；Gram/MSE 冲突率 `0.567475`；两折均接受的 case 为 30/35。
- calibration-only 1-block：3869 行提案、1779 行 Gram 接受，接受率
  `0.459809`；冲突率 `0.413027`。
- 合成、静态/运行时合规与 L3 单元测试：`18 passed`（另有既有 15 项
  Linear 合规测试通过）。

## 复现

```powershell
.\.venv\Scripts\python.exe evaluator\linear_candidate_screen.py `
  --cache artifacts\real_model_suite\cache\qwen2.5-0.5b__seq128__calib2__test4__layersall__schema1.pt `
  --solution solution.py --layers 0 5 11 17 23 `
  --roles q k v o fc_gate fc_up proj --stage L3 `
  --output artifacts\real_model_suite\l3-global-lrh-stratified-qwen.json `
  --report logs\execution\2026-08-30-l3-global-lrh-stratified.md

.\.venv\Scripts\python.exe evaluator\activation_lrh_diagnostic.py `
  --cache artifacts\real_model_suite\cache\qwen2.5-0.5b__seq128__calib2__test4__layersall__schema1.pt `
  --solution solution.py --layers 0 5 11 17 23 `
  --roles q k v o fc_gate fc_up proj `
  --output artifacts\real_model_suite\l3-global-lrh-diagnostic-qwen.json `
  --report logs\execution\2026-08-30-l3-global-lrh-diagnostic.md

.\.venv\Scripts\python.exe evaluator\real_model_suite.py `
  --models qwen2.5-0.5b --primary-model qwen2.5-0.5b `
  --panel-profile qwen-official --device cpu --algorithm-device cpu `
  --cache-mode read --solution solution.py --candidate-name v107-l3-global-lrh `
  --output artifacts\real_model_suite\v107-l3-global-lrh-qwen-full.json `
  --report logs\execution\2026-08-30-v107-l3-global-lrh-qwen-full.md
```

本目录同时保留 4-block 与 1-block 的 JSON/报告，便于后续 L4 以当前精度
parent 为基准；`solution.py` 是归档时的 4-block 精度版本。
