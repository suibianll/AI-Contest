# v151 proj ROAB-off targeted role experiment

- 日期：2026-09-02
- 父版本：pre-A3 v147 fixed combination（v140 Linear + v86 Attention）
- 候选：只对 `rows < channels` 的 Linear 矩阵禁用 ROAB；同时移除额外 A3 residual pass，保持
  pre-A3 时间边界
- 候选源码 SHA256：`65577F422E0DB1AAC9D3E27EC1DAB4EC5501FB8A6804C9349159268966CF8D25`
- 父 workbench SHA256：`800CA10EC3414E4FE886B93CA62BD4A350D26BBA015287DF7E8DF2DD871AC23D`

## Canonical proxy-v2 targeted smoke

命令（两次分别替换 `--solution`）：

```powershell
.venv\Scripts\python.exe evaluator\official_eval.py `
  --solution <candidate.py> --name <name> `
  --cache artifacts/official_eval/cache/qwen2.5-0.5b-proxy-v2.pt `
  --cache-mode read --linear-cases 14 --attention-cases 1 `
  --output artifacts/official_eval/<name>-targeted.json `
  --report logs/official_eval/<name>-targeted.md
```

14 Linear cases cover two layers × seven roles；1 Attention case只作 v86 冻结回归。完整校准
调用图仍为 168 个 Weight state + 24 个 Attention state。

| Candidate | Linear | Attention | Overall | API total | Wall |
|---|---:|---:|---:|---:|---:|
| pre-A3 parent | 0.582528216 | 0.942927486 | 0.606554834 | 201.258s | 209.078s |
| v151 | 0.582528216 | 0.942927486 | 0.606554834 | 193.213s | 199.430s |

逐 role（两层）完全相同：

```text
q=.670757  k=.808461  v=.756484  o=.683818
fc_gate=.396959  fc_up=.368327  proj=.392891
```

结论：proj ROAB-off 在 Qwen targeted panel 是 no-op；约 8 秒本地差异不能当作 runtime 提升。

## External hif4 causal GPT-2 smoke

外部仓库：<https://github.com/youxilee/hif4>，commit
`dd5ee6515323169dbd4133b3d4fd1ff1cb7be646`。

```powershell
<repo-python> -u real_data_eval.py --layers 4 --seq 128 --calib 2 --test 2 --mode amax6 --config current
```

| Candidate | q | k | v | o | fc | proj | Attention |
|---|---:|---:|---:|---:|---:|---:|---:|
| pre-A3 parent | 0.6802 | 0.7443 | 0.6505 | 0.7032 | 0.5658 | 0.5029 | 0.4540 |
| v151 | 0.6802 | 0.7443 | 0.6505 | 0.7032 | 0.5658 | 0.5658 | 0.4540 |

外部结果只改善 proj `+0.0629`，q/k/v/o/fc/Attention 不变。该方向对 GPT-2 有效但不转移到
Qwen，故 v151 仅保留为 cross-model control，不晋级为 root parent。

## Next decision

保留 v151 rejected 快照；下一实验改为 `fc_gate/fc_up` 专属 expansive scale/encoder，保留
BOAT，不关闭 BOAT。仍冻结 q/k/v/o 和 v86 Attention，并使用新的跨候选 role diff 报告。
