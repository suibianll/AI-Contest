# v155 L5a permutation-stability candidate

- Status: `RETAINED` (local diagnostic control; not the selected submission)
- Parent: pre-A3 local attribution parent `workbench/pre-a3-v147-parent.py`, SHA
  `800CA10EC3414E4FE886B93CA62BD4A350D26BBA015287DF7E8DF2DD871AC23D`
- Unique change: for expansive `rows > channels` Linear blocks, derive channel pressure after
  the existing BOAT transform, apply a fixed four-quartile low/high interleave, and admit it only
  when both calibration folds improve by at least their absolute disagreement. The rule is
  parameter-free at deployment, keeps the existing HiF4 codec and call graph, and leaves
  Attention plus static q/k/v/o/proj controls unchanged.
- Source SHA256: `816ECBF5E253745C5EBFD04233BD04A2B772CF1510641393C7900CDAFA0EB4CC`

## Local evidence

Protocol: `proxy-v2`; Qwen2.5-0.5B; cache
`artifacts/official_eval/cache/qwen2.5-0.5b-proxy-v2.pt`; CUDA; calibration lengths
`[10,128,512,1024,1024]`; input codec `e4m3-subnormal-ceil-v1`; shared call graph
`168 weight + 24 attention` states. Local seconds are same-host diagnostics and are not an
official-time conversion.

The formal single-file effect-panel command was:

```powershell
.venv\Scripts\python.exe evaluator\official_eval.py --solution solutions\20260902_v155_l5a-permutation-stability_scoreNA_timeNA\solution.py --name v155-l5a-perm-stability-effect --cache artifacts\official_eval\cache\qwen2.5-0.5b-proxy-v2.pt --cache-mode read --effect-panel --focus-linear-roles fc --baseline-json artifacts\official_eval\l3-fc-parent-effect.json --output artifacts\official_eval\v155-l5a-perm-stability-effect.json --report logs\official_eval\v155-l5a-perm-stability-effect.md
```

| panel | Linear | Attention | Overall | API total | wall |
|---|---:|---:|---:|---:|---:|
| effect (56 + 5) | 0.588162284 | 0.757433277 | 0.602036955 | 207.196s | 219.182s |
| default-equivalent workbench (168 + 120) | 0.570998953 | 0.724734669 | 0.635055502 | 248.121s | 280.763s |

The default-equivalent paired replay versus the pre-A3 parent is the decision evidence:

| scope | cases | mean Δgain | positive / negative / unchanged | median MSE ratio |
|---|---:|---:|---:|---:|
| Linear overall | 168 | +0.000116536 | 4 / 0 / 164 | 1.000000 |
| focus `fc` | 48 | +0.000407876 | 4 / 0 / 44 | 1.000000 |
| controls (`q/k/v/o/proj`) | 120 | 0 | 0 / 0 / 120 | 1.000000 |
| Attention | 120 | 0 | 0 / 0 / 120 | 1.000000 |

The effect panel gives the same qualitative result (`Linear +0.000139055`, focus `fc +0.000486693`,
2/0/54 overall and 2/0/14 focus). The evaluator-only decomposition identifies strong W/A
coupling rather than an operand-MSE win: default `fc` W-only and A-only arms are negative while
the Both arm is positive; this is a coordinate interaction signal, not evidence that the
permutation independently improves W or A.

The accepted default cases are only four (`L2 fc_gate`, `L7 fc_gate`, `L16 fc_up`, `L19 fc_gate`),
so the rule has low recall. The original quartile rule without the fold-disagreement gate had mixed
signs and was rejected as unstable. No arbitrary layer list or tuned threshold was added.

## Official fields

- Official score: `unregistered` (the user-confirmed repository anchor remains v86 `16744`;
  the separate un-synchronized high-score fact is `17816`)
- Official time: `NA`
- Official status: `unregistered`

## Cross-model check and decision

The same cached GPT-2 panel was run against the exact pre-A3 parent (72 Linear + 60 Attention).
The candidate moved Linear `0.519793773→0.519641076` (`−0.000152696`), with `ffn_in/fc`
`0.458206636→0.457290459` and all other roles unchanged; Attention stayed `0.411099959`.
This is a structural regression, not a reproducible gain. The Qwen positive therefore remains a
low-recall Qwen-local coordinate signal.

Do not use v155 as the submission: its Qwen gain is only `+0.000116536` on four default cases and
the strict GPT-2 pair is negative. Keep the snapshot as a reproducible negative control while v156
is submitted. Do not replace the root `solution.py`, do not claim an official improvement, and do
not spend another run on permutation thresholds. Any later mechanism should branch from the exact
v86 baseline and target a materially larger Linear effect with the same paired error-source report.
