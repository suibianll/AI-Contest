# 2026-09-02 NVFP4 evaluator input cache

## Outcome

`evaluator/official_eval.py` now persists exact-profile NVFP4 carrier/scale evaluator inputs. It
does not cache candidate state or outputs. Parent and candidate evaluations with identical dense
source, protocol, codec, scenario, panel, and case limits reuse the same prepared input tensors.

## Commands

Initial build, without candidate API calls:

```powershell
python evaluator/official_eval.py --cache-mode read --nvfp4-cache-mode write --compact-panel --linear-only --output artifacts/official_eval/nvfp4-cache-build-check.json --report logs/official_eval/nvfp4-cache-build-check.md
```

Forced cache hit, also without candidate API calls:

```powershell
python evaluator/official_eval.py --cache-mode read --nvfp4-cache-mode read --compact-panel --linear-only --output artifacts/official_eval/nvfp4-cache-hit-check.json --report logs/official_eval/nvfp4-cache-hit-check.md
```

## Evidence

- Dense source cache: `10,984,305,646` bytes.
- Prepared compact-linear NVFP4 cache: `476,399,887` bytes.
- Initial dense load + selected NVFP4 encoding + persistent write: `9.278575s`.
- Forced prepared-cache load: `0.202392s`.
- Preparation reduction: approximately `97.8%`.
- Hit result records `data_source=nvfp4_cache` and `data_metadata.nvfp4_cache_hit=true`.
- Unit tests: `30 passed` in `tests/test_official_eval.py` using a repository-local pytest temp root
  because the host's default pytest temp directory denied access.

The two capture-only invocations initially returned process code 1 despite writing valid outputs,
because the CLI treated an intentionally empty `results` list as failure. The evaluator now treats
the documented capture-only mode as success; this is an execution-status fix, not a scoring change.

## Attention and complete-panel verification

Attention-only compact cache:

- Build: `7.381841s`, `59,184,287` bytes.
- Forced read hit: `0.053160s`, `data_source=nvfp4_cache`.

Complete default-panel audit (current root `solution.py`):

- Input cache: `both-default-nvfp4.pt`, built once in `19.321546s`, `2,872,472,567` bytes.
- Panel: `168 Linear + 120 Attention`; calibration: `168 weight + 24 attention`; all six APIs ran.
- Cached preparation: `1.185913s`; API total: `617.842032s`; candidate wall: `669.348815s`.
- Local proxy: Linear `0.570268537`, Attention `0.724718506`, overall `0.634622690`.
- Linear tail: median `0.572989`, worst-quartile mean `0.309062`, positive/negative/zero `166/2/0`,
  minimum `-0.562535`.
- Official score/time: `unregistered/NA`; the local wall time is not an official timeout decision.

Raw evidence: `artifacts/official_eval/nvfp4-cache-attention-build-check.json`,
`artifacts/official_eval/nvfp4-cache-attention-hit-check.json`,
`artifacts/official_eval/nvfp4-cache-default-build-check.json`, and
`artifacts/official_eval/root-nvfp4-full-20260902.json`.
