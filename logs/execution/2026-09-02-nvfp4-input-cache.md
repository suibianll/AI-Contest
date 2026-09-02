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
