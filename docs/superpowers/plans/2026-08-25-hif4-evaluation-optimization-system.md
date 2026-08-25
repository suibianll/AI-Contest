# HiF4 Evaluation Optimization System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Torch-only, dual-track local system that evaluates HiF4 candidates, protects holdout data, and promotes or rolls back an immutable Champion.

**Architecture:** A focused Python package owns independent HiF4 scoring, compliance checks, campaign state, subprocess execution, statistical gates, and a file-backed registry. CUDA performs accuracy screening; CPU performs authoritative local timing and final promotion checks. The current v9 `solution.py` is snapshotted as the initial Champion, while a narrow `CandidateGenerator` protocol preserves a path to later automatic optimization.

**Tech Stack:** Python 3.12, PyTorch 2.x, pytest, argparse, JSON, pathlib, multiprocessing/subprocess, hashlib/HMAC.

**Spec:** `docs/superpowers/specs/2026-08-25-hif4-evaluation-optimization-system-design.md`

## Global Constraints

- Execute actual PyTorch candidate code; never use a NumPy simulation backend or silently fall back when Torch/CUDA is unavailable.
- Preserve the downloaded `solution.py` v9 SHA256 `a6b8b858156164333d1d3ca25c6233b4845061f40a16d4cf74695ecdbb9041f7` as the initial Champion.
- Use CUDA only for accuracy screening; only CPU timing may satisfy the runtime promotion gate.
- Independently implement the standard HiF4 baseline; do not call candidate-private quantization helpers for baseline results.
- Compare candidate and Champion on identical paired cases.
- Never expose holdout seeds or the campaign secret in reports; consume a holdout attempt even when an evaluation crashes.
- Do not calculate calibration `A @ W` and use that output to fit activation quantization.
- Keep all downloaded files not explicitly modified by a task intact.
- Use a project-local `.venv`; install packages through a configurable mainland-China mirror and record resolved versions.

---

## File Map

- `requirements.txt`: runtime/test dependency ranges.
- `.gitignore`: virtual environment, campaigns, registry runtime state, caches, and reports.
- `config/default.json`: tier sizes, statistical gates, timeouts, dtypes, and device tolerances.
- `hif4_system/models.py`: shared frozen dataclasses and JSON-safe serialization.
- `hif4_system/formats.py`: NVFP4 and independent standard HiF4 encode/decode/validation.
- `hif4_system/scoring.py`: Linear/Attention reference outputs and competition scores.
- `hif4_system/suites.py`: deterministic Torch-only synthetic Linear/Attention cases.
- `hif4_system/solution_loader.py`: isolated module loading and six-interface discovery.
- `hif4_system/compliance.py`: static and runtime candidate checks.
- `hif4_system/statistics.py`: Torch bootstrap, summaries, paired comparisons, and gates.
- `hif4_system/campaign.py`: dev/holdout seed selection, budget reservation, and manifest updates.
- `hif4_system/worker.py`: one-process evaluation protocol.
- `hif4_system/runner.py`: subprocess lifecycle, CUDA accuracy track, and CPU validation track.
- `hif4_system/registry.py`: immutable snapshots, Champion pointer, promotion, history, and rollback.
- `hif4_system/optimizer.py`: future candidate-generation protocol with no holdout access.
- `hif4_system/cli.py` and root `cli.py`: command implementation and stable entry point.
- `hif4_generalization_eval.py`: compatibility wrapper routing to the Torch-only CLI.
- `tests/`: focused unit and end-to-end tests.

---

### Task 1: Project Runtime, Shared Models, and Configuration

**Files:**
- Create: `.gitignore`
- Create: `requirements.txt`
- Create: `config/default.json`
- Create: `hif4_system/__init__.py`
- Create: `hif4_system/models.py`
- Create: `hif4_system/config.py`
- Test: `tests/test_config_models.py`

**Interfaces:**
- Produces: `EvaluationConfig`, `CaseResult`, `TimingResult`, `RunResult`, `load_config(path: Path | None) -> EvaluationConfig`, and `to_jsonable(value: object) -> object`.
- `EvaluationConfig` exposes `tier(name)`, `thresholds`, `timeouts`, `device_tolerance`, and `bootstrap_rounds`.

- [ ] **Step 1: Write configuration/model tests**

```python
def test_default_config_is_torch_only():
    cfg = load_config(None)
    assert set(cfg.tiers) == {"smoke", "standard", "soak"}
    assert cfg.backends == ("torch",)
    assert cfg.tier("smoke").calibration_samples == 1

def test_case_result_round_trip_is_json_safe():
    row = CaseResult("seed-00", "linear", "balanced", 0, False, "fp32", 1.0, 0.5, 0.5)
    assert json.loads(json.dumps(to_jsonable(row)))["score"] == 0.5
```

- [ ] **Step 2: Run the tests and confirm the missing package failure**

Run: `.venv\Scripts\python -m pytest tests/test_config_models.py -v`  
Expected: collection fails because `hif4_system.config` and `hif4_system.models` do not exist.

- [ ] **Step 3: Implement immutable models and validated config loading**

Use frozen dataclasses. Reject unknown tiers, non-positive timeouts, threshold rates outside `[0, 1]`, and any backend other than the exact tuple `("torch",)`.

```python
@dataclass(frozen=True)
class CaseResult:
    seed_id: str
    kind: str
    scenario: str
    test_index: int
    causal: bool
    compute_dtype: str
    mse_std: float
    mse_player: float
    score: float

@dataclass(frozen=True)
class TimingResult:
    player_quant_seconds: float
    wall_seconds: float
    peak_rss_bytes: int | None = None
```

- [ ] **Step 4: Add dependency and runtime ignore files**

`requirements.txt` contains `torch>=2.5,<3` and `pytest>=8,<10`. `.gitignore` excludes `.venv/`, `__pycache__/`, `.pytest_cache/`, `campaigns/`, `registry/`, `reports/`, and `*.tmp`, but not source/config/tests.

- [ ] **Step 5: Run tests and commit**

Run: `.venv\Scripts\python -m pytest tests/test_config_models.py -v`  
Expected: PASS.  
Commit: `feat: add evaluation system configuration models`

---

### Task 2: Independent Torch Formats and Competition Scoring

**Files:**
- Create: `hif4_system/formats.py`
- Create: `hif4_system/scoring.py`
- Test: `tests/test_formats.py`
- Test: `tests/test_scoring.py`

**Interfaces:**
- Produces: `quantize_to_nvfp4(x)`, `dequantize_nvfp4(value, scale)`, `standard_hif4_quantize(x)`, `dequantize_hif4(params)`, `validate_hif4_params(params, logical_shape)`, `attention_output(q, k, v, q_heads, kv_heads, head_dim, causal)`, and `competition_score(reference, baseline, player)`.
- `standard_hif4_quantize` returns only `scale_factor`, `scale_lv2`, `scale_lv3`, `sign`, and `mantissa` tensors.

- [ ] **Step 1: Write format boundary tests**

```python
def test_standard_hif4_never_emits_nan_scale_code():
    x = torch.tensor([[0.0] * 63 + [1.0e30]], dtype=torch.float32)
    params = standard_hif4_quantize(x)
    validate_hif4_params(params, x.shape)
    assert torch.isfinite(dequantize_hif4(params)).all()

def test_hif4_shape_contract_for_128_channels():
    params = standard_hif4_quantize(torch.randn(3, 128))
    assert params["scale_factor"].shape == (3, 2)
    assert params["scale_lv2"].shape == (3, 16)
    assert params["scale_lv3"].shape == (3, 16, 2)
```

- [ ] **Step 2: Run format tests and verify failure**

Run: `.venv\Scripts\python -m pytest tests/test_formats.py -v`  
Expected: FAIL because `hif4_system.formats` is missing.

- [ ] **Step 3: Implement independent NVFP4 and standard HiF4 paths**

Use float32 working tensors, BF16-aligned NVFP4 dequantization, block size 16 for NVFP4, and block size 64 for HiF4. Encode E6M2 scales with round-to-nearest-even and clamp codes to `[0, 254]`. Validate exact keys, shapes, allowed dtypes, devices, and finite values without importing `solution.py`.

- [ ] **Step 4: Write scoring tests**

```python
def test_competition_score_matches_rule():
    ref = torch.tensor([0.0, 2.0])
    std = torch.tensor([1.0, 3.0])
    player = torch.tensor([0.5, 2.5])
    assert competition_score(ref, std, player) == pytest.approx(0.75)

@pytest.mark.parametrize("causal", [False, True])
def test_gqa_attention_shape(causal):
    out = attention_output(torch.randn(7, 256), torch.randn(7, 128),
                           torch.randn(7, 128), 4, 2, 64, causal)
    assert out.shape == (7, 256)
```

- [ ] **Step 5: Implement operator scoring**

`competition_score` computes float32 MSE and `(mse_std - mse_player) / max(mse_std, 1e-30)`. `attention_output` reshapes to heads, repeats KV heads for GQA, applies `1/sqrt(head_dim)`, applies an upper-triangular causal mask when requested, softmaxes in float32, and returns `[seq_len, q_heads * head_dim]`.

- [ ] **Step 6: Run tests and commit**

Run: `.venv\Scripts\python -m pytest tests/test_formats.py tests/test_scoring.py -v`  
Expected: PASS.  
Commit: `feat: add independent torch hif4 scoring`

---

### Task 3: Torch-Only Synthetic Suites and Candidate Evaluation

**Files:**
- Create: `hif4_system/suites.py`
- Create: `hif4_system/solution_loader.py`
- Create: `hif4_system/evaluator.py`
- Test: `tests/fixtures/minimal_solution.py`
- Test: `tests/test_suites_evaluator.py`

**Interfaces:**
- Produces: `build_suite(seed: int, tier: TierConfig, device: torch.device) -> EvaluationSuite`, `load_solution(path: Path) -> SolutionAPI`, and `evaluate_solution(api, suite, device, dtypes, causal_modes) -> RunResult`.
- `SolutionAPI` contains exactly the six competition callables.

- [ ] **Step 1: Write deterministic-suite and paired-evaluation tests**

```python
def test_suite_is_deterministic_on_cpu(default_config):
    left = build_suite(101, default_config.tier("smoke"), torch.device("cpu"))
    right = build_suite(101, default_config.tier("smoke"), torch.device("cpu"))
    assert torch.equal(left.linear[0].weight[0], right.linear[0].weight[0])

def test_minimal_solution_produces_linear_and_attention_cases(smoke_suite):
    api = load_solution(Path("tests/fixtures/minimal_solution.py"))
    result = evaluate_solution(api, smoke_suite, torch.device("cpu"), ("fp32",), (False, True))
    assert {row.kind for row in result.cases} == {"linear", "attention"}
```

- [ ] **Step 2: Run tests and verify missing evaluator failure**

Run: `.venv\Scripts\python -m pytest tests/test_suites_evaluator.py -v`  
Expected: FAIL because suite/evaluator modules do not exist.

- [ ] **Step 3: Implement Torch-only deterministic suites**

Port the existing scenario definitions from `hif4_generalization_eval.py` without any imports of `hif4_*numpy_sim.py`. Generate all randomness with an explicit CPU `torch.Generator`, then move immutable case tensors to the requested device. Cover the exact modes in the design spec and keep tier sizes in `config/default.json`.

- [ ] **Step 4: Implement loader and evaluator**

Load each solution under a unique module name derived from SHA256. Evaluate player calibration/quantization and the independent baseline on the same decoded NVFP4 tensors. Record player quantization time only around the six candidate calls; synchronize CUDA immediately before and after timed regions. Return stable case keys `(seed_id, kind, scenario, test_index, causal, compute_dtype)`.

- [ ] **Step 5: Run tests and commit**

Run: `.venv\Scripts\python -m pytest tests/test_suites_evaluator.py -v`  
Expected: PASS.  
Commit: `feat: evaluate torch hif4 candidates on paired suites`

---

### Task 4: Compliance Checks, Statistics, and Campaign Discipline

**Files:**
- Create: `hif4_system/compliance.py`
- Create: `hif4_system/statistics.py`
- Create: `hif4_system/campaign.py`
- Test: `tests/test_compliance.py`
- Test: `tests/test_statistics.py`
- Test: `tests/test_campaign.py`

**Interfaces:**
- Produces: `check_static(path) -> ComplianceReport`, `validate_state(value) -> None`, `summarize(cases, rounds, seed) -> Summary`, `compare(candidate, champion, rounds, seed) -> Comparison`, `decide(summary, comparison, timing, thresholds, authoritative_timing) -> Decision`, and `Campaign.reserve(split, tier) -> SeedReservation`.

- [ ] **Step 1: Write static/runtime compliance tests**

```python
def test_file_io_is_rejected(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text("def f():\n    return open('x')\n", encoding="utf-8")
    assert "file_io" in check_static(bad).violations

def test_valid_nested_cpu_state_is_accepted():
    validate_state({"mode": "safe", "x": torch.ones(3), "flags": [True, 2]})
```

- [ ] **Step 2: Write Torch-bootstrap and paired-key tests**

```python
def test_bootstrap_is_repeatable_without_numpy(sample_cases):
    assert summarize(sample_cases, 200, 17) == summarize(sample_cases, 200, 17)

def test_pairing_rejects_mismatched_case_sets(candidate_cases, champion_cases):
    with pytest.raises(ValueError, match="paired case keys"):
        compare(candidate_cases[:-1], champion_cases, 100, 9)
```

- [ ] **Step 3: Write campaign budget tests**

```python
def test_failed_holdout_reservation_still_consumes_budget(tmp_path):
    campaign = Campaign.create(tmp_path, max_holdout_uses=1)
    reservation = campaign.reserve("holdout", "smoke")
    campaign.finish(reservation, status="crashed", report=None)
    with pytest.raises(HoldoutBudgetExhausted):
        campaign.reserve("holdout", "smoke")
```

- [ ] **Step 4: Run tests and verify missing-module failures**

Run: `.venv\Scripts\python -m pytest tests/test_compliance.py tests/test_statistics.py tests/test_campaign.py -v`  
Expected: FAIL during imports.

- [ ] **Step 5: Implement compliance, Torch statistics, and campaign state**

AST checks reject direct file I/O APIs and flag suspicious Linear calibration matrix products for review. Runtime state validation enforces allowed scalar/container/CPU tensor types, finite tensor values, depth at most 8, and node count at most 4096. Statistics use `torch.randint` and `torch.quantile`. Campaign writes JSON atomically, stores a 32-byte `.holdout_secret`, derives seeds with HMAC-SHA256, records only a seed commitment, and locks thresholds after the first holdout reservation.

- [ ] **Step 6: Run tests and commit**

Run: `.venv\Scripts\python -m pytest tests/test_compliance.py tests/test_statistics.py tests/test_campaign.py -v`  
Expected: PASS.  
Commit: `feat: enforce evaluation discipline and promotion gates`

---

### Task 5: Isolated Worker and Dual-Track Runner

**Files:**
- Create: `hif4_system/worker.py`
- Create: `hif4_system/runner.py`
- Test: `tests/fixtures/crashing_solution.py`
- Test: `tests/fixtures/hanging_solution.py`
- Test: `tests/test_runner.py`

**Interfaces:**
- Produces: `WorkerRequest`, `WorkerResponse`, `run_isolated(request, timeout_seconds) -> WorkerResponse`, `run_accuracy_track(...) -> TrackReport`, and `run_cpu_track(...) -> TrackReport`.
- Worker JSON contains only paths, config primitives, results, timing, environment metadata, and errors; tensors never cross the process boundary.

- [ ] **Step 1: Write crash, timeout, and authority tests**

```python
def test_worker_captures_candidate_crash(tmp_path):
    response = run_isolated(request_for("tests/fixtures/crashing_solution.py"), 10)
    assert response.status == "crashed"
    assert "RuntimeError" in response.error

def test_cuda_timing_is_not_authoritative(cuda_track_report):
    assert cuda_track_report.device_type == "cuda"
    assert cuda_track_report.authoritative_timing is False
```

- [ ] **Step 2: Run runner tests and verify missing-module failure**

Run: `.venv\Scripts\python -m pytest tests/test_runner.py -v`  
Expected: FAIL because runner/worker modules do not exist.

- [ ] **Step 3: Implement subprocess protocol**

Launch `.venv\Scripts\python -m hif4_system.worker` with a request JSON path and response temp path. Use `subprocess.Popen`, capture UTF-8 stdout/stderr, terminate on timeout, and replace the final response atomically. The worker records Torch version, CUDA runtime, device name, CPU, Python, thread count, source hash, and config hash.

- [ ] **Step 4: Implement dual-track orchestration**

`run_accuracy_track` requires explicit CUDA availability when device is `cuda`; it never changes to CPU automatically. `run_cpu_track` forces CPU and marks timing authoritative. Both reserve campaign seeds before launching the worker. A full validation first evaluates candidate and Champion on GPU dev, then CPU dev, and only after passing both reserves holdout.

- [ ] **Step 5: Run tests and commit**

Run: `.venv\Scripts\python -m pytest tests/test_runner.py -v`  
Expected: PASS; CUDA-specific tests skip only when the test is explicitly marked `requires_cuda`.  
Commit: `feat: add isolated gpu and cpu evaluation tracks`

---

### Task 6: Immutable Champion Registry

**Files:**
- Create: `hif4_system/registry.py`
- Test: `tests/test_registry.py`

**Interfaces:**
- Produces: `Registry.initialize(solution_path, expected_sha256)`, `register_candidate(solution_path, reports) -> CandidateRecord`, `promote(candidate_id, expected_sha256)`, `history()`, `rollback(version_id | None)`, and `champion() -> VersionRecord`.

- [ ] **Step 1: Write initialization, mutation-rejection, and rollback tests**

```python
def test_initial_champion_preserves_v9_hash(tmp_path, v9_path):
    registry = Registry(tmp_path)
    record = registry.initialize(v9_path, EXPECTED_V9_SHA256)
    assert record.sha256 == EXPECTED_V9_SHA256
    assert sha256_file(record.solution_path) == EXPECTED_V9_SHA256

def test_promotion_rejects_changed_candidate(tmp_path, registry_with_candidate):
    candidate = registry_with_candidate.pending()[0]
    candidate.solution_path.write_text("changed", encoding="utf-8")
    with pytest.raises(HashMismatch):
        registry_with_candidate.promote(candidate.id, candidate.sha256)
```

- [ ] **Step 2: Run registry tests and verify missing-module failure**

Run: `.venv\Scripts\python -m pytest tests/test_registry.py -v`  
Expected: FAIL because `hif4_system.registry` does not exist.

- [ ] **Step 3: Implement immutable snapshots and atomic pointers**

Version IDs are `<UTC timestamp>-<sha256[:12]>`. Copy source and reports into a new directory, fsync files, then atomically replace `champion.json`. Never edit version directories after registration. Promotion requires passed GPU dev, CPU dev, and holdout reports bound to the exact candidate hash. Rollback creates a new pointer-history event rather than deleting versions.

- [ ] **Step 4: Run tests and commit**

Run: `.venv\Scripts\python -m pytest tests/test_registry.py -v`  
Expected: PASS.  
Commit: `feat: add immutable champion registry and rollback`

---

### Task 7: CLI, Compatibility Wrapper, and Optimizer Boundary

**Files:**
- Create: `hif4_system/cli.py`
- Create: `hif4_system/optimizer.py`
- Create: `cli.py`
- Modify: `hif4_generalization_eval.py`
- Test: `tests/test_cli.py`
- Test: `tests/test_optimizer_protocol.py`

**Interfaces:**
- Produces CLI commands `init`, `evaluate`, `validate`, `promote`, `history`, and `rollback`.
- Produces `CandidateGenerator.generate(champion: Path, feedback: DevFeedback, output_dir: Path) -> Sequence[GeneratedCandidate]` and `DevFeedback` without holdout fields.

- [ ] **Step 1: Write CLI and optimizer-boundary tests**

```python
def test_init_cli_freezes_v9(cli_runner, v9_path, tmp_path):
    result = cli_runner("init", "--champion", str(v9_path), "--root", str(tmp_path))
    assert result.returncode == 0
    assert EXPECTED_V9_SHA256 in result.stdout

def test_dev_feedback_cannot_contain_holdout_cases():
    assert "holdout" not in {field.name for field in dataclasses.fields(DevFeedback)}
```

- [ ] **Step 2: Run tests and verify failure**

Run: `.venv\Scripts\python -m pytest tests/test_cli.py tests/test_optimizer_protocol.py -v`  
Expected: FAIL because CLI/optimizer modules do not exist.

- [ ] **Step 3: Implement CLI with stable exit codes**

Return `0` for success, `2` for invalid arguments/config, `3` for compliance rejection, `4` for evaluation failure/timeout, and `5` for failed promotion gates. Print a short human summary plus the absolute JSON report path. `evaluate --device cuda` errors when CUDA is unavailable. `validate` executes GPU dev, CPU dev, and holdout in order and never promotes implicitly.

- [ ] **Step 4: Replace the legacy evaluator with a compatibility wrapper**

Keep its accepted Torch arguments where practical, reject every `--backend` value other than `torch`, and route to the new runner. Remove all imports and branches referring to `hif4_numpy_sim`, `hif4_v4_numpy_sim`, or `hif4_v5_linear_sim`.

- [ ] **Step 5: Implement the non-operational optimizer protocol**

Define immutable `DevFeedback` and `GeneratedCandidate` dataclasses plus a `CandidateGenerator` protocol. Do not provide an automatic generator in phase one. The feedback object contains Champion hash, candidate hash, dev summaries, grouped failure categories, and remaining dev budget only.

- [ ] **Step 6: Run tests and commit**

Run: `.venv\Scripts\python -m pytest tests/test_cli.py tests/test_optimizer_protocol.py -v`  
Expected: PASS.  
Commit: `feat: expose hif4 evaluation lifecycle cli`

---

### Task 8: End-to-End Verification and Initial Champion Bootstrap

**Files:**
- Create: `tests/test_end_to_end.py`
- Create: `README.md`
- Modify: `config/default.json`
- Runtime output: `registry/`, `campaigns/default/`, `reports/`

**Interfaces:**
- Consumes all prior tasks.
- Produces a locally initialized registry whose Champion snapshot hash equals the downloaded v9 hash and CPU/GPU smoke reports bound to that snapshot.

- [ ] **Step 1: Write end-to-end smoke assertions**

```python
def test_v9_cpu_smoke_report_is_auditable(v9_cpu_smoke_report):
    assert v9_cpu_smoke_report["metadata"]["candidate_sha256"] == EXPECTED_V9_SHA256
    assert v9_cpu_smoke_report["metadata"]["device_type"] == "cpu"
    assert v9_cpu_smoke_report["metadata"]["authoritative_timing"] is True
    assert v9_cpu_smoke_report["summary"]["case_count"] > 0
```

- [ ] **Step 2: Run the complete test suite**

Run: `.venv\Scripts\python -m pytest -q`  
Expected: all non-CUDA tests pass; CUDA tests pass when a compatible NVIDIA runtime is present or are explicitly skipped with the hardware reason.

- [ ] **Step 3: Initialize v9 as Champion**

Run: `.venv\Scripts\python cli.py init --champion solution_v9_champion.py --root .`
Expected: stdout reports the exact v9 SHA256 and the immutable snapshot path.

- [ ] **Step 4: Run GPU and CPU smoke evaluations**

Run: `.venv\Scripts\python cli.py evaluate solution.py --tier smoke --device cuda --split dev`  
Expected: CUDA accuracy report with `authoritative_timing=false`.  
Run: `.venv\Scripts\python cli.py evaluate solution.py --tier smoke --device cpu --split dev`  
Expected: CPU report with `authoritative_timing=true`.

- [ ] **Step 5: Verify Torch-only behavior and audit artifacts**

Run: `rg -n "numpy|backend.*auto|hif4_.*_sim" hif4_system hif4_generalization_eval.py cli.py`  
Expected: no simulation imports or fallback branches; documentation strings may mention that NumPy backends are unsupported. Verify reports contain source/config hashes, environment, seed commitment, summary, timing, and gates without raw holdout seeds.

- [ ] **Step 6: Document operations and commit**

Document mirror-based environment creation, CUDA detection, all CLI commands, report interpretation, campaign reset policy, and rollback.  
Run: `.venv\Scripts\python -m pytest -q`  
Expected: PASS.  
Commit: `docs: complete hif4 evaluation system bootstrap`

---

## Plan Self-Review Results

- Spec coverage: all fourteen design sections map to Tasks 1–8; automatic optimization is intentionally limited to the Task 7 protocol in phase one.
- Type consistency: shared result/config types originate in Task 1; evaluator, statistics, runner, registry, and CLI consume the same names.
- Scope: one deployable local evaluation lifecycle; multi-machine services, automatic submission, and a concrete optimizer remain excluded.
- Verification: every component has a failing-test step, minimal implementation step, passing-test step, and commit checkpoint.
