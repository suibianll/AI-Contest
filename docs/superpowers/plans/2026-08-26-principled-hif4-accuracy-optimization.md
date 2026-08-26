# Principled HiF4 Accuracy Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 以当前官方 10250 分、127 秒的 `solution.py` 为不可丢失的 Champion，构建结构化正交旋转、完整二阶 HiF4 舍入、真实 Attention 损失学习及可审计的人工官方反馈闭环，在 300 秒硬时限内获得可归因的实质提分。

**Architecture:** `solution.py` 保持为唯一算法真源和自包含官方提交文件，新增算法以受 feature flag 控制的独立分区嵌入其中；本地工程只负责候选导出、评测、合规检查、计时和人工官方结果登记。实施分五个可独立验收的阶段：先固化基线与候选基础设施，再交付固定 H64/H8 旋转，再交付 Weight/Activation 二阶求解，然后引入学习型 butterfly 与 V 偏差抑制，最后执行完整发布门禁。

**Tech Stack:** Python 3.9+、PyTorch（CPU 为权威计时，CUDA 用于开发加速）、pytest、现有 `hif4_system` schema v2 评测器、标准库 `argparse/json/hashlib/pathlib`；禁止 NumPy 模拟。

**Spec:** `docs/superpowers/specs/2026-08-26-principled-hif4-accuracy-architecture-design.md`

## Global Constraints

- 保持官方六个 API 的名称、参数顺序和返回结构不变。
- 所有官方输出仍为合法 HiF4 五字段；所有 state 必须是有限、无梯度、CPU、contiguous、dense-strided 数据。
- Linear 校准与候选选择不得计算 `A @ W` 或等价的完整 Linear 输出并用于反推激活量化。
- Linear 的 A/W 变换和 Attention 的 Q/K 变换在未量化空间必须严格保持算子不变；V 不做不可恢复的旋转、排列或 centering。
- 当前官方 10250 分实现必须固化为逐 case 和整文件双重回退。
- 官方硬时限为 300 秒；内部提交目标为 220～235 秒；本地预测超过 270 秒的候选不得进入人工提交队列。
- 第一批只提交 C0～C6；每个候选只引入表中指定机制，不混入学习型 butterfly、V 偏差项或 Linear 交叉项。
- 正式 holdout 在 C1/C2/C3 主机制完成开发集与数学测试前不消耗；同一个候选只允许一次 holdout 决策。
- 官方结果只能由用户手工提交后回填；工具不得登录、点击或自动提交竞赛网站。
- 官方提交额度为每日 30 次；首轮只使用约 7 次，预留余量用于最高分候选复验与第二阶段单变量对照。
- `solution.py` 运行时不得读写文件、导入本地工程模块或依赖 NumPy。
- 每个算法候选必须记录源码 SHA256、本地 Linear/Attention 分项、负分率、最差十分位、CPU 时间、预测官方时间、官方总分和官方实际时间。

---

## File Structure

### Algorithm source of truth

- `solution.py` — 唯一可提交算法真源；新增 feature flag、H64/H8/butterfly、二阶量化、Attention 直接目标和所有安全回退。
- `solution_10250_champion.py` — Task 1 对当前 `solution.py` 的不可变快照；只允许在官方新 Champion 确认后新增下一版本文件，不覆盖该文件。

### Offline experiment system

- `hif4_system/candidates.py` — C0～C6 及后续单变量候选定义、flag 校验和确定性命名。
- `hif4_system/official.py` — 人工官方分数/时间的追加式记录、SHA 绑定和 Champion 查询。
- `tools/export_submission.py` — 将一个候选 flag 块写入 `solution.py` 副本，执行 AST/静态合规检查并生成 manifest；不修改工作区 `solution.py`。
- `hif4_system/cli.py` — 增加 `export-candidate`、`record-official` 和 `official-history` 命令。
- `README.md` — 记录候选导出、CPU 评测、人工提交和回填命令。

### Tests

- `tests/test_official_feedback.py` — 官方反馈的追加性、SHA 绑定、时间门禁和 Champion 选择。
- `tests/test_candidate_export.py` — 候选矩阵、flag 注入、单文件自包含、manifest 和静态合规。
- `tests/test_rotations.py` — H64/H8/butterfly 正交性、Linear 不变量、MHA/GQA logits 不变量。
- `tests/test_second_order.py` — Hessian、阻尼 Cholesky、GPTQ 误差反馈、层级坐标下降和回退单调性。
- `tests/test_attention_objective.py` — non-causal 真实损失、学习/验证分离、STE 与真实离散复评、V 均值偏差目标。
- `tests/test_solution_regression.py` — 六接口、HiF4/state 合法性、feature-off 等价、CPU 时间预算。
- `tests/algorithm_helpers.py` — 测试专用的确定性 NVFP4 case 构造、临时 flag 注入加载、HiF4/state 深比较；生产代码不导入该文件。

---

## Phase A — Baseline and Experiment Infrastructure

### Task 1: Freeze the 10250 Champion and add append-only official feedback

**Files:**
- Create: `solution_10250_champion.py`
- Create: `hif4_system/official.py`
- Create: `tests/test_official_feedback.py`
- Modify: `hif4_system/cli.py:320-713`

**Interfaces:**
- Consumes: `solution.py`; CLI root directory semantics already used by `hif4_system.cli`.
- Produces: `OfficialResult`, `record_official_result(root: Path, result: OfficialResult) -> Path`, `load_official_results(root: Path) -> tuple[OfficialResult, ...]`, `best_official_result(root: Path) -> OfficialResult | None`.

- [ ] **Step 1: Copy and hash the current official Champion**

Run:

```powershell
Copy-Item -LiteralPath solution.py -Destination solution_10250_champion.py
Get-FileHash -Algorithm SHA256 solution.py
Get-FileHash -Algorithm SHA256 solution_10250_champion.py
```

Expected: the two SHA256 values are identical. Record that value as `source_sha256` in the first C0 result; do not reuse the old v9 hash.

- [ ] **Step 2: Write failing tests for append-only official records**

```python
from pathlib import Path

import pytest

from hif4_system.official import (
    OfficialResult,
    best_official_result,
    load_official_results,
    record_official_result,
)


def test_official_results_are_sha_bound_and_append_only(tmp_path: Path) -> None:
    source = tmp_path / "C0" / "solution.py"
    source.parent.mkdir()
    source.write_text("x = 1\n", encoding="utf-8")
    first = OfficialResult.from_submission(
        candidate_id="C0", source=source, score=10250.0, runtime_seconds=127.0
    )
    path = record_official_result(tmp_path, first)
    assert path.name == "official_results.jsonl"
    assert load_official_results(tmp_path) == (first,)
    assert best_official_result(tmp_path) == first

    source.write_text("x = 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA256"):
        record_official_result(tmp_path, first)


def test_predicted_over_270_seconds_is_not_submittable() -> None:
    assert not OfficialResult.submission_allowed(270.01)
    assert OfficialResult.submission_allowed(270.0)
```

- [ ] **Step 3: Run the tests and confirm the module is absent**

Run: `python -m pytest tests/test_official_feedback.py -v`

Expected: FAIL during import with `ModuleNotFoundError: hif4_system.official`.

- [ ] **Step 4: Implement the immutable result model and atomic append**

```python
@dataclass(frozen=True)
class OfficialResult:
    candidate_id: str
    source_path: str
    source_sha256: str
    score: float
    runtime_seconds: float
    recorded_at_utc: str

    @staticmethod
    def submission_allowed(predicted_seconds: float) -> bool:
        return math.isfinite(predicted_seconds) and predicted_seconds <= 270.0

    @classmethod
    def from_submission(
        cls, candidate_id: str, source: Path, score: float, runtime_seconds: float
    ) -> "OfficialResult":
        payload = source.read_bytes()
        if not math.isfinite(score) or runtime_seconds <= 0.0:
            raise ValueError("official score must be finite and runtime must be positive")
        return cls(
            candidate_id=candidate_id,
            source_path=str(source.resolve()),
            source_sha256=hashlib.sha256(payload).hexdigest(),
            score=float(score),
            runtime_seconds=float(runtime_seconds),
            recorded_at_utc=datetime.now(timezone.utc).isoformat(),
        )
```

`record_official_result` must re-hash `source_path`, reject a mismatch, serialize one compact JSON object per line through a temporary sibling file, then atomically replace `official_results.jsonl`. `best_official_result` sorts by descending score, then ascending runtime, then recording time.

- [ ] **Step 5: Add manual-only CLI commands**

Add parsers with these exact arguments:

```python
record = subparsers.add_parser("record-official")
record.add_argument("--candidate-id", required=True)
record.add_argument("--source", required=True)
record.add_argument("--score", required=True, type=float)
record.add_argument("--runtime-seconds", required=True, type=float)
record.add_argument("--root", default=".")
record.set_defaults(handler=_cmd_record_official)

official_history = subparsers.add_parser("official-history")
official_history.add_argument("--root", default=".")
official_history.set_defaults(handler=_cmd_official_history)
```

The record command only stores values supplied by the user; it must contain no browser, HTTP or submission code.

- [ ] **Step 6: Verify the task**

Run:

```powershell
python -m pytest tests/test_official_feedback.py tests/test_cli.py -v
python cli.py record-official --candidate-id C0 --source solution_10250_champion.py --score 10250 --runtime-seconds 127 --root artifacts/manual-official
python cli.py official-history --root artifacts/manual-official
```

Expected: all tests PASS; history prints exactly one SHA-bound C0 record.

- [ ] **Step 7: Commit**

```powershell
git add solution_10250_champion.py hif4_system/official.py hif4_system/cli.py tests/test_official_feedback.py tests/test_cli.py
git commit -m "feat: freeze 10250 champion and record official feedback"
```

### Task 2: Add deterministic candidate flags and standalone export

**Files:**
- Create: `hif4_system/candidates.py`
- Create: `tools/export_submission.py`
- Create: `tests/test_candidate_export.py`
- Modify: `solution.py:15-50`
- Modify: `hif4_system/cli.py:320-713`

**Interfaces:**
- Consumes: `check_static(path: Path)` from `hif4_system.compliance`; SHA utilities from Task 1.
- Produces: `CandidateSpec`, `candidate_matrix() -> dict[str, CandidateSpec]`, `all_candidates() -> dict[str, CandidateSpec]`, `export_candidate(source: Path, spec: CandidateSpec, output_root: Path, predicted_seconds: float) -> Path`.

- [ ] **Step 1: Write failing tests for C0–C6 isolation**

```python
def test_first_wave_matrix_changes_only_declared_mechanisms() -> None:
    matrix = candidate_matrix()
    assert tuple(matrix) == ("C0", "C1", "C2", "C3", "C4", "C5", "C6")
    assert matrix["C0"].enabled == frozenset()
    assert matrix["C1"].enabled == frozenset({"linear_h64"})
    assert matrix["C2"].enabled == frozenset({"attention_h64"})
    assert matrix["C3"].enabled == frozenset({"weight_second_order"})
    assert matrix["C4"].enabled == frozenset({"linear_h64", "attention_h64"})
    assert matrix["C5"].enabled == frozenset({"linear_h64", "weight_second_order"})
    assert matrix["C6"].enabled == frozenset(
        {"linear_h64", "attention_h64", "weight_second_order"}
    )


def test_export_is_standalone_hash_bound_and_does_not_modify_source(tmp_path: Path) -> None:
    before = SOURCE.read_bytes()
    exported = export_candidate(SOURCE, candidate_matrix()["C1"], tmp_path, 220.0)
    assert SOURCE.read_bytes() == before
    assert exported.name == "solution.py"
    assert check_static(exported).passed
    compile(exported.read_text(encoding="utf-8"), str(exported), "exec")
    manifest = json.loads(exported.with_name("manifest.json").read_text(encoding="utf-8"))
    assert manifest["candidate_id"] == "C1"
    assert manifest["enabled"] == ["linear_h64"]
    assert manifest["source_sha256"] == hashlib.sha256(exported.read_bytes()).hexdigest()
```

- [ ] **Step 2: Run tests and verify the missing interfaces**

Run: `python -m pytest tests/test_candidate_export.py -v`

Expected: FAIL importing `hif4_system.candidates`.

- [ ] **Step 3: Define the exact flag block and candidate model**

Insert this block near the constants in `solution.py`:

```python
# BEGIN HIF4_CANDIDATE_FLAGS
_ENABLE_LINEAR_H64 = False
_ENABLE_LINEAR_H8 = False
_ENABLE_ATTENTION_H64 = False
_ENABLE_ATTENTION_H8 = False
_ENABLE_WEIGHT_SECOND_ORDER = False
_ENABLE_ACTIVATION_SECOND_ORDER = False
_ENABLE_LINEAR_CROSS_TERM = False
_ENABLE_LEARNED_BUTTERFLY = False
_ENABLE_V_BIAS = False
# END HIF4_CANDIDATE_FLAGS
```

Define:

```python
@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    enabled: frozenset[str]

    def __post_init__(self) -> None:
        unknown = self.enabled.difference(ALLOWED_FEATURES)
        if unknown:
            raise ValueError(f"unknown candidate features: {sorted(unknown)}")
```

The first-wave matrix must be a literal ordered tuple converted to a dict; no generated combinatorial grid is permitted.

Keep second-wave registration explicit and empty at this stage:

```python
_SECOND_WAVE: dict[str, CandidateSpec] = {}


def all_candidates() -> dict[str, CandidateSpec]:
    combined = candidate_matrix()
    combined.update(_SECOND_WAVE)
    return combined
```

Later tasks add named entries to `_SECOND_WAVE`; they do not alter the seven-entry `candidate_matrix()` used for the factorial readout.

- [ ] **Step 4: Implement marker replacement, compliance gate and manifest**

```python
def export_candidate(
    source: Path,
    spec: CandidateSpec,
    output_root: Path,
    predicted_seconds: float,
) -> Path:
    if predicted_seconds > 270.0 or not math.isfinite(predicted_seconds):
        raise ValueError("predicted official runtime exceeds 270 seconds")
    text = source.read_text(encoding="utf-8")
    rendered = replace_flag_block(text, spec.enabled)
    target_dir = output_root / spec.candidate_id
    target_dir.mkdir(parents=True, exist_ok=False)
    target = target_dir / "solution.py"
    target.write_text(rendered, encoding="utf-8", newline="\n")
    report = check_static(target)
    if not report.passed:
        raise ValueError(f"candidate failed static compliance: {report.violations}")
    sha256 = hashlib.sha256(target.read_bytes()).hexdigest()
    write_manifest(target_dir / "manifest.json", spec, sha256, predicted_seconds)
    return target
```

`replace_flag_block` must require exactly one begin marker and one end marker. The manifest additionally initializes `local_metrics` and `official_result` to `null`, so later commands update metadata rather than source.

- [ ] **Step 5: Add `export-candidate` CLI wiring**

```python
export = subparsers.add_parser("export-candidate")
export.add_argument("candidate_id", choices=tuple(all_candidates()))
export.add_argument("--source", default="solution.py")
export.add_argument("--predicted-seconds", required=True, type=float)
export.add_argument("--output-root", default="artifacts/submissions")
export.set_defaults(handler=_cmd_export_candidate)
```

- [ ] **Step 6: Verify C0 export is byte-stable outside the flag block**

Run:

```powershell
python -m pytest tests/test_candidate_export.py tests/test_compliance.py -v
python cli.py export-candidate C0 --source solution.py --predicted-seconds 127 --output-root artifacts/submissions
python -m py_compile artifacts/submissions/C0/solution.py
```

Expected: PASS; `manifest.json` SHA equals the exported file SHA; source `solution.py` remains unchanged.

- [ ] **Step 7: Commit**

```powershell
git add solution.py hif4_system/candidates.py hif4_system/cli.py tools/export_submission.py tests/test_candidate_export.py
git commit -m "feat: export isolated HiF4 submission candidates"
```

---

## Phase B — Exact Structured Rotations

### Task 3: Implement deterministic H64/H8 rotation primitives

**Files:**
- Modify: `solution.py:89-128`
- Create: `tests/test_rotations.py`

**Interfaces:**
- Consumes: `_HIF4_BLOCK_SIZE == 64`, PyTorch tensors with last dimension divisible by 64.
- Produces: `_fwht_last_dim(x: torch.Tensor) -> torch.Tensor`, `_rotation_signs(seed: int, groups: int, width: int, device: torch.device) -> torch.Tensor`, `_apply_rotation(x: torch.Tensor, signs: torch.Tensor, mode: int) -> torch.Tensor`; modes `0=identity`, `1=H64`, `2=H8`.

- [ ] **Step 1: Write orthogonality and determinism tests**

```python
@pytest.mark.parametrize("mode", [1, 2])
def test_rotation_preserves_inner_products(mode: int) -> None:
    torch.manual_seed(7)
    x = torch.randn(11, 128)
    y = torch.randn(13, 128)
    signs = solution._rotation_signs(19, 2, 64, x.device)
    xr = solution._apply_rotation(x, signs, mode)
    yr = solution._apply_rotation(y, signs, mode)
    torch.testing.assert_close(x @ y.T, xr @ yr.T, rtol=2e-5, atol=2e-5)


def test_rotation_signs_are_deterministic_without_global_rng_mutation() -> None:
    torch.manual_seed(123)
    before = torch.random.get_rng_state()
    left = solution._rotation_signs(5, 3, 64, torch.device("cpu"))
    right = solution._rotation_signs(5, 3, 64, torch.device("cpu"))
    assert torch.equal(left, right)
    assert torch.equal(before, torch.random.get_rng_state())
```

- [ ] **Step 2: Run tests and observe missing helpers**

Run: `python -m pytest tests/test_rotations.py -v`

Expected: FAIL with missing `_rotation_signs` or `_apply_rotation`.

- [ ] **Step 3: Implement normalized in-place-free FWHT**

```python
def _fwht_last_dim(x: torch.Tensor) -> torch.Tensor:
    width = int(x.shape[-1])
    if width <= 0 or width & (width - 1):
        raise ValueError("FWHT width must be a positive power of two")
    y = x
    stride = 1
    while stride < width:
        shape = y.shape[:-1] + (-1, 2, stride)
        pair = y.reshape(shape)
        left, right = pair.unbind(-2)
        y = torch.stack((left + right, left - right), dim=-2).reshape_as(y)
        stride *= 2
    return y * (float(width) ** -0.5)
```

`_apply_rotation` reshapes the last dimension into 64-wide groups. H64 applies `x * signs` followed by one 64-wide FWHT. H8 reshapes each group to `[..., 8, 8]`, applies the matching signs, then applies FWHT only on the final 8-wide axis. Identity returns `x` unchanged. `_rotation_signs` uses an integer hash of `(seed, group, channel)` rather than the global Torch RNG and returns exactly `-1.0/+1.0`.

- [ ] **Step 4: Add direct Linear invariance coverage**

```python
def test_paired_linear_rotation_is_exact_before_quantization() -> None:
    torch.manual_seed(11)
    a = torch.randn(17, 128)
    w = torch.randn(23, 128)
    signs = solution._rotation_signs(31, 2, 64, a.device)
    ar = solution._apply_rotation(a, signs, 1)
    wr = solution._apply_rotation(w, signs, 1)
    torch.testing.assert_close(a @ w.T, ar @ wr.T, rtol=2e-5, atol=2e-5)
```

- [ ] **Step 5: Run focused and formatting tests**

Run: `python -m pytest tests/test_rotations.py tests/test_formats.py -v`

Expected: all PASS on CPU.

- [ ] **Step 6: Commit**

```powershell
git add solution.py tests/test_rotations.py
git commit -m "feat: add deterministic structured Hadamard rotations"
```

### Task 4: Integrate fixed H64 into the Linear calibration and online path

**Files:**
- Modify: `solution.py:870-1187`
- Modify: `tests/test_rotations.py`
- Create: `tests/test_solution_regression.py`
- Create: `tests/algorithm_helpers.py`

**Interfaces:**
- Consumes: Task 3 rotation helpers; existing `_linear_candidate_metrics`, `_dense_to_hif4`, `_cpu_state_tensor`.
- Produces: `_linear_rotation_candidates(groups: int, device: torch.device) -> tuple[tuple[int, torch.Tensor | None], ...]`, `_transform_linear_activation(dense: torch.Tensor, multiplier: torch.Tensor | None, permutation: torch.Tensor | None, rotation_mode: int, rotation_signs: torch.Tensor | None) -> torch.Tensor`, activation state keys `rotation_mode: int`, `rotation_signs: torch.Tensor | None`, `rotation_version: int`.

- [ ] **Step 1: Add tests for feature-off equivalence and state legality**

```python
def test_linear_rotation_state_is_cpu_finite_and_reproducible() -> None:
    calibrated = call_linear_calibration(solution, seed=41, in_features=128)
    state = calibrated["activation_state"]
    validate_state(state)
    assert state["rotation_mode"] in (0, 1)
    if state["rotation_mode"] == 1:
        assert state["rotation_signs"].device.type == "cpu"
        assert state["rotation_signs"].shape == (2, 64)
    first = call_dynamic_activation(solution, state, seed=43)
    second = call_dynamic_activation(solution, clone_state(state), seed=43)
    assert_hif4_equal(first, second)


def test_linear_feature_off_matches_frozen_champion() -> None:
    current = load_solution_with_flags(linear_h64=False)
    frozen = load_solution(ROOT / "solution_10250_champion.py")
    assert_linear_public_outputs_equal(current, frozen, seed=47)
```

Implement six imported helpers in `tests/algorithm_helpers.py`: `load_solution(path: Path) -> ModuleType` unwraps `.module` from `hif4_system.solution_loader.load_solution`; `load_solution_with_flags(**enabled: bool) -> ModuleType` writes a marker-replaced source into a retained `TemporaryDirectory`; `call_linear_calibration(module: ModuleType, seed: int, in_features: int) -> dict[str, object]` selects the first matching deterministic smoke `LinearCase`; `call_dynamic_activation(module: ModuleType, state: dict[str, object], seed: int) -> dict[str, torch.Tensor]` uses that case's first test pair; `assert_hif4_equal(left, right) -> None` compares every HiF4 tensor field; and `assert_linear_public_outputs_equal(left, right, seed: int) -> None` deep-compares both public Linear calls and nested state. Case generation calls `build_suite(seed, load_config(None).tier("smoke"), torch.device("cpu"), tier_name="smoke", split="dev")`. Deep comparison checks every tensor and nested state value, not only reconstructed MSE.

- [ ] **Step 2: Run the tests before state integration**

Run: `python -m pytest tests/test_solution_regression.py::test_linear_rotation_state_is_cpu_finite_and_reproducible -v`

Expected: FAIL because the new state keys do not exist.

- [ ] **Step 3: Add fixed seed candidates to Linear calibration**

```python
_FIXED_ROTATION_SEEDS = (3, 17, 29, 43, 71, 97)


def _linear_rotation_candidates(groups: int, device: torch.device):
    candidates = [(0, None)]
    if _ENABLE_LINEAR_H64:
        candidates.extend(
            (1, _rotation_signs(seed, groups, 64, device))
            for seed in _FIXED_ROTATION_SEEDS
        )
    return tuple(candidates)
```

For each already-selected `(D, P)`, rotate sampled activation and sampled Weight with the same candidate. Rank with the existing legal Linear proxy plus the new full Hessian proxy once Task 7 exists. At this stage selection must require mean proxy improvement of at least 0.5% and no calibration sample worse than 1.0%; identity remains first and wins ties.

- [ ] **Step 4: Apply the selected transform exactly once**

In calibration, construct:

```python
weight_smooth = (weight * best_d.unsqueeze(0)).index_select(-1, best_perm)
weight_smooth = _apply_rotation(weight_smooth, best_rotation_signs, best_rotation_mode)
```

In dynamic activation, preserve the current D/P order and apply rotation before `_dense_to_hif4` through the new helper:

```python
dense = _dequantize_nvfp4_float32(activation_quant, activation_scale)
dense = _transform_linear_activation(
    dense, multiplier, permutation, rotation_mode, rotation_signs_on_device
)
```

Store signs through `_cpu_state_tensor`; do not regenerate them online from a seed.

- [ ] **Step 5: Verify public API regression and local Linear gain gate**

Run:

```powershell
python -m pytest tests/test_rotations.py tests/test_solution_regression.py tests/test_suites_evaluator.py -v
python cli.py export-candidate C1 --source solution.py --predicted-seconds 190 --output-root artifacts/task4
python cli.py audit --candidate artifacts/task4/C1/solution.py --incumbent solution_10250_champion.py --tier smoke --split dev --root artifacts/linear-h64-smoke
```

Expected: tests PASS; audit is paired and reports Linear cases separately. A negative smoke result does not delete the implementation because C1 still needs an isolated official candidate, but identity fallback must prevent per-calibration proxy regression.

- [ ] **Step 6: Commit**

```powershell
git add solution.py tests/test_rotations.py tests/test_solution_regression.py
git commit -m "feat: integrate exact H64 rotation into linear quantization"
```

### Task 5: Integrate fixed H64 into Attention with correct GQA sharing

**Files:**
- Modify: `solution.py:1189-2238`
- Modify: `tests/test_rotations.py`
- Modify: `tests/test_solution_regression.py`

**Interfaces:**
- Consumes: Task 3 helpers; existing `_attention_true_metrics`, `_transform_attention_pair`, six public APIs.
- Produces: `_expand_kv_rotation_to_q(signs: torch.Tensor, q_num_heads: int, kv_num_heads: int) -> torch.Tensor`; q/k state keys `rotation_mode`, `rotation_signs`, `rotation_version`; V state remains unrotated.

- [ ] **Step 1: Add MHA and GQA logit invariance tests**

```python
@pytest.mark.parametrize("q_heads,kv_heads,head_dim", [(4, 4, 64), (8, 2, 64), (4, 1, 128)])
def test_attention_rotation_preserves_mha_and_gqa_logits(
    q_heads: int, kv_heads: int, head_dim: int
) -> None:
    torch.manual_seed(53)
    tokens = 19
    q = torch.randn(tokens, q_heads, head_dim)
    k = torch.randn(tokens, kv_heads, head_dim)
    signs = solution._rotation_signs(17, kv_heads * (head_dim // 64), 64, q.device)
    k_signs = signs.reshape(kv_heads, head_dim // 64, 64)
    q_signs = k_signs.repeat_interleave(q_heads // kv_heads, dim=0)
    qr = rotate_heads(q, q_signs, mode=1)
    kr = rotate_heads(k, k_signs, mode=1)
    baseline = gqa_logits(q, k, kv_heads)
    rotated = gqa_logits(qr, kr, kv_heads)
    torch.testing.assert_close(baseline, rotated, rtol=2e-5, atol=2e-5)


def rotate_heads(x: torch.Tensor, signs: torch.Tensor, mode: int) -> torch.Tensor:
    tokens, heads, width = x.shape
    flat_signs = signs.reshape(heads * (width // 64), 64)
    rotated = solution._apply_rotation(x.reshape(tokens, heads * width), flat_signs, mode)
    return rotated.reshape_as(x)


def gqa_logits(q: torch.Tensor, k: torch.Tensor, kv_heads: int) -> torch.Tensor:
    expanded_k = k.repeat_interleave(q.shape[1] // kv_heads, dim=1)
    return torch.einsum("thd,shd->hts", q, expanded_k)
```

- [ ] **Step 2: Run the focused invariance test**

Run: `python -m pytest tests/test_rotations.py -k "attention_rotation" -v`

Expected: FAIL because `rotate_heads` integration helpers are absent.

- [ ] **Step 3: Implement KV-to-Q sharing and candidate scoring**

The per-KV-head sign shape is `[kv_num_heads, head_dim // 64, 64]`. Query signs are produced only by:

```python
def _expand_kv_rotation_to_q(signs, q_num_heads, kv_num_heads):
    if q_num_heads % kv_num_heads != 0:
        raise ValueError("q_num_heads must be divisible by kv_num_heads")
    return signs.repeat_interleave(q_num_heads // kv_num_heads, dim=0)
```

Extend `_transform_attention_pair` so its final step applies the shared rotation to Q and K after the current exact D/P/K-centering transform. Score Identity and six H64 seeds using `_attention_true_metrics` on non-causal output only. Select H64 only if mean ratio improves by at least 0.5%, worst sample does not worsen by more than 0.5%, and both MHA/GQA metadata remain valid.

- [ ] **Step 4: Persist separate Q/K signs and keep V unchanged**

q_state stores expanded Query signs; k_state stores KV signs. Both are detached CPU contiguous FP32 tensors. v_state must contain neither `rotation_mode != 0` nor `rotation_signs`; add an assertion before return:

```python
assert int(v_state.get("rotation_mode", 0)) == 0
assert v_state.get("rotation_signs") is None
```

- [ ] **Step 5: Verify all Attention public paths**

Run:

```powershell
python -m pytest tests/test_rotations.py tests/test_solution_regression.py tests/test_scoring.py tests/test_suites_evaluator.py -v
python cli.py export-candidate C2 --source solution.py --predicted-seconds 195 --output-root artifacts/task5
python cli.py audit --candidate artifacts/task5/C2/solution.py --incumbent solution_10250_champion.py --tier smoke --split dev --root artifacts/attention-h64-smoke
```

Expected: PASS; no causal cases are introduced; V reconstructed values are byte-identical to feature-off behavior.

- [ ] **Step 6: Commit**

```powershell
git add solution.py tests/test_rotations.py tests/test_solution_regression.py
git commit -m "feat: add GQA-safe H64 attention rotation"
```

### Task 6: Add isolated H8 alternatives, rotation selection telemetry, and export C1/C2/C4

**Files:**
- Modify: `solution.py:15-50,955-2238`
- Modify: `hif4_system/candidates.py`
- Modify: `tools/export_submission.py`
- Modify: `tests/test_candidate_export.py`
- Modify: `tests/test_solution_regression.py`

**Interfaces:**
- Consumes: Tasks 2–5.
- Produces: `linear_h8` and `attention_h8` second-wave features; entries `R1={linear_h8}` and `R2={attention_h8}` added to `all_candidates()`; `build_manifest(spec: CandidateSpec, source_sha256: str, predicted_seconds: float) -> dict[str, object]`; `attach_local_metrics(manifest: dict[str, object], metrics: dict[str, object]) -> dict[str, object]`; manifest metrics keys `rotation_mode_counts`, `fallback_rate`, `predicted_official_seconds`.

- [ ] **Step 1: Write failing tests for large-tensor H8 selection and manifest telemetry**

```python
def test_large_tensor_rotation_budget_uses_h8_or_identity() -> None:
    module = load_solution_with_flags(linear_h8=True)
    state = call_linear_calibration(module, seed=59, in_features=4096)["activation_state"]
    assert state["rotation_mode"] in (0, 2)


def test_manifest_accepts_rotation_telemetry(tmp_path: Path) -> None:
    manifest = build_manifest(candidate_matrix()["C4"], "a" * 64, 230.0)
    updated = attach_local_metrics(
        manifest,
        {"rotation_mode_counts": {"identity": 2, "h64": 6, "h8": 1}, "fallback_rate": 2 / 9},
    )
    assert updated["local_metrics"]["fallback_rate"] == pytest.approx(2 / 9)
```

- [ ] **Step 2: Run the focused tests**

Run: `python -m pytest tests/test_candidate_export.py tests/test_solution_regression.py -k "rotation_budget or rotation_telemetry" -v`

Expected: FAIL because the H8 budget policy and telemetry are absent.

- [ ] **Step 3: Add deterministic budget policy**

Use exact thresholds, with no adaptive hidden-data branch:

```python
def _rotation_modes_for_numel(
    numel: int, h64_enabled: bool, h8_enabled: bool
) -> tuple[int, ...]:
    modes = [0]
    if h8_enabled:
        modes.append(2)
    if h64_enabled:
        modes.append(1)
    return tuple(modes)
```

H8 has two fixed sign seeds `(17, 43)` and is scored by the same operator selector. Tie order is Identity, H8, H64 to choose the cheaper path when scores are equal within `1e-7`. C1/C2 never enable H8；R1/R2 是首轮检查点之后才使用的隔离 H8 对照。

- [ ] **Step 4: Attach local evaluator summaries without changing submission source**

Add a CLI helper that reads a completed audit report, verifies candidate SHA, then atomically updates only `manifest.json`. It records Linear/Attention mean score, negative count, worst decile, mode counts, fallback rate and CPU wall time.

- [ ] **Step 5: Run standard paired audits and export the first rotation candidates**

Run:

```powershell
python -m pytest tests/test_candidate_export.py tests/test_solution_regression.py tests/test_rotations.py -v
python cli.py export-candidate C1 --source solution.py --predicted-seconds 190 --output-root artifacts/first-wave
python cli.py export-candidate C2 --source solution.py --predicted-seconds 195 --output-root artifacts/first-wave
python cli.py export-candidate C4 --source solution.py --predicted-seconds 230 --output-root artifacts/first-wave
python cli.py audit --candidate artifacts/first-wave/C1/solution.py --incumbent solution_10250_champion.py --tier standard --split dev --root artifacts/audits/C1
python cli.py audit --candidate artifacts/first-wave/C2/solution.py --incumbent solution_10250_champion.py --tier standard --split dev --root artifacts/audits/C2
python cli.py audit --candidate artifacts/first-wave/C4/solution.py --incumbent solution_10250_champion.py --tier standard --split dev --root artifacts/audits/C4
```

Expected: every source passes compliance and each audit produces paired per-kind metrics. Any predicted official time above 270 seconds blocks export.

- [ ] **Step 6: Commit**

```powershell
git add solution.py hif4_system/candidates.py tools/export_submission.py tests/test_candidate_export.py tests/test_solution_regression.py
git commit -m "feat: add isolated H8 candidates and rotation telemetry"
```

---

## Phase C — Full Second-Order HiF4 Quantization

### Task 7: Build stable 64×64 Hessian factors and the exact quadratic objective

**Files:**
- Modify: `solution.py:325-545`
- Create: `tests/test_second_order.py`

**Interfaces:**
- Consumes: FP32 samples whose final width is divisible by 64.
- Produces: `_block_hessian(samples: torch.Tensor, damping_ratio: float) -> torch.Tensor`, `_factor_hessian(hessian: torch.Tensor, max_rank: int = 8) -> dict[str, torch.Tensor | int]`, `_quadratic_error(error: torch.Tensor, hessian: torch.Tensor) -> torch.Tensor`, `_apply_hessian_factor(error: torch.Tensor, factor: dict[str, Any]) -> torch.Tensor`.

- [ ] **Step 1: Write Hessian correctness and failure-path tests**

```python
def test_block_hessian_matches_direct_gram() -> None:
    torch.manual_seed(61)
    x = torch.randn(37, 128)
    h = solution._block_hessian(x, damping_ratio=0.0)
    expected = torch.stack(((x[:, :64].T @ x[:, :64]) / 37, (x[:, 64:].T @ x[:, 64:]) / 37))
    torch.testing.assert_close(h, expected)


def test_factor_falls_back_to_finite_diag_rank8() -> None:
    h = torch.ones(2, 64, 64)
    h[0, 0, 0] = float("nan")
    factor = solution._factor_hessian(h, max_rank=8)
    assert factor["mode"] in (1, 2)
    for value in factor.values():
        if torch.is_tensor(value):
            assert torch.isfinite(value).all()


def test_factored_quadratic_error_matches_dense() -> None:
    torch.manual_seed(67)
    x = torch.randn(80, 64)
    h = solution._block_hessian(x, 1e-4)
    e = torch.randn(5, 1, 64)
    factor = solution._factor_hessian(h)
    torch.testing.assert_close(
        solution._apply_hessian_factor(e, factor),
        solution._quadratic_error(e, h),
        rtol=2e-4,
        atol=2e-4,
    )
```

- [ ] **Step 2: Run tests before implementation**

Run: `python -m pytest tests/test_second_order.py -v`

Expected: FAIL due to missing Hessian helpers.

- [ ] **Step 3: Implement block Gram matrices and damping escalation**

```python
def _block_hessian(samples: torch.Tensor, damping_ratio: float) -> torch.Tensor:
    x = samples.to(torch.float32).reshape(-1, samples.shape[-1] // 64, 64)
    h = torch.einsum("ngi,ngj->gij", x, x) / float(max(int(x.shape[0]), 1))
    diagonal_mean = h.diagonal(dim1=-2, dim2=-1).mean(dim=-1, keepdim=True)
    eye = torch.eye(64, dtype=h.dtype, device=h.device).unsqueeze(0)
    return torch.nan_to_num(h) + eye * diagonal_mean.clamp_min(_EPS).unsqueeze(-1) * damping_ratio
```

`_factor_hessian` tries Cholesky with damping ratios `(1e-6, 1e-5, 1e-4, 1e-3, 1e-2)`. If all fail, use `torch.linalg.eigh` on the finite symmetrized matrix, retain the largest eight non-negative eigenpairs, and store `diag = clamp(diagonal - low_rank_diagonal, min=_EPS)`. `mode=1` means Cholesky and `mode=2` means diag+rank.

- [ ] **Step 4: Implement dense and factored loss with identical batch semantics**

```python
def _quadratic_error(error: torch.Tensor, hessian: torch.Tensor) -> torch.Tensor:
    return torch.einsum("...gi,gij,...gj->...g", error, hessian, error)
```

The factored path computes `||e @ L||²` for Cholesky and `sum(diag * e²) + ||e @ U||²` for diag+rank. Return one loss per leading item and 64-channel group; never silently sum rows or groups.

- [ ] **Step 5: Verify numerics on CPU and CUDA when present**

Run:

```powershell
python -m pytest tests/test_second_order.py -v
python -m pytest tests/test_formats.py tests/test_scoring.py -v
```

Expected: PASS; rank fallback is deterministic and finite. CUDA-specific parametrization skips cleanly when unavailable.

- [ ] **Step 6: Commit**

```powershell
git add solution.py tests/test_second_order.py
git commit -m "feat: add stable block Hessian objectives"
```

### Task 8: Implement GPTQ-style mantissa feedback and hierarchy coordinate descent

**Files:**
- Modify: `solution.py:325-793`
- Modify: `tests/test_second_order.py`

**Interfaces:**
- Consumes: Task 7 Hessian/factor helpers; existing `_solve_exact_hierarchy`, `_pack_hif4_params`, E6M2 encode/decode.
- Produces: `_gptq_round_block(values: torch.Tensor, scale: torch.Tensor, hierarchy: torch.Tensor, hessian: torch.Tensor) -> torch.Tensor`, `_coordinate_descent_hierarchy(values: torch.Tensor, initial: dict[str, torch.Tensor], hessian: torch.Tensor, sweeps: int = 2) -> dict[str, torch.Tensor]`, `_second_order_dense_to_hif4(dense: torch.Tensor, hessian: torch.Tensor, max_blocks: int) -> dict[str, torch.Tensor]`.

- [ ] **Step 1: Write monotonicity and legality tests**

```python
def test_gptq_and_coordinate_descent_never_raise_full_quadratic_loss() -> None:
    torch.manual_seed(71)
    values = torch.randn(9, 64) * torch.linspace(0.1, 4.0, 64)
    samples = torch.randn(96, 64) @ torch.diag(torch.linspace(0.2, 3.0, 64))
    h = solution._block_hessian(samples, 1e-4)
    baseline = solution._dense_to_hif4(values)
    improved = solution._second_order_dense_to_hif4(values, h, max_blocks=9)
    validate_hif4(improved, values.shape)
    base_error = values - solution._dequantize_hif4(baseline)
    new_error = values - solution._dequantize_hif4(improved)
    assert torch.all(
        solution._quadratic_error(new_error.reshape(9, 1, 64), h)
        <= solution._quadratic_error(base_error.reshape(9, 1, 64), h) + 1e-6
    )


def test_each_hierarchy_sweep_is_monotone() -> None:
    losses = run_coordinate_descent_with_trace(seed=73, sweeps=2)
    assert losses[1] <= losses[0] + 1e-7
    assert losses[2] <= losses[1] + 1e-7
```

- [ ] **Step 2: Run the focused tests**

Run: `python -m pytest tests/test_second_order.py -k "gptq or hierarchy_sweep" -v`

Expected: FAIL because the second-order solver is absent.

- [ ] **Step 3: Implement sequential mantissa decisions with error feedback**

For a fixed scale/hierarchy, derive each coordinate's legal two mantissa neighbors from the HiF4 lattice. Process coordinates in descending Hessian diagonal order. For candidate `q` at position `i`, score the updated complete residual by the 64-dimensional quadratic objective; choose the lower score and propagate the selected residual through the remaining active coordinates using the damped inverse-Hessian column:

```python
delta = selected_value - working[..., i]
denom = inverse_h[..., i, i].clamp_min(_EPS)
working[..., remaining] -= (
    delta.unsqueeze(-1)
    * inverse_h[..., i, remaining]
    / denom.unsqueeze(-1)
)
quantized[..., i] = selected_value
```

Use `torch.linalg.cholesky_inverse` only on 64×64 damped factors. If it is non-finite, return the current exact-hierarchy block unchanged.

- [ ] **Step 4: Implement two-sweep eight-group hierarchy coordinate descent**

For group indices `0..7`, enumerate the eight legal `(lv2, lv3)` states already supported by `_solve_exact_hierarchy`. Replace one 8-value group at a time, reconstruct the full 64-value block, and score the full quadratic form. Commit only strict decreases larger than `1e-9`; perform exactly two forward sweeps. Preserve the globally shared E6M2 scale while updating group fields.

- [ ] **Step 5: Add E6M2 neighbor evaluation and hard-block cap**

Evaluate standard scale plus code offsets `(-2, -1, 1, 2, 3)`. Rank blocks by current full quadratic loss, refine at most `max_blocks`, and copy the baseline fields for all other blocks. At the final field level, compare complete baseline and candidate reconstruction and select baseline whenever candidate loss is higher or non-finite.

- [ ] **Step 6: Verify exact legality and deterministic output**

Run:

```powershell
python -m pytest tests/test_second_order.py tests/test_formats.py -v
python -m pytest tests/test_second_order.py -v --maxfail=1
```

Expected: all PASS twice with identical trace values.

- [ ] **Step 7: Commit**

```powershell
git add solution.py tests/test_second_order.py
git commit -m "feat: add second-order HiF4 error-feedback solver"
```

### Task 9: Integrate second-order Weight quantization and export C3/C5/C6

**Files:**
- Modify: `solution.py:870-1160`
- Modify: `tests/test_second_order.py`
- Modify: `tests/test_solution_regression.py`

**Interfaces:**
- Consumes: Tasks 4, 7 and 8; `_ENABLE_WEIGHT_SECOND_ORDER` from Task 2.
- Produces: Weight solver metadata in activation state: `weight_solver_mode`, `hessian_damping`, `second_order_blocks`, `second_order_fallback_blocks`.

- [ ] **Step 1: Add tests proving the full Weight objective cannot regress**

```python
def test_weight_second_order_is_gated_by_full_hessian_objective() -> None:
    module = load_solution_with_flags(weight_second_order=True)
    case = make_correlated_linear_case(seed=79, in_features=128, out_features=96)
    result = module.hif4_calibration_and_quantize_weight(*case.weight, case.calibration)
    state = result["activation_state"]
    assert state["second_order_fallback_blocks"] >= 0
    assert state["second_order_blocks"] >= state["second_order_fallback_blocks"]
    assert_weight_quadratic_not_worse(module, case, result)


def test_weight_second_order_feature_off_matches_champion() -> None:
    current = load_solution_with_flags(weight_second_order=False, linear_h64=False)
    frozen = load_solution(ROOT / "solution_10250_champion.py")
    assert_linear_public_outputs_equal(current, frozen, seed=83)
```

- [ ] **Step 2: Run focused tests before integration**

Run: `python -m pytest tests/test_second_order.py tests/test_solution_regression.py -k "weight_second_order" -v`

Expected: FAIL because calibration does not publish or use second-order metadata.

- [ ] **Step 3: Build `H_A` after the selected exact transform**

For every sampled activation apply the selected `D^-1`, permutation and rotation, concatenate only sampled rows, then call `_block_hessian`. This operation may compute `A.T @ A`; it must never multiply those samples by Weight. Select full Cholesky for at most 64 groups and diag+rank-8 beyond that.

- [ ] **Step 4: Run the second-order solver only on bounded hard blocks**

Use exact caps:

```python
weight_second_order_cap = 32_768 if weight.numel() <= 4_194_304 else 8_192
```

Generate both the current Champion candidate and second-order candidate. Compare per row/group with `_quadratic_error`; copy all five baseline HiF4 fields where the new objective is not lower. Store counts in activation state through Python integers.

- [ ] **Step 5: Run paired standard evaluation and export candidates**

Run:

```powershell
python -m pytest tests/test_second_order.py tests/test_solution_regression.py tests/test_compliance.py -v
python cli.py export-candidate C3 --source solution.py --predicted-seconds 210 --output-root artifacts/first-wave
python cli.py export-candidate C5 --source solution.py --predicted-seconds 245 --output-root artifacts/first-wave
python cli.py export-candidate C6 --source solution.py --predicted-seconds 260 --output-root artifacts/first-wave
python cli.py audit --candidate artifacts/first-wave/C3/solution.py --incumbent solution_10250_champion.py --tier standard --split dev --root artifacts/audits/C3
python cli.py audit --candidate artifacts/first-wave/C5/solution.py --incumbent solution_10250_champion.py --tier standard --split dev --root artifacts/audits/C5
python cli.py audit --candidate artifacts/first-wave/C6/solution.py --incumbent solution_10250_champion.py --tier standard --split dev --root artifacts/audits/C6
```

Expected: all candidates are under the 270-second predicted gate and retain independent flag sets.

- [ ] **Step 6: Commit**

```powershell
git add solution.py tests/test_second_order.py tests/test_solution_regression.py
git commit -m "feat: integrate full-Hessian weight quantization"
```

### Task 10: Add bounded second-order Activation quantization

**Files:**
- Modify: `solution.py:1118-1187`
- Modify: `hif4_system/candidates.py`
- Modify: `tests/test_second_order.py`
- Modify: `tests/test_solution_regression.py`

**Interfaces:**
- Consumes: quantized `weight_hat`, Task 7 factor format, Task 8 solver.
- Produces: activation state keys `activation_hessian_factor`, `activation_second_order_ratio`, `activation_second_order_cap`; second-wave feature `activation_second_order`; candidate `A1={activation_second_order}` added to `all_candidates()`.

- [ ] **Step 1: Write tests for factor serialization and online fallback**

```python
def test_activation_hessian_state_is_legal_and_online_result_is_finite() -> None:
    module = load_solution_with_flags(activation_second_order=True)
    calibrated = call_linear_calibration(module, seed=89, in_features=128)
    validate_state(calibrated["activation_state"])
    params = call_dynamic_activation(module, calibrated["activation_state"], seed=97)
    validate_hif4(params)
    assert torch.isfinite(module._dequantize_hif4(params)).all()


def test_activation_second_order_falls_back_when_factor_is_invalid() -> None:
    module = load_solution_with_flags(activation_second_order=True)
    state = call_linear_calibration(module, seed=101, in_features=64)["activation_state"]
    state["activation_hessian_factor"]["diag"][0] = float("nan")
    actual = call_dynamic_activation(module, state, seed=103)
    state["activation_second_order_ratio"] = 0.0
    fallback = call_dynamic_activation(module, state, seed=103)
    assert_hif4_equal(actual, fallback)
```

- [ ] **Step 2: Run the new tests**

Run: `python -m pytest tests/test_second_order.py -k "activation_hessian or activation_second_order" -v`

Expected: FAIL because factor state is absent.

- [ ] **Step 3: Construct `H_W` without runtime Linear outputs**

After final Weight quantization, reconstruct `weight_hat`, reshape its input dimension into groups, and compute each `64×64` Gram matrix with `einsum("ogi,ogj->gij", ...)`. Factor it with Task 7 and move every tensor through `_cpu_state_tensor`. Use Cholesky only when serialized state is at most 2 MiB; otherwise store diag+rank-8.

- [ ] **Step 4: Bound online second-order work**

Rank activation blocks using the current weighted reconstruction loss. Apply one GPTQ pass to the worst 20%, capped at 8192 blocks; use 10% and cap 2048 when `activation_quant.numel() > 262_144`. Compare full factored loss per block and copy all five current fields when the second-order result is not strictly better.

- [ ] **Step 5: Verify CPU timing and no state mutation**

Run:

```powershell
python -m pytest tests/test_second_order.py tests/test_solution_regression.py tests/test_suites_evaluator.py -v
python cli.py export-candidate A1 --source solution.py --predicted-seconds 260 --output-root artifacts/second-wave
python cli.py audit --candidate artifacts/second-wave/A1/solution.py --incumbent solution_10250_champion.py --tier standard --split dev --root artifacts/activation-second-order
```

Expected: state-copy test PASS; authoritative CPU projected time is recorded. If projection exceeds 270 seconds, reduce only the documented block ratio/cap and rerun the same paired audit.

- [ ] **Step 6: Commit**

```powershell
git add solution.py hif4_system/candidates.py tests/test_second_order.py tests/test_solution_regression.py
git commit -m "feat: add bounded second-order activation rounding"
```

### Task 11: Add the Linear cross-term as an isolated second-wave experiment

**Files:**
- Modify: `solution.py:870-1187`
- Modify: `hif4_system/candidates.py`
- Modify: `tests/test_second_order.py`
- Modify: `评测算法与初赛任务书判题流程对比分析.md`

**Interfaces:**
- Consumes: transformed calibration inputs, quantized Weight, candidate Activation residuals; `_ENABLE_LINEAR_CROSS_TERM`.
- Produces: `_linear_cross_term(delta_a: torch.Tensor, a: torch.Tensor, delta_w: torch.Tensor, weight_hat: torch.Tensor) -> torch.Tensor`; candidate `X1` containing only `weight_second_order + activation_second_order + linear_cross_term`.

- [ ] **Step 1: Write an algebraic equivalence test outside submission code**

```python
def test_cross_term_matches_expanded_output_error_for_test_only() -> None:
    torch.manual_seed(107)
    a = torch.randn(9, 64)
    w = torch.randn(7, 64)
    ah = a + 0.03 * torch.randn_like(a)
    wh = w + 0.03 * torch.randn_like(w)
    direct = (ah @ wh.T - a @ w.T).square().sum()
    da, dw = ah - a, wh - w
    separated = (
        (a @ dw.T).square().sum()
        + (da @ wh.T).square().sum()
        + solution._linear_cross_term(da, a, dw, wh)
    )
    torch.testing.assert_close(direct, separated, rtol=2e-5, atol=2e-5)
```

The `@` operations appear only in this mathematical test. Production `_linear_cross_term` must use the trace form from the spec.

- [ ] **Step 2: Run the equivalence test**

Run: `python -m pytest tests/test_second_order.py -k cross_term -v`

Expected: FAIL because `_linear_cross_term` is absent.

- [ ] **Step 3: Implement the trace form and rule guard**

```python
def _linear_cross_term(delta_a, a, delta_w, weight_hat):
    left = torch.einsum("ni,nj->ij", delta_a, a)
    right = torch.einsum("oi,oj->ij", delta_w, weight_hat)
    return 2.0 * torch.einsum("ij,ji->", left, right)
```

Before enabling this feature, extend static compliance with an AST test that the Linear calibration call graph contains no matrix multiplication (`@`, `torch.matmul`, `torch.mm`, `torch.bmm`, `einsum` output signatures combining activation rows with Weight rows). Document why the trace factors are legal inputs rather than reconstructed Linear outputs.

- [ ] **Step 4: Register and export only the isolated X1 candidate**

`X1.enabled` is exactly `{"weight_second_order", "activation_second_order", "linear_cross_term"}`. Do not add it to C0–C6. Run standard dev audit and a fresh compliance scan before any holdout or official submission.

- [ ] **Step 5: Verify and commit**

Run:

```powershell
python -m pytest tests/test_second_order.py tests/test_compliance.py tests/test_candidate_export.py -v
python cli.py export-candidate X1 --source solution.py --predicted-seconds 260 --output-root artifacts/second-wave
```

Expected: PASS and export succeeds only if the production Linear call graph remains legal.

```powershell
git add solution.py hif4_system/candidates.py tests/test_second_order.py tests/test_compliance.py 评测算法与初赛任务书判题流程对比分析.md
git commit -m "feat: isolate legal linear cross-term experiment"
```

---

## Official Checkpoint 1 — Manual C0–C6 Factorial Readout

This checkpoint is intentionally between fixed mechanisms and learned mechanisms. It consumes seven of the daily 30 manual submissions and determines which branch deserves further work.

- [ ] Confirm every `artifacts/first-wave/C*/manifest.json` has a unique source SHA and predicted time at most 270 seconds.
- [ ] Submit C0, C1, C2, C3, C4, C5 and C6 manually; record the returned total score and runtime with `python cli.py record-official ...`.
- [ ] Compute effects from total scores: Linear rotation `C1-C0`, Attention rotation `C2-C0`, Weight second-order `C3-C0`, rotation interaction `C4-C1-C2+C0`, Linear/second-order interaction `C5-C1-C3+C0`, full residual `C6-C4-C5+C1`.
- [ ] Repeat the highest-scoring candidate once. Promote it only if the repeat remains above C0 and runtime remains below 300 seconds.
- [ ] If every mechanism is below C0, retain the code behind flags, keep `solution_10250_champion.py` as Champion, and diagnose local/official rank disagreement before executing Phase D.

Use this exact PowerShell pattern for each result:

```powershell
python cli.py record-official --candidate-id C1 --source artifacts/first-wave/C1/solution.py --score 11000 --runtime-seconds 190 --root artifacts/manual-official
```

Replace the example values `11000` and `190` with the actual official values shown by the website. The command itself never submits anything.

---

## Phase D — Learned Attention Rotation and V Bias Control

### Task 12: Implement orthogonal butterfly layers and discrete final selection

**Files:**
- Modify: `solution.py:89-128,1189-2238`
- Modify: `hif4_system/candidates.py`
- Modify: `tests/test_rotations.py`
- Create: `tests/test_attention_objective.py`

**Interfaces:**
- Consumes: best fixed H64 initialization and true Attention metric from Tasks 3 and 5.
- Produces: `_apply_butterfly(x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor`, `_learn_attention_rotation(samples: Sequence[dict[str, torch.Tensor]], initial_signs: torch.Tensor, metadata: dict[str, int]) -> dict[str, Any]`; candidate `B1={attention_h64, learned_butterfly}` added to `all_candidates()`.

- [ ] **Step 1: Write butterfly orthogonality and split tests**

```python
def test_butterfly_preserves_inner_products() -> None:
    torch.manual_seed(109)
    x = torch.randn(7, 64)
    y = torch.randn(5, 64)
    angles = torch.randn(6, 8) * 0.1
    xr = solution._apply_butterfly(x, angles)
    yr = solution._apply_butterfly(y, angles)
    torch.testing.assert_close(x @ y.T, xr @ yr.T, rtol=3e-5, atol=3e-5)


def test_attention_learning_and_validation_windows_do_not_overlap() -> None:
    train, validation = split_attention_samples(make_attention_samples(tokens=64, count=1))
    assert set(train[0]["token_ids"]).isdisjoint(validation[0]["token_ids"])
    assert len(train[0]["token_ids"]) >= 16
    assert len(validation[0]["token_ids"]) >= 16
```

- [ ] **Step 2: Run tests before implementation**

Run: `python -m pytest tests/test_rotations.py tests/test_attention_objective.py -k "butterfly or validation_windows" -v`

Expected: FAIL due to missing butterfly and split helpers.

- [ ] **Step 3: Implement six orthogonal butterfly stages**

Each stage contains disjoint 2×2 rotations. Stage strides are `(1, 2, 4, 8, 16, 32)`; each stage shares eight angles cyclically across its 32 pairs. Apply row-vector rotations using:

```python
left_new = left * torch.cos(theta) - right * torch.sin(theta)
right_new = left * torch.sin(theta) + right * torch.cos(theta)
```

The function must allocate no 64×64 dense matrix. Angle state shape is `[kv_heads, head_dim // 64, 6, 8]`.

- [ ] **Step 4: Implement train/validation splitting and the true objective**

With two or more calibration samples, use even-index samples for learning and odd-index samples for validation. With one sample, use the first half of sampled tokens for learning and the second half for validation. Define:

```python
def _attention_ratio_objective(candidate_mse, standard_mse, tail_weight=0.25):
    ratio = candidate_mse / standard_mse.clamp_min(1e-12)
    return ratio.mean() + tail_weight * torch.relu(ratio.max() - 1.0)
```

All Attention calls are non-causal and use the existing GQA output routine.

- [ ] **Step 5: Optimize with STE but select with real HiF4**

Initialize angles so the learned transform composes after the best H64. Run Adam for exactly 16 steps with learning rate `0.03`, clip angle gradients to norm `1.0`, and stop immediately on non-finite loss. The STE helper is:

```python
def _ste_hif4(x: torch.Tensor) -> torch.Tensor:
    quantized = _dequantize_hif4(_dense_to_hif4(x))
    return x + (quantized - x).detach()
```

After optimization, detach angles and re-run Identity, fixed H64 and learned butterfly through genuine `_dense_to_hif4` on validation data. Learned rotation wins only if mean ratio improves at least 1%, every validation sample is no worse than fixed H64 by more than 0.2%, and all outputs are finite.

- [ ] **Step 6: Verify NaN fallback and serialized state**

Run:

```powershell
python -m pytest tests/test_rotations.py tests/test_attention_objective.py tests/test_solution_regression.py -v
python cli.py export-candidate B1 --source solution.py --predicted-seconds 260 --output-root artifacts/second-wave
python cli.py audit --candidate artifacts/second-wave/B1/solution.py --incumbent solution_10250_champion.py --tier standard --split dev --root artifacts/learned-butterfly
```

Expected: PASS; injected non-finite optimization returns fixed H64; stored angles are finite CPU contiguous tensors without gradients.

- [ ] **Step 7: Commit**

```powershell
git add solution.py hif4_system/candidates.py tests/test_rotations.py tests/test_attention_objective.py tests/test_solution_regression.py
git commit -m "feat: learn validated butterfly attention rotations"
```

### Task 13: Add V mean-bias-aware quantization under true Attention gating

**Files:**
- Modify: `solution.py:1270-1453,1553-2238`
- Modify: `hif4_system/candidates.py`
- Modify: `tests/test_attention_objective.py`
- Modify: `tests/test_solution_regression.py`

**Interfaces:**
- Consumes: existing V standard/refined candidates and `_attention_true_metrics`.
- Produces: `_v_bias_loss(reference: torch.Tensor, reconstructed: torch.Tensor, bias_weight: float) -> torch.Tensor`; V state keys `bias_weight`, `bias_candidate_enabled`; candidate `V1={v_bias}` added to `all_candidates()`.

- [ ] **Step 1: Write tests for mean-error sensitivity and V coordinate preservation**

```python
def test_v_bias_loss_penalizes_coherent_token_error() -> None:
    reference = torch.zeros(32, 64)
    coherent = torch.full_like(reference, 0.1)
    alternating = coherent.clone()
    alternating[1::2].neg_()
    coherent_loss = solution._v_bias_loss(reference, coherent, 0.5)
    alternating_loss = solution._v_bias_loss(reference, alternating, 0.5)
    assert coherent_loss > alternating_loss


def test_v_path_never_applies_rotation_or_centering() -> None:
    module = load_solution_with_flags(v_bias=True, attention_h64=True)
    states = call_attention_calibration(module, seed=113)
    assert states["v_state"].get("rotation_signs") is None
    assert states["v_state"].get("center_mode", 0) == 0
```

- [ ] **Step 2: Run focused tests**

Run: `python -m pytest tests/test_attention_objective.py -k "v_bias or v_path" -v`

Expected: FAIL because `_v_bias_loss` and V policy metadata are absent.

- [ ] **Step 3: Implement the bias-aware objective**

```python
def _v_bias_loss(reference, reconstructed, bias_weight):
    error = reconstructed.to(torch.float32) - reference.to(torch.float32)
    token_count = max(int(error.shape[0]), 1)
    local = error.square().sum()
    coherent = float(token_count) * error.mean(dim=0).square().sum()
    return local + float(bias_weight) * coherent
```

Evaluate bias weights `(0.0, 0.25, 0.5)` only when `_ENABLE_V_BIAS` is true. Reuse the current HiF4 field candidates; this task changes their ranking objective, not the format or V coordinates.

- [ ] **Step 4: Gate the selected V candidate with real non-causal Attention output**

Compare standard V, current local-MSE V and bias-aware V while Q/K remain fixed at their selected transforms. Require mean Attention ratio improvement of 0.5%, no sample degradation above 0.5%, and no increase in the worst ratio. Otherwise store `bias_weight=0.0` and use the current Champion V fields.

- [ ] **Step 5: Verify and commit**

Run:

```powershell
python -m pytest tests/test_attention_objective.py tests/test_solution_regression.py tests/test_scoring.py -v
python cli.py export-candidate V1 --source solution.py --predicted-seconds 260 --output-root artifacts/second-wave
python cli.py audit --candidate artifacts/second-wave/V1/solution.py --incumbent solution_10250_champion.py --tier standard --split dev --root artifacts/v-bias
```

Expected: PASS; Attention cases report the V policy enable rate and remain finite.

```powershell
git add solution.py hif4_system/candidates.py tests/test_attention_objective.py tests/test_solution_regression.py
git commit -m "feat: add attention-gated V bias suppression"
```

---

## Phase E — Full Verification, Holdout, and Champion Promotion

### Task 14: Run compliance, generalization, timing and documentation gates

**Files:**
- Modify: `README.md`
- Modify: `HIF4_泛化评估环境说明.md`
- Modify: `HiF4_最优算法设计.md`
- Modify: `tests/test_end_to_end.py`

**Interfaces:**
- Consumes: all prior tasks and the winning official mechanism from Checkpoint 1.
- Produces: one SHA-bound release candidate, complete local report set, one holdout decision and reproducible manual submission instructions.

- [ ] **Step 1: Add an end-to-end release gate test**

```python
def test_release_candidate_has_complete_auditable_artifacts(tmp_path: Path) -> None:
    exported = export_candidate(
        ROOT / "solution.py", candidate_matrix()["C6"], tmp_path / "exports", 235.0
    )
    run = _run_cli(
        tmp_path / "audit", "audit", "--candidate", str(exported),
        "--incumbent", str(ROOT / "solution_10250_champion.py"),
        "--tier", "smoke", "--split", "dev",
    )
    assert run.returncode in (0, 5)
    manifest = json.loads(exported.with_name("manifest.json").read_text(encoding="utf-8"))
    assert manifest["source_sha256"] == _sha256(exported)
    assert manifest["predicted_official_seconds"] <= 270.0
    assert check_static(exported).passed
```

- [ ] **Step 2: Materialize the winning official flags into root `solution.py`**

Read `best_official_result(Path("artifacts/manual-official"))`, verify its recorded source still has the same SHA, and copy only its exact candidate flag block into root `solution.py`. Do not copy generated manifests or artifact paths. Re-export the same feature set and assert the resulting file has identical algorithm text and SHA. If Checkpoint 1 has no official winner above C0, keep all root flags false so `solution.py` remains behaviorally equivalent to the 10250 Champion.

- [ ] **Step 3: Run the full unit and smoke suite**

Run:

```powershell
python -m pytest -q
python -m py_compile solution.py solution_10250_champion.py
```

Expected: every test PASS; both submissions parse under the configured Python version.

- [ ] **Step 4: Run standard and soak paired development audits**

Run:

```powershell
python cli.py audit --candidate solution.py --incumbent solution_10250_champion.py --tier standard --split dev --root artifacts/release-standard
python cli.py audit --candidate solution.py --incumbent solution_10250_champion.py --tier soak --split dev --root artifacts/release-soak
```

Acceptance gates:

- overall paired mean delta is positive;
- neither Linear nor Attention mean delta is negative by more than 0.2 percentage points;
- negative-case count does not increase;
- worst-decile score does not fall by more than 0.5 percentage points;
- CPU projected official time is at most 270 seconds;
- feature selection and fallback rates are nonzero where the winning mechanism is expected to act.

- [ ] **Step 5: Consume one anonymous holdout decision**

Run only after Step 3 passes:

```powershell
python cli.py audit --candidate solution.py --incumbent solution_10250_champion.py --tier standard --split holdout --root artifacts/release-holdout
```

Expected: the campaign records the holdout reservation without writing raw seeds. A failed holdout is recorded and not rerun under the same candidate SHA.

- [ ] **Step 6: Document the exact operator pipeline and manual loop**

Update documentation with these ordered paths:

1. Linear: NVFP4 dequantize → D/P → selected rotation → bounded second-order HiF4 → per-block Champion fallback.
2. Attention Q/K: exact centering/D/P → shared KV/GQA rotation → true non-causal candidate gate → HiF4.
3. Attention V: original coordinates → bias-aware candidate ranking → true output gate → HiF4.
4. Offline: export → compliance → paired CPU audit → optional CUDA accuracy → holdout → manual official submit → SHA-bound result record → Champion decision.

Include exact CLI commands from Tasks 1, 2, 6, 9 and this task. State explicitly that no local score is an estimate of official absolute points; it is only a paired ranking signal.

- [ ] **Step 7: Re-run release verification after documentation and final source changes**

Run:

```powershell
git diff --check
python -m pytest -q
python cli.py validate --candidate solution.py --tier standard --root artifacts/final-validation
git status --short
```

Expected: no whitespace errors; all tests PASS; validation passes all configured tracks; status contains only intended files.

- [ ] **Step 8: Commit the verified release candidate**

```powershell
git add solution.py README.md HIF4_泛化评估环境说明.md HiF4_最优算法设计.md tests/test_end_to_end.py
git commit -m "feat: validate principled HiF4 optimization candidate"
```

---

## Official Checkpoint 2 — Learned and Second-Wave Candidates

After Phase E passes, spend official submissions only around the best first-wave mechanism:

- fixed H64 winner vs H8 equivalent;
- fixed H64 winner vs learned butterfly with every other flag identical;
- second-order full Cholesky vs diag+rank-8 with every other flag identical;
- V standard vs V bias-aware with every other flag identical;
- pure Hessian winner vs isolated X1 cross-term only after compliance review.

For each comparison, submit one variant, record score/runtime, and repeat only the winner. Promote a new Champion when its repeated official score is higher and both runs are below 300 seconds. Preserve prior Champion files and registry history; never overwrite `solution_10250_champion.py`.

## Final Acceptance Checklist

- [ ] Six official APIs and all HiF4/state validators pass.
- [ ] C0 with all features off is field-for-field equivalent to `solution_10250_champion.py`.
- [ ] H64/H8/butterfly mathematical invariants pass for Linear, MHA and GQA.
- [ ] Every second-order accepted block has non-increasing full quadratic loss.
- [ ] Cholesky, learned rotation and non-finite failures take deterministic fallback paths.
- [ ] V never changes coordinate system.
- [ ] Production Linear calibration call graph contains no forbidden output reconstruction.
- [ ] Exported submissions are standalone and statically compliant.
- [ ] Standard, soak and the single authorized holdout reports are SHA-bound and auditable.
- [ ] Predicted official runtime is at most 270 seconds; observed official runtime is below 300 seconds.
- [ ] Official results are manually entered and append-only.
- [ ] The winning official candidate is repeated before Champion promotion.
