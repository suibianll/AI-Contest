# HiF4 Solution Archive Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure the repository around one active `solution.py`, a separate real-GPT evaluator, and immutable score/time-labelled archives for the historical v9 and current 10250-point solutions.

**Architecture:** The repository root owns the single editable submission. `evaluator/` owns the minimal local evaluation runtime, while `solutions/` owns immutable version directories plus a human-maintained comparison table. Existing `docs/superpowers/` material remains unchanged except for this implementation plan.

**Tech Stack:** Python 3.12, PyTorch 2.x, Transformers 4.48–5.x, Markdown, Git, PowerShell

**Spec:** `docs/superpowers/specs/2026-08-26-solution-archive-workflow-design.md`

## Global Constraints

- Keep root `solution.py` as the only active, editable submission.
- Keep evaluator code separate under `evaluator/`.
- Archive directories use `YYYYMMDD_vNNN_topic_scoreSCORE_timeTIMEs`, with official score and time at the end.
- Archived `solution.py` files are immutable byte-for-byte snapshots.
- Do not restore the deleted legacy evaluator, tests, reports, or simulation scripts.
- Do not add an archive script, database, promotion system, fallback architecture, or unit tests.
- Preserve every file under `docs/superpowers/`.
- Historical unknown values use explicit `NA` or approximate labels; never invent precise results.

---

### Task 1: Separate the local evaluator

**Files:**
- Move: `real_data_eval.py` → `evaluator/real_data_eval.py`
- Move: `nvfp4_sim.py` → `evaluator/nvfp4_sim.py`
- Move: `requirements.txt` → `evaluator/requirements.txt`
- Modify: `evaluator/real_data_eval.py`

**Interfaces:**
- Consumes: root `solution.py`; a Hugging Face GPT-2 model name or local path.
- Produces: `python evaluator/real_data_eval.py --solution solution.py --model gpt2` with Linear q/k/v/o/fc/proj and Attention scores.

- [ ] **Step 1: Create the evaluator directory and move the three runtime files**

Use PowerShell `New-Item` and `Move-Item` with explicit paths after confirming each destination does not exist. Do not copy any old `hif4_system/` file.

- [ ] **Step 2: Point the evaluator's default solution path back to the repository root**

Change:

```python
default=Path(__file__).resolve().parent / "solution.py"
```

to:

```python
default=Path(__file__).resolve().parents[1] / "solution.py"
```

- [ ] **Step 3: Verify the isolated evaluator entry point**

Run:

```powershell
.\.venv\Scripts\python.exe -B evaluator\real_data_eval.py --help
.\.venv\Scripts\python.exe -B -c "from pathlib import Path; from evaluator.real_data_eval import load_solution; print(load_solution(Path('solution.py')).__name__)"
.\.venv\Scripts\python.exe -m pip check
```

Expected: help lists `--solution`, `--model`, `--layers`, `--seq`, `--calib`, `--test`, `--mode`, and `--kv-heads`; solution loading prints `_hif4_evaluated_solution`; pip reports no broken requirements.

### Task 2: Restore and archive the historical v9 solution

**Files:**
- Restore temporarily: `solution_v9_champion.py`
- Create: `solutions/20260825_v000_v9-baseline_score9000plus_timeNA/solution.py`
- Create: `solutions/20260825_v000_v9-baseline_score9000plus_timeNA/result.md`

**Interfaces:**
- Consumes: Git-tracked `solution_v9_champion.py` at `HEAD`, expected SHA256 `a6b8b858156164333d1d3ca25c6233b4845061f40a16d4cf74695ecdbb9041f7`.
- Produces: immutable v000 archive with explicit approximate score and unknown time labels.

- [ ] **Step 1: Restore the tracked v9 source from Git**

Run:

```powershell
git restore --source=HEAD --worktree -- solution_v9_champion.py
Get-FileHash -Algorithm SHA256 solution_v9_champion.py
```

Expected SHA256: `A6B8B858156164333D1D3CA25C6233B4845061F40A16D4CF74695ECDBB9041F7`.

- [ ] **Step 2: Create the v000 archive and move the restored source into it**

Create `solutions/20260825_v000_v9-baseline_score9000plus_timeNA/`, then move `solution_v9_champion.py` to that directory as `solution.py`. There must be no `solution_v9_champion.py` left at repository root.

- [ ] **Step 3: Write the v000 result record**

Create `result.md` with these exact facts:

```markdown
# v000 — v9 baseline

- Date: 2026-08-25
- Source SHA256: `a6b8b858156164333d1d3ca25c6233b4845061f40a16d4cf74695ecdbb9041f7`
- Change: Original downloaded v9 baseline.
- Hypothesis: Historical baseline; no new mechanism was introduced.
- Local evaluator: NA — this version predates the real-GPT evaluator workflow.
- Local Linear q/k/v/o/fc/proj: NA
- Local Attention: NA
- Local runtime: NA
- Official score: approximately 9000+
- Official runtime: NA
- Official score delta: NA because the preceding official version is unavailable.
- Status: `accepted`
- Conclusion: Preserve as the pre-10250 historical baseline. Do not use its approximate official score for precise delta calculations.
- Next direction: Compare the 10250 baseline against this version only qualitatively.
```

- [ ] **Step 4: Verify the v000 archive**

Run Python syntax compilation on the archived solution and verify its SHA256 equals the expected v9 hash.

### Task 3: Archive the current official Champion

**Files:**
- Create: `solutions/20260826_v001_current-baseline_score10250_time127s/solution.py`
- Create: `solutions/20260826_v001_current-baseline_score10250_time127s/result.md`
- Preserve: `solution.py`

**Interfaces:**
- Consumes: current root `solution.py`, SHA256 `4acc1b8a4e751a75a68158efa19b2ee44f9d5a9e8821a05659d7425723f90ed5`.
- Produces: immutable v001 archive while leaving the same bytes active at root.

- [ ] **Step 1: Create the v001 directory and copy the active solution**

Copy root `solution.py` to `solutions/20260826_v001_current-baseline_score10250_time127s/solution.py`. Do not move or modify root `solution.py`.

- [ ] **Step 2: Write the v001 result record**

Create `result.md` with these exact facts:

```markdown
# v001 — current baseline

- Date: 2026-08-26
- Source SHA256: `4acc1b8a4e751a75a68158efa19b2ee44f9d5a9e8821a05659d7425723f90ed5`
- Change: Improved calibration and dynamic HiF4 quantization relative to v9.
- Hypothesis: More accurate scale search, smoothing, permutation, and output-aware calibration improve official score within the time limit.
- Local evaluator: NA — this official result predates adoption of the current real-GPT archive workflow.
- Local Linear q/k/v/o/fc/proj: NA
- Local Attention: NA
- Local runtime: NA
- Official score: 10250
- Official runtime: 127 seconds
- Official score delta: NA because v000 has only an approximate score.
- Status: `champion`
- Conclusion: This is the active official baseline for all subsequent single-mechanism experiments.
- Next direction: Run the real-GPT evaluator before every new official submission and compare local component changes against this baseline.
```

- [ ] **Step 3: Verify active/archive identity**

Run `Get-FileHash -Algorithm SHA256` on root and archived `solution.py`. Both must equal `4ACC1B8A4E751A75A68158EFA19B2EE44F9D5A9E8821A05659D7425723F90ED5`.

### Task 4: Create the human-maintained archive index

**Files:**
- Create: `solutions/README.md`

**Interfaces:**
- Consumes: v000 and v001 `result.md` records.
- Produces: one comparison table and the manual optimization workflow.

- [ ] **Step 1: Write the archive table**

Create this table with relative links:

```markdown
# HiF4 Solution Archive

Root `solution.py` is the only active submission. Archived source files are immutable.

| Version | Date | Topic | Local Linear | Local Attention | Local Time | Official Score | Official Time | Delta | Status | Directory |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| v000 | 2026-08-25 | v9 baseline | NA | NA | NA | ~9000+ | NA | NA | accepted | [archive](20260825_v000_v9-baseline_score9000plus_timeNA/) |
| v001 | 2026-08-26 | current baseline | NA | NA | NA | 10250 | 127s | NA | champion | [archive](20260826_v001_current-baseline_score10250_time127s/) |
```

- [ ] **Step 2: Add the manual workflow below the table**

Document this exact sequence: modify root `solution.py`; run the real-GPT evaluator; submit the same file officially; create the next score/time-labelled archive regardless of success or failure; verify SHA256; add `result.md`; update the table; use local/official direction agreement to select the next single-mechanism experiment.

### Task 5: Verify and commit the repository structure

**Files:**
- Verify: `solution.py`
- Verify: `evaluator/real_data_eval.py`
- Verify: `evaluator/nvfp4_sim.py`
- Verify: `evaluator/requirements.txt`
- Verify: `solutions/README.md`
- Verify: both archived `solution.py` and `result.md` files
- Preserve: `docs/superpowers/**`

**Interfaces:**
- Consumes: outputs of Tasks 1–4.
- Produces: a committed minimal repository layout ready for iterative local/official evaluation.

- [ ] **Step 1: Run syntax and dependency checks**

Run:

```powershell
.\.venv\Scripts\python.exe -m py_compile solution.py evaluator\real_data_eval.py evaluator\nvfp4_sim.py solutions\20260825_v000_v9-baseline_score9000plus_timeNA\solution.py solutions\20260826_v001_current-baseline_score10250_time127s\solution.py
.\.venv\Scripts\python.exe evaluator\real_data_eval.py --help
.\.venv\Scripts\python.exe -m pip check
```

Expected: compilation exits 0, evaluator help renders, and pip reports no broken requirements.

- [ ] **Step 2: Verify hashes and whitelist layout**

Verify:

- v000 archive SHA256 is `a6b8b858156164333d1d3ca25c6233b4845061f40a16d4cf74695ecdbb9041f7`.
- root active and v001 archive SHA256 are both `4acc1b8a4e751a75a68158efa19b2ee44f9d5a9e8821a05659d7425723f90ed5`.
- repository root contains only `.git/`, `.venv/`, `docs/`, `evaluator/`, `solutions/`, and `solution.py`.
- `docs/superpowers/` still contains all pre-existing plans/specs plus this workflow spec and plan.

- [ ] **Step 3: Remove generated `__pycache__/` directories**

Resolve each generated cache directory, confirm it is beneath the repository root and not beneath `.venv/`, then delete it. Re-run the root whitelist check.

- [ ] **Step 4: Commit the implementation**

Stage the intended repository restructure, including the previously approved deletions, evaluator move, both solution archives, archive records, and index. Do not stage `.venv/`.

```powershell
git add -A -- . ':(exclude).venv'
git commit -m "chore: organize solution archive workflow"
```
