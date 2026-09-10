"""Build L-MC1 from the v237 root.

The change moves one computation, it does not replace it:

  parent  calibration stores only `gram_diag_mean` (a scalar it already had);
          every dynamic call rebuilds G from h_inv with
              inverse = cholesky_inverse(cholesky(h_inv))
              ridge   = mean(diag(inverse)) - gram_diag_mean
              metric  = inverse; metric.diagonal().sub_(ridge)

  L-MC1   calibration builds G once and stores it (already ridge-subtracted);
          every dynamic call loads it.

Two appended shadows, no parent byte touched:

  `_em1_compile_metric`  gains the inverse.  It runs on the same device as the
  dynamic path (both come from the evaluator's single --algorithm-device), which
  matters: measured on this machine, CPU and CUDA `cholesky_inverse` are NOT
  bitwise equal (relative difference ~1e-6), while a float32 device round trip
  is exact.  The metric is therefore computed on the run device, stored as a CPU
  tensor (validate_state requires CPU), and moved back at call time -- exact.

  `_em1_metric`  loses the inverse entirely.  It returns the stored tensor
  read-only; nothing downstream writes to it, and the ridge is already applied,
  so the "subtract the ridge again on the second call" hazard the plan names
  cannot arise.

Equivalence rests on three things, all checked in verify.py rather than argued:

  * the same expression on the same input: the calibration computes exactly the
    sequence the dynamic path used to;
  * the same activation set: the parent's em1 arm switched off when the Cholesky
    failed, so calibration only stores a metric when that same Cholesky
    succeeded, and the dynamic arm requires the stored tensor;
  * no mutation: the descent only reshapes and multiplies the metric, so a state
    shared across calls yields the same answer on every call.

Shadows are resolved from module globals when the parent's own callers run --
`_em1_compile_metric` from the em1 weight-calibration wrapper, `_em1_metric`
from the live `_em1_dynamic_descent` (which is itself L-TF2's shadow, since
v237's append is the live definition).
"""

from pathlib import Path
import ast
import difflib
import hashlib
import json


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

PARENT = ROOT / "solutions/20260910_v237_linear-tf2-first-pass-gradient-reuse_scoreNA_timeNA/solution.py"
CANDIDATE = HERE / "candidate" / "solution.py"
MODULE_OUT = HERE / "implementation.generated.py"
BUILD_JSON = HERE / "build.json"

EXPECTED_PARENT_SHA256 = (
    "ecb1f9e510b5507e1a2dc8b95f8a84e9b51864a420b3828537a328813e2ce554"
)
EXPECTED_PARENT_BYTES = 516697

COMPILE_START = b"def _em1_compile_metric(\n"
COMPILE_END = b"\n@torch.no_grad()\ndef _em1_metric(\n"
METRIC_START = b"def _em1_metric(\n"
METRIC_END = b"\n@torch.no_grad()\ndef _em1_dynamic_descent(\n"

# --- _em1_compile_metric -----------------------------------------------------

OLD_COMPILE_STORE = """    gram_diag_mean = float(gram.diagonal().mean())

    state["em1"] = {
"""

NEW_COMPILE_STORE = """    gram_diag_mean = float(gram.diagonal().mean())

    # L-MC1: build G here, once, instead of on every dynamic call.  This is the
    # same expression on the same input the dynamic path used, run on the run
    # device so the stored value is what that path would have produced.
    metric = None
    try:
        inverse = torch.cholesky_inverse(
            torch.linalg.cholesky(h_inv.to(device=device, dtype=torch.float32))
        )
        ridge = float(inverse.diagonal().mean()) - gram_diag_mean
        inverse.diagonal().sub_(ridge)
        # Deliberately NOT _cpu_state_tensor: that helper calls .contiguous(),
        # and cholesky_inverse returns a transposed-stride tensor (stride (1, n),
        # the LAPACK layout).  The layout is part of the arithmetic -- a
        # contiguous copy carries the same values but makes the downstream .mm()
        # take a different cuBLAS path and round differently, which was measured
        # at the full five-field level before this was fixed.  nan_to_num and the
        # device round trip both preserve strides.
        metric = torch.nan_to_num(
            inverse.detach().to(device="cpu", dtype=torch.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
    except RuntimeError:
        # The parent's em1 arm switched itself off when the Cholesky failed;
        # storing nothing keeps the candidate's activation set identical.
        metric = None

    state["em1"] = {
"""

OLD_COMPILE_PAYLOAD = """        "gram_diag_mean": gram_diag_mean,
        "channels": channels,
"""

NEW_COMPILE_PAYLOAD = """        "gram_diag_mean": gram_diag_mean,
        "metric": metric,
        "channels": channels,
"""

# --- _em1_metric -------------------------------------------------------------

OLD_METRIC_BODY = """    h_inv = state.get("h_inv")
    if not torch.is_tensor(h_inv):
        return None
    h_inv = h_inv.to(device=device, dtype=torch.float32)
    if h_inv.ndim != 2 or tuple(h_inv.shape) != (channels, channels):
        return None
    try:
        inverse = torch.cholesky_inverse(torch.linalg.cholesky(h_inv))
    except RuntimeError:
        return None
    # c = mean(diag(h_inv^{-1} - gram)).  Calibration stored only
    # mean(diag(gram)) -- a scalar it already had -- so the inverse is built
    # exactly once per dynamic call instead of once per calibration *and* once
    # per call.  The decomposition is exact in the reals; in fp32 the two means
    # carry ~1e-7 relative rounding, far below the ridge's own scale.
    gram_diag_mean = payload.get("gram_diag_mean")
    if not isinstance(gram_diag_mean, (int, float)):
        return None
    ridge = float(inverse.diagonal().mean()) - float(gram_diag_mean)
    metric = inverse
    metric.diagonal().sub_(ridge)
"""

NEW_METRIC_BODY = """    # L-MC1: the calibration already built G -- same expression, same input,
    # run device, ridge applied -- so this path only loads it.  No Cholesky, no
    # inverse, and no write: the tensor is returned as the caller's operand, and
    # the descent only reshapes and multiplies it.
    metric = payload.get("metric")
    if not torch.is_tensor(metric):
        return None
    metric = metric.to(device=device, dtype=torch.float32)
    if metric.ndim != 2 or tuple(metric.shape) != (channels, channels):
        return None
"""

HEADER = '''# ---------------------------------------------------------------------------
# L-MC1 -- the fixed metric rebuild moves from the dynamic path into calibration.
#
# `_em1_metric` rebuilt G on every dynamic call:
#
#     inverse = torch.cholesky_inverse(torch.linalg.cholesky(h_inv))
#     ridge   = mean(diag(inverse)) - gram_diag_mean
#     metric  = inverse; metric.diagonal().sub_(ridge)
#
# Nothing in that depends on the activation -- only on `h_inv` and the scalar
# `gram_diag_mean`, both fixed once calibration finishes -- so it is rebuilt
# work, and the card builds it once and stores it.
#
# The two shadows below carry the change.  `_em1_compile_metric` gains the
# inverse; `_em1_metric` loses it and loads the stored tensor instead.
#
# The stored tensor is computed on the run device and stored as a CPU tensor,
# because `validate_state` requires CPU.  That order matters: measured here, CPU
# and CUDA `cholesky_inverse` are not bitwise equal (relative difference ~1e-6),
# while a float32 CUDA->CPU->CUDA round trip is exact.  Computing on CPU would
# have changed the arithmetic; computing on the run device and round-tripping
# does not.
#
# The activation set is preserved rather than widened: the parent's em1 arm
# switched off when the Cholesky raised, so calibration stores a metric only when
# the same Cholesky succeeded, and the dynamic arm requires a stored tensor.
#
# Nothing downstream writes to the metric -- the descent reshapes and multiplies
# it -- so the ridge is applied exactly once even when one state dict serves many
# calls.  verify.py checks that by comparing the stored bytes across calls.
#
# The definitions below shadow the parent's, which stay in the file as dead
# code -- the append-only build never rewrites a parent byte.
# ---------------------------------------------------------------------------


'''


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def cut(payload: bytes, start: bytes, end: bytes, label: str) -> str:
    """Cuts from `start` to the first `end` after it.

    The end anchor is deliberately not required to be unique: v237 carries two
    `_em1_dynamic_descent` definitions (the parent's and L-TF2's shadow), so a
    whole-file uniqueness check on the descent header would always fail.  What
    matters is that the slice begins at the one definition this card edits.
    """

    if payload.count(start) != 1:
        raise SystemExit(f"{label}: start anchor occurs {payload.count(start)} times")
    begin = payload.index(start)
    stop = payload.find(end, begin)
    if stop < 0:
        raise SystemExit(f"{label}: end anchor not found after the start")
    text = payload[begin:stop].decode("utf-8")
    if "def " in text[len(start) :]:
        raise SystemExit(f"{label}: the slice holds more than one definition")
    return text


def substitute(text: str, old: str, new: str, label: str) -> str:
    if text.count(old) != 1:
        raise SystemExit(f"{label}: needle occurs {text.count(old)} times, expected 1")
    fixed = text.replace(old, new)
    if old in fixed:
        raise SystemExit(f"{label}: needle survived the substitution")
    return fixed


def main() -> dict:
    parent_bytes = PARENT.read_bytes()
    digest = sha256_bytes(parent_bytes)
    if digest != EXPECTED_PARENT_SHA256:
        raise SystemExit(f"parent SHA256 mismatch: {digest}")
    if len(parent_bytes) != EXPECTED_PARENT_BYTES:
        raise SystemExit(f"parent byte count mismatch: {len(parent_bytes)}")

    compile_text = cut(parent_bytes, COMPILE_START, COMPILE_END, "_em1_compile_metric")
    metric_text = cut(parent_bytes, METRIC_START, METRIC_END, "_em1_metric")

    fixed_compile = substitute(
        compile_text, OLD_COMPILE_STORE, NEW_COMPILE_STORE, "compile store"
    )
    fixed_compile = substitute(
        fixed_compile, OLD_COMPILE_PAYLOAD, NEW_COMPILE_PAYLOAD, "compile payload"
    )
    fixed_metric = substitute(
        metric_text, OLD_METRIC_BODY, NEW_METRIC_BODY, "metric body"
    )

    if "cholesky_inverse" in fixed_metric:
        raise SystemExit("the dynamic shadow still builds an inverse")
    if fixed_metric.count("cholesky_inverse") != 0:
        raise SystemExit("unexpected inverse in the dynamic shadow")
    if "cholesky_inverse" not in fixed_compile:
        raise SystemExit("the calibration shadow does not build the inverse")
    if "sub_(" in fixed_metric:
        raise SystemExit("the dynamic shadow still mutates its metric")

    changed_lines = 0
    byte_delta = 0
    for before, after in ((compile_text, fixed_compile), (metric_text, fixed_metric)):
        opcodes = [
            opcode
            for opcode in difflib.SequenceMatcher(
                None, before.splitlines(), after.splitlines()
            ).get_opcodes()
            if opcode[0] != "equal"
        ]
        changed_lines += sum(max(o[2] - o[1], o[4] - o[3]) for o in opcodes)
        byte_delta += len(after.encode("utf-8")) - len(before.encode("utf-8"))

    for text, label in ((fixed_compile, "compile"), (fixed_metric, "metric")):
        ast.parse(text)  # syntax check before anything is written

    module_text = (
        HEADER
        + fixed_compile.rstrip("\n")
        + "\n\n\n"
        + fixed_metric.rstrip("\n")
        + "\n"
    )
    module_bytes = module_text.encode("utf-8")
    if module_text.count("def _em1_compile_metric(") != 1:
        raise SystemExit("module should define _em1_compile_metric once")
    if module_text.count("def _em1_metric(") != 1:
        raise SystemExit("module should define _em1_metric once")

    candidate_bytes = (
        parent_bytes.rstrip(b"\n") + b"\n\n\n" + module_bytes.rstrip(b"\n") + b"\n"
    )
    if not candidate_bytes.startswith(parent_bytes.rstrip(b"\n")):
        raise SystemExit("candidate is not an append of the parent bytes")
    if candidate_bytes.count(b"def _em1_compile_metric(") != 2:
        raise SystemExit("candidate should hold the parent definition plus the shadow")
    if candidate_bytes.count(b"def _em1_metric(") != 2:
        raise SystemExit("candidate should hold the parent definition plus the shadow")

    CANDIDATE.parent.mkdir(parents=True, exist_ok=True)
    CANDIDATE.write_bytes(candidate_bytes)
    MODULE_OUT.write_bytes(module_bytes)

    info = {
        "parent_path": str(PARENT.relative_to(ROOT)).replace("\\", "/"),
        "parent_sha256": digest,
        "parent_bytes": len(parent_bytes),
        "implementation_sha256": sha256_bytes(module_bytes),
        "implementation_bytes": len(module_bytes),
        "candidate_sha256": sha256_bytes(candidate_bytes),
        "candidate_bytes": len(candidate_bytes),
        "substitutions_performed": 3,
        "substitution_sites": [
            "_em1_compile_metric: store the metric",
            "_em1_compile_metric: add the metric key to the payload",
            "_em1_metric: load the metric instead of rebuilding it",
        ],
        "changed_lines": changed_lines,
        "byte_delta": byte_delta,
        "edit": (
            "build G once in calibration and store it (ridge applied); load it in "
            "the dynamic path instead of rebuilding it"
        ),
    }
    BUILD_JSON.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    return info


if __name__ == "__main__":
    for key, value in main().items():
        print(f"{key}={value}")
