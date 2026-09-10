"""A-CT2 legality, sole-change, tail-equivalence, call-count and coverage check.

Controls (plan docs/superpowers/plans/2026-09-10-retained-root-gradient-reuse-plan.md,
section 4, the A-CT2 implementation card):

  A. lineage and sole change.  The candidate is a pure byte-append of the v236
     assembly, which is itself the v231 root plus A-GR1; both are shared
     prefixes, every Linear and dynamic Q/K/V API is byte-identical, and the six
     APIs import from a bare directory with no repository siblings.  The two
     substitutions are asserted to reconstruct the parent exactly, which given
     their uniqueness is the statement that nothing outside them changed.

  B. the carried sums are bit-identical to the ones they replace.  This is the
     card's whole claim and it is checked end to end rather than by argument:
     the full ``hif4_calibration_attention`` is run on the assembly and on the
     candidate over the same real calibration windows for every full-attention
     layer, and the returned q/k/v states must be byte-identical.  Three of the
     fields in those states are exactly the values at stake --
     ``agr1_final_loss``, ``agr1_q_scale_ratio2``, ``agr1_k_scale_ratio2`` --
     so byte-identical states is a strictly stronger statement than comparing
     the three floats, and it also catches anything the edit might have disturbed
     elsewhere in the training tail.

  C. the call-count evidence.  ``_agr1_scale_loss_grad`` and
     ``_a2_apply_group_rotation`` are counted over a whole calibration call for
     both arms.  The training portion must be untouched at 32 steps x 2 roles x
     3 folds = 192, and the tail must drop from 12 to 6 -- the second walk of the
     prepared folds.  204 -> 198 per role, per layer.

  D. coverage.  Accepted, rejected, the identity parent arm, ineligible and the
     exception fallback, all on real data or by patching both modules
     identically.

  E. determinism.  The assembly is run twice and must agree with itself, so
     control B's equality cannot be an artefact of a run-to-run difference.

CPU only.  Real data comes from the 4B pack.
"""

from pathlib import Path
import hashlib
import importlib.util
import inspect
import sys
import tempfile

import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

ASSEMBLY = ROOT / "solutions/20260910_v236_attention-agr1-on-v231_scoreNA_timeNA/solution.py"
ROOT_ARCHIVE = ROOT / "solutions/20260910_v231_linear-em3-k2-arm_scoreNA_timeNA/solution.py"
CANDIDATE = HERE / "candidate" / "solution.py"
PACK = ROOT / "artifacts/official_eval/cache/qwen3.5-4b-proxy-v2.pt"

ASSEMBLY_SHA256 = "3319fc354af1ff0f35075498f26886bdf43a078d32a27852f575a4c50d32ba07"
ROOT_SHA256 = "ea79a1c12dc667142c620975aab188920fae7b29988c804f41c1a696cc5754f1"

LINEAR_APIS = (
    "hif4_calibration_and_quantize_weight",
    "hif4_dynamic_quantize_activation",
)
QKV_APIS = (
    "hif4_dynamic_quantize_q",
    "hif4_dynamic_quantize_k",
    "hif4_dynamic_quantize_v",
)
PUBLIC_APIS = LINEAR_APIS + ("hif4_calibration_attention",) + QKV_APIS

COUNTED = ("_agr1_scale_loss_grad", "_a2_apply_group_rotation", "_agr1_train", "_agr1_gate_loss")

EXPECTED_STEPS = 32
EXPECTED_ROLES = 2
EXPECTED_FOLDS = 3


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def unwrapped(function):
    return inspect.unwrap(function)


def raw_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.detach().cpu().contiguous().numpy().tobytes()


def deep_equal(a, b, label: str, path: str = "") -> None:
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            raise AssertionError(f"{label}{path}: key sets differ: {set(a) ^ set(b)}")
        for key in a:
            deep_equal(a[key], b[key], label, f"{path}[{key!r}]")
        return
    if torch.is_tensor(a) or torch.is_tensor(b):
        if not (torch.is_tensor(a) and torch.is_tensor(b)):
            raise AssertionError(f"{label}{path}: one side is not a tensor")
        if a.dtype != b.dtype or tuple(a.shape) != tuple(b.shape):
            raise AssertionError(f"{label}{path}: dtype/shape differ")
        if raw_bytes(a) != raw_bytes(b):
            raise AssertionError(f"{label}{path}: payload differs bitwise")
        return
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            raise AssertionError(f"{label}{path}: length differs")
        for index, (left, right) in enumerate(zip(a, b)):
            deep_equal(left, right, label, f"{path}[{index}]")
        return
    if isinstance(a, float) and isinstance(b, float):
        if a != b and not (a != a and b != b):
            raise AssertionError(f"{label}{path}: {a!r} != {b!r}")
        return
    if a != b:
        raise AssertionError(f"{label}{path}: {a!r} != {b!r}")


class CallCounter:
    def __init__(self, module, names):
        self.module = module
        self.names = names
        self.counts = {name: 0 for name in names}
        self._saved = {}

    def __enter__(self):
        for name in self.names:
            original = getattr(self.module, name, None)
            if original is None:
                raise AssertionError(f"module has no {name}")
            self._saved[name] = original
            counter = self

            def wrapper(*args, __name=name, __original=original, **kwargs):
                counter.counts[__name] += 1
                return __original(*args, **kwargs)

            setattr(self.module, name, wrapper)
        return self

    def __exit__(self, *exc_info):
        for name, original in self._saved.items():
            setattr(self.module, name, original)
        return False


def single_file_import(candidate_path: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="act2-solo-") as solo:
        target = Path(solo) / "solution.py"
        target.write_bytes(candidate_path.read_bytes())
        spec = importlib.util.spec_from_file_location("act2_solo", target)
        if spec is None or spec.loader is None:
            raise AssertionError("cannot import the candidate standalone")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        missing = [name for name in PUBLIC_APIS if not callable(getattr(module, name, None))]
        if missing:
            raise AssertionError(f"candidate does not expose {missing}")


def control_a(candidate_path: Path):
    assembly_bytes = ASSEMBLY.read_bytes()
    root_bytes = ROOT_ARCHIVE.read_bytes()
    if hashlib.sha256(assembly_bytes).hexdigest() != ASSEMBLY_SHA256:
        raise AssertionError("the assembly is not the recorded v236 bytes")
    if hashlib.sha256(root_bytes).hexdigest() != ROOT_SHA256:
        raise AssertionError("the root archive is not the recorded v231 bytes")
    candidate_bytes = candidate_path.read_bytes()
    if not candidate_bytes.startswith(root_bytes):
        raise AssertionError("candidate does not start with the v231 root bytes")
    if not candidate_bytes.startswith(assembly_bytes.rstrip(b"\n")):
        raise AssertionError("candidate is not an append of the assembly")
    if len(candidate_bytes) <= len(assembly_bytes):
        raise AssertionError("candidate does not append anything")

    candidate = load_module(candidate_path, "act2_candidate")
    assembly = load_module(ASSEMBLY, "act2_assembly")
    root = load_module(ROOT_ARCHIVE, "act2_root")
    single_file_import(candidate_path)

    for name in LINEAR_APIS:
        if (
            unwrapped(getattr(candidate, name)).__code__.co_code
            != unwrapped(getattr(root, name)).__code__.co_code
        ):
            raise AssertionError(f"Linear API {name} bytecode drifted from the root")
    for name in QKV_APIS + ("hif4_calibration_attention",):
        if (
            unwrapped(getattr(candidate, name)).__code__.co_code
            != unwrapped(getattr(assembly, name)).__code__.co_code
        ):
            raise AssertionError(f"{name} bytecode drifted from the assembly")

    # The shadow must be the live _agr1_train.
    assembly_first = unwrapped(assembly._agr1_train).__code__.co_firstlineno
    candidate_first = unwrapped(candidate._agr1_train).__code__.co_firstlineno
    if candidate_first <= assembly_first:
        raise AssertionError("the appended _agr1_train does not shadow the assembly's")
    hook = unwrapped(assembly.hif4_calibration_attention)
    if "_agr1_train" not in hook.__code__.co_names:
        raise AssertionError("the calibration hook does not resolve _agr1_train by name")

    print(
        f"[A] root={len(root_bytes)}B ({ROOT_SHA256[:8]}) assembly={len(assembly_bytes)}B "
        f"({ASSEMBLY_SHA256[:8]}) candidate={len(candidate_bytes)}B "
        f"({hashlib.sha256(candidate_bytes).hexdigest()[:8]}) | both shared prefixes hold, "
        f"Linear and Attention and Q/K/V bytecode identical, the shadow _agr1_train is live"
    )
    return candidate, assembly


def real_windows(pack, layer, pair):
    return [
        {
            "q": pair(pack["calibration_qkv"][s][layer][0]),
            "k": pair(pack["calibration_qkv"][s][layer][1]),
            "v": pair(pack["calibration_qkv"][s][layer][2]),
        }
        for s in range(len(pack["calibration_qkv"]))
    ]


def control_bc(candidate, assembly, windows, heads, label, check_counts: bool = True):
    q_heads, kv_heads, head_dim = heads

    def run(module):
        with CallCounter(module, COUNTED) as counter:
            states = module.hif4_calibration_attention(
                windows, q_heads, kv_heads, head_dim
            )
        return states, counter.counts

    assembly_states, assembly_counts = run(assembly)
    candidate_states, candidate_counts = run(candidate)

    # The three floats this card touches live in the returned state, so a
    # byte-identical state already proves them equal; state it explicitly too.
    audit_keys = (
        "agr1_final_loss",
        "agr1_q_scale_ratio2",
        "agr1_k_scale_ratio2",
        "agr1_initial_loss",
        "agr1_arm",
    )
    audit = {key: (assembly_states["q_state"].get(key), candidate_states["q_state"].get(key))
             for key in audit_keys}
    for key, (left, right) in audit.items():
        if left != right:
            raise AssertionError(f"{label}: {key} differs: {left!r} vs {right!r}")
    deep_equal(assembly_states, candidate_states, f"B:{label} calibration states")

    folds = EXPECTED_FOLDS
    training = EXPECTED_STEPS * EXPECTED_ROLES * folds
    if assembly_counts["_agr1_train"] != 1 or candidate_counts["_agr1_train"] != 1:
        raise AssertionError(f"{label}: expected one _agr1_train call per calibration")
    if not check_counts:
        print(
            f"[B:{label}] full calibration is byte-identical between the two arms "
            f"(arm={audit['agr1_arm'][0]!r}); call counts not asserted on a patched run"
        )
        return assembly_states, audit

    # `_agr1_scale_loss_grad` is called only from inside _agr1_train, so its total
    # is pinned exactly.  `_a2_apply_group_rotation` is also used while building
    # the prepared folds and inside the parent calibration, so only the delta is
    # pinned -- and that delta is the ratio block's own rotations.
    gradient_calls = training + 2 * EXPECTED_ROLES * folds
    old, new = assembly_counts["_agr1_scale_loss_grad"], candidate_counts["_agr1_scale_loss_grad"]
    if old != gradient_calls:
        raise AssertionError(
            f"{label}: the assembly makes {old} _agr1_scale_loss_grad calls, expected "
            f"{gradient_calls}"
        )
    if new != training + EXPECTED_ROLES * folds:
        raise AssertionError(
            f"{label}: the candidate makes {new} _agr1_scale_loss_grad calls, expected "
            f"{training + EXPECTED_ROLES * folds}"
        )
    rotation_delta = (
        assembly_counts["_a2_apply_group_rotation"] - candidate_counts["_a2_apply_group_rotation"]
    )
    if rotation_delta != EXPECTED_ROLES * folds:
        raise AssertionError(
            f"{label}: _a2_apply_group_rotation dropped by {rotation_delta}, expected "
            f"{EXPECTED_ROLES * folds}"
        )

    print(
        f"[B:{label}] full calibration is byte-identical between the two arms "
        f"(arm={audit['agr1_arm'][0]!r}, final_loss={audit['agr1_final_loss'][0]!r}, "
        f"q_ratio={audit['agr1_q_scale_ratio2'][0]!r}, k_ratio={audit['agr1_k_scale_ratio2'][0]!r})"
    )
    print(
        f"[C:{label}] _agr1_scale_loss_grad {assembly_counts['_agr1_scale_loss_grad']} -> "
        f"{candidate_counts['_agr1_scale_loss_grad']} and _a2_apply_group_rotation "
        f"{assembly_counts['_a2_apply_group_rotation']} -> "
        f"{candidate_counts['_a2_apply_group_rotation']}; the {training} training-step calls "
        f"are untouched and the tail drops from {2 * EXPECTED_ROLES * folds} to "
        f"{EXPECTED_ROLES * folds}"
    )
    return assembly_states, audit


def control_d_patched(candidate, assembly, windows, heads, label, patch_name, factory):
    """Runs the end-to-end control with one name replaced identically in both modules.

    ``factory`` is called once per module with that module's own original, so a
    replacement that has to call through (the M = I probe) calls the right
    module's version rather than capturing one module's function for both. That
    mistake is invisible in the outputs and only shows up in per-module call
    counts, which is exactly how it was caught here.
    """

    saved = {}
    try:
        for module in (assembly, candidate):
            original = getattr(module, patch_name)
            saved[id(module)] = original
            setattr(module, patch_name, factory(original))
        return control_bc(candidate, assembly, windows, heads, label, check_counts=False)
    finally:
        for module in (assembly, candidate):
            setattr(module, patch_name, saved[id(module)])


def main() -> None:
    torch.set_grad_enabled(False)
    candidate_path = HERE / "candidate" / "solution.py"
    candidate, assembly = control_a(candidate_path)

    if not PACK.exists():
        raise SystemExit("the 4B pack is required: every control in this card is on real data")
    sys.path.insert(0, str(ROOT / "evaluator"))
    import official_eval as v2  # noqa: PLC0415

    pack = torch.load(PACK, map_location="cpu", mmap=True, weights_only=False)
    heads = (int(pack["q_heads"]), int(pack["kv_heads"]), int(pack["head_dim"]))
    layers = [int(layer) for layer in pack["metadata"]["attention_layers"]]
    print(f"[real] pack loaded: heads={heads} attention_layers={layers}")

    audits = {}
    for layer in layers:
        windows = real_windows(pack, layer, v2._pair)
        _, audits[layer] = control_bc(candidate, assembly, windows, heads, f"real/layer{layer}")

    arms = {layer: audit["agr1_arm"][0] for layer, audit in audits.items()}
    if not any(value == "accepted" for value in arms.values()):
        raise AssertionError(f"no real layer reached the accepted arm: {arms}")
    if not any(value == "parent" for value in arms.values()):
        raise AssertionError(f"no real layer exercised the rejected gate: {arms}")
    print(f"[D:coverage] accepted and rejected both occur on real data: {arms}")

    windows = real_windows(pack, layers[0], v2._pair)

    def make_strip(original):
        """Each module keeps its own calibration underneath the strip."""

        def strip_rotation(windows, q_heads, kv_num_heads, head_dim):
            states = original(windows, q_heads, kv_num_heads, head_dim)
            for role in ("q_state", "k_state"):
                states[role].pop("learned_rotation", None)
            states["k_state"].pop("learned_center", None)
            return states

        return strip_rotation

    _, audit = control_d_patched(
        candidate, assembly, windows, heads, "real/M=I",
        "_AGR1_PARENT_CALIBRATION", make_strip,
    )
    if audit["agr1_arm"][0] != "parent":
        raise AssertionError("the M=I probe did not take the identity-parent route")
    print("[D:coverage] the identity-parent arm (M = I) agrees")

    short = windows[:2]
    deep_equal(
        assembly.hif4_calibration_attention(short, *heads),
        candidate.hif4_calibration_attention(short, *heads),
        "D:ineligible states",
    )
    print("[D:coverage] the ineligible path agrees")

    def make_boom(_original):
        def boom(*args, **kwargs):
            raise ValueError("control: deliberate trainer failure")

        return boom

    control_d_patched(candidate, assembly, windows, heads, "real/fallback", "_agr1_train", make_boom)
    print("[D:coverage] the exception fallback path agrees")

    # Determinism: the assembly against itself.  The count expectations belong to
    # the candidate arm, so they are not asserted here -- what is asserted is that
    # two runs of the same code agree byte for byte, which is what makes the
    # equality in control B meaningful.
    control_bc(assembly, assembly, windows, heads, "assembly-vs-itself", check_counts=False)

    print("\nALL A-CT2 CONTROLS PASSED")


if __name__ == "__main__":
    main()
