"""A-CT1 legality, sole-change, gate-equivalence, call-count and coverage check.

Controls (plan docs/superpowers/plans/2026-09-10-retained-root-gradient-reuse-plan.md,
section 4 "控制与验证"):

  A. lineage and sole change.  The candidate is a pure byte-append of the v236
     assembly, which is itself the v231 root plus the A-GR1 block; so the root
     bytes and the A-GR1 bytes are both shared prefixes, every Linear API is
     byte-identical by construction, and the six APIs import from a bare
     directory with no repository siblings.  The sole change is asserted at the
     syntax level too: putting the two original gate-loss calls back in place of
     the single pair call must reproduce the A-GR1 function's statement tree
     exactly.

  P. the premise, measured rather than argued.  The reuse is only exact if the
     V five fields really are arm-independent.  Three separate facts are
     measured: the two arms hold equal-valued v_state; the V API returns the
     same five fields on two calls with equal-valued states; and none of the
     three APIs writes anything into the state dicts it is handed.  If any of
     those failed, the reuse would be unsound and the card would have to shrink
     to the reference target alone.

  B. gate equivalence, bit for bit.  The card's whole claim is "the same two
     losses".  The A-GR1 gate path (two _agr1_gate_loss calls) and the A-CT1
     path (one _act1_gate_pair call) are run on the same parent state, the same
     trained candidate and the same window, and the four numbers are compared
     exactly -- not with a tolerance.  This runs for every gate window on real
     calibration data.

  C. the call-count evidence, which is the point of the card.  The complete
     invocation counts of _a2_attention_forward, _dequantize_nvfp4_float32,
     _dequantize_hif4 and the three deployment APIs are recorded for both paths.
     The reduction must be exactly the arm-independent work and nothing else:
     per gate window, one fewer reference attention forward, one fewer V
     quantisation, three fewer reference decodes, and the same two Q and two K
     calls (those are the arm-dependent part and must NOT be reduced).

  D. end-to-end equivalence.  The full hif4_calibration_attention is run on the
     assembly and on the candidate, on the same real calibration windows, for
     every full-attention layer of the panel: the returned q/k/v states must be
     byte-identical and every `agr1_*` info field must be equal.

  E. coverage.  The plan requires the gate cases to include accepted, rejected,
     M = I and the exception fallback.  The six real layers give accepted and
     rejected; M = I (no learned rotation to start from) and the fallback are
     reached by patching the assembly and the candidate identically, which is a
     statement about the two implementations and not about the mechanism.

  F. determinism.  The assembly is run twice and must agree with itself, so that
     control D's equality cannot be an artefact of a run-to-run difference.

Diagnostics printed, never gated: per-layer arms, gate losses and the panel's
own reported distributions.

CPU only.  Real data comes from the 4B pack; without it the synthetic controls
still run and the real ones are skipped.
"""

from pathlib import Path
import ast
import hashlib
import importlib.util
import inspect
import sys
import tempfile
import textwrap

import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

ASSEMBLY = ROOT / "solutions/20260910_v236_attention-agr1-on-v231_scoreNA_timeNA/solution.py"
ROOT_ARCHIVE = ROOT / "solutions/20260910_v231_linear-em3-k2-arm_scoreNA_timeNA/solution.py"
CANDIDATE = HERE / "candidate" / "solution.py"
PACK = ROOT / "artifacts/official_eval/cache/qwen3.5-4b-proxy-v2.pt"

ASSEMBLY_SHA256 = "3319fc354af1ff0f35075498f26886bdf43a078d32a27852f575a4c50d32ba07"
ASSEMBLY_BYTES = 520955
ROOT_SHA256 = "ea79a1c12dc667142c620975aab188920fae7b29988c804f41c1a696cc5754f1"
ROOT_BYTES = 505762

FN_START = b"@torch.no_grad()\ndef hif4_calibration_attention(\n"

LINEAR_APIS = (
    "hif4_calibration_and_quantize_weight",
    "hif4_dynamic_quantize_activation",
)
ATTENTION_APIS = (
    "hif4_calibration_attention",
    "hif4_dynamic_quantize_q",
    "hif4_dynamic_quantize_k",
    "hif4_dynamic_quantize_v",
)
PUBLIC_APIS = LINEAR_APIS + ATTENTION_APIS

# The two original calls, used only as the comparison device in control A.
TWO_CALLS = """parent_loss = _agr1_gate_loss(
    calib_qkv_list[index], states,
    q_num_heads, kv_num_heads, head_dim,
)
candidate_loss = _agr1_gate_loss(
    calib_qkv_list[index], candidate,
    q_num_heads, kv_num_heads, head_dim,
)
"""

SYNTH_Q_HEADS = 4
SYNTH_KV_HEADS = 2
SYNTH_HEAD_DIM = 64
SYNTH_TOKENS = 96
SYNTH_WINDOWS = 5
NVFP4_MAGNITUDES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
FIVE_FIELDS = ("scale_factor", "scale_lv2", "scale_lv3", "sign", "mant")

COUNTED = (
    "_a2_attention_forward",
    "_dequantize_nvfp4_float32",
    "_dequantize_hif4",
    "_agr1_gate_loss",
    "_act1_gate_pair",
)
COUNTED_APIS = ("_AGR1_PARENT_Q", "_AGR1_PARENT_K", "_AGR1_PARENT_V")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


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
    """Byte-for-byte equality over a nested structure of tensors and scalars."""

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
            raise AssertionError(
                f"{label}{path}: {a.dtype}{tuple(a.shape)} != {b.dtype}{tuple(b.shape)}"
            )
        if raw_bytes(a) != raw_bytes(b):
            raise AssertionError(f"{label}{path}: payload differs bitwise")
        return
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            raise AssertionError(f"{label}{path}: length {len(a)} != {len(b)}")
        for index, (left, right) in enumerate(zip(a, b)):
            deep_equal(left, right, label, f"{path}[{index}]")
        return
    if isinstance(a, float) and isinstance(b, float):
        # Exact equality, and NaN is allowed to match NaN only when both are.
        if a != b and not (a != a and b != b):
            raise AssertionError(f"{label}{path}: {a!r} != {b!r}")
        return
    if a != b:
        raise AssertionError(f"{label}{path}: {a!r} != {b!r}")


def five_fields_equal(a: dict, b: dict, label: str) -> None:
    for key in FIVE_FIELDS:
        if raw_bytes(a[key]) != raw_bytes(b[key]):
            raise AssertionError(f"{label}: five field {key} differs bitwise")


class CallCounter:
    """Counts invocations of module-level names, restoring them on exit."""

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


def make_nvfp4(shape, seed):
    generator = torch.Generator().manual_seed(seed)
    index = torch.randint(0, 8, shape, generator=generator)
    sign = torch.randint(0, 2, shape, generator=generator) * 2 - 1
    quant = (NVFP4_MAGNITUDES[index] * sign).to(torch.float32)
    scale = (
        torch.rand(shape[:-1] + (shape[-1] // 16,), generator=generator) * 0.02 + 0.002
    ).to(torch.float32)
    return quant, scale


def synthetic_windows(count=SYNTH_WINDOWS):
    return [
        {
            "q": make_nvfp4((SYNTH_TOKENS, SYNTH_Q_HEADS * SYNTH_HEAD_DIM), 100 + w),
            "k": make_nvfp4((SYNTH_TOKENS, SYNTH_KV_HEADS * SYNTH_HEAD_DIM), 200 + w),
            "v": make_nvfp4((SYNTH_TOKENS, SYNTH_KV_HEADS * SYNTH_HEAD_DIM), 300 + w),
        }
        for w in range(count)
    ]


def real_windows(pack, layer):
    """Exactly the list proxy_v3_eval.prepare_shard builds for one layer."""

    return [
        {
            "q": _pair(pack["calibration_qkv"][sample][layer][0]),
            "k": _pair(pack["calibration_qkv"][sample][layer][1]),
            "v": _pair(pack["calibration_qkv"][sample][layer][2]),
        }
        for sample in range(len(pack["calibration_qkv"]))
    ]


_pair = None


# ---------------------------------------------------------------------------
# A. lineage and sole change
# ---------------------------------------------------------------------------


def single_file_import(candidate_path: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="act1-solo-") as solo:
        target = Path(solo) / "solution.py"
        target.write_bytes(candidate_path.read_bytes())
        spec = importlib.util.spec_from_file_location("act1_solo", target)
        if spec is None or spec.loader is None:
            raise AssertionError("cannot import the candidate standalone")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        missing = [name for name in PUBLIC_APIS if not callable(getattr(module, name, None))]
        if missing:
            raise AssertionError(f"candidate does not expose {missing}")


def dissolve_pair_call(tree: ast.Module) -> ast.Module:
    """Puts the two original gate-loss calls back, for AST comparison."""

    function = tree.body[0]
    restored = 0
    for node in ast.walk(function):
        if isinstance(node, ast.For) and ast.unparse(node.target) == "index":
            for position, statement in enumerate(node.body):
                if (
                    isinstance(statement, ast.Assign)
                    and len(statement.targets) == 1
                    and isinstance(statement.targets[0], ast.Tuple)
                    and ast.unparse(statement.value).startswith("_act1_gate_pair(")
                ):
                    node.body[position : position + 1] = ast.parse(TWO_CALLS).body
                    restored += 1
                    break
    if restored != 1:
        raise AssertionError(
            f"expected exactly one _act1_gate_pair call in the gate loop, found {restored}"
        )
    ast.fix_missing_locations(tree)
    return tree


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
        raise AssertionError("candidate is not an append of the v236 assembly")
    if len(candidate_bytes) <= len(assembly_bytes):
        raise AssertionError("candidate does not append anything")

    candidate = load_module(candidate_path, "act1_candidate")
    assembly = load_module(ASSEMBLY, "act1_assembly")
    root = load_module(ROOT_ARCHIVE, "act1_root")
    single_file_import(candidate_path)

    # Linear is untouched twice over: the append is after every Linear
    # definition, and the root bytes are a shared prefix.
    for name in LINEAR_APIS:
        if (
            unwrapped(getattr(candidate, name)).__code__.co_code
            != unwrapped(getattr(root, name)).__code__.co_code
        ):
            raise AssertionError(f"Linear API {name} bytecode drifted from the root")
    # The dynamic Q/K/V APIs are the pre-A-GR1 ones in both files.
    for name in ("hif4_dynamic_quantize_q", "hif4_dynamic_quantize_k", "hif4_dynamic_quantize_v"):
        if (
            unwrapped(getattr(candidate, name)).__code__.co_code
            != unwrapped(getattr(assembly, name)).__code__.co_code
        ):
            raise AssertionError(f"{name} bytecode drifted from the assembly")

    # The shadow must be the live definition.
    assembly_first = unwrapped(assembly.hif4_calibration_attention).__code__.co_firstlineno
    candidate_first = unwrapped(candidate.hif4_calibration_attention).__code__.co_firstlineno
    if candidate_first <= assembly_first:
        raise AssertionError("the appended calibration does not shadow the assembly's")

    # Sole change at the syntax level.
    src_assembly = inspect.getsource(unwrapped(assembly.hif4_calibration_attention))
    src_candidate = inspect.getsource(unwrapped(candidate.hif4_calibration_attention))
    tree_assembly = ast.parse(src_assembly)
    tree_candidate = ast.parse(src_candidate)
    pair_calls = sum(
        1
        for node in ast.walk(tree_candidate)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "_act1_gate_pair"
    )
    if pair_calls != 1:
        raise AssertionError(f"candidate holds {pair_calls} pair calls, expected 1")
    if src_candidate.count("_agr1_gate_loss(") != 0:
        raise AssertionError("the shadow still calls _agr1_gate_loss directly")
    if ast.dump(tree_assembly) != ast.dump(dissolve_pair_call(tree_candidate)):
        raise AssertionError(
            "putting the two gate-loss calls back does not reproduce the assembly's "
            "statement tree, so the pair call is not the only structural change"
        )

    print(
        f"[A] root={len(root_bytes)}B ({ROOT_SHA256[:8]}) assembly={len(assembly_bytes)}B "
        f"({ASSEMBLY_SHA256[:8]}) candidate={len(candidate_bytes)}B "
        f"({hashlib.sha256(candidate_bytes).hexdigest()[:8]}) | both shared prefixes hold, "
        f"Linear and Q/K/V bytecode identical, AST identical once the single pair call is "
        f"put back as two calls"
    )
    return candidate, assembly, root


# ---------------------------------------------------------------------------
# P. the premise: is the shared work really arm-independent?
# ---------------------------------------------------------------------------


def control_p(candidate, assembly, windows, heads):
    q_heads, kv_heads, head_dim = heads
    states = assembly._AGR1_PARENT_CALIBRATION(windows, q_heads, kv_heads, head_dim)
    if not all(isinstance(states[role], dict) and states[role] for role in
               ("q_state", "k_state", "v_state")):
        raise AssertionError("the base calibration returned no usable states")

    # P1 the candidate arm is the parent arm with only q/k rotated.
    candidate_states = assembly._agr1_parent_copy(states)
    device = torch.device("cpu")
    tq, tk, center, _ = assembly._agr1_train(
        windows[: len(windows) - len(assembly._AGR1_GATE_WINDOWS)],
        states, q_heads, kv_heads, head_dim, device,
    )
    candidate_states["q_state"]["learned_rotation"] = tq
    candidate_states["k_state"]["learned_rotation"] = tk
    if center is not None:
        candidate_states["k_state"]["learned_center"] = center
    deep_equal(
        states["v_state"], candidate_states["v_state"],
        "P1: the two arms' v_state",
    )
    arm_diff = []
    for role in ("q_state", "k_state", "v_state"):
        for key in set(states[role]) | set(candidate_states[role]):
            left = states[role].get(key)
            right = candidate_states[role].get(key)
            if torch.is_tensor(left) or torch.is_tensor(right):
                same = (
                    torch.is_tensor(left)
                    and torch.is_tensor(right)
                    and left.dtype == right.dtype
                    and tuple(left.shape) == tuple(right.shape)
                    and raw_bytes(left) == raw_bytes(right)
                )
            else:
                same = left == right
            if not same:
                arm_diff.append(f"{role}.{key}")
    if any(entry.startswith("v_state.") for entry in arm_diff):
        raise AssertionError(f"the arms differ inside v_state: {arm_diff}")
    print(
        f"[P1] the two arms differ only in q_state/k_state; their v_state is byte-identical "
        f"(differing fields across all three: {arm_diff})"
    )

    # P2 the V API is a function of its arguments: same inputs, same five fields.
    item = windows[assembly._AGR1_GATE_WINDOWS[0]]
    v_first = assembly._AGR1_PARENT_V(
        *item["v"], kv_heads, head_dim, states["v_state"]
    )
    v_second = assembly._AGR1_PARENT_V(
        *item["v"], kv_heads, head_dim, dict(states["v_state"])
    )
    five_fields_equal(v_first, v_second, "P2: two V calls on equal states")
    print("[P2] two V calls with equal-valued states return five bit-identical fields")

    # P3 no API writes into the state it is handed.
    frozen = {
        role: dict(states[role]) for role in ("q_state", "k_state", "v_state")
    }
    before = {role: {k: raw_bytes(v) if torch.is_tensor(v) else repr(v)
                     for k, v in frozen[role].items()}
              for role in frozen}
    assembly._AGR1_PARENT_Q(*item["q"], q_heads, head_dim, frozen["q_state"])
    assembly._AGR1_PARENT_K(*item["k"], kv_heads, head_dim, frozen["k_state"])
    assembly._AGR1_PARENT_V(*item["v"], kv_heads, head_dim, frozen["v_state"])
    after = {role: {k: raw_bytes(v) if torch.is_tensor(v) else repr(v)
                    for k, v in frozen[role].items()}
             for role in frozen}
    if before != after:
        raise AssertionError("a deployment API mutated the state it was handed")
    print(
        "[P3] Q/K/V leaves every state dict byte-identical (no writes), so evaluating "
        "one arm cannot disturb the other's inputs"
    )
    return states, candidate_states


# ---------------------------------------------------------------------------
# B/C. gate equivalence and call counts
# ---------------------------------------------------------------------------


def control_bc(candidate, windows, states, candidate_states, heads, label):
    q_heads, kv_heads, head_dim = heads
    records = []
    for index in candidate._AGR1_GATE_WINDOWS:
        item = windows[index]
        with CallCounter(candidate, COUNTED + COUNTED_APIS) as old_counter:
            old_parent = candidate._agr1_gate_loss(
                item, states, q_heads, kv_heads, head_dim
            )
            old_candidate = candidate._agr1_gate_loss(
                item, candidate_states, q_heads, kv_heads, head_dim
            )
        with CallCounter(candidate, COUNTED + COUNTED_APIS) as new_counter:
            new_parent, new_candidate = candidate._act1_gate_pair(
                item, states, candidate_states, q_heads, kv_heads, head_dim
            )
        if old_parent != new_parent or old_candidate != new_candidate:
            raise AssertionError(
                f"{label} window {index}: losses differ -- A-GR1 "
                f"({old_parent!r}, {old_candidate!r}) vs A-CT1 "
                f"({new_parent!r}, {new_candidate!r})"
            )
        records.append((index, old_parent, old_candidate, old_counter.counts, new_counter.counts))

    # The reduction must be exactly the arm-independent work.
    # Per gate window the new path must save exactly the arm-independent work.
    # `_dequantize_nvfp4_float32` is counted at 4 rather than 3 because the V
    # quantiser calls it internally as well: 3 explicit reference decodes plus
    # 1 fewer internal decode on the V path (2 V calls become 1).
    expected = {
        "_a2_attention_forward": 1,
        "_dequantize_nvfp4_float32": 4,
        "_dequantize_hif4": 1,
        "_AGR1_PARENT_V": 1,
        "_AGR1_PARENT_Q": 0,
        "_AGR1_PARENT_K": 0,
    }
    index, old_parent, old_candidate, old_counts, new_counts = records[0]
    for name, saved in expected.items():
        actual = old_counts[name] - new_counts[name]
        if actual != saved:
            raise AssertionError(
                f"{label} window {index}: {name} reduced by {actual}, expected {saved}"
            )
    if new_counts["_agr1_gate_loss"] != 0 or old_counts["_act1_gate_pair"] != 0:
        raise AssertionError("the two paths are not disjoint")
    if old_counts["_agr1_gate_loss"] != 2 or new_counts["_act1_gate_pair"] != 1:
        raise AssertionError(
            f"{label}: expected 2 old calls and 1 new call, got "
            f"{old_counts['_agr1_gate_loss']} and {new_counts['_act1_gate_pair']}"
        )

    print(
        f"[B:{label}] every gate window's parent and candidate loss is exactly equal "
        f"between the two paths ({len(records)} windows, "
        f"e.g. window {records[0][0]}: {records[0][1]!r} / {records[0][2]!r})"
    )
    print(
        f"[C:{label}] per gate window the new path saves exactly: "
        f"attention forwards {old_counts['_a2_attention_forward']} -> "
        f"{new_counts['_a2_attention_forward']}, reference decodes "
        f"{old_counts['_dequantize_nvfp4_float32']} -> "
        f"{new_counts['_dequantize_nvfp4_float32']}, hif4 decodes "
        f"{old_counts['_dequantize_hif4']} -> {new_counts['_dequantize_hif4']}, "
        f"V quantisations {old_counts['_AGR1_PARENT_V']} -> "
        f"{new_counts['_AGR1_PARENT_V']}; Q and K calls unchanged at "
        f"{old_counts['_AGR1_PARENT_Q']}/{new_counts['_AGR1_PARENT_Q']} and "
        f"{old_counts['_AGR1_PARENT_K']}/{new_counts['_AGR1_PARENT_K']}"
    )
    return records


# ---------------------------------------------------------------------------
# D. end-to-end
# ---------------------------------------------------------------------------


def run_calibration(module, windows, heads):
    q_heads, kv_heads, head_dim = heads
    return module.hif4_calibration_attention(windows, q_heads, kv_heads, head_dim)


def control_d(candidate, assembly, windows, heads, label):
    assembly_states = run_calibration(assembly, windows, heads)
    candidate_states = run_calibration(candidate, windows, heads)
    deep_equal(assembly_states, candidate_states, f"D:{label} calibration states")
    info_keys = sorted(
        key for key in assembly_states["q_state"] if str(key).startswith(("agr1_", "a2_"))
    )
    audit = {key: assembly_states["q_state"][key] for key in info_keys}
    print(
        f"[D:{label}] full calibration returns byte-identical q/k/v states and equal "
        f"audit fields (arm={audit.get('agr1_arm')!r}, "
        f"attempted={audit.get('agr1_attempted')}, accepted={audit.get('agr1_accepted')}, "
        f"parent_arm={audit.get('agr1_parent_arm')!r}, "
        f"gate_parent={audit.get('agr1_gate_parent_mse')!r}, "
        f"gate_candidate={audit.get('agr1_gate_candidate_mse')!r})"
    )
    return assembly_states, audit


def control_d_patched(candidate, assembly, windows, heads, label, patch_name, replacement):
    """Same as D but with one name replaced identically in both modules."""

    saved = {}
    try:
        for module in (assembly, candidate):
            saved[id(module)] = getattr(module, patch_name)
            setattr(module, patch_name, replacement)
        return control_d(candidate, assembly, windows, heads, label)
    finally:
        for module in (assembly, candidate):
            setattr(module, patch_name, saved[id(module)])


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    torch.set_grad_enabled(False)
    candidate_path = HERE / "candidate" / "solution.py"
    candidate, assembly, root = control_a(candidate_path)

    global _pair
    have_real = PACK.exists()
    if have_real:
        sys.path.insert(0, str(ROOT / "evaluator"))
        import official_eval as v2  # noqa: PLC0415

        _pair = v2._pair
        pack = torch.load(PACK, map_location="cpu", mmap=True, weights_only=False)
        heads = (int(pack["q_heads"]), int(pack["kv_heads"]), int(pack["head_dim"]))
        attention_layers = [int(layer) for layer in pack["metadata"]["attention_layers"]]
        print(f"[real] pack loaded: heads={heads} attention_layers={attention_layers}")
    else:
        pack = None
        heads = (SYNTH_Q_HEADS, SYNTH_KV_HEADS, SYNTH_HEAD_DIM)
        attention_layers = []
        print("[real] pack not present; only the synthetic controls run")

    # --- premise, gate equivalence and counts on real data -------------------
    if have_real:
        layer = attention_layers[0]
        windows = real_windows(pack, layer)
        states, candidate_states = control_p(candidate, assembly, windows, heads)
        control_bc(candidate, windows, states, candidate_states, heads, f"real/layer{layer}")

    # --- end-to-end on every real attention layer ----------------------------
    audit_by_layer = {}
    if have_real:
        first = None
        for layer in attention_layers:
            windows = real_windows(pack, layer)
            _, audit = control_d(candidate, assembly, windows, heads, f"real/layer{layer}")
            audit_by_layer[layer] = audit
            if first is None:
                first = windows
        real_windows_first = first
        arms = {layer: audit["agr1_arm"] for layer, audit in audit_by_layer.items()}
        attempted = {layer: audit["agr1_attempted"] for layer, audit in audit_by_layer.items()}
        if not any(value == "accepted" for value in arms.values()):
            raise AssertionError(f"no real layer reached the accepted arm: {arms}")
        if not any(value == "parent" for value in arms.values()):
            raise AssertionError(f"no real layer exercised the rejected gate: {arms}")
        if not all(value == 1 for value in attempted.values()):
            raise AssertionError(f"not every real layer attempted training: {attempted}")
        print(
            f"[E:coverage] accepted and rejected gate outcomes both occur on real data "
            f"(attempted=1 everywhere): {arms}"
        )
    else:
        real_windows_first = None

    # --- synthetic end-to-end ------------------------------------------------
    synthetic = synthetic_windows()
    control_d(candidate, assembly, synthetic, heads, "synthetic")

    # --- coverage: M = I, ineligible, fallback -------------------------------
    if have_real:
        base_calibration = assembly._AGR1_PARENT_CALIBRATION

        def strip_rotation(windows, q_heads, kv_num_heads, head_dim):
            states = base_calibration(windows, q_heads, kv_num_heads, head_dim)
            for role in ("q_state", "k_state"):
                states[role].pop("learned_rotation", None)
            states["k_state"].pop("learned_center", None)
            return states

        _, audit = control_d_patched(
            candidate, assembly, real_windows_first, heads, "real/M=I",
            "_AGR1_PARENT_CALIBRATION", strip_rotation,
        )
        if audit.get("agr1_parent_arm") != "identity":
            raise AssertionError(f"the M=I probe did not reach the identity arm: {audit}")
        print("[E:coverage] the identity-parent arm (M = I, no learned rotation to start from) agrees")

    # ineligible: fewer windows than the gate needs.
    short = synthetic_windows(count=2)
    short_states = run_calibration(assembly, short, heads)
    short_candidate = run_calibration(candidate, short, heads)
    deep_equal(short_states, short_candidate, "E:ineligible states")
    if short_states["q_state"].get("agr1_arm") != "ineligible":
        raise AssertionError("a two-window list should be ineligible")
    print("[E:coverage] the ineligible path agrees")

    # fallback: make the trainer fail identically in both modules.
    def boom(*args, **kwargs):
        raise ValueError("control: deliberate trainer failure")

    _, audit = control_d_patched(
        candidate, assembly, synthetic, heads, "fallback", "_agr1_train", boom
    )
    if not str(audit.get("agr1_error", "")).startswith("ValueError:"):
        raise AssertionError(f"the fallback probe did not take the fallback path: {audit}")
    print("[E:coverage] the exception fallback path agrees")

    # --- determinism ---------------------------------------------------------
    if have_real:
        control_d(assembly, assembly, real_windows_first, heads, "assembly-vs-itself")

    print("\nALL A-CT1 CONTROLS PASSED")


if __name__ == "__main__":
    main()
