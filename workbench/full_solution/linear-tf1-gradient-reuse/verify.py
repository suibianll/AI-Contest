"""L-TF1 legality, sole-change, output-equivalence and call-count check.

Controls (plan 2026-09-10-linear-correctness-and-runtime-plan.md, section 3):

  A. lineage and sole change.  The candidate is a pure byte-append of the
     retained v230 root, so every Attention API, the parent weight encoder and
     every shared helper are identical by construction; the six APIs import from
     a bare directory with no repository siblings; the appended definition --
     not the parent's -- is the one the parent's own dynamic hook resolves at
     call time.  The sole change is then asserted at the syntax level: parsing
     both descents and dissolving the one inserted ``if _pass:`` node back into
     its parent block must reproduce the parent's statement tree *exactly*.
     build.py makes the same check; it is repeated here so the archive's
     verification does not rest on the builder's own say-so.

  B. output equivalence, bit for bit.  This is the whole claim of the card --
     "if the candidate claims a pure time change, it must prove the hard codes
     and the outputs equivalent" (plan section 3).  Parent and candidate are run
     on the same state and the same activation and the five HiF4 fields are
     compared byte by byte, not with a tolerance, on: the compiled synthetic
     layer, the real layer-0/q activations, a state with no ``em1`` payload, and
     the out-of-scope bypass.  Comparison is on the raw bytes so that a NaN can
     never compare equal by accident.

  C. the premise, measured rather than argued.  premise.py establishes
     statically that nothing between the pre-loop gradient and the pass-0
     recomputation changes ``deployed`` or ``reference``.  Here the same claim
     is measured: ``Tensor.mm`` is wrapped and the two products that make up the
     parent's pre-loop gradient are compared, bit for bit, against the two that
     make up its pass-0 recomputation.

  D. the call-count evidence, which is the point of the card.  The complete
     sequence of ``Tensor.mm`` invocations is recorded for both arms.  The
     candidate's sequence must be the parent's with exactly the two products of
     the pass-0 gradient removed and nothing else changed.  At K = 1 that is
     20 -> 18 products per invocation.

  E. the failure paths, unchanged.  Two of them, because they are the only
     places the guarded branch could in principle behave differently:
       E1 a non-finite gradient before the loop -- both arms must take the
          ``nonfinite-gradient`` diagnostic and return the result untouched;
       E2 a non-finite gradient at pass 1 under K = 2 -- the only situation in
          which the guarded branch actually executes the abort.  The abort is
          not merely inferred from matching outputs: the mm totals show the loop
          stopped where it should.

  F. determinism, state round-trip, and the K > 1 arm.  The shipped root is
     K = 1, where the guard never fires, so the K = 2 arm is exercised by
     setting ``_EM1_PASSES`` in the loaded modules -- a test-time manipulation of
     the in-memory module only; nothing on disk is touched.  Both arms must
     again agree bit for bit, and the parent's later passes must still recompute
     in the parent's own order.

Diagnostics printed, never gated: the mm shape sequence, the two gradient
product pairs, and the accepted-group counts.

CPU only.  The real layer comes from the calibration cache if it is present;
without it the synthetic controls still run and the real ones are skipped.
"""

from pathlib import Path
import ast
import hashlib
import importlib.util
import inspect
import sys
import tempfile

import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

CARD_PARENT_SHA256 = (
    "0f1af6dbc207ff32b2c6be16987e9c4fe50f3f10747de26782ef52a6f2fab7bc"
)
CARD_PARENT_BYTES = 505496

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

# Anchors for cutting the parent's own function text back out of the root.
FN_START = b"@torch.no_grad()\ndef _em1_dynamic_descent(\n"
FN_END = b"\n_EM1_PARENT_LINEAR_CALIBRATION = hif4_calibration_and_quantize_weight"

# K is fixed by this card: L-TF1 changes *when* the gradient is computed and must
# not move the pass count, so it is asserted against a caller-supplied
# expectation rather than read back off the module.
EXPECTED_PASSES = 1
EXPECTED_GROUPS_PER_BLOCK = 16

# Tensor.mm products per invocation:
#   pre-loop gradient        2
#   each pass: gradient      2 (candidate: only for _pass > 0)
#              g_delta      16 (one per group step)
MM_PER_PASS_STEPS = EXPECTED_GROUPS_PER_BLOCK
MM_PER_GRADIENT = 2

CACHE_DIR = ROOT / "artifacts/official_eval/cache/proxy-v3-calibration"
PACK = ROOT / "artifacts/official_eval/cache/qwen3.5-4b-proxy-v2.pt"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def load_solution(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def unwrapped(function):
    """Strip torch's ``no_grad`` decorator so source and code reads are real.

    ``torch.no_grad()`` is a class-based decorator whose ``__call__`` returns a
    closure defined in ``torch/utils/_contextlib.py``.  ``inspect.getsource`` on
    the returned wrapper therefore yields torch's own decorator source -- the
    same text for *any* two decorated functions -- so comparing wrappers proves
    nothing.  Every source or code-object comparison here unwraps first.
    """

    return inspect.unwrap(function)


def live_descent(module):
    """The ``_em1_dynamic_descent`` the module's own dynamic hook will call."""

    return unwrapped(module._em1_dynamic_descent)


def parent_function_text() -> bytes:
    root_bytes = (ROOT / "solution.py").read_bytes()
    if root_bytes.count(FN_START) != 1 or root_bytes.count(FN_END) != 1:
        raise AssertionError("parent function anchors are not unique")
    start = root_bytes.index(FN_START)
    return root_bytes[start : root_bytes.index(FN_END, start)]


def raw_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.detach().cpu().contiguous().numpy().tobytes()


def bit_identical(a: torch.Tensor, b: torch.Tensor, label: str) -> None:
    """Byte-for-byte equality, deliberately NaN-safe.

    ``torch.equal`` reports NaN != NaN, so a NaN-vs-NaN comparison would look
    like a difference and -- worse -- a comparison that *passed* could be hiding
    a NaN.  Comparing raw bytes has neither problem.
    """

    if a.dtype != b.dtype:
        raise AssertionError(f"{label}: dtype {a.dtype} != {b.dtype}")
    if tuple(a.shape) != tuple(b.shape):
        raise AssertionError(f"{label}: shape {tuple(a.shape)} != {tuple(b.shape)}")
    if raw_bytes(a) != raw_bytes(b):
        try:
            worst = float((a.float() - b.float()).abs().max())
            detail = f"max |difference| {worst:.6e}"
        except (RuntimeError, TypeError):
            detail = "difference could not be quantified"
        raise AssertionError(f"{label}: payload differs bitwise ({detail})")


def same_params(a: dict, b: dict, label: str) -> None:
    if set(a) != set(b):
        raise AssertionError(f"{label}: key sets differ: {set(a) ^ set(b)}")
    for key in a:
        if not torch.is_tensor(a[key]) or not torch.is_tensor(b[key]):
            raise AssertionError(f"{label}: field {key} is not a tensor")
        bit_identical(a[key], b[key], f"{label}: field {key}")


def assert_legal_params(params: dict, rows: int, channels: int) -> None:
    expected = {
        "scale_factor": (rows, channels // 64, 1, 1, 1),
        "scale_lv2": (rows, channels // 64, 8, 1, 1),
        "scale_lv3": (rows, channels // 64, 8, 2, 1),
        "sign": (rows, channels // 64, 8, 2, 4),
        "mant": (rows, channels // 64, 8, 2, 4),
    }
    if set(params) != set(expected):
        raise AssertionError(f"unexpected HiF4 keys: {set(params)}")
    for key, shape in expected.items():
        value = params[key]
        if not torch.is_tensor(value):
            raise AssertionError(f"{key} is not a tensor")
        if tuple(value.shape) != shape:
            raise AssertionError(f"{key} shape {tuple(value.shape)} != {shape}")
        if not bool(torch.isfinite(value).all()):
            raise AssertionError(f"{key} holds non-finite values")
    mant = params["mant"].to(torch.float32)
    if bool((mant < 0).any()) or bool((mant > 7 * 0.25 + 1e-6).any()):
        raise AssertionError("mantissa outside [0, 1.75]")
    codes = mant * 4.0
    if float((codes - codes.round()).abs().max()) > 1e-4:
        raise AssertionError("mantissa is not a multiple of 0.25")
    if bool((params["sign"].abs() > 1).any()):
        raise AssertionError("sign outside {-1, 0, 1}")


def dissolve_pass_guard(tree: ast.Module) -> ast.Module:
    """Removes an inserted ``if _pass:`` node by splicing its body back in.

    Comparison device only: if dissolving the guard makes one tree equal the
    other, the guard is the only structural difference between them.
    """

    function = tree.body[0]
    for node in ast.walk(function):
        if isinstance(node, ast.For) and ast.unparse(node.target) == "_pass":
            first = node.body[0]
            if (
                isinstance(first, ast.If)
                and ast.unparse(first.test) == "_pass"
                and first.orelse == []
            ):
                index = node.body.index(first)
                node.body[index : index + 1] = first.body
                ast.fix_missing_locations(tree)
                return tree
    raise AssertionError("no `if _pass:` guard found to dissolve")


class MmRecorder:
    """Records every ``Tensor.mm`` product, optionally poisoning one of them.

    Poisoning is how control E2 reaches the pass-1 abort without waiting for
    real arithmetic to overflow: one product is replaced by a NaN tensor so the
    guard's finiteness test fires at a known index in each arm.
    """

    def __init__(self, poison_index: int | None = None, poison_value: float = float("nan")):
        self.poison_index = poison_index
        self.poison_value = poison_value
        self.shapes: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
        self.results: list[torch.Tensor] = []
        self._original = None

    def __enter__(self):
        self._original = torch.Tensor.mm
        recorder = self

        def mm(tensor, other):
            recorder.shapes.append((tuple(tensor.shape), tuple(other.shape)))
            product = recorder._original(tensor, other)
            if (
                recorder.poison_index is not None
                and len(recorder.shapes) == recorder.poison_index
            ):
                product = product.clone()
                product.fill_(recorder.poison_value)
            recorder.results.append(product)
            return product

        torch.Tensor.mm = mm
        return self

    def __exit__(self, *exc_info):
        torch.Tensor.mm = self._original
        return False

    @property
    def count(self) -> int:
        return len(self.shapes)


def run_recorded(module, activation_quant, activation_scale, state, poison_index=None):
    """One dynamic call, with the mm sequence recorded and the state untouched."""

    with MmRecorder(poison_index=poison_index) as recorder:
        output = module.hif4_dynamic_quantize_activation(
            activation_quant, activation_scale, dict(state)
        )
    return output, recorder


# ---------------------------------------------------------------------------
# A. lineage and sole change
# ---------------------------------------------------------------------------


def single_file_import(candidate_path: Path) -> None:
    """Import the candidate from a directory holding nothing else."""

    with tempfile.TemporaryDirectory(prefix="tf1-solo-") as solo:
        target = Path(solo) / "solution.py"
        target.write_bytes(candidate_path.read_bytes())
        spec = importlib.util.spec_from_file_location("tf1_solo", target)
        if spec is None or spec.loader is None:
            raise AssertionError("cannot import the candidate standalone")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        missing = [name for name in PUBLIC_APIS if not callable(getattr(module, name, None))]
        if missing:
            raise AssertionError(f"candidate does not expose {missing}")


def control_a(candidate_path: Path):
    root_bytes = (ROOT / "solution.py").read_bytes()
    root_digest = hashlib.sha256(root_bytes).hexdigest()
    if root_digest != CARD_PARENT_SHA256:
        print(
            f"[A] NOTE the live root is {root_digest[:8]}, not the card parent "
            f"{CARD_PARENT_SHA256[:8]}; the candidate is still compared against "
            "the bytes it was built from"
        )
    candidate_bytes = candidate_path.read_bytes()
    if len(root_bytes) < CARD_PARENT_BYTES:
        raise AssertionError("live root is shorter than the card parent")
    if candidate_bytes[:CARD_PARENT_BYTES] != root_bytes[:CARD_PARENT_BYTES]:
        first = next(
            (
                i
                for i in range(min(len(candidate_bytes), CARD_PARENT_BYTES))
                if candidate_bytes[i] != root_bytes[i]
            ),
            CARD_PARENT_BYTES,
        )
        raise AssertionError(f"candidate diverges from the parent at byte {first}")
    if candidate_bytes == root_bytes:
        raise AssertionError("candidate is byte-identical to the root")
    if len(candidate_bytes) <= CARD_PARENT_BYTES:
        raise AssertionError("candidate does not append anything")

    candidate = load_solution(candidate_path, "tf1_candidate")
    parent = load_solution(ROOT / "solution.py", "tf1_parent")
    single_file_import(candidate_path)

    # The append is the whole candidate, so an Attention API cannot differ.
    for name in ATTENTION_APIS:
        if (
            unwrapped(getattr(candidate, name)).__code__.co_code
            != unwrapped(getattr(parent, name)).__code__.co_code
        ):
            raise AssertionError(f"Attention API {name} bytecode drifted")

    # The shadow must be the live definition: the parent's hook resolves the
    # name from module globals when it runs, so the later definition wins.
    parent_first = unwrapped(parent._em1_dynamic_descent).__code__.co_firstlineno
    candidate_first = unwrapped(candidate._em1_dynamic_descent).__code__.co_firstlineno
    hook = unwrapped(parent.hif4_dynamic_quantize_activation)
    if "_em1_dynamic_descent" not in hook.__code__.co_names:
        raise AssertionError("the parent hook does not resolve the descent by name")
    if candidate_first <= parent_first:
        raise AssertionError("the appended descent does not shadow the parent's")

    # Sole change at the syntax level, computed here rather than trusted from
    # build.py: dissolving the guard must give back the parent's statement tree.
    src_parent = inspect.getsource(live_descent(parent))
    src_candidate = inspect.getsource(live_descent(candidate))
    tree_parent = ast.parse(src_parent)
    tree_candidate = ast.parse(src_candidate)
    guard_count = sum(
        1
        for node in ast.walk(tree_candidate)
        if isinstance(node, ast.If) and ast.unparse(node.test) == "_pass"
    )
    if guard_count != 1:
        raise AssertionError(
            f"candidate holds {guard_count} `if _pass:` guards, expected 1"
        )
    if ast.dump(tree_parent) != ast.dump(dissolve_pass_guard(tree_candidate)):
        raise AssertionError(
            "dissolving the guard does not reproduce the parent's statement tree, "
            "so the guard is not the only structural change"
        )
    if src_candidate.count("_pass") != src_parent.count("_pass") + 1:
        raise AssertionError("the guard is not the only new mention of _pass")

    print(
        f"[A] parent={len(root_bytes)}B ({root_digest[:8]}) "
        f"candidate={len(candidate_bytes)}B "
        f"({hashlib.sha256(candidate_bytes).hexdigest()[:8]}) "
        f"shared prefix={CARD_PARENT_BYTES}B attention bytecode identical "
        f"AST identical once the single `if _pass:` guard is dissolved"
    )
    return candidate, parent


# ---------------------------------------------------------------------------
# B. output equivalence
# ---------------------------------------------------------------------------


def make_layer(device, seed, rows, channels, windows=2, samples=12):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    weight_quant = torch.randn((rows, channels), generator=generator).to(device)
    weight_scale = torch.ones((rows, channels // 16), dtype=torch.float32, device=device)
    activations = [
        (
            torch.randn((samples, channels), generator=generator).to(device),
            torch.ones((samples, channels // 16), dtype=torch.float32, device=device),
        )
        for _ in range(windows)
    ]
    return weight_quant, weight_scale, activations


def control_b(candidate, parent, device):
    weight_quant, weight_scale, activations = make_layer(
        device, 311, rows=64, channels=2560
    )
    parent_cal = parent.hif4_calibration_and_quantize_weight(
        weight_quant, weight_scale, activations
    )
    candidate_cal = candidate.hif4_calibration_and_quantize_weight(
        weight_quant, weight_scale, activations
    )
    same_params(
        parent_cal["weight_params"], candidate_cal["weight_params"], "calibration"
    )
    state = candidate_cal["activation_state"]
    if state.get("em1_arm") != "compiled":
        raise AssertionError(f"narrow layer arm is {state.get('em1_arm')}")
    payload = state.get("em1")
    if not isinstance(payload, dict) or tuple(payload["h"].shape) != (2560, 2560):
        raise AssertionError("compiled H has the wrong shape")

    # The diagnostics are written into the state dict that was passed in, so
    # keep a handle on each arm's own copy rather than rebuilding a throwaway.
    parent_state = dict(state)
    candidate_state = dict(state)
    parent_live = parent.hif4_dynamic_quantize_activation(
        activations[0][0], activations[0][1], parent_state
    )
    candidate_live = candidate.hif4_dynamic_quantize_activation(
        activations[0][0], activations[0][1], candidate_state
    )
    same_params(parent_live, candidate_live, "compiled synthetic layer")
    assert_legal_params(candidate_live, 12, 2560)
    if candidate_state.get("em1_dynamic_arm") != "applied":
        raise AssertionError(
            f"the compiled synthetic layer took {candidate_state.get('em1_dynamic_arm')}"
        )

    # No em1 payload -> both must fall through to the parent dynamic output.
    stripped = {key: value for key, value in state.items() if key != "em1"}
    same_params(
        parent.hif4_dynamic_quantize_activation(
            activations[0][0], activations[0][1], dict(stripped)
        ),
        candidate.hif4_dynamic_quantize_activation(
            activations[0][0], activations[0][1], dict(stripped)
        ),
        "parent-off state",
    )

    # Out of scope: in_features > 4096 stores nothing and both must match.
    wide_quant, wide_scale, wide_acts = make_layer(device, 313, rows=64, channels=4160)
    wide = candidate.hif4_calibration_and_quantize_weight(
        wide_quant, wide_scale, wide_acts
    )
    wide_state = wide["activation_state"]
    if wide_state.get("em1_arm") != "out-of-scope":
        raise AssertionError(f"wide layer arm is {wide_state.get('em1_arm')}")
    if "em1" in wide_state:
        raise AssertionError("wide layer stored a metric payload")
    same_params(
        parent.hif4_dynamic_quantize_activation(
            wide_acts[0][0], wide_acts[0][1], dict(wide_state)
        ),
        candidate.hif4_dynamic_quantize_activation(
            wide_acts[0][0], wide_acts[0][1], dict(wide_state)
        ),
        "out-of-scope layer",
    )

    print(
        "[B] parent and candidate agree byte for byte on the compiled layer "
        f"({candidate_state.get('em1_accepted_steps')} steps, "
        f"{candidate_state.get('em1_accepted_groups')} group moves, "
        f"{candidate_state.get('em1_changed_mantissa')} mantissa codes changed), "
        "on a state with no em1 payload, and on the out-of-scope bypass"
    )
    return state, activations


def control_b_real(candidate, parent, device, pair, real_state):
    parent_state = dict(real_state)
    candidate_state = dict(real_state)
    parent_live = parent.hif4_dynamic_quantize_activation(
        pair[0], pair[1], parent_state
    )
    candidate_live = candidate.hif4_dynamic_quantize_activation(
        pair[0], pair[1], candidate_state
    )
    same_params(parent_live, candidate_live, "real layer0/q")
    rows = int(pair[0].shape[0])
    assert_legal_params(candidate_live, rows, 2560)
    if candidate_state.get("em1_dynamic_arm") != "applied":
        raise AssertionError(
            f"the real layer took {candidate_state.get('em1_dynamic_arm')}"
        )
    print(
        f"[B:real/layer0/q] parent and candidate agree byte for byte on {rows} real "
        f"rows of the cached layer-0/q state "
        f"({candidate_state.get('em1_accepted_steps')} steps, "
        f"{candidate_state.get('em1_accepted_groups')} group moves, "
        f"{candidate_state.get('em1_accepted_rows')} rows kept, "
        f"{candidate_state.get('em1_changed_mantissa')} mantissa codes changed)"
    )


# ---------------------------------------------------------------------------
# C. the premise, measured
# ---------------------------------------------------------------------------


def control_c(parent, device, label, state, act_pair):
    """The parent's pre-loop gradient equals its pass-0 recomputation, bitwise.

    Only the two products of each gradient are compared, so this is a statement
    about the arithmetic and not about the surrounding loop.
    """

    _, recorder = run_recorded(parent, act_pair[0], act_pair[1], state)
    if recorder.count < 2 * MM_PER_GRADIENT:
        raise AssertionError(f"{label}: only {recorder.count} mm products recorded")
    pre_a, pre_b = recorder.results[0], recorder.results[1]
    pass_a, pass_b = recorder.results[2], recorder.results[3]
    bit_identical(pre_a, pass_a, f"{label}: pre-loop vs pass-0 (deployed-reference)@metric")
    bit_identical(pre_b, pass_b, f"{label}: pre-loop vs pass-0 reference@h_matrix")
    bit_identical(pre_a + pre_b, pass_a + pass_b, f"{label}: summed gradient")
    print(
        f"[C:{label}] the two gradient products are bit-identical across the two "
        f"evaluations (shapes {recorder.shapes[0][0]}@{recorder.shapes[0][1]}, "
        f"{recorder.shapes[1][0]}@{recorder.shapes[1][1]}); product 0 identical: "
        f"{raw_bytes(pre_a) == raw_bytes(pass_a)}, product 1 identical: "
        f"{raw_bytes(pre_b) == raw_bytes(pass_b)}"
    )


# ---------------------------------------------------------------------------
# D. call-count evidence
# ---------------------------------------------------------------------------


def expected_mm_count(passes: int, reuse_first_pass: bool) -> int:
    gradients = passes - 1 if reuse_first_pass else passes
    return MM_PER_GRADIENT + gradients * MM_PER_GRADIENT + passes * MM_PER_PASS_STEPS


def control_d(candidate, parent, device, label, state, act_pair, passes):
    _, parent_rec = run_recorded(parent, act_pair[0], act_pair[1], state)
    _, candidate_rec = run_recorded(candidate, act_pair[0], act_pair[1], state)

    want_parent = expected_mm_count(passes, reuse_first_pass=False)
    want_candidate = expected_mm_count(passes, reuse_first_pass=True)
    if parent_rec.count != want_parent:
        raise AssertionError(
            f"{label}: parent made {parent_rec.count} mm products, expected {want_parent}"
        )
    if candidate_rec.count != want_candidate:
        raise AssertionError(
            f"{label}: candidate made {candidate_rec.count} mm products, expected "
            f"{want_candidate}"
        )
    if parent_rec.count - candidate_rec.count != MM_PER_GRADIENT:
        raise AssertionError(
            f"{label}: the candidate saves {parent_rec.count - candidate_rec.count} "
            f"products, expected exactly {MM_PER_GRADIENT}"
        )

    # The saving must be the pass-0 gradient and nothing else: removing the
    # parent's products 2 and 3 (0-indexed) must reproduce the candidate's whole
    # sequence of operand shapes.
    parent_shapes = [
        shape for index, shape in enumerate(parent_rec.shapes)
        if index not in (2, 3)
    ]
    if parent_shapes != candidate_rec.shapes:
        raise AssertionError(
            f"{label}: the candidate's product sequence is not the parent's with "
            "the pass-0 gradient removed"
        )
    print(
        f"[D:{label}] K={passes}: parent {parent_rec.count} mm products, candidate "
        f"{candidate_rec.count}; the candidate's sequence equals the parent's with "
        f"products 2 and 3 removed and no other change"
    )
    return parent_rec, candidate_rec


# ---------------------------------------------------------------------------
# E. failure paths
# ---------------------------------------------------------------------------


def control_e1(candidate, parent, device, state, act_pair):
    """A non-finite gradient before the loop must behave identically."""

    poison_quant = act_pair[0].clone()
    poison_quant.view(-1)[0] = float("nan")
    parent_state = dict(state)
    candidate_state = dict(state)
    parent_out = parent.hif4_dynamic_quantize_activation(
        poison_quant, act_pair[1], parent_state
    )
    candidate_out = candidate.hif4_dynamic_quantize_activation(
        poison_quant, act_pair[1], candidate_state
    )
    arms = {
        "parent": parent_state.get("em1_dynamic_arm"),
        "candidate": candidate_state.get("em1_dynamic_arm"),
    }
    if arms["parent"] != "nonfinite-gradient" or arms["candidate"] != "nonfinite-gradient":
        raise AssertionError(
            f"a non-finite reference should give the nonfinite-gradient diagnostic, "
            f"got {arms}"
        )
    same_params(parent_out, candidate_out, "E1 non-finite reference output")
    print(
        f"[E1] a NaN reference takes the parent's own `nonfinite-gradient` path in "
        f"both arms, and both return the result untouched, byte for byte"
    )


def control_e2(candidate, parent, device, label, state, act_pair, passes=2):
    """A non-finite gradient at pass 1: the only place the guard's abort runs.

    ``_EM1_PASSES`` is raised to 2 for the duration of this control only, in
    both modules, because it is the sole configuration in which the guarded
    branch executes.  It is restored afterwards, and the restore is checked.

    The poisoned index is the *second* pass's first gradient product -- that is,
    everything the arm issues before it, plus one.  It is derived from the
    expected call counts rather than written out by hand, so the two arms' in
    dices move with their own arithmetic: the candidate is two products behind
    the parent for the whole run, exactly the two the card removes.
    """

    # Everything issued before the second pass's gradient: the pre-loop
    # gradient, then the first pass.  A parent pass costs its gradient plus the
    # steps; the candidate's first pass costs the steps alone.
    parent_poison = (
        MM_PER_GRADIENT
        + (passes - 1) * (MM_PER_GRADIENT + MM_PER_PASS_STEPS)
        + 1
    )
    candidate_poison = MM_PER_GRADIENT + (passes - 1) * MM_PER_PASS_STEPS + 1
    if parent_poison - candidate_poison != MM_PER_GRADIENT:
        raise AssertionError(
            f"E2 {label}: the two poison indices differ by "
            f"{parent_poison - candidate_poison}, expected {MM_PER_GRADIENT}"
        )
    if parent_poison != 21 or candidate_poison != 19:
        raise AssertionError(
            f"E2 {label}: expected the pre-computed indices 21 and 19, got "
            f"{parent_poison} and {candidate_poison}"
        )

    saved = (int(candidate._EM1_PASSES), int(parent._EM1_PASSES))
    try:
        candidate._EM1_PASSES = passes
        parent._EM1_PASSES = passes
        parent_out, parent_rec = run_recorded(
            parent, act_pair[0], act_pair[1], state, poison_index=parent_poison
        )
        candidate_out, candidate_rec = run_recorded(
            candidate, act_pair[0], act_pair[1], state, poison_index=candidate_poison
        )
    finally:
        candidate._EM1_PASSES, parent._EM1_PASSES = saved
    if (int(candidate._EM1_PASSES), int(parent._EM1_PASSES)) != saved:
        raise AssertionError("the E2 probe leaked into a module's K")

    # Both arms must have reached the abort, which is visible in the count: the
    # poisoned product is the first of the two that make up the second pass's
    # gradient, so a run that aborted issued exactly poison_index + 1 products
    # and no group-step products for the second pass at all.  A run that did not
    # abort would have issued far more.
    parent_want = parent_poison + MM_PER_GRADIENT - 1
    candidate_want = candidate_poison + MM_PER_GRADIENT - 1
    if parent_rec.count != parent_want:
        raise AssertionError(
            f"E2 {label}: the parent issued {parent_rec.count} products, so it did "
            f"not stop after the second pass's gradient (want {parent_want})"
        )
    if candidate_rec.count != candidate_want:
        raise AssertionError(
            f"E2 {label}: the candidate issued {candidate_rec.count} products, so it "
            f"did not stop after the second pass's gradient (want {candidate_want})"
        )
    if parent_out.keys() != candidate_out.keys():
        raise AssertionError(f"E2 {label}: the two arms returned different keys")
    same_params(parent_out, candidate_out, f"E2 {label} aborted output")
    print(
        f"[E2:{label}] K={passes} with a non-finite gradient at the second pass: "
        f"parent issues {parent_rec.count} products (abort at index {parent_poison}) "
        f"and candidate {candidate_rec.count} (abort at index {candidate_poison}) -- "
        f"both stop before any second-pass group step, and their outputs are "
        f"byte-identical"
    )


def passes_before(passes: int) -> int:
    return max(passes - 1, 0)


# ---------------------------------------------------------------------------
# F. determinism, round-trip, and the K > 1 arm
# ---------------------------------------------------------------------------


def control_f_determinism(candidate, device, label, state, act_pair, scratch: Path):
    first = candidate.hif4_dynamic_quantize_activation(act_pair[0], act_pair[1], dict(state))
    second = candidate.hif4_dynamic_quantize_activation(act_pair[0], act_pair[1], dict(state))
    same_params(first, second, f"{label}: determinism")
    torch.save(state, scratch)
    reloaded = torch.load(scratch, map_location="cpu", weights_only=False)
    third = candidate.hif4_dynamic_quantize_activation(act_pair[0], act_pair[1], reloaded)
    same_params(first, third, f"{label}: state round-trip")
    scratch.unlink()
    print(f"[F:{label}] determinism and state round-trip hold")


def control_f_k2(candidate, parent, device, label, state, act_pair):
    """The K = 2 arm, where the guarded branch actually recomputes."""

    original_candidate = int(candidate._EM1_PASSES)
    original_parent = int(parent._EM1_PASSES)
    try:
        candidate._EM1_PASSES = 2
        parent._EM1_PASSES = 2
        parent_out = parent.hif4_dynamic_quantize_activation(
            act_pair[0], act_pair[1], dict(state)
        )
        candidate_out = candidate.hif4_dynamic_quantize_activation(
            act_pair[0], act_pair[1], dict(state)
        )
        same_params(parent_out, candidate_out, f"{label}: K=2 output")
    finally:
        candidate._EM1_PASSES = original_candidate
        parent._EM1_PASSES = original_parent
    if int(candidate._EM1_PASSES) != EXPECTED_PASSES:
        raise AssertionError("the K=2 probe leaked into the module's K")
    print(
        f"[F:{label}] K=2 (set in memory only, restored to {EXPECTED_PASSES}): the "
        f"parent's later passes still recompute and both arms agree byte for byte"
    )


def control_f_shape(candidate, label, state, expected_passes):
    if int(candidate._EM1_PASSES) != expected_passes:
        raise AssertionError(
            f"K drifted: {candidate._EM1_PASSES}, L-TF1 expects {expected_passes}"
        )
    if int(candidate._EM1_GROUPS_PER_BLOCK) != EXPECTED_GROUPS_PER_BLOCK:
        raise AssertionError("groups per block drifted")
    cap = expected_passes * EXPECTED_GROUPS_PER_BLOCK
    steps = int(state.get("em1_accepted_steps", -1))
    moves = int(state.get("em1_accepted_groups", -1))
    print(
        f"[F:{label}] passes={candidate._EM1_PASSES} accepted_steps={steps} "
        f"(cap {cap}) accepted_group_moves={moves}"
    )
    if steps != cap:
        raise AssertionError(f"{label}: ran {steps} steps, expected {cap}")
    if moves <= 0:
        raise AssertionError(f"{label}: not one group move was kept")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    torch.set_grad_enabled(False)
    device = torch.device("cpu")
    candidate_path = HERE / "candidate" / "solution.py"
    candidate, parent = control_a(candidate_path)

    state, activations = control_b(candidate, parent, device)
    act_pair = activations[0]

    control_c(parent, device, "synthetic", state, act_pair)
    control_d(candidate, parent, device, "synthetic", state, act_pair, EXPECTED_PASSES)
    control_e1(candidate, parent, device, state, act_pair)

    live = dict(state)
    candidate.hif4_dynamic_quantize_activation(act_pair[0], act_pair[1], live)
    control_f_shape(candidate, "synthetic", live, EXPECTED_PASSES)
    control_f_determinism(
        candidate, device, "synthetic", state, act_pair,
        HERE / "_tf1_state_roundtrip.pt",
    )
    control_f_k2(candidate, parent, device, "synthetic", state, act_pair)

    caches = sorted(CACHE_DIR.glob(f"{CARD_PARENT_SHA256[:16]}-linear-*.pt"))
    if caches and PACK.exists():
        sys.path.insert(0, str(ROOT / "evaluator"))
        import official_eval as v2  # noqa: PLC0415

        # The six shard caches partition the layers rather than indexing them in
        # order, so the layer-0 entry has to be looked for across all of them.
        pack = torch.load(PACK, map_location="cpu", mmap=True, weights_only=False)
        entry = None
        for path in caches:
            payload = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
            entry = next(
                (
                    item
                    for item in payload["weight_states"]
                    if int(item["layer"]) == 0 and str(item["role"]) == "q"
                ),
                None,
            )
            if entry is not None:
                break
        if entry is None:
            raise AssertionError("no cached weight state for layer 0 / role q")

        # The cached state is the *parent's* own calibration output -- the six
        # caches are named by parent SHA -- so using it directly gives the
        # dynamic controls a state the candidate has never produced.  The
        # calibration hook is byte-identical anyway (control A), so recompiling
        # it here would only re-run code already proven equal.
        real_state = dict(entry["state"])
        if real_state.get("em1_arm") != "compiled" or not isinstance(
            real_state.get("em1"), dict
        ):
            raise AssertionError("the cached layer-0/q state is not in the compiled arm")
        if tuple(real_state["em1"]["h"].shape) != (2560, 2560):
            raise AssertionError("the cached H is not (2560, 2560)")

        raw = pack["test_activations"]["q"][1][0].to(torch.float32).contiguous()
        pair = tuple(value.to(device) for value in v2._pair(raw))

        control_b_real(candidate, parent, device, pair, real_state)
        control_c(parent, device, "real/layer0/q", real_state, pair)
        control_d(
            candidate, parent, device, "real/layer0/q", real_state, pair, EXPECTED_PASSES
        )
        control_e1(candidate, parent, device, real_state, pair)
        real_live = dict(real_state)
        candidate.hif4_dynamic_quantize_activation(pair[0], pair[1], real_live)
        control_f_shape(candidate, "real/layer0/q", real_live, EXPECTED_PASSES)
        control_f_determinism(
            candidate, device, "real/layer0/q", real_state, pair,
            HERE / "_tf1_state_roundtrip.pt",
        )
        control_e2(
            candidate, parent, device, "real/layer0/q", real_state, pair, passes=2
        )
        control_f_k2(candidate, parent, device, "real/layer0/q", real_state, pair)
    else:
        print(
            "[real] calibration cache or pack not present; the real layer-0/q "
            "controls were skipped and the synthetic ones stand alone"
        )
        control_e2(candidate, parent, device, "synthetic", state, act_pair, passes=2)

    print("\nALL L-TF1 CONTROLS PASSED")


if __name__ == "__main__":
    main()
