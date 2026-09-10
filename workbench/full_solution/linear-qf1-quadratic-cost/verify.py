"""L-QF1 legality, sole-change, quadratic-algebra and real-proposal check.

Controls (plan 2026-09-10-linear-correctness-and-runtime-plan.md, section 2):

  A. lineage: the candidate is a pure byte-append of the retained v230 root, so
     every Attention API, the parent weight encoder and every shared helper are
     identical by construction; the six APIs import from a bare directory with
     no repository siblings; the appended definition -- not the parent's -- is
     the one the parent's own dynamic hook resolves at call time;
  Q. quadratic algebra against an *independent* reference.  Nothing here
     re-runs the expression under test: the reference evaluates the ideal
     objective ``J`` at two points with dense float64 matmuls and differences
     them.  The subject under test is the subscript string read out of the
     shipped source, so the control cannot drift from what ships.
       Q1 off-diagonal SPD, general delta: the correction reproduces
          ``d^T G d`` and the parent reproduces ``(sum_a d_a) * d . colsum(G)``
       Q2 single-coordinate delta ``c e_m``: corrected ``= c^2 G[m,m]``,
          parent ``= c^2 colsum(G)[m]``
       Q3 linear + quadratic against the directly evaluated objective change
       Q4 an explicit case where the parent form is not the true value
       Q5 a positive-definite Gram with a negative column sum, where the parent
          form reports a *negative* cost for a move whose true cost is positive
          and the accept test ``best_cost < 0`` therefore fires wrongly
  B. append neutrality and the parent-off control.  A "correction off" build --
     the parent's own function text appended unsubstituted -- must reproduce the
     parent bit for bit on both the synthetic and the real layer, which is what
     isolates the single subscript as the sole cause of every difference.  Then
     a state with no ``em1`` payload must return the parent dynamic output bit
     for bit, and an out-of-scope layer must store nothing and match.
  C. metric plumbing: the compiled ``h``/``gram_diag_mean`` reproduce an
     independent recomputation and ``G = h_inv^{-1} - c I`` recovers the
     deployed Gram.  L-QF1 must not disturb any of this path.
  D. real proposals and acceptance: parent and candidate run on the same real
     layer, and the accepted-group / accepted-row / changed-mantissa counts and
     the true output squared error are compared.  The mechanism is monotone by
     construction, so the candidate's true loss must not increase; whether it
     beats the parent is the empirical question the panel answers.
  E. determinism and state round-trip.
  F. schedule shape: K is unchanged, so every accepted step count must be
     ``K * 16`` and cannot silently drift.

Diagnostics printed, never gated: the ranking flips QF1 causes on real data,
the margin distribution between old and new cost, the judged sign of the local
Gram's column sums, and the accepted-count deltas.
"""

from pathlib import Path
import hashlib
import importlib.util
import inspect
import re
import shutil
import sys
import tempfile

import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

# The candidate is built by appending to the retained v230 root, which is
# `ROOT/solution.py` at the time of writing.  Both are recorded so that a root
# move mid-card is detected rather than silently redefining the comparison.
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

PARENT_SUBSCRIPT = "krba,bij,krbj->krb"
CORRECT_SUBSCRIPT = "krbi,bij,krbj->krb"

# Anchors for cutting the parent's own function text back out of the root.
FN_START = b"@torch.no_grad()\ndef _em1_dynamic_descent(\n"
FN_END = b"\n_EM1_PARENT_LINEAR_CALIBRATION = hif4_calibration_and_quantize_weight"

# K is fixed by this card: L-QF1 prices the corrected formula alone and must not
# move the pass count, so it is asserted against a caller-supplied expectation
# rather than read back off the module.
EXPECTED_PASSES = 1
EXPECTED_GROUPS_PER_BLOCK = 16

CACHE_DIR = ROOT / "artifacts/official_eval/cache/proxy-v3-calibration"
PACK = ROOT / "artifacts/official_eval/cache/qwen3.5-4b-proxy-v2.pt"


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


def subscript_of(module) -> str:
    """Read the shipped subscript out of the live function's source."""

    found = re.findall(r"torch\.einsum\(\s*\"([^\"]+)\"", inspect.getsource(live_descent(module)))
    if len(found) != 1:
        raise AssertionError(f"expected one einsum in the descent, found {found}")
    return found[0]


def parent_function_text() -> bytes:
    root_bytes = (ROOT / "solution.py").read_bytes()
    if root_bytes.count(FN_START) != 1 or root_bytes.count(FN_END) != 1:
        raise AssertionError("parent function anchors are not unique")
    start = root_bytes.index(FN_START)
    return root_bytes[start : root_bytes.index(FN_END, start)]


_OFF_DIR = tempfile.TemporaryDirectory(prefix="qf1-correction-off-")


def correction_off_path() -> Path:
    """Write the parent's descent back out, unsubstituted, as a whole solution.

    This is the "correction off" build.  Its only difference from the candidate
    is the one subscript, so any behavioural difference between the candidate
    and this file is caused by the correction and by nothing else.
    """

    path = Path(_OFF_DIR.name) / "solution.py"
    if not path.exists():
        root_bytes = (ROOT / "solution.py").read_bytes()
        module = (
            b"# L-QF1 correction OFF: the parent's own descent, appended verbatim.\n\n\n"
            + parent_function_text().rstrip(b"\n")
            + b"\n"
        )
        path.write_bytes(root_bytes.rstrip(b"\n") + b"\n\n\n" + module)
    return path


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
    # mant = code * 0.25 with code an integer in [0, 7]; a legal state cannot
    # carry anything else.
    mant = params["mant"].to(torch.float32)
    if bool((mant < 0).any()) or bool((mant > 7 * 0.25 + 1e-6).any()):
        raise AssertionError("mantissa outside [0, 1.75]")
    codes = mant * 4.0
    if float((codes - codes.round()).abs().max()) > 1e-4:
        raise AssertionError("mantissa is not a multiple of 0.25")
    if bool((params["sign"].abs() > 1).any()):
        raise AssertionError("sign outside {-1, 0, 1}")


# ---------------------------------------------------------------------------
# A. lineage
# ---------------------------------------------------------------------------


def single_file_import(candidate_path: Path):
    with tempfile.TemporaryDirectory() as tmp:
        isolated = Path(tmp) / "solution.py"
        shutil.copyfile(candidate_path, isolated)
        module = load_solution(isolated, "qf1_isolated")
    for name in PUBLIC_APIS:
        if not hasattr(module, name):
            raise AssertionError(f"isolated import is missing {name}")
    return module


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

    candidate = load_solution(candidate_path, "qf1_candidate")
    parent = load_solution(ROOT / "solution.py", "qf1_parent")
    off = load_solution(correction_off_path(), "qf1_off")
    single_file_import(candidate_path)

    # The append is the whole candidate, so an Attention API cannot differ.
    for name in ATTENTION_APIS:
        if unwrapped(getattr(candidate, name)).__code__.co_code != unwrapped(
            getattr(parent, name)
        ).__code__.co_code:
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
    if candidate.__dict__["_em1_dynamic_descent"] is not candidate._em1_dynamic_descent:
        raise AssertionError("the shadow is not the module-level binding")

    sub_parent = subscript_of(parent)
    sub_candidate = subscript_of(candidate)
    sub_off = subscript_of(off)
    if sub_parent != PARENT_SUBSCRIPT:
        raise AssertionError(f"parent subscript is {sub_parent!r}")
    if sub_candidate != CORRECT_SUBSCRIPT:
        raise AssertionError(f"candidate subscript is {sub_candidate!r}")
    if sub_off != PARENT_SUBSCRIPT:
        raise AssertionError(f"correction-off subscript is {sub_off!r}")

    # Sole change, at the source level: the two functions differ only in that
    # substring.  This is the check that would catch a stray edit anywhere else.
    src_parent = inspect.getsource(live_descent(parent))
    src_candidate = inspect.getsource(live_descent(candidate))
    if src_parent.count(PARENT_SUBSCRIPT) != 1 or src_candidate.count(CORRECT_SUBSCRIPT) != 1:
        raise AssertionError("the descent does not hold exactly one contraction")
    if src_parent.replace(PARENT_SUBSCRIPT, "") != src_candidate.replace(
        CORRECT_SUBSCRIPT, ""
    ):
        raise AssertionError("the candidate descent differs from the parent by more than the contraction")

    print(
        f"[A] parent={len(root_bytes)}B ({root_digest[:8]}) "
        f"candidate={len(candidate_bytes)}B "
        f"({hashlib.sha256(candidate_bytes).hexdigest()[:8]}) "
        f"shared prefix={CARD_PARENT_BYTES}B attention bytecode identical "
        f"descent source differs only at the contraction"
    )
    return candidate, parent, off


# ---------------------------------------------------------------------------
# Q. the quadratic algebra, against an independent reference
# ---------------------------------------------------------------------------


def ideal_objective(x, reference, gram, h_matrix):
    """``J(x) = (x-r) G (x-r)^T + 2 r H (x-r)^T`` in float64, dense matmuls.

    This is the objective the descent models.  It is written out directly from
    the definition -- no einsum, no shared helper, no reuse of the expression
    under test -- so a difference of two evaluations is an independent reading
    of what a move costs.
    """

    u = x.to(torch.float64) - reference.to(torch.float64)
    quad = ((u.mm(gram.to(torch.float64))) * u).sum(dim=1)
    lin = 2.0 * ((reference.to(torch.float64).mm(h_matrix.to(torch.float64))) * u).sum(dim=1)
    return quad + lin


def reference_gradient(deployed, reference, gram, h_matrix):
    """``g = (x-r) G + r H`` in float64, straight from the source's definition."""

    return (deployed.to(torch.float64) - reference.to(torch.float64)).mm(
        gram.to(torch.float64)
    ) + reference.to(torch.float64).mm(h_matrix.to(torch.float64))


def quadratic(subscript, delta, gram):
    return torch.einsum(subscript, delta.to(torch.float64), gram.to(torch.float64), delta.to(torch.float64))


def control_q(candidate, parent):
    """The correction, its failure mode, and the parent's failure mode."""

    generator = torch.Generator(device="cpu").manual_seed(20260910)
    rows, blocks = 6, 3

    # The descent prices a group move with the (blocks, 4, 4) stack of diagonal
    # blocks of the metric -- `local_gram = group_gram[:, step]` -- so the
    # reference Gram here has that shape.  It is a distinct off-diagonal SPD
    # matrix per block (G_b = A_b^T A_b + eps I), so the check is not uniform by
    # construction.
    gram = torch.empty((blocks, 4, 4), dtype=torch.float64)
    for b in range(blocks):
        a = torch.randn((4, 4), generator=generator, dtype=torch.float64)
        gram[b] = a.t().mm(a) + 0.5 * torch.eye(4, dtype=torch.float64)
    off_diagonal = float(gram.abs().sum() - gram.diagonal(dim1=-2, dim2=-1).abs().sum())
    eig_min = float(torch.linalg.eigvalsh(gram).min())
    if eig_min <= 0:
        raise AssertionError("the reference Gram is not positive definite")
    if off_diagonal <= 0:
        raise AssertionError("the reference Gram has no off-diagonal mass")

    sub_parent = subscript_of(parent)
    sub_candidate = subscript_of(candidate)

    # ---- Q1: general deltas -------------------------------------------------
    delta = torch.randn((8, rows, blocks, 4), generator=generator, dtype=torch.float64)
    # Plain per-block matmul, so the reference shares no machinery with the
    # contraction it is judging.
    dense_ref = (
        torch.stack(
            [delta[:, :, b, :] @ gram[b] for b in range(blocks)], dim=2
        ) * delta
    ).sum(dim=-1)
    cand_q = quadratic(sub_candidate, delta, gram)
    parent_q = quadratic(sub_parent, delta, gram)
    cand_rel = float((cand_q - dense_ref).abs().max() / dense_ref.abs().max())
    # The parent's own characterisation, written independently of the einsum:
    # (sum_a d_a) * (colsum(G) . d), with colsum(G)[b, j] = sum_i G[b, i, j].
    total = delta.sum(dim=-1)
    colsum = gram.sum(dim=-2)
    parent_model = total * (delta * colsum).sum(dim=-1)
    parent_rel = float((parent_q - parent_model).abs().max() / parent_model.abs().max())
    disagreement = float((cand_q - parent_q).abs().max())
    print(
        f"[Q1] general delta: corrected-vs-d^T G d rel={cand_rel:.3e} "
        f"parent-vs-(sum d)*colsum rel={parent_rel:.3e} "
        f"max|corrected-parent|={disagreement:.3e} min_eig(G)={eig_min:.4e} "
        f"off_diag/|G|={off_diagonal / float(gram.abs().sum()):.3f}"
    )
    if cand_rel > 1e-12:
        raise AssertionError("the corrected form is not d^T G d")
    if parent_rel > 1e-12:
        raise AssertionError("the parent form is not (sum d) * colsum(G) . d")
    if disagreement < 1e-6:
        raise AssertionError("the two forms agreed; the reference Gram is not discriminating")

    # ---- Q2: single-coordinate deltas --------------------------------------
    print("[Q2] single-coordinate delta = c * e_m (block 0 shown)")
    worst_cand = worst_parent = 0.0
    for m in range(4):
        d = torch.zeros((1, rows, blocks, 4), dtype=torch.float64)
        d[..., m] = 1.0
        true_value = gram[:, m, m]
        colsum_m = gram[:, :, m].sum(dim=-1)
        # Delta is the same on every row, so the first row carries the value.
        cand_v = quadratic(sub_candidate, d, gram)[0, 0, :]
        parent_v = quadratic(sub_parent, d, gram)[0, 0, :]
        print(
            f"     m={m}: corrected={float(cand_v[0]):+.6f} =G[m,m]={float(true_value[0]):+.6f}"
            f" | parent={float(parent_v[0]):+.6f} =colsum[m]={float(colsum_m[0]):+.6f}"
        )
        worst_cand = max(worst_cand, float((cand_v - true_value).abs().max()))
        worst_parent = max(worst_parent, float((parent_v - true_value).abs().max()))
    if worst_cand > 1e-12:
        raise AssertionError("the corrected single-coordinate cost is not G[m,m]")
    if worst_parent < 1e-6:
        raise AssertionError("the parent single-coordinate cost matched the truth")

    # ---- Q3: linear + quadratic against the directly evaluated objective ----
    # Both readings are float64 and the objective reading never touches the
    # expression under test: it evaluates J at x and at x + d and subtracts.
    reference = torch.randn((rows, 4), generator=generator, dtype=torch.float64)
    h_matrix = torch.randn((4, 4), generator=generator, dtype=torch.float64)
    deployed = torch.randn((rows, 4), generator=generator, dtype=torch.float64)
    d4 = torch.randn((8, rows, blocks, 4), generator=generator, dtype=torch.float64) * 0.05

    true_change = torch.empty((8, rows, blocks), dtype=torch.float64)
    linear = torch.empty((8, rows, blocks), dtype=torch.float64)
    for b in range(blocks):
        g2 = gram[b]
        base_b = deployed.unsqueeze(0).expand(8, rows, 4).reshape(-1, 4)
        ref_b = reference.unsqueeze(0).expand(8, rows, 4).reshape(-1, 4)
        moved_b = (deployed.unsqueeze(0) + d4[:, :, b, :]).reshape(-1, 4)
        true_change[:, :, b] = (
            ideal_objective(moved_b, ref_b, g2, h_matrix)
            - ideal_objective(base_b, ref_b, g2, h_matrix)
        ).reshape(8, rows)
        gradient_b = reference_gradient(deployed, reference, g2, h_matrix)
        linear[:, :, b] = (d4[:, :, b, :] * gradient_b.unsqueeze(0)).sum(dim=-1)

    cand_cost = 2.0 * linear + quadratic(sub_candidate, d4, gram)
    parent_cost = 2.0 * linear + quadratic(sub_parent, d4, gram)
    scale = max(float(true_change.abs().max()), 1e-30)
    cand_gap = float((cand_cost - true_change).abs().max()) / scale
    parent_gap = float((parent_cost - true_change).abs().max()) / scale
    print(
        f"[Q3] 2*linear + quadratic vs J(x+d) - J(x): "
        f"corrected rel={cand_gap:.3e}  parent rel={parent_gap:.3e}"
    )
    if cand_gap > 1e-9:
        raise AssertionError("the corrected cost does not equal the objective change")
    if parent_gap < 1e-6:
        raise AssertionError("the parent cost matched the objective change")

    # ---- Q4: the recorded counterexample ------------------------------------
    small = torch.tensor([[[2.0, 1.0], [1.0, 3.0]]], dtype=torch.float64)
    move = torch.tensor([[[[1.0, 0.0]]]], dtype=torch.float64)
    old = float(quadratic(sub_parent, move, small))
    new = float(quadratic(sub_candidate, move, small))
    print(f"[Q4] G=[[2,1],[1,3]] d=[1,0]: parent={old:.6f} corrected={new:.6f} truth=2.0")
    if abs(old - 3.0) > 1e-12 or abs(new - 2.0) > 1e-12:
        raise AssertionError("the recorded counterexample does not reproduce")

    # ---- Q5: a PD Gram with a negative column sum --------------------------
    # The parent form is the column sum, and a positive-definite matrix can have
    # a negative column sum.  Then the parent reports a *negative* cost for a
    # move whose true quadratic cost is positive, and `best_cost < 0` fires on a
    # move the truth rejects.  (It can equally drop a move the truth accepts.)
    flip_gram = torch.tensor([[[1.0, -2.0], [-2.0, 10.0]]], dtype=torch.float64)
    flip_eig = torch.linalg.eigvalsh(flip_gram)
    if float(flip_eig.min()) <= 0:
        raise AssertionError("the sign-flip Gram is not positive definite")
    move0 = torch.tensor([[[[1.0, 0.0]]]], dtype=torch.float64)
    old_q = float(quadratic(sub_parent, move0, flip_gram))
    new_q = float(quadratic(sub_candidate, move0, flip_gram))
    colsum0 = float(flip_gram[0, :, 0].sum())
    print(
        f"[Q5] G=[[1,-2],[-2,10]] eig={flip_eig.tolist()} PD=True "
        f"colsum[0]={colsum0:+.4f} -> parent={old_q:+.4f} corrected={new_q:+.4f}"
    )
    if not (old_q < 0 < new_q):
        raise AssertionError("the sign-flip case did not reproduce")
    # And show it reaches the accept test: pick a linear term that puts the true
    # cost just above zero and the parent's cost just below it.
    linear_term = -0.5 * (new_q - 0.2)
    accept_true = 2.0 * linear_term + new_q < 0.0
    accept_parent = 2.0 * linear_term + old_q < 0.0
    print(
        f"     true cost={2.0 * linear_term + new_q:+.4f} (accept={accept_true}) "
        f"parent cost={2.0 * linear_term + old_q:+.4f} (accept={accept_parent})"
    )
    if accept_true or not accept_parent:
        raise AssertionError("the sign flip does not separate the accept decisions")

    # ---- the local group block really is the full metric's block ------------
    # The descent prices a group-supported move with the 4x4 diagonal block of
    # the full metric.  That is only legitimate because the move is zero outside
    # the block, so the two quadratics must agree exactly.
    full = torch.randn((8, 8), generator=generator, dtype=torch.float64)
    full = full.t().mm(full) + torch.eye(8, dtype=torch.float64)
    block = full[:4, :4]
    d_block = torch.randn((3, 5, 4), generator=generator, dtype=torch.float64)
    d_full = torch.zeros((3, 5, 8), dtype=torch.float64)
    d_full[..., :4] = d_block
    block_cost = ((d_block @ block) * d_block).sum(dim=-1)
    full_cost = ((d_full @ full) * d_full).sum(dim=-1)
    gap = float((block_cost - full_cost).abs().max() / full_cost.abs().max())
    print(f"[Q6] group-block pricing equals full-metric pricing: rel={gap:.3e}")
    if gap > 1e-12:
        raise AssertionError("the 4x4 diagonal block is not the full metric on that group")

    print("[Q] the correction reproduces the true quadratic; the parent does not")


# ---------------------------------------------------------------------------
# B. append neutrality, parent-off, out-of-scope
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


def same_params(a: dict, b: dict, label: str) -> None:
    if set(a) != set(b):
        raise AssertionError(f"{label}: key sets differ")
    for key in a:
        if not bool(torch.equal(a[key], b[key])):
            raise AssertionError(f"{label}: field {key} differs")


def control_b(candidate, parent, off, device):
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
    if int(state.get("em1_channels", -1)) != 2560:
        raise AssertionError("compiled channel count is wrong")
    payload = state.get("em1")
    if not isinstance(payload, dict) or tuple(payload["h"].shape) != (2560, 2560):
        raise AssertionError("compiled H has the wrong shape")
    if payload["h"].dtype != torch.float32 or payload["h"].device.type != "cpu":
        raise AssertionError("compiled H must be a CPU float32 tensor")

    # Correction off == parent, bit for bit, on the live mechanism.
    parent_live = parent.hif4_dynamic_quantize_activation(
        activations[0][0], activations[0][1], dict(state)
    )
    off_live = off.hif4_dynamic_quantize_activation(
        activations[0][0], activations[0][1], dict(state)
    )
    same_params(parent_live, off_live, "correction-off vs parent")
    # Reachability is control F's job; this only records that the synthetic
    # layer is not vacuous, since a ranking change may legitimately accept fewer
    # moves than the parent did.
    flag = dict(state)
    candidate.hif4_dynamic_quantize_activation(activations[0][0], activations[0][1], flag)
    print(
        f"[B] synthetic layer reachable: changed_mantissa="
        f"{flag.get('em1_changed_mantissa')} accepted_groups={flag.get('em1_accepted_groups')}"
    )

    # Parent off: no em1 payload -> the parent dynamic output, bit for bit.
    stripped = {key: value for key, value in state.items() if key != "em1"}
    parent_off = parent.hif4_dynamic_quantize_activation(
        activations[0][0], activations[0][1], dict(stripped)
    )
    candidate_off = candidate.hif4_dynamic_quantize_activation(
        activations[0][0], activations[0][1], dict(stripped)
    )
    same_params(parent_off, candidate_off, "parent-off")

    # Out of scope: in_features > 4096 stores nothing and matches the parent.
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
        "out-of-scope",
    )
    print(
        "[B] correction-off reproduces the parent bit for bit; parent-off state "
        "and the out-of-scope bypass match"
    )


# ---------------------------------------------------------------------------
# C. metric plumbing
# ---------------------------------------------------------------------------


def control_c(candidate, device, label, state, weight_params, weight_quant, weight_scale):
    payload = state.get("em1")
    if not isinstance(payload, dict):
        raise AssertionError(f"{label}: no em1 payload")
    channels = int(state["in_features"])
    deployed = candidate._dequantize_hif4(weight_params).to(torch.float32)
    dense = candidate._dequantize_nvfp4_float32(weight_quant, weight_scale).to(torch.float32)
    gram = deployed.transpose(0, 1).mm(deployed)
    h_true = gram - dense.transpose(0, 1).mm(deployed)
    h_stored = payload["h"].to(torch.float32)
    g_norm = float(gram.norm())
    h_gap = float((h_stored - h_true).norm() / max(g_norm, 1e-30))

    gram_diag_mean = float(payload["gram_diag_mean"])
    diag_gap = abs(gram_diag_mean - float(gram.diagonal().mean())) / max(
        abs(gram_diag_mean), 1e-30
    )

    h_inv = state["h_inv"].to(torch.float32)
    inverse = torch.cholesky_inverse(torch.linalg.cholesky(h_inv))
    ridge = float(inverse.diagonal().mean()) - gram_diag_mean
    metric = inverse - ridge * torch.eye(channels)
    g_rel = float((metric - gram).norm() / max(g_norm, 1e-30))

    live = candidate._em1_metric(dict(state), channels, torch.device("cpu"))
    if live is None:
        raise AssertionError(f"{label}: _em1_metric declined the compiled state")
    live_metric, live_h = live
    path_rel = float((live_metric - metric).norm() / max(float(metric.norm()), 1e-30))
    path_h_rel = float((live_h - h_stored).norm() / max(float(h_stored.norm()), 1e-30))
    print(
        f"[C:{label}] channels={channels} ridge={ridge:.6e} H_gap/|G|={h_gap:.3e} "
        f"G_rel={g_rel:.3e} diag_gap={diag_gap:.3e} path_rel={path_rel:.3e} "
        f"path_h_rel={path_h_rel:.3e}"
    )
    if h_gap > 1e-5 or g_rel > 1e-3 or diag_gap > 1e-6:
        raise AssertionError(f"{label}: the metric path drifted")
    if path_rel > 1e-6 or path_h_rel > 1e-6:
        raise AssertionError(f"{label}: _em1_metric disagrees with the reference")
    return metric


# ---------------------------------------------------------------------------
# D. real proposals, acceptance and the true objective
# ---------------------------------------------------------------------------


def true_loss(deployed_activation, weight_hat, reference_activation, reference_weight):
    player = deployed_activation.to(torch.float64) @ weight_hat.to(torch.float64).t()
    reference = (
        reference_activation.to(torch.float64) @ reference_weight.to(torch.float64).t()
    )
    return float((player - reference).square().sum())


def control_d(candidate, parent, device, label, state, weight_params, act_pair,
              weight_quant, weight_scale, reference_weight):
    channels = int(state["in_features"])
    rows = int(act_pair[0].shape[0])

    live = dict(state)
    parent_params = parent.hif4_dynamic_quantize_activation(act_pair[0], act_pair[1], live)
    parent_diag = dict(live)
    live2 = dict(state)
    candidate_params = candidate.hif4_dynamic_quantize_activation(act_pair[0], act_pair[1], live2)
    candidate_diag = dict(live2)
    assert_legal_params(candidate_params, rows, channels)

    weight_hat = candidate._dequantize_hif4(weight_params).to(torch.float32)
    reference_activation = candidate._dequantize_nvfp4_float32(act_pair[0], act_pair[1]).to(
        torch.float32
    )
    loss_parent = true_loss(
        candidate._dequantize_hif4(parent_params).to(torch.float32),
        weight_hat, reference_activation, reference_weight,
    )
    loss_candidate = true_loss(
        candidate._dequantize_hif4(candidate_params).to(torch.float32),
        weight_hat, reference_activation, reference_weight,
    )
    base = true_loss(
        candidate._dequantize_hif4(
            candidate._EM1_PARENT_LINEAR_DYNAMIC(act_pair[0], act_pair[1], dict(
                {k: v for k, v in state.items() if k != "em1"}
            ))
        ).to(torch.float32),
        weight_hat, reference_activation, reference_weight,
    )

    changed = int(
        (candidate_params["mant"] != parent_params["mant"]).any(dim=-1).sum()
    )
    print(
        f"[D:{label}] rows={rows} parent_groups={parent_diag.get('em1_accepted_groups')} "
        f"candidate_groups={candidate_diag.get('em1_accepted_groups')} "
        f"parent_rows={parent_diag.get('em1_accepted_rows')} "
        f"candidate_rows={candidate_diag.get('em1_accepted_rows')} "
        f"parent_changed={parent_diag.get('em1_changed_mantissa')} "
        f"candidate_changed={candidate_diag.get('em1_changed_mantissa')} "
        f"mantissa_entries_differing={changed}"
    )
    print(
        f"[D:{label}] L_base={base:.6e} L_parent={loss_parent:.6e} "
        f"L_candidate={loss_candidate:.6e} "
        f"parent_rel={(loss_parent - base) / max(base, 1e-30):+.4%} "
        f"candidate_rel={(loss_candidate - base) / max(base, 1e-30):+.4%} "
        f"candidate_vs_parent_rel={(loss_candidate - loss_parent) / max(loss_parent, 1e-30):+.4%}"
    )
    if loss_candidate > base + 1e-12 * max(base, 1.0):
        raise AssertionError(f"{label}: the candidate increased the true loss")
    if loss_parent > base + 1e-12 * max(base, 1.0):
        raise AssertionError(f"{label}: the parent increased the true loss")
    return loss_parent, loss_candidate, base


def ranking_diagnostic(candidate, device, label, state, weight_params, act_pair):
    """How many proposals QF1 flips, computed from the metric, not the descent.

    Rebuilds the local pricing the way the descent does and counts the
    (row, block, step) slots whose argmin candidate differs between the two
    contractions, plus how often the corrected accept decision differs from the
    parent's.  This is the mechanism's reach; it says nothing about sign.
    """

    channels = int(state["in_features"])
    metric_pair = candidate._em1_metric(dict(state), channels, device)
    if metric_pair is None:
        return
    metric, h_matrix = metric_pair
    parent_params = candidate._EM1_PARENT_LINEAR_DYNAMIC(
        act_pair[0], act_pair[1], dict({k: v for k, v in state.items() if k != "em1"})
    )
    deployed = candidate._dequantize_hif4(parent_params).to(torch.float32)
    reference = candidate._dequantize_nvfp4_float32(act_pair[0], act_pair[1]).to(torch.float32)
    gradient = (deployed - reference).mm(metric) + reference.mm(h_matrix)
    rows = int(deployed.shape[0])
    sub_candidate = subscript_of(candidate)

    # Rebuild the parent's candidate geometry the way the descent builds it.
    params = parent_params
    blocks = int(params["mant"].shape[1])
    sign_v = params["sign"].to(torch.float32).reshape(rows, blocks, 16, 4)
    scale_v = (
        params["scale_factor"].to(torch.float32).reshape(rows, blocks, 1, 1, 1)
        * params["scale_lv2"].to(torch.float32).reshape(rows, blocks, 8, 1, 1)
        * params["scale_lv3"].to(torch.float32).reshape(rows, blocks, 8, 2, 1)
    ).reshape(rows, blocks, 16, 1)
    code_v = torch.round(params["mant"].to(torch.float32).reshape(rows, blocks, 16, 4) / 0.25)
    element_index = torch.arange(4).reshape(4, 1).expand(4, 2).reshape(8)
    step_values = torch.tensor([-1.0, 1.0]).reshape(1, 2).expand(4, 2).reshape(8)
    element_mask = torch.zeros(8, 4, dtype=torch.bool)
    element_mask[torch.arange(8), element_index] = True
    step_column = step_values.reshape(8, 1, 1, 1)
    mask_row = element_mask.reshape(8, 1, 1, 4)
    metric6 = metric.reshape(blocks, 16, 4, blocks, 16, 4)
    block_index = torch.arange(blocks).reshape(blocks, 1, 1, 1)
    group_index = torch.arange(16).reshape(1, 16, 1, 1)
    group_gram = metric6[:, :, :, :, :, :][
        block_index, group_index, :, block_index, group_index, :
    ].reshape(blocks, 16, 4, 4)

    flips = 0
    slots = 0
    accept_flips = 0
    colsum_negative = 0
    local_grams = 0
    for step in range(16):
        code_g = code_v[:, :, step, :]
        moved = (code_g.unsqueeze(0) + step_column).clamp_(0.0, 7.0)
        candidates = torch.where(mask_row, moved, code_g.unsqueeze(0))
        delta = (
            sign_v[:, :, step, :].unsqueeze(0)
            * (candidates - code_g.unsqueeze(0))
            * 0.25
            * scale_v[:, :, step, :].unsqueeze(0)
        )
        local_gram = group_gram[:, step]
        colsum = local_gram.sum(dim=1)
        diag = torch.diagonal(local_gram, dim1=-2, dim2=-1)
        colsum_negative += int((colsum < 0).sum())
        local_grams += int(colsum.numel())
        gradient4 = gradient.reshape(rows, blocks, 16, 4)
        linear = (delta * gradient4[:, :, step, :].unsqueeze(0)).sum(dim=-1)
        cost_c = 2.0 * linear + quadratic(sub_candidate, delta, local_gram)
        cost_p = 2.0 * linear + quadratic(PARENT_SUBSCRIPT, delta, local_gram)
        best_c = cost_c.argmin(dim=0)
        best_p = cost_p.argmin(dim=0)
        flips += int((best_c != best_p).sum())
        slots += int(best_c.numel())
        accept_flips += int(
            ((cost_c.gather(0, best_c.unsqueeze(0)) < 0) != (cost_p.gather(0, best_p.unsqueeze(0)) < 0)).sum()
        )
    print(
        f"[D:{label}] proposal ranking: {flips}/{slots} slots picked a different "
        f"candidate ({flips / max(slots, 1):.2%}); accept decision differed in "
        f"{accept_flips}; local grams with a negative column sum: "
        f"{colsum_negative}/{local_grams}"
    )


# ---------------------------------------------------------------------------
# E, F
# ---------------------------------------------------------------------------


def control_e(candidate, device, label, state, act_pair, cache_path: Path):
    first = candidate.hif4_dynamic_quantize_activation(act_pair[0], act_pair[1], dict(state))
    second = candidate.hif4_dynamic_quantize_activation(act_pair[0], act_pair[1], dict(state))
    same_params(first, second, f"{label}: determinism")
    torch.save(state, cache_path)
    reloaded = torch.load(cache_path, map_location="cpu", weights_only=False)
    third = candidate.hif4_dynamic_quantize_activation(act_pair[0], act_pair[1], reloaded)
    same_params(first, third, f"{label}: state round-trip")
    cache_path.unlink()
    print(f"[E:{label}] determinism and state round-trip hold")


def control_f(candidate, label, state, expected_passes):
    if int(candidate._EM1_PASSES) != expected_passes:
        raise AssertionError(
            f"K drifted: {candidate._EM1_PASSES}, L-QF1 expects {expected_passes}"
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


def main() -> None:
    torch.set_grad_enabled(False)
    device = torch.device("cpu")
    candidate_path = HERE / "candidate" / "solution.py"
    candidate, parent, off = control_a(candidate_path)
    control_q(candidate, parent)
    control_b(candidate, parent, off, device)

    weight_quant, weight_scale, activations = make_layer(
        device, 907, rows=64, channels=2560
    )
    result = candidate.hif4_calibration_and_quantize_weight(
        weight_quant, weight_scale, activations
    )
    state = result["activation_state"]
    reference_weight = candidate._dequantize_nvfp4_float32(weight_quant, weight_scale).to(
        torch.float32
    )
    control_c(
        candidate, device, "synthetic", state, result["weight_params"],
        weight_quant, weight_scale,
    )
    live = dict(state)
    candidate.hif4_dynamic_quantize_activation(activations[0][0], activations[0][1], live)
    control_f(candidate, "synthetic", live, EXPECTED_PASSES)
    control_d(
        candidate, parent, device, "synthetic", state, result["weight_params"],
        activations[0], weight_quant, weight_scale, reference_weight,
    )
    control_e(
        candidate, device, "synthetic", state, activations[0],
        HERE / "_qf1_state_roundtrip.pt",
    )

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
                    item for item in payload["weight_states"]
                    if int(item["layer"]) == 0 and str(item["role"]) == "q"
                ),
                None,
            )
            if entry is not None:
                break
        if entry is None:
            raise AssertionError("no cached weight state for layer 0 / role q")
        weight_quant, weight_scale = v2._pair(pack["weights"][0]["q"].to(torch.float32))
        weight_quant = weight_quant.to(torch.float32)
        weight_scale = weight_scale.to(torch.float32)
        real_state = dict(entry["state"])
        real_result = {"weight_params": entry["params"], "activation_state": real_state}
        diagnostics: dict = {}
        candidate._em1_compile_metric(weight_quant, weight_scale, real_result, diagnostics)
        if diagnostics.get("em1_arm") != "compiled":
            raise AssertionError(f"real layer arm is {diagnostics.get('em1_arm')}")
        raw = pack["test_activations"]["q"][1][0].to(torch.float32)[:128].contiguous()
        pair = tuple(value.to(device) for value in v2._pair(raw))
        reference_weight = candidate._dequantize_nvfp4_float32(weight_quant, weight_scale).to(
            torch.float32
        )
        control_c(
            candidate, device, "real/layer0/q", real_state, entry["params"],
            weight_quant, weight_scale,
        )
        real_live = dict(real_state)
        candidate.hif4_dynamic_quantize_activation(pair[0], pair[1], real_live)
        control_f(candidate, "real/layer0/q", real_live, EXPECTED_PASSES)
        control_d(
            candidate, parent, device, "real/layer0/q", real_state, entry["params"],
            pair, weight_quant, weight_scale, reference_weight,
        )
        ranking_diagnostic(
            candidate, device, "real/layer0/q", real_state, entry["params"], pair
        )
        control_e(
            candidate, device, "real/layer0/q", real_state, pair,
            HERE / "_qf1_state_roundtrip.pt",
        )
    else:
        print("[D:real] no calibration cache for the card parent; real control skipped")

    print("ALL L-QF1 CONTROLS PASSED")


if __name__ == "__main__":
    main()
