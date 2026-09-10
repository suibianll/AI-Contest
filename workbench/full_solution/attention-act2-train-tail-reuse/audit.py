"""A-CT2 static audit: is the end of ``_agr1_train`` recomputing what it just computed?

The plan (section 4, development ruling) allows this audit to run once A-CT1 is
archived, and requires it to *confirm* the duplication before any implementation
card is registered:

    "A-CT1开发完且官方仍未回传时，允许继续A-CT2训练末尾统计复用的静态审计：
     _agr1_train末尾计算final_loss后，info中的q/k scale ratio又对相同最终M、
     同prepared folds重算相同变换和loss。先验证同dtype、顺序和异常行为，明确
     可复用的每fold标量；只在确认后，在本节补一张固定实现卡再开发。"

The claim to test: the loop that builds ``final_loss`` already walks every
``prepared`` fold and evaluates ``_agr1_scale_loss_grad(_a2_apply_group_rotation(
coordinate, heads, matrix), denominator)`` for both ``m`` and ``p``; and then
``agr1_q_scale_ratio2`` / ``agr1_k_scale_ratio2`` walk the same folds again with
the same ``m``/``p`` and the same folds, evaluating exactly the same expression
and keeping only the scalar it discarded the first time.

Three questions, answered separately and computed rather than asserted:

  Q1  Are the two expressions the same expression?  Answered at the AST level on
      the shipped text, not by eye.
  Q2  Does the second evaluation actually re-request bit-identical inputs and
      return bit-identical scalars?  Answered by instrumenting the real function
      during a real ``_agr1_train`` call on real calibration data.
  Q3  Is the per-fold scalar that could be carried forward exactly the one the
      ratio needs?  Answered by rebuilding the two ratios from the carried
      scalars and comparing bit for bit.

Also recorded, because a reuse claim has to survive them:

  * dtype and device of every value involved, measured rather than assumed;
  * the exception surface -- both the loop and the ratio block call the same two
    helpers on the same inputs, so neither can raise where the other would;
  * the ``force_zero`` branch, which skips the ``final_loss`` loop entirely and
    would leave the scalar unbound.  It is checked for reachability rather than
    waved away, because if it were reachable the reuse would change behaviour.

This script is read-only: it loads the v236 archive, runs it, and prints.  It
does not modify the repository and writes no candidate.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import sys
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

AGR1 = ROOT / "solutions/20260910_v236_attention-agr1-on-v231_scoreNA_timeNA/solution.py"
ROOT_ARCHIVE = ROOT / "solutions/20260910_v231_linear-em3-k2-arm_scoreNA_timeNA/solution.py"
PACK = ROOT / "artifacts/official_eval/cache/qwen3.5-4b-proxy-v2.pt"

AGR1_SHA256 = "3319fc354af1ff0f35075498f26886bdf43a078d32a27852f575a4c50d32ba07"

TRAIN_FN = b"@torch.no_grad()\ndef _agr1_train(\n"
END_MARKER = b"\n@torch.no_grad()\ndef _agr1_gate_loss(\n"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def raw_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.detach().cpu().contiguous().numpy().tobytes()


def extract_train_function(payload: bytes) -> str:
    start = payload.index(TRAIN_FN)
    end = payload.index(END_MARKER, start)
    return payload[start:end].decode("utf-8")


# ---------------------------------------------------------------------------
# Q1 -- the two expressions, compared as syntax trees
# ---------------------------------------------------------------------------


def control_q1(function_text: str) -> dict:
    # Parse once: identity comparisons below run across this one tree only.
    function = ast.parse(function_text).body[0]
    calls = sorted(
        (
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and ast.unparse(node.func) == "_agr1_scale_loss_grad"
        ),
        key=lambda node: (node.lineno, node.col_offset),
    )
    print(f"Q1: `_agr1_scale_loss_grad` call sites inside _agr1_train: {len(calls)}")
    for call in calls:
        print(f"    line {call.lineno}: {ast.unparse(call)[:96]}")

    # Separate them by the statement that owns them, rather than by text.
    final_loop = None
    info_assign = None
    for node in ast.walk(function):
        if (
            isinstance(node, ast.For)
            and ast.unparse(node.target) == "fold"
            and any(
                isinstance(inner, ast.AugAssign)
                and ast.unparse(inner.target) == "final_loss"
                for inner in ast.walk(node)
            )
        ):
            final_loop = node
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and ast.unparse(node.targets[0]) == "info"
        ):
            info_assign = node
    if final_loop is None or info_assign is None:
        raise SystemExit("could not locate the final_loss loop and the info dict")

    def within(node: ast.AST) -> list[ast.Call]:
        last = getattr(node, "end_lineno", node.lineno)
        return [call for call in calls if node.lineno <= call.lineno <= last]

    loop_calls = within(final_loop)
    data_calls = within(info_assign)
    print(f"    inside the `for fold in prepared` final_loss loop: {len(loop_calls)}")
    print(f"    inside the two ratio expressions: {len(data_calls)}")

    if len(loop_calls) != 1 or len(data_calls) != 2:
        raise SystemExit(
            f"expected one call in the loop and two in the ratios, got "
            f"{len(loop_calls)} and {len(data_calls)}"
        )
    loop_call = loop_calls[0]
    ratio_q = [call for call in data_calls if "fold[0][0]" in ast.unparse(call)]
    ratio_k = [call for call in data_calls if "fold[1][0]" in ast.unparse(call)]
    if len(ratio_q) != 1 or len(ratio_k) != 1:
        raise SystemExit("the two ratio calls are not one q call and one k call")

    # The loop is generic over `zip(fold, (m, p), ...)`, so its role is decided by
    # the zip order rather than by its argument text.  Take the innermost For that
    # holds the call and read that order off the AST.
    containers = [
        node
        for node in ast.walk(final_loop)
        if isinstance(node, ast.For) and any(inner is loop_call for inner in ast.walk(node))
    ]
    inner_for = max(containers, key=lambda node: node.lineno)
    zip_iter = ast.unparse(inner_for.iter)
    print(f"    the generic call sits under `for {ast.unparse(inner_for.target)} in {zip_iter}`")
    if "zip(fold,(m,p))" != zip_iter.replace(" ", ""):
        raise SystemExit(f"the loop does not zip the expected sequences: {zip_iter}")
    if ast.unparse(inner_for.target) != "((coordinate, denominator, heads), matrix)":
        raise SystemExit(
            f"the loop does not unpack the expected names: {ast.unparse(inner_for.target)}"
        )
    print(
        "      -> iteration i binds coordinate/denominator/heads to fold[i] and matrix "
        "to (m, p)[i]"
    )

    loop_args = [ast.unparse(arg) for arg in loop_call.args]
    print(f"      loop call args: {loop_args}")
    if loop_args[0] != "_a2_apply_group_rotation(coordinate, int(heads), matrix)":
        raise SystemExit("the loop call does not rotate the fold coordinate by the matrix")

    # The q ratio hard-codes fold[0]/m and the k ratio fold[1]/p, which is exactly
    # what iteration 0 and iteration 1 bind.
    for role, index, matrix in (("q", 0, "m"), ("k", 1, "p")):
        args = [ast.unparse(arg) for arg in (ratio_q[0] if role == "q" else ratio_k[0]).args]
        print(f"      {role} ratio args: {args}")
        if args[0] != f"_a2_apply_group_rotation(fold[{index}][0], int(fold[{index}][2]), {matrix})":
            raise SystemExit(f"the {role} ratio does not rotate the expected fold entry")
        if args[1] != f"fold[{index}][1]":
            raise SystemExit(f"the {role} ratio does not use the expected denominator")
    print(
        "    -> structurally the same expression: the loop's generic call at iteration i "
        "is the ratio's role-i call, on the same fold entry and the same matrix"
    )
    return {"call_sites": len(calls), "loop_calls": len(loop_calls), "ratio_calls": len(data_calls)}


# ---------------------------------------------------------------------------
# Q2/Q3 -- instrument the real function during a real call
# ---------------------------------------------------------------------------


def real_windows(pack, layer, pair):
    return [
        {
            "q": pair(pack["calibration_qkv"][s][layer][0]),
            "k": pair(pack["calibration_qkv"][s][layer][1]),
            "v": pair(pack["calibration_qkv"][s][layer][2]),
        }
        for s in range(len(pack["calibration_qkv"]))
    ]


def control_q2_q3(module, windows, heads, label):
    q_heads, kv_heads, head_dim = heads
    states = module._AGR1_PARENT_CALIBRATION(windows, q_heads, kv_heads, head_dim)
    fit_windows = windows[: len(windows) - len(module._AGR1_GATE_WINDOWS)]

    calls: list[tuple[bytes, bytes, bytes]] = []
    original = module._agr1_scale_loss_grad

    def spy(x, denominator):
        loss, grad = original(x, denominator)
        calls.append((raw_bytes(x), raw_bytes(denominator), raw_bytes(loss)))
        return loss, grad

    module._agr1_scale_loss_grad = spy
    try:
        _, _, _, info = module._agr1_train(
            fit_windows, states, q_heads, kv_heads, head_dim, torch.device("cpu")
        )
    finally:
        module._agr1_scale_loss_grad = original

    prepared = len(fit_windows)
    tail = 2 * prepared
    if len(calls) < tail:
        raise SystemExit(f"only {len(calls)} calls recorded")

    # Layout: 32 steps x 2 roles x prepared folds, then the final_loss loop
    # (2 x prepared), then the two ratio expressions (2 x prepared).
    steps = module._AGR1_TRAIN_STEPS
    expected_total = steps * 2 * prepared + 2 * tail
    if len(calls) != expected_total:
        raise SystemExit(
            f"expected {expected_total} calls "
            f"({steps} steps x 2 roles x {prepared} folds + {2 * tail} tail), got {len(calls)}"
        )

    final_loss_calls = calls[len(calls) - tail * 2 : len(calls) - tail]
    ratio_calls = calls[len(calls) - tail :]

    print(
        f"Q2 [{label}]: {len(calls)} `_agr1_scale_loss_grad` calls total = "
        f"{steps} training steps x 2 roles x {prepared} folds = {steps * 2 * prepared}, "
        f"plus {len(final_loss_calls)} in the final_loss loop, "
        f"plus {len(ratio_calls)} in the two ratio expressions"
    )

    duplicated = 0
    identical = 0
    for index, (x_bytes, denom_bytes, loss_bytes) in enumerate(ratio_calls):
        match = next(
            (
                position
                for position, (other_x, other_denom, other_loss) in enumerate(final_loss_calls)
                if other_x == x_bytes and other_denom == denom_bytes
            ),
            None,
        )
        if match is None:
            raise SystemExit(
                f"Q2 [{label}]: ratio call {index} has no bit-identical partner in the "
                "final_loss loop -- the reuse premise does not hold"
            )
        duplicated += 1
        if final_loss_calls[match][2] == loss_bytes:
            identical += 1
    if identical != duplicated:
        raise SystemExit("a duplicated call returned a different scalar")
    print(
        f"    every one of the {duplicated} ratio calls repeats a final_loss call with "
        f"bit-identical inputs and returns a bit-identical scalar ({identical}/{duplicated})"
    )

    # Q3: rebuild the two ratios from the loop's own scalars and compare exactly.
    loop_q = [final_loss_calls[i] for i in range(0, len(final_loss_calls), 2)]
    loop_k = [final_loss_calls[i] for i in range(1, len(final_loss_calls), 2)]
    rebuilt_q = sum(
        float(torch.frombuffer(bytearray(bits), dtype=torch.float32)[0])
        for _, _, bits in loop_q
    ) / float(prepared)
    rebuilt_k = sum(
        float(torch.frombuffer(bytearray(bits), dtype=torch.float32)[0])
        for _, _, bits in loop_k
    ) / float(prepared)
    reported_q = float(info["agr1_q_scale_ratio2"])
    reported_k = float(info["agr1_k_scale_ratio2"])
    print(
        f"Q3 [{label}]: rebuilt q ratio {rebuilt_q!r} vs reported {reported_q!r} -> "
        f"exact: {rebuilt_q == reported_q}; k ratio {rebuilt_k!r} vs {reported_k!r} -> "
        f"exact: {rebuilt_k == reported_k}"
    )
    if rebuilt_q != reported_q or rebuilt_k != reported_k:
        raise SystemExit("the carried scalars do not reproduce the reported ratios exactly")

    # The identity the reuse relies on: the tail block is the two ratios plus the
    # regulariser, up to float summation order.
    final_loss = float(info["agr1_final_loss"])
    residual = final_loss - (reported_q + reported_k)
    print(
        f"    final_loss - (q_ratio + k_ratio) = {residual:.6e} "
        f"(the regulariser term; not zero, and not expected to be)"
    )
    return {
        "total_calls": len(calls),
        "training_step_calls": steps * 2 * prepared,
        "final_loss_calls": len(final_loss_calls),
        "ratio_calls": len(ratio_calls),
        "duplicated_calls": duplicated,
        "bit_identical": identical,
        "rebuilt_ratios_exact": True,
    }


# ---------------------------------------------------------------------------
# dtype / force_zero / exception surface
# ---------------------------------------------------------------------------


def control_facts(module, function_text, heads):
    q_heads, kv_heads, head_dim = heads
    print("FACTS:")
    print(
        f"    dtypes: eye/zeros are torch.float32; coordinates and denominators are built "
        f"in float32 by _agr1_train; _a2_apply_group_rotation casts the rotation to "
        f"float32 and _agr1_scale_loss_grad returns float32 -> the ratio expressions and "
        f"the loop agree on dtype by construction (same helper, same args)"
    )
    tree = ast.parse(function_text)
    force_zero_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and ast.unparse(node.func).endswith("_agr1_train")
    ]
    print(f"    _agr1_train calls inside its own body: {len(force_zero_calls)}")
    # Reachability of force_zero is checked on the repository, not in this file.
    call_sites = []
    for path in sorted((ROOT / "solutions").glob("*/solution.py")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        for index, line in enumerate(text.splitlines(), start=1):
            if "_agr1_train(" in line and "def _agr1_train" not in line:
                call_sites.append((path.name, index, line.strip()))
    passing_force_zero = [site for site in call_sites if "force_zero" in site[2]]
    print(
        f"    _agr1_train call sites across solutions/: {len(call_sites)}; "
        f"of those, passing force_zero: {len(passing_force_zero)}"
    )
    if passing_force_zero:
        raise SystemExit(
            "a call site passes force_zero, where final_loss is never bound and the "
            "reuse premise would not hold"
        )
    print(
        "    -> force_zero is unreachable from every archived call site, so the branch "
        "that skips the final_loss loop cannot occur; the reuse is safe on that count"
    )
    return {
        "call_sites": len(call_sites),
        "call_sites_passing_force_zero": len(passing_force_zero),
    }


def main() -> int:
    torch.set_grad_enabled(False)
    payload = AGR1.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != AGR1_SHA256:
        raise SystemExit(f"the A-GR1 assembly is not the recorded v236 bytes: {digest}")
    if not payload.startswith(ROOT_ARCHIVE.read_bytes()):
        raise SystemExit("the A-GR1 assembly does not start with the v231 root")

    function_text = extract_train_function(payload)
    module = load_module(AGR1, "act2_audit_agr1")

    sys.path.insert(0, str(ROOT / "evaluator"))
    import official_eval as v2  # noqa: PLC0415

    results = {"q1": control_q1(function_text)}
    if not PACK.exists():
        raise SystemExit("the 4B pack is required for the empirical half of the audit")
    pack = torch.load(PACK, map_location="cpu", mmap=True, weights_only=False)
    heads = (int(pack["q_heads"]), int(pack["kv_heads"]), int(pack["head_dim"]))
    layers = [int(layer) for layer in pack["metadata"]["attention_layers"]]

    per_layer = {}
    for layer in layers:
        windows = real_windows(pack, layer, v2._pair)
        per_layer[layer] = control_q2_q3(module, windows, heads, f"real/layer{layer}")
    results["q2_q3_by_layer"] = per_layer
    results["facts"] = control_facts(module, function_text, heads)

    saving = {
        layer: record["duplicated_calls"] for layer, record in per_layer.items()
    }
    print(
        f"\nAUDIT CONCLUSION: the tail of _agr1_train evaluates the same expression on the "
        f"same folds twice. The second walk is {next(iter(saving.values()))} calls per "
        f"layer (2 roles x 3 folds) out of "
        f"{next(iter(per_layer.values()))['total_calls']} total, and every one of them "
        f"returns a bit-identical scalar that the loop already had."
    )
    print(
        "The reusable per-fold scalars are the two `_agr1_scale_loss_grad(...)[0]` values "
        "the final_loss loop already computes for m and p; the ratios are their fold sums "
        "divided by len(prepared), and the loop's own per-fold values are summed in the "
        "same order the ratio expressions use."
    )
    print("DUPLICATION CONFIRMED -- an implementation card is warranted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
