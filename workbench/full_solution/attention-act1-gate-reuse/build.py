"""Build A-CT1 from the A-GR1 assembly on the retained v231 root.

A-GR1's gate evaluates each gate window twice, once for the parent arm and once
for the candidate arm.  Both calls run ``_agr1_gate_loss``, which -- per call --
decodes the dense reference Q/K/V out of the window, quantises and decodes
parent-side Q/K/V, and runs two attention forwards.  The two arms differ only in
the Q/K rotation and the K center they carry:

    candidate = _agr1_parent_copy(states)
    candidate["q_state"]["learned_rotation"] = tq
    candidate["k_state"]["learned_rotation"] = tk
    candidate["k_state"]["learned_center"]   = center   # when present

``v_state`` is copied and therefore equal-valued, and the V API reads its state
without writing it (``_check_attention_state`` only reads), so both arms
quantise V to the same five fields.  The dense reference Q/K/V and the reference
attention output ``target`` do not depend on the arm at all.  Three of the four
attention forwards and one of the two V quantisations per window are therefore
recomputed work, not arm-dependent work.

A-CT1 replaces the two calls with one:

    parent_loss, candidate_loss = _act1_gate_pair(
        calib_qkv_list[index], states, candidate,
        q_num_heads, kv_num_heads, head_dim,
    )

and ``_act1_gate_pair`` computes the reference decode, ``target`` and the parent
V decode once, then evaluates both arms' Q/K and their own attention forward
against that same ``target``.  Every window's two losses, the strict
``candidate_loss < parent_loss`` test, the AND across windows, and the final
state compilation are untouched.

The cache lives for the duration of one call and nothing else: no global input
cache, no write into any deployment state, and no caching of anything that moves
with M (the Q/K decode is redone per arm, which is exactly the part the two arms
disagree about).

Build strategy -- derived, not hand-written:

  1. take the A-GR1 ``hif4_calibration_attention`` text out of the v236 bytes
     (v236 is the v231 root plus the A-GR1 block byte for byte, verified with
     cmp at card start, so its text *is* the assembly);
  2. substitute exactly one block: the two gate-loss calls become one pair call;
  3. append the new ``_act1_gate_pair`` helper and the substituted function.

Sole-change evidence, computed rather than asserted:

  * byte level -- one substitution, reported as a byte delta and a line diff;
  * syntax level -- replacing the inserted single ``Assign`` node back with the
    two original call statements must reproduce the A-GR1 function's statement
    tree exactly;
  * lineage -- the candidate starts with the v231 root bytes, and with the whole
    v236 assembly.

The appended definitions shadow the A-GR1 ones, which stay in the file untouched
-- the append-only build never rewrites a parent byte.
"""

from pathlib import Path
import ast
import difflib
import hashlib
import json
import textwrap


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

ASSEMBLY = ROOT / "solutions/20260910_v236_attention-agr1-on-v231_scoreNA_timeNA/solution.py"
ROOT_ARCHIVE = ROOT / "solutions/20260910_v231_linear-em3-k2-arm_scoreNA_timeNA/solution.py"
CANDIDATE = HERE / "candidate" / "solution.py"
MODULE_OUT = HERE / "implementation.generated.py"
BUILD_JSON = HERE / "build.json"

EXPECTED_ASSEMBLY_SHA256 = (
    "3319fc354af1ff0f35075498f26886bdf43a078d32a27852f575a4c50d32ba07"
)
EXPECTED_ASSEMBLY_BYTES = 520955
EXPECTED_ROOT_SHA256 = (
    "ea79a1c12dc667142c620975aab188920fae7b29988c804f41c1a696cc5754f1"
)
EXPECTED_ROOT_BYTES = 505762

# The A-GR1 shadow of hif4_calibration_attention is the last definition in the
# assembly and runs to end of file, so its text is cut with rindex.
FN_START = b"@torch.no_grad()\ndef hif4_calibration_attention(\n"

OLD_GATE = """            parent_loss = _agr1_gate_loss(
                calib_qkv_list[index], states,
                q_num_heads, kv_num_heads, head_dim,
            )
            candidate_loss = _agr1_gate_loss(
                calib_qkv_list[index], candidate,
                q_num_heads, kv_num_heads, head_dim,
            )
"""

NEW_GATE = """            parent_loss, candidate_loss = _act1_gate_pair(
                calib_qkv_list[index], states, candidate,
                q_num_heads, kv_num_heads, head_dim,
            )
"""

HEADER = '''# ---------------------------------------------------------------------------
# A-CT1 -- the arm-independent work inside the A-GR1 gate is done once.
#
# A-GR1's gate calls _agr1_gate_loss twice for each gate window, once with the
# parent states and once with the candidate states.  Each call re-derives the
# dense reference Q/K/V from the window, recomputes the reference attention
# output `target`, and re-quantises V.  None of that depends on the arm: the two
# arms differ only in the Q/K rotation and the K center they carry, and the V
# state is equal-valued in both because the candidate is a copy of the parent
# with only q_state/k_state touched.
#
# A-CT1 collapses the pair into one call:
#
#   A-GR1    for index in _AGR1_GATE_WINDOWS:
#                parent_loss = _agr1_gate_loss(
#                    calib_qkv_list[index], states,
#                    q_num_heads, kv_num_heads, head_dim,
#                )
#                candidate_loss = _agr1_gate_loss(
#                    calib_qkv_list[index], candidate,
#                    q_num_heads, kv_num_heads, head_dim,
#                )
#
#   A-CT1    for index in _AGR1_GATE_WINDOWS:
#                parent_loss, candidate_loss = _act1_gate_pair(
#                    calib_qkv_list[index], states, candidate,
#                    q_num_heads, kv_num_heads, head_dim,
#                )
#
# Per gate window the new call does one reference decode instead of two, one
# reference attention forward instead of two, and one V quantisation instead of
# two.  The Q/K quantisation is still done once per arm, because that is the
# part the arms disagree about, and each arm still gets its own attention
# forward against the shared target.  Over the two shipped gate windows that is
# two fewer reference forwards, two fewer V quantisations and two fewer sets of
# reference decodes.
#
# Nothing else moves.  Every window's two losses, the strict
# `candidate_loss < parent_loss` per window, the AND across windows, the info
# fields and the final state compilation are the A-GR1 code, byte for byte.
# The shared work is held in a local for the duration of the call -- no global
# cache, no write into any deployment state, and nothing that moves with M is
# cached.
#
# The definitions below shadow the A-GR1 ones, which stay in the file as dead
# code -- the append-only build never rewrites a parent byte.
# ---------------------------------------------------------------------------


@torch.no_grad()
def _act1_gate_pair(
    item: dict,
    parent_states: dict,
    candidate_states: dict,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
):
    """Both A-GR1 gate losses for one window, sharing the arm-independent work.

    Returns ``(parent_loss, candidate_loss)``.  Each arm's Q/K are still
    quantised and decoded from its own state, and each arm still gets its own
    attention forward; only the reference decode, the reference `target` and the
    parent-side V are computed once.  The V five fields are arm-independent
    because the two arms hold equal-valued ``v_state`` -- the candidate is a
    copy of the parent with only ``q_state``/``k_state`` modified -- and because
    the V API reads its state without writing it.
    """

    reference = [
        _dequantize_nvfp4_float32(*item[role]).to(torch.float32)[None]
        for role in ("q", "k", "v")
    ]
    target = _a2_attention_forward(
        reference[0], reference[1], reference[2], q_heads, kv_heads, head_dim
    )
    v_hat = _dequantize_hif4(
        _AGR1_PARENT_V(*item["v"], kv_heads, head_dim, parent_states["v_state"])
    ).to(torch.float32)[None]
    losses = []
    for states in (parent_states, candidate_states):
        q_hat = _dequantize_hif4(
            _AGR1_PARENT_Q(*item["q"], q_heads, head_dim, states["q_state"])
        ).to(torch.float32)[None]
        k_hat = _dequantize_hif4(
            _AGR1_PARENT_K(*item["k"], kv_heads, head_dim, states["k_state"])
        ).to(torch.float32)[None]
        actual = _a2_attention_forward(
            q_hat, k_hat, v_hat, q_heads, kv_heads, head_dim
        )
        losses.append(float((actual - target).square().mean().item()))
    return losses[0], losses[1]


'''


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def extract_gate_function(assembly_bytes: bytes) -> bytes:
    """The A-GR1 shadow of hif4_calibration_attention: the last def, to EOF."""

    start = assembly_bytes.rindex(FN_START)
    text = assembly_bytes[start:]
    if b"_AGR1_GATE_WINDOWS" not in text:
        raise SystemExit("the last hif4_calibration_attention is not the A-GR1 one")
    if b"def " in text[len(FN_START) :]:
        raise SystemExit("the extracted A-GR1 function is not the last definition")
    return text


def dissolve_gate_pair(tree: ast.Module) -> ast.Module:
    """Puts the two original gate-loss calls back in place of the pair call.

    Comparison device only: if that reproduces the A-GR1 tree exactly, then the
    pair call is the only structural difference between the two functions.
    """

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
                    reference = ast.parse(textwrap.dedent(OLD_GATE)).body
                    node.body[position : position + 1] = reference
                    restored += 1
                    break
    if restored != 1:
        raise SystemExit(
            f"expected exactly one _act1_gate_pair call in the gate loop, found {restored}"
        )
    ast.fix_missing_locations(tree)
    return tree


def main() -> dict:
    assembly_bytes = ASSEMBLY.read_bytes()
    assembly_digest = sha256_bytes(assembly_bytes)
    if assembly_digest != EXPECTED_ASSEMBLY_SHA256:
        raise SystemExit(
            f"assembly SHA256 mismatch: {assembly_digest} != {EXPECTED_ASSEMBLY_SHA256}"
        )
    if len(assembly_bytes) != EXPECTED_ASSEMBLY_BYTES:
        raise SystemExit(
            f"assembly byte count mismatch: {len(assembly_bytes)} != {EXPECTED_ASSEMBLY_BYTES}"
        )

    root_bytes = ROOT_ARCHIVE.read_bytes()
    root_digest = sha256_bytes(root_bytes)
    if root_digest != EXPECTED_ROOT_SHA256 or len(root_bytes) != EXPECTED_ROOT_BYTES:
        raise SystemExit(f"root archive mismatch: {root_digest} / {len(root_bytes)}")
    if assembly_bytes[:EXPECTED_ROOT_BYTES] != root_bytes:
        raise SystemExit("the assembly does not start with the v231 root bytes")

    function_bytes = extract_gate_function(assembly_bytes)
    function_text = function_bytes.decode("utf-8")

    if function_text.count(OLD_GATE) != 1:
        raise SystemExit(
            f"the A-GR1 gate loop holds {function_text.count(OLD_GATE)} copies of the "
            "two-call block, expected exactly 1"
        )
    if NEW_GATE in function_text:
        raise SystemExit("the assembly already holds the paired gate call")

    fixed_text = function_text.replace(OLD_GATE, NEW_GATE)

    a = function_text.encode("utf-8")
    b = fixed_text.encode("utf-8")
    line_diff = [
        opcode
        for opcode in difflib.SequenceMatcher(
            None, function_text.splitlines(), fixed_text.splitlines()
        ).get_opcodes()
        if opcode[0] != "equal"
    ]
    changed_lines = sum(max(o[2] - o[1], o[4] - o[3]) for o in line_diff)

    parent_tree = ast.parse(function_text)
    candidate_tree = ast.parse(fixed_text)
    if ast.dump(parent_tree) != ast.dump(dissolve_gate_pair(candidate_tree)):
        raise SystemExit(
            "the candidate's AST is not the A-GR1 function's once the pair call is "
            "put back as two calls -- the edit is not the sole structural change"
        )
    tree_identical_after_dissolving_pair = True

    module_text = HEADER + fixed_text.rstrip("\n") + "\n"
    module_bytes = module_text.encode("utf-8")

    if module_text.count("def _act1_gate_pair(") != 1:
        raise SystemExit("the generated module should define _act1_gate_pair once")
    if module_text.count("def hif4_calibration_attention(") != 1:
        raise SystemExit("the generated module should define the shadow once")
    if fixed_text.count("_agr1_gate_loss(") != 0:
        raise SystemExit("the shadow should no longer call _agr1_gate_loss directly")

    candidate_bytes = (
        assembly_bytes.rstrip(b"\n") + b"\n\n\n" + module_bytes.rstrip(b"\n") + b"\n"
    )

    if not candidate_bytes.startswith(root_bytes):
        raise SystemExit("candidate does not start with the v231 root bytes")
    if not candidate_bytes.startswith(assembly_bytes.rstrip(b"\n")):
        raise SystemExit("candidate is not an append of the assembly bytes")
    if candidate_bytes.count(b"def hif4_calibration_attention(") != 4:
        raise SystemExit(
            "expected four definitions of hif4_calibration_attention "
            "(base, v195, A-GR1, A-CT1 shadow)"
        )
    if candidate_bytes.count(b"def _act1_gate_pair(") != 1:
        raise SystemExit("candidate should hold exactly one _act1_gate_pair definition")

    CANDIDATE.parent.mkdir(parents=True, exist_ok=True)
    CANDIDATE.write_bytes(candidate_bytes)
    MODULE_OUT.write_bytes(module_bytes)

    info = {
        "assembly_path": str(ASSEMBLY.relative_to(ROOT)).replace("\\", "/"),
        "assembly_sha256": assembly_digest,
        "assembly_bytes": len(assembly_bytes),
        "root_path": str(ROOT_ARCHIVE.relative_to(ROOT)).replace("\\", "/"),
        "root_sha256": root_digest,
        "root_bytes": len(root_bytes),
        "assembly_starts_with_root": True,
        "implementation_sha256": sha256_bytes(module_bytes),
        "implementation_bytes": len(module_bytes),
        "candidate_sha256": sha256_bytes(candidate_bytes),
        "candidate_bytes": len(candidate_bytes),
        "extracted_agr1_function_bytes": len(a),
        "substitutions_performed": 1,
        "extracted_function_changed_lines": changed_lines,
        "extracted_function_byte_delta": len(b) - len(a),
        "ast_identical_after_dissolving_pair_call": tree_identical_after_dissolving_pair,
        "edit": "replace the two per-window _agr1_gate_loss calls with one _act1_gate_pair call",
    }
    BUILD_JSON.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    return info


if __name__ == "__main__":
    for key, value in main().items():
        print(f"{key}={value}")
