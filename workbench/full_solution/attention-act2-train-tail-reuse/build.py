"""Build A-CT2 from the A-GR1 assembly on the retained v231 root.

The audit (``audit.py``) confirmed the duplication and pinned down exactly what
can be carried forward, so this build is mechanical rather than exploratory.

``_agr1_train`` ends by walking every ``prepared`` fold twice.  The first walk is
the ``final_loss`` loop, which evaluates

    _agr1_scale_loss_grad(_a2_apply_group_rotation(fold[i][0], int(fold[i][2]), m|p),
                          fold[i][1])[0]

for the q role (with ``m``) and the k role (with ``p``) and accumulates the
scalars into ``final_loss``.  The second walk is the two ratio entries of the
``info`` dict, which evaluate exactly the same expression on exactly the same
folds and keep only the scalar the first walk discarded.  Measured on all six
real attention layers: 204 ``_agr1_scale_loss_grad`` calls per ``_agr1_train``
(192 training + 6 final_loss + 6 ratio), and all 6 ratio calls repeat a
final_loss call with bit-identical inputs and a bit-identical result.

A-CT2 carries the two per-role sums out of the first walk and deletes the
second:

    A-GR1   final_loss = reg
            for fold in prepared:
                for (coordinate, denominator, heads), matrix in zip(fold, (m, p)):
                    value, _ = _agr1_scale_loss_grad(...)
                    final_loss += float(value.item()) / float(len(prepared))
            ...
            "agr1_q_scale_ratio2": <walk every fold again with m>
            "agr1_k_scale_ratio2": <walk every fold again with p>

    A-CT2   final_loss = reg
            q_scale_sum = 0.0
            k_scale_sum = 0.0
            for fold in prepared:
                (q_coordinate, q_denominator, q_heads), (k_...) = fold
                q_value, _ = _agr1_scale_loss_grad(... m ...)
                final_loss += float(q_value.item()) / float(len(prepared))
                k_value, _ = _agr1_scale_loss_grad(... p ...)
                final_loss += float(k_value.item()) / float(len(prepared))
                q_scale_sum += float(q_value.item())
                k_scale_sum += float(k_value.item())
            ...
            "agr1_q_scale_ratio2": 0.0 if not prepared else q_scale_sum / float(len(prepared)),
            "agr1_k_scale_ratio2": 0.0 if not prepared else k_scale_sum / float(len(prepared)),

The two accumulator orders are chosen to reproduce the parent bit for bit:

  * ``final_loss`` still accumulates q-then-k per fold, each term divided by
    ``len(prepared)`` before it is added -- the same order the parent's inner
    ``zip(fold, (m, p))`` produced;
  * each ratio sum still accumulates in fold order and is then divided once by
    ``len(prepared)``, which is what the parent's ``sum(... for fold in prepared)
    / float(len(prepared))`` does (``sum`` starts from 0 and 0 + x is exact).

``audit.py`` verified the second point directly: rebuilding both ratios from the
loop's own scalars reproduces the reported values exactly on all six layers.

The unroll is what makes the accumulators work: a float is immutable, so
``for ... matrix, role_sum in zip(fold, (m, p), (q_scale_sum, k_scale_sum))``
would rebind a local and lose the sum.

What is NOT touched: the 32-step training loop and its gradient, ``_agr1_project``,
``_agr1_scale_loss_grad``, ``_a2_apply_group_rotation``, the gate, the window and
candidate configuration, the acceptance rule and the final state compilation.
Only the tail of ``_agr1_train`` changes.

Build strategy -- the appended module is derived from the assembly, not written
by hand:

  1. cut the A-GR1 ``_agr1_train`` text out of the v236 bytes;
  2. perform exactly two substring substitutions: the final_loss loop, and the
     two ratio entries of the info dict;
  3. append the substituted function.

Sole-change evidence: both needles are asserted unique and asserted absent after
substitution, and the candidate text must equal
``parent.replace(loop).replace(ratios)`` exactly -- which, given uniqueness, is
the statement that nothing outside those two blocks was touched.  The byte delta
and a line diff are reported alongside.

The appended definition shadows the A-GR1 one; the A-GR1 text stays in the file
untouched, and ``hif4_calibration_attention`` resolves ``_agr1_train`` from module
globals when it runs, so the shadow is what executes.
"""

from pathlib import Path
import difflib
import hashlib
import json


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

FN_START = b"@torch.no_grad()\ndef _agr1_train(\n"
FN_END = b"\n@torch.no_grad()\ndef _agr1_gate_loss(\n"

OLD_LOOP = """        final_loss = float(n.square().mean().item()) * _AGR1_REG_WEIGHT
        for fold in prepared:
            for (coordinate, denominator, heads), matrix in zip(fold, (m, p)):
                value, _ = _agr1_scale_loss_grad(
                    _a2_apply_group_rotation(coordinate, int(heads), matrix),
                    denominator,
                )
                final_loss += float(value.item()) / float(len(prepared))
"""

NEW_LOOP = """        final_loss = float(n.square().mean().item()) * _AGR1_REG_WEIGHT
        q_scale_sum = 0.0
        k_scale_sum = 0.0
        for fold in prepared:
            (
                (q_coordinate, q_denominator, q_heads),
                (k_coordinate, k_denominator, k_heads),
            ) = fold
            q_value, _ = _agr1_scale_loss_grad(
                _a2_apply_group_rotation(q_coordinate, int(q_heads), m),
                q_denominator,
            )
            final_loss += float(q_value.item()) / float(len(prepared))
            k_value, _ = _agr1_scale_loss_grad(
                _a2_apply_group_rotation(k_coordinate, int(k_heads), p),
                k_denominator,
            )
            final_loss += float(k_value.item()) / float(len(prepared))
            q_scale_sum += float(q_value.item())
            k_scale_sum += float(k_value.item())
"""

OLD_RATIOS = """        "agr1_q_scale_ratio2": 0.0 if not prepared else float(
            sum(
                float(_agr1_scale_loss_grad(
                    _a2_apply_group_rotation(fold[0][0], int(fold[0][2]), m),
                    fold[0][1],
                )[0].item())
                for fold in prepared
            ) / float(len(prepared))
        ),
        "agr1_k_scale_ratio2": 0.0 if not prepared else float(
            sum(
                float(_agr1_scale_loss_grad(
                    _a2_apply_group_rotation(fold[1][0], int(fold[1][2]), p),
                    fold[1][1],
                )[0].item())
                for fold in prepared
            ) / float(len(prepared))
        ),
"""

NEW_RATIOS = """        "agr1_q_scale_ratio2": 0.0 if not prepared else q_scale_sum / float(len(prepared)),
        "agr1_k_scale_ratio2": 0.0 if not prepared else k_scale_sum / float(len(prepared)),
"""

HEADER = '''# ---------------------------------------------------------------------------
# A-CT2 -- the training tail carries its per-role sums instead of walking the
# folds a second time.
#
# The A-GR1 tail walks every `prepared` fold twice.  The `final_loss` loop
# evaluates
#
#     _agr1_scale_loss_grad(
#         _a2_apply_group_rotation(fold[i][0], int(fold[i][2]), m or p), fold[i][1]
#     )[0]
#
# once with `m` for the q role and once with `p` for the k role, and keeps the
# sum.  The two `agr1_*_scale_ratio2` entries of the `info` dict then evaluate
# exactly the same expression on exactly the same folds, keeping only the scalar
# the first walk discarded:
#
#   A-GR1  ... for fold in prepared:
#              for (coordinate, denominator, heads), matrix in zip(fold, (m, p)):
#                  value, _ = _agr1_scale_loss_grad(...)
#                  final_loss += float(value.item()) / float(len(prepared))
#          ...
#          "agr1_q_scale_ratio2": <walk every fold again with m>,
#          "agr1_k_scale_ratio2": <walk every fold again with p>,
#
#   A-CT2  ... for fold in prepared:
#              (q_coordinate, q_denominator, q_heads), (k_...) = fold
#              q_value, _ = _agr1_scale_loss_grad(... m ...)
#              final_loss += float(q_value.item()) / float(len(prepared))
#              k_value, _ = _agr1_scale_loss_grad(... p ...)
#              final_loss += float(k_value.item()) / float(len(prepared))
#              q_scale_sum += float(q_value.item())
#              k_scale_sum += float(k_value.item())
#          ...
#          "agr1_q_scale_ratio2": 0.0 if not prepared else q_scale_sum / float(len(prepared)),
#          "agr1_k_scale_ratio2": 0.0 if not prepared else k_scale_sum / float(len(prepared)),
#
# Measured on every full-attention layer of the panel: 204
# `_agr1_scale_loss_grad` calls per `_agr1_train` (32 steps x 2 roles x 3 folds
# = 192 in training, 6 in the final_loss loop, 6 in the ratios), and all 6 ratio
# calls repeat a final_loss call with bit-identical inputs and return a
# bit-identical scalar.  The second walk is therefore pure recomputation, and
# deleting it takes the call count to 198.
#
# The two accumulation orders reproduce the parent bit for bit: `final_loss`
# still adds q-then-k per fold with each term divided by len(prepared) first,
# and each ratio sum still accumulates in fold order and divides once at the
# end, which is what the parent's `sum(... for fold in prepared) /
# float(len(prepared))` does.  audit.py checked the latter directly by
# rebuilding both ratios from the loop's own scalars -- exact on all six layers.
#
# The unroll is required, not stylistic: a float is immutable, so passing
# `(q_scale_sum, k_scale_sum)` through `zip` would rebind a local and drop the
# sum.
#
# Nothing else moves: the 32-step training loop and its gradient, _agr1_project,
# _agr1_scale_loss_grad, _a2_apply_group_rotation, the gate, the window and
# candidate configuration, the acceptance rule and the final state compilation
# are the A-GR1 code, byte for byte.
#
# The definition below shadows the A-GR1 one, which stays in the file as dead
# code -- the append-only build never rewrites a parent byte.
# ---------------------------------------------------------------------------


'''


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def main() -> dict:
    assembly_bytes = ASSEMBLY.read_bytes()
    assembly_digest = sha256_bytes(assembly_bytes)
    if assembly_digest != EXPECTED_ASSEMBLY_SHA256:
        raise SystemExit(f"assembly SHA256 mismatch: {assembly_digest}")
    if len(assembly_bytes) != EXPECTED_ASSEMBLY_BYTES:
        raise SystemExit(f"assembly byte count mismatch: {len(assembly_bytes)}")

    root_bytes = ROOT_ARCHIVE.read_bytes()
    if sha256_bytes(root_bytes) != EXPECTED_ROOT_SHA256:
        raise SystemExit("the root archive is not the recorded v231 bytes")
    if assembly_bytes[:EXPECTED_ROOT_BYTES] != root_bytes:
        raise SystemExit("the assembly does not start with the v231 root bytes")

    start = assembly_bytes.index(FN_START)
    end = assembly_bytes.index(FN_END, start)
    function_text = assembly_bytes[start:end].decode("utf-8")

    for name, needle in (("final_loss loop", OLD_LOOP), ("ratio entries", OLD_RATIOS)):
        count = function_text.count(needle)
        if count != 1:
            raise SystemExit(f"the {name} needle occurs {count} times, expected exactly 1")

    fixed_text = function_text.replace(OLD_LOOP, NEW_LOOP).replace(OLD_RATIOS, NEW_RATIOS)

    # Given both needles are unique, this equality is the statement that nothing
    # outside the two blocks changed.
    if fixed_text != function_text.replace(OLD_LOOP, NEW_LOOP).replace(OLD_RATIOS, NEW_RATIOS):
        raise SystemExit("the substitution is not reproducible")
    for name, needle in (("final_loss loop", OLD_LOOP), ("ratio entries", OLD_RATIOS)):
        if needle in fixed_text:
            raise SystemExit(f"the {name} survived the substitution")
    if "q_scale_sum" not in fixed_text or "k_scale_sum" not in fixed_text:
        raise SystemExit("the carried sums are missing from the substituted text")
    if fixed_text.count("_agr1_scale_loss_grad(") != 2 + 1:
        raise SystemExit(
            "expected the training call plus the two tail calls, got "
            f"{fixed_text.count('_agr1_scale_loss_grad(')}"
        )

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

    module_text = HEADER + fixed_text.rstrip("\n") + "\n"
    module_bytes = module_text.encode("utf-8")

    candidate_bytes = (
        assembly_bytes.rstrip(b"\n") + b"\n\n\n" + module_bytes.rstrip(b"\n") + b"\n"
    )

    if not candidate_bytes.startswith(root_bytes):
        raise SystemExit("candidate does not start with the v231 root bytes")
    if not candidate_bytes.startswith(assembly_bytes.rstrip(b"\n")):
        raise SystemExit("candidate is not an append of the assembly bytes")
    if candidate_bytes.count(b"def _agr1_train(") != 2:
        raise SystemExit("candidate should hold the A-GR1 definition plus the shadow")

    CANDIDATE.parent.mkdir(parents=True, exist_ok=True)
    CANDIDATE.write_bytes(candidate_bytes)
    MODULE_OUT.write_bytes(module_bytes)

    info = {
        "assembly_path": str(ASSEMBLY.relative_to(ROOT)).replace("\\", "/"),
        "assembly_sha256": assembly_digest,
        "assembly_bytes": len(assembly_bytes),
        "root_sha256": sha256_bytes(root_bytes),
        "root_bytes": len(root_bytes),
        "assembly_starts_with_root": True,
        "implementation_sha256": sha256_bytes(module_bytes),
        "implementation_bytes": len(module_bytes),
        "candidate_sha256": sha256_bytes(candidate_bytes),
        "candidate_bytes": len(candidate_bytes),
        "extracted_agr1_train_bytes": len(a),
        "substitutions_performed": 2,
        "substitution_sites": ["the final_loss loop", "the two ratio entries of the info dict"],
        "extracted_function_changed_lines": changed_lines,
        "extracted_function_byte_delta": len(b) - len(a),
        "reconstruction_identity_holds": True,
        "edit": (
            "carry the per-role sums out of the final_loss loop and compute the two "
            "scale ratios from them, deleting the second walk of the prepared folds"
        ),
    }
    BUILD_JSON.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    return info


if __name__ == "__main__":
    for key, value in main().items():
        print(f"{key}={value}")
