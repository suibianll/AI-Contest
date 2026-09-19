"""v264 = v255 + the full +/-1 pattern move set in the L-EM descent.

The live `_em1_dynamic_descent` proposes, per group step, exactly eight moves:
one element of the 4-element group moves by +/-1.  This card extends the
candidate set to all 2^4 = 16 sign patterns over the four elements -- the
original eight single-element moves plus the eight multi-element ones.  The
exact quadratic `delta^T G delta` and the whole-row joint acceptance are
unchanged, so every accepted move still strictly decreases the ideal-target J
and the mechanism stays monotone on the true deployed output error.

The append-only shadow is the L-QF1 function block taken from the v251 build
(v250 + L-QF1; the same block is live inside v255, which only appends the
attention/A-GR1/constant blocks after it).  Taking it from v251 keeps the copy
free of the A-GR1 block, so re-appending cannot duplicate the attention
wrapper.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PARENT = ROOT / "solution.py"
V250 = ROOT / "solutions/20260911_v250_linear-lem3-k6_scoreNA_timeNA/solution.py"
V251 = HERE / "candidate_v251" / "solution.py"
OUT = HERE / "candidate_v264" / "solution.py"

OLD_BUILD = """    element_index = torch.arange(4, device=device).reshape(4, 1).expand(4, 2).reshape(8)
    step_values = torch.tensor([-1.0, 1.0], device=device).reshape(1, 2).expand(4, 2).reshape(8)
    element_mask = torch.zeros(8, 4, dtype=torch.bool, device=device)
    element_mask[torch.arange(8, device=device), element_index] = True
    step_column = step_values.reshape(8, 1, 1, 1)
    mask_row = element_mask.reshape(8, 1, 1, 4)
"""

NEW_BUILD = """    # L-EM move-set extension: all 2^4 +/-1 patterns over the four elements of
    # a group (the parent's eight single-element moves are the subset with one
    # nonzero entry).  The exact quadratic and the whole-row joint acceptance
    # are unchanged, so the descent stays monotone on the true output error.
    step_pattern = torch.tensor(
        [
            [a, b, c, d]
            for a in (-1.0, 1.0)
            for b in (-1.0, 1.0)
            for c in (-1.0, 1.0)
            for d in (-1.0, 1.0)
        ],
        device=device,
    )
    step_column = step_pattern.reshape(16, 1, 1, 4)
"""

OLD_MOVE = """            moved = (code_g.unsqueeze(0) + step_column).clamp_(0.0, _EM1_CODE_MAX)
            candidates = torch.where(mask_row, moved, code_g.unsqueeze(0))
"""

NEW_MOVE = """            moved = (code_g.unsqueeze(0) + step_column).clamp_(0.0, _EM1_CODE_MAX)
            candidates = moved
"""

OLD_STEP = """            code_step = torch.where(
                keep.unsqueeze(-1),
                element_mask[best_index].to(torch.float32)
                * step_values[best_index].unsqueeze(-1),
                torch.zeros(rows, blocks, 4, device=device, dtype=torch.float32),
            )
"""

NEW_STEP = """            code_step = torch.where(
                keep.unsqueeze(-1),
                step_pattern[best_index],
                torch.zeros(rows, blocks, 4, device=device, dtype=torch.float32),
            )
"""

HEADER = '''


# ---------------------------------------------------------------------------
# L-EM move-set extension on the v255 line.
#
# The live descent proposes eight moves per group step -- one element of the
# element group moves by +/-1.  This shadow proposes all 2^4 = 16 sign
# patterns over the group, which keeps the eight parent moves and adds the
# eight multi-element ones.  The candidate cost is still the exact
# `2 <delta, g> + delta^T G delta` and the acceptance is still the exact
# whole-row joint check, so monotonicity on the true deployed output error is
# preserved and no new state or dynamic input is required.
# ---------------------------------------------------------------------------


'''


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    parent_bytes = PARENT.read_bytes()
    v250 = V250.read_bytes()
    v251 = V251.read_bytes()
    assert v251.startswith(v250), "v251 is not v250 + append"
    block = v251[len(v250):]
    assert block.count(b"def _em1_dynamic_descent(") == 1, "block must hold exactly one shadow"
    assert b"# A-GR1" not in block, "block must not carry the A-GR1 append"
    text = block.decode("utf-8")
    assert text.count(OLD_BUILD) == 1
    assert text.count(OLD_MOVE) == 1
    assert text.count(OLD_STEP) == 1

    fixed = (
        text.replace(OLD_BUILD, NEW_BUILD)
        .replace(OLD_MOVE, NEW_MOVE)
        .replace(OLD_STEP, NEW_STEP)
    )
    assert fixed.count("step_pattern") == 3
    assert "element_mask" not in fixed
    assert "mask_row" not in fixed
    assert "step_values" not in fixed
    assert fixed.count("def _em1_dynamic_descent(") == 1

    candidate = parent_bytes + HEADER.encode("utf-8") + fixed.encode("utf-8")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes(candidate)
    assert candidate.startswith(parent_bytes)
    assert candidate.count(b"def _em1_dynamic_descent(") == (
        parent_bytes.count(b"def _em1_dynamic_descent(") + 1
    )
    assert candidate.count(b"def _agr1_train(") == parent_bytes.count(b"def _agr1_train(")
    print(f"parent    bytes={len(parent_bytes)} sha={sha256(parent_bytes)[:16]}")
    print(f"block     bytes={len(block)}")
    print(f"candidate bytes={len(candidate)} sha={sha256(candidate)[:16]}")
    print("APPEND-ONLY OK, 16-pattern move set OK, A-GR1 not duplicated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
