"""Build v205 attn-no-c764-rotation-search: pricing ablation.

Removes the C76.4 variable-size (H16/H32/H64) head-local signed Hadamard
rotation search from the v189 Attention calibration stack by flipping the
existing ``_ATTN_ROTATION_ENABLED`` module flag to ``False``.  Everything
else (dual-track Q/K candidate search, A1 final gate, R3 learned rotation,
logit gain, pair-matrix smooth, A3 V importance candidates) is untouched:
the flag gates exactly one block (solution.py:10464-10540) and the A3 arm
downstream recomputes ``a1_v_hats``/``base_causal`` itself when they are
``None``, so no gate input changes semantics.

solution.py has mixed line endings; the flag region (lines ~429-435) is
CRLF, so patching uses \r\n anchors there.
"""

from pathlib import Path
import hashlib
import json


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PARENT = ROOT / "solution.py"
CANDIDATE_DIR = HERE / "candidate"
CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)

source = PARENT.read_bytes().decode("utf-8")


def replace_once(old: str, new: str, label: str) -> None:
    global source
    if source.count(old) != 1:
        raise RuntimeError(f"anchor for {label} is not unique; rebuild needs review")
    source = source.replace(old, new, 1)


anchor = "\r\n".join([
    "# changing the v074 default or its feature-off equivalence test.",
    "_ATTN_ROTATION_ENABLED = True",
])
replacement = "\r\n".join([
    "# changing the v074 default or its feature-off equivalence test.",
    "# v205 ablation: disable the C76.4 variable H16/H32/H64 rotation search.",
    "_ATTN_ROTATION_ENABLED = False",
])
replace_once(anchor, replacement, "c764-flag-off")

candidate = CANDIDATE_DIR / "solution.py"
candidate.write_bytes(source.encode("utf-8"))

config = {
    "run_id": "attn-no-c764-rotation-search",
    "version": "v205",
    "mechanism": (
        "Ablation: _ATTN_ROTATION_ENABLED True->False, removing the C76.4 "
        "variable-size (H16/H32/H64) head-local signed Hadamard rotation "
        "search (3 block sizes x 4 seeds x deployed-MSE gate) from the v189 "
        "Attention calibration stack; A1 winner passes through unchanged, "
        "all other mechanisms identical to root"
    ),
    "purpose": "pricing ablation for the C76.4 search (~28-30% of Attention calibration per root-time-audit), not a score candidate",
    "parent": "solution.py",
    "parent_sha256": hashlib.sha256(PARENT.read_bytes()).hexdigest(),
    "source_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
    "config": {
        "_ATTN_ROTATION_ENABLED": False,
        "_ATTN_ROTATION_BLOCKS": [16, 32, 64],
        "_ATTN_ROTATION_SEEDS": [0, 1, 2, 3],
        "_ATTN_ROTATION_GQA_ONLY": True,
    },
    "candidate_count": 1,
    "official_status": "unregistered/NA",
}
(HERE / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
print(json.dumps(config, indent=2))
