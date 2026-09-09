"""Build v204 linear-no-rank2-residual: root minus the L-R2 rank-2 residual.

Subtractive ablation for official pricing of the rank-2 residual segment.
The root already carries a module-level switch ``_WEIGHT_RESIDUAL_RANK``
(default 2 = v166 rank-1 + L-R2 rank-2).  With rank=1 the fused-update path
keeps only the legacy rank-1 pair (``[D,1]``); the rank-2 complement solver
``_rank2_residual_complement`` and the ``[L-R2]`` print are skipped entirely,
and every downstream use of ``residual_u``/``residual_v`` is shape-generic
over the rank dimension.

The patch is a single anchored constant flip; the anchor is unique because
the only other occurrences are ``_WEIGHT_RESIDUAL_RANK >= 2`` guards.
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


replace_once(
    "_WEIGHT_RESIDUAL_RANK = 2",
    "_WEIGHT_RESIDUAL_RANK = 1  # v204 ablation: price the L-R2 rank-2 segment",
    "rank-switch",
)

candidate = CANDIDATE_DIR / "solution.py"
candidate.write_bytes(source.encode("utf-8"))

config = {
    "run_id": "linear-no-rank2-residual",
    "version": "v204",
    "mechanism": (
        "Subtractive pricing ablation: root with _WEIGHT_RESIDUAL_RANK 2 -> 1, "
        "removing the L-R2 rank-2 residual complement (power-iteration pair "
        "solver, fused rank-2 update, [L-R2] print) while keeping the v166 "
        "rank-1 residual segment and all other behavior unchanged"
    ),
    "switch_form": "module-level constant _WEIGHT_RESIDUAL_RANK (root line 78)",
    "parent": "solution.py",
    "parent_sha256": hashlib.sha256(PARENT.read_bytes()).hexdigest(),
    "source_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
    "config": {
        "_WEIGHT_RESIDUAL_RANK": 1,
        "_WEIGHT_RESIDUAL_COEFF": 0.25,
        "_WEIGHT_RESIDUAL_POWER_ITERS": 128,
    },
    "candidate_count": 1,
    "official_status": "unregistered/NA",
}
(HERE / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
print(json.dumps(config, indent=2))
