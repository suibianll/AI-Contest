"""Build v195: aggregate the K-center gradient across all training windows.

Parent bug: inside ``_a2_train_rotation`` the per-step window loop executes
``grad_center = dk3.sum(dim=0)``, overwriting the value every window, so the
Adam update for ``center`` only ever sees the last training window's gradient
(``grad_theta`` is accumulated correctly).  The fix zeroes ``grad_center`` at
the start of each step and accumulates ``dk3.sum(dim=0)`` across windows.
Everything else (steps, lr, Cayley rotation, losses, gate) is untouched.
"""

from pathlib import Path
import hashlib
import json


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PARENT = ROOT / "solution.py"
CANDIDATE_DIR = HERE / "candidate"
CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)

with open(PARENT, "r", encoding="utf-8", newline="") as handle:
    source = handle.read()

anchor_step = "        grad_theta = torch.zeros_like(theta)\r\n"
replacement_step = (
    "        grad_theta = torch.zeros_like(theta)\r\n"
    "        grad_center = torch.zeros_like(center)\r\n"
)
if source.count(anchor_step) != 1:
    raise RuntimeError("The _a2_train_rotation step-reset anchor changed")
source = source.replace(anchor_step, replacement_step, 1)

anchor_window = "            grad_center = dk3.sum(dim=0)\r\n"
replacement_window = "            grad_center = grad_center + dk3.sum(dim=0)\r\n"
if source.count(anchor_window) != 1:
    raise RuntimeError("The _a2_train_rotation grad_center anchor changed")
source = source.replace(anchor_window, replacement_window, 1)

candidate = CANDIDATE_DIR / "solution.py"
with open(candidate, "w", encoding="utf-8", newline="") as handle:
    handle.write(source)

config = {
    "run_id": "attn-a2-center-gradient-aggregate",
    "version": "v195",
    "mechanism": "a2-k-center-gradient-aggregated-across-training-windows",
    "parent": "solution.py",
    "parent_sha256": hashlib.sha256(PARENT.read_bytes()).hexdigest(),
    "source_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
    "change": (
        "_a2_train_rotation: reset grad_center per step and accumulate "
        "dk3.sum(dim=0) over all training windows (was: overwritten each "
        "window, so only the last window shaped the center update)"
    ),
    "frozen": "steps/lr/clip/reg/Cayley/gate/last-window selection unchanged",
    "candidate_count": 1,
    "official_status": "unregistered/NA",
}
(HERE / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
print(json.dumps(config, indent=2))
