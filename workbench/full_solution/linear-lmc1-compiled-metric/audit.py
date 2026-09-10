"""L-MC1 audit: does moving the metric rebuild into calibration remove work or add it?

The card's premise is that ``_em1_metric`` rebuilds ``G`` from the calibration
state on every dynamic call:

    inverse = torch.cholesky_inverse(torch.linalg.cholesky(h_inv))
    ridge   = mean(diag(inverse)) - gram_diag_mean
    metric  = inverse; metric.diagonal().sub_(ridge)

and that the rebuild depends only on state, never on the activation, so it can be
done once at calibration and stored.

That is true about *dependency*, and the root's own comment says the current
split was a deliberate trade:

    "removes an n^3 Cholesky inverse from every calibration call -- measured at
     39.9 ms (in=4096) / 15.3 ms (in=2560), i.e. ~2.8 s over the official 144
     in-scope calibrations"

So the card does not remove an inverse, it *moves* it -- and whether that is a
win depends on how many dynamic calls there are relative to calibrations:

  * calibration runs once per (layer, role): 24 x 7 = 168 calls, 144 of them in
    scope (``proj`` is 9216 channels, above the 4096 cap);
  * the dynamic API runs once per *case*.

Locally the panel has 336 Linear cases, so the move would save inverses.  The
official shape is the one that matters and the repository records it as **50
Linear samples** against the same 144 calibrations -- which inverts the sign.

This script measures the local side from the pack rather than assuming it, and
reads the official side the only way it can be read: from the numbers the
repository already recorded, cited here so the arithmetic can be checked.

CPU only, reads the pack, writes no candidate.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

PACK = ROOT / "artifacts/official_eval/cache/qwen3.5-4b-proxy-v2.pt"
ROOT_SOLUTION = ROOT / "solution.py"

EM1_MAX_CHANNELS = 4096
HIF4_BLOCK_SIZE = 64

# What the repository records about the official shape.
OFFICIAL_LINEAR_SAMPLES = 50
OFFICIAL_IN_SCOPE_CALIBRATIONS = 144
OFFICIAL_DYNAMIC_NOTE = "README: '官方样例数按用户确认的 50 Linear + 250 Attention 记录'"
OFFICIAL_CALIBRATION_NOTE = (
    "the root's own comment in _em1_compile_metric: '~2.8 s over the official "
    "144 in-scope calibrations'"
)


def load_pack():
    return torch.load(PACK, map_location="cpu", mmap=True, weights_only=False)


def count_from_pack(pack):
    """Counts in-scope calibrations and in-scope dynamic calls the panel produces."""

    layers = int(pack["layers"])
    roles = [str(role) for role in pack["roles"]]
    hidden = int(pack["hidden_size"])

    # Channel count per role, taken from the pack's own weight shapes.
    per_role = {}
    for role in roles:
        weight = pack["weights"][0][role]
        shape = tuple(weight[0].shape) if isinstance(weight, (tuple, list)) else tuple(weight.shape)
        channels = int(shape[-1])
        per_role[role] = channels

    in_scope_roles = [
        role
        for role, channels in per_role.items()
        if channels <= EM1_MAX_CHANNELS and channels % HIF4_BLOCK_SIZE == 0
    ]
    calibrations_total = layers * len(roles)
    calibrations_in_scope = layers * len(in_scope_roles)

    # The panel's Linear cases: 24 layers x 7 roles x 2 windows per the 4B guide.
    windows = 2
    cases_total = layers * len(roles) * windows
    cases_in_scope = layers * len(in_scope_roles) * windows

    return {
        "layers": layers,
        "roles": roles,
        "hidden_size": hidden,
        "channels_per_role": per_role,
        "in_scope_roles": in_scope_roles,
        "out_of_scope_roles": [r for r in roles if r not in in_scope_roles],
        "calibrations_total": calibrations_total,
        "calibrations_in_scope": calibrations_in_scope,
        "local_cases_total": cases_total,
        "local_cases_in_scope": cases_in_scope,
    }


def main() -> int:
    torch.set_grad_enabled(False)
    if not PACK.exists():
        raise SystemExit("the 4B pack is required for the local counts")
    pack = load_pack()
    counts = count_from_pack(pack)

    local_dynamic = counts["local_cases_in_scope"]
    local_calibrations = counts["calibrations_in_scope"]
    local_net = local_dynamic - local_calibrations

    official_dynamic = OFFICIAL_LINEAR_SAMPLES
    official_calibrations = OFFICIAL_IN_SCOPE_CALIBRATIONS
    official_net = official_dynamic - official_calibrations

    print(f"channels per role: {counts['channels_per_role']}")
    print(f"in-scope roles: {counts['in_scope_roles']} (out: {counts['out_of_scope_roles']})")
    print()
    print(
        f"LOCAL panel: calibrations(in-scope)={local_calibrations}, "
        f"dynamic(in-scope)={local_dynamic} -> the move saves {local_net:+d} inverses"
    )
    print(
        f"OFFICIAL shape (recorded): calibrations(in-scope)={official_calibrations}, "
        f"dynamic={official_dynamic} -> the move saves {official_net:+d} inverses"
    )
    print()
    if official_net < 0:
        print(
            "VERDICT: in the official shape the move ADDS work. Moving the inverse from "
            "the dynamic path to calibration removes one inverse per dynamic call but adds "
            "one per calibration, and there are far more calibrations (144) than dynamic "
            "calls (50). The card's own premise -- that the rebuild is repeated work worth "
            "eliminating -- holds on the local panel and inverts in the shape that is "
            "actually scored."
        )
    else:
        print("VERDICT: the move reduces work in the official shape; continue the audit.")

    payload = {
        "card": "L-MC1",
        "premise_verified": {
            "metric_depends_only_on_state": True,
            "evidence": (
                "_em1_metric reads h_inv and the stored gram_diag_mean scalar; the "
                "activation enters only after G is built"
            ),
        },
        "counts_local": counts,
        "counts_official": {
            "dynamic_calls": official_dynamic,
            "calibrations_in_scope": official_calibrations,
            "dynamic_note": OFFICIAL_DYNAMIC_NOTE,
            "calibration_note": OFFICIAL_CALIBRATION_NOTE,
        },
        "net_inverses": {
            "local": local_net,
            "official": official_net,
        },
        "verdict": "NET_WORK_INCREASES_OFFICIALLY" if official_net < 0 else "CONTINUE",
    }
    (HERE / "audit.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {HERE / 'audit.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
