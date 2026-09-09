# ---------------------------------------------------------------------------
# v219 / C76.2: fixed output-Fisher importance for Q/K.
#
# The statistic is calibration-only: attention Jacobian/value deviation
# weights are used to build legal one-dimensional Q/K importance vectors.
# Use one fixed blend value and the existing deterministic state comparison;
# no blend, role, or other parameter neighborhood is scanned.  The deployed
# state format and all dynamic APIs remain unchanged.
# ---------------------------------------------------------------------------


_ATTN_FISHER_IMPORTANCE = True
_ATTN_FISHER_BLEND_VALUES = (0.50,)
