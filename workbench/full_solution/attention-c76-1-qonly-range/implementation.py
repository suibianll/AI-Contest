# ---------------------------------------------------------------------------
# v218 / C76.1: one fixed Q-only headwise range permutation.
#
# The existing A1 output selector normally keeps Q and K on the same
# hierarchy-aware ordering.  This candidate enables only the first
# independent Q-only range permutation candidate and fixes the candidate cap
# at one.  K ordering, V, and all dynamic APIs remain unchanged.
# ---------------------------------------------------------------------------


_ATTN_OUTPUT_HEADWISE_PERMUTATION = True
_ATTN_OUTPUT_HEADWISE_MAX_CANDIDATES = 1
