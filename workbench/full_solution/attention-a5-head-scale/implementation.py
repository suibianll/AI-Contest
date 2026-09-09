# ---------------------------------------------------------------------------
# v217 / A5: one fixed reciprocal temperature factor for Q/K.
#
# The existing A1 output selector contains this product-preserving parameter
# family but production keeps it disabled.  This candidate enables exactly one
# pre-registered factor, 1.25: Q-side smoothing is multiplied by the factor
# and K-side smoothing by its reciprocal.  No alternate factor is scanned,
# V is unchanged, and the deployed dynamic APIs remain the existing paths.
# ---------------------------------------------------------------------------


_ATTN_OUTPUT_HEAD_SCALE = True
_ATTN_OUTPUT_HEAD_SCALE_FACTORS = (1.25,)

