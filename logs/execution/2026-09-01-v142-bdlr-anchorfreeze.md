# v142 BDLR anchor-freeze execution

v142 kept the rank-4 cross-block `H/D` column sketch but froze selected anchor coordinates during the coordinate update. The complete run returned Linear `0.2825593037789877`, Attention `0.7159419612310174`, API `211.46008620189968 s`, wall `234.84197629999835 s`.

The time objective remained under the local proxy, but precision was still far below v140. Per-layer inspection showed that the BDLR-adjusted calibration activation changed the subsequent output-supervised W selection, especially for the first Q projection. v143 therefore leaves the existing W/calibration path unchanged and applies BDLR only to the final dynamic activation API.
