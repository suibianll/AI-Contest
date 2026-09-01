# v130 official update

Status: **TIMEOUT / OFFICIAL FAILURE (user-confirmed)**.

- Change: first `Q(A)`/`A@W` output-supervised weight refinement.
- Local `official-shape-v1`: Linear `0.471837`, Attention `0.836579`, API
  `295.437s`, wall `317.607s`.
- Official outcome (user confirmed 2026-09-01): **timeout**, time `>300s`, no
  score returned.

The result shows that a local API total just below 300 seconds is not a safe
official-time predictor.  The expensive Attention path is the primary suspect:
this run spends `115.461s` in Attention calibration and `34.459s` in dynamic
Q/K/V, versus the official-pass v86's local `55.347s` and `5.761s`.
