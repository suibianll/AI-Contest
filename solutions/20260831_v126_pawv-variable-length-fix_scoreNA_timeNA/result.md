# v126 result

- Date: 2026-08-31
- Parent: v125 C1c rank-8 / max-blocks-8 precision-only root
- Change: PAWV variable-length calibration fix; length-keyed diagonals; exact dynamic-length lookup; unseen-length fallback; remove unused full `P^TP/eigh`
- Source SHA256 (canonical LF): `47e2e3ab76c6deaac8de47bbcbd8f689cf5989dc8ff9e9081a887ec89e819b08`
- Local full Linear/Attention/panel/time: `NA` (not rerun)
- Official score/time: `NA`
- Targeted regression: `1 passed in 3.28s`; full public calibration API on
  `[10,128,512,1024,1024]` passed in `11.3977s` locally
- Status: `active-repaired; official-not-tested; runtime inherited invalid`
- Evidence: [`v126 execution log`](../../logs/execution/2026-08-31-v126-pawv-variable-length-fix.md)

The archived `solution.py` is an exact copy of the tested root source. v125 full-layer values are
not assigned to v126 because direct diagonal reduction can change V tie-breaking.
