# L6e compressed cross-block checkpoint

- Date: 2026-08-31
- Parent: v118 L6d structured block-circulant factor
- Parent normalized LF SHA256: `ec44cf79abcd5170c1667ef7e50fb0a494753c3a96c1b6fcceca9f5030630251`
- Fixed cache: Qwen2.5-0.5B, 24 layers, `seq=128`, `calib=2`, `test=4`, `amax6`,
  CPU, cache read; evaluator source/data revisions are those recorded in the v118
  full-layer report.

## Precision ledger

| candidate | screen Linear | full Linear | Attention | panel | API time | decision |
|---|---:|---:|---:|---:|---:|---|
| v115 L6a rank-16 | 0.53284175 | 0.5090910148 | 0.8420394885 | 295.6806514001 | 716.482861s | accepted parent |
| v116 L6b wide rank-4 | 0.5330906465 | 0.5093045894 | 0.8420394885 | 295.7340450430 | 739.424609s | accepted parent |
| v117 L6c full `G_64` hierarchy | 0.5332946034 | 0.5095117268 | 0.8420394885 | 295.7858293956 | 2019.475204s | accepted parent |
| v118 L6d structured factor | 0.53337532 | 0.5096012555 | 0.8420394885 | 295.8082115559 | 2249.746436s | **highest parent** |

All four directions are positive on the fixed Qwen path. The gains are incremental and
concentrated in the wide projection: v116 improves `proj(d=4864)`, v117 adds all-role
non-regression through hierarchy coordinates, and v118 adds `proj` only (`+0.0006267005`
versus v117). No Attention regression was observed.

## Cross-block statistics

The L5e frame is unchanged by L6a–L6d (these candidates alter activation proposals,
not BOAT coordinates), so its revalidated off-block ratios still apply:

\[
\rho_{off}(G)=\frac{\|G-\operatorname{blockdiag}_{64}(G)\|_F}{\|G\|_F}.
\]

For the 30 width-896 components, the mean is `0.76125` for the transformed weight frame
and `0.88382` for calibration activations. Thus the block-diagonal `gram64` operand omits
substantial coupling; the positive L6 increments are consistent with recovering only a
small compressed portion of that residual, not with exhausting the available headroom.

## Proposal recall and exact objective

On real Qwen layer 23 `proj(d=4864)`, using the four cached validation windows and the
same v118 source, each 128-row activation selected four high-loss blocks, giving
`4*128*4 = 2048` structured row proposals. The exact deployed-Gram gate retained 71
rows, so the observed proposal recall was

\[
\operatorname{recall}_{proposal}=\frac{71}{2048}=0.03466797\;(3.47\%).
\]

For the same rows, the deployed objective

\[
J_{64}(E)=\sum_r e_r^T G_q e_r
\]

fell from `7489.2359619` to `7481.8167725`, an aggregate relative decrease of
`0.0009906470` (`0.0991%`). Per-window relative decreases were `0.0763%`, `0.0638%`,
`0.1378%`, and `0.1181%`. These are exact `G_q` measurements, not proxy MSE.

## State cost

For the wide `d=4864` shape, the incremental compressed state is

\[
4\cdot64\cdot64\cdot4 + 76\cdot4\cdot4 = 66{,}752\text{ bytes},
\]

from four FP32 kernels and 76 distance coefficient rows. The existing v117 state already
contains the dense deployed Gram (`4864^2*4 = 94,633,984` bytes), `gram64`, and the rank-4
factor; v118 total activation-state tensor bytes measured from calibration are
`96,043,200`, of which the structured addition is only `66,752` bytes. L6d therefore
compresses the *new* cross-block operand but does not remove the pre-existing dense `G_q`
baseline.

## Checkpoint decision

L6a–L6d all produced positive full-layer Qwen parents, so the blanket statement “cross-block
compression is not actionable” is false. However, L6d's `3.47%` proposal recall and
`0.0991%` exact-objective reduction show that the current frozen proposal is narrow, while
the 2249.75-second API time is unsuitable for submission. L6 is complete as an accuracy
exploration queue. The next active plan must keep v118 as its precision parent and target:

1. vectorized/sparse structured coordinate proposals with identical exact-gate semantics;
2. elimination or low-rank replacement of the pre-existing dense deployment-Gram state;
3. cross-fold and multi-model validation of the `proj` gain before any broader route; and
4. only after those checks, the C1 `<420s` runtime compression gate.
