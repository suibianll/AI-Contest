# v139 output-aware gain rejected

v139 changed only calibration of the already deployed Linear activation gain.
It added no dynamic API operation and inherited v138 static Attention unchanged.

- Linear: `0.5072782560` (`-0.0000412489` versus v138)
- Attention: `0.7159419612` (identical to v138)
- Weight calibration: `134.7817732s`
- Dynamic activation: `16.5206781s`
- Attention calibration: `36.9703411s`
- Dynamic Q/K/V: `5.1164202s`
- API total: `193.3892126s`
- Wall: `217.1957354s`

Decision: reject. The next Linear experiment keeps the same four refined
blocks and one sweep, but changes block ranking from raw residual correlation
to curvature-normalized predicted output decrease.
