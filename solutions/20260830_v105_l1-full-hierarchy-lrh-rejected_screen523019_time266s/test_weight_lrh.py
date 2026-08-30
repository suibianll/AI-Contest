from __future__ import annotations

import torch

import solution


def test_lrh_writes_full_hierarchy_atomically_and_is_monotone() -> None:
    torch.manual_seed(19)
    weight = torch.randn(32, 256) * 0.08
    parent = solution._dense_to_hif4(weight, offsets=solution._BASE_OFFSETS)
    activation = torch.randn(64, 256) * 0.2

    refined = solution._polish_weight_lrh(weight, parent, activation)
    before = activation @ (solution._dequantize_hif4(parent) - weight).T
    after = activation @ (solution._dequantize_hif4(refined) - weight).T

    assert torch.isfinite(solution._dequantize_hif4(refined)).all()
    assert float(after.square().mean()) <= float(before.square().mean()) + 1.0e-7
    for key in ("scale_factor", "scale_lv2", "scale_lv3", "sign", "mant"):
        assert tuple(refined[key].shape) == tuple(parent[key].shape)
        assert torch.isfinite(refined[key].to(torch.float32)).all()

    # A hierarchy update must not leave mantissas paired with a stale
    # denominator: decoding the returned fields is the only value used by the
    # deployed representation.
    decoded = solution._dequantize_hif4(refined)
    assert decoded.shape == weight.shape
