import importlib.util
from pathlib import Path

import torch


def _load_candidate():
    path = (
        Path(__file__).resolve().parents[1]
        / "workbench"
        / "linear_static_actorder_hdiag_recovered_solution.py"
    )
    spec = importlib.util.spec_from_file_location("static_actorder_recovered", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_hdiag_order_uses_complete_64_channel_blocks():
    module = _load_candidate()
    weight_quant = torch.zeros(2, 128)
    params = {
        "scale_factor": torch.ones(2, 2, 1, 1, 1),
        "scale_lv2": torch.ones(2, 2, 8, 1, 1),
        "scale_lv3": torch.ones(2, 2, 8, 2, 1),
        "sign": torch.ones(2, 2, 8, 2, 4),
        "mant": torch.cat(
            [
                torch.full((2, 1, 8, 2, 4), 0.25),
                torch.full((2, 1, 8, 2, 4), 1.75),
            ],
            dim=1,
        ),
    }
    order = module._static_actorder_hdiag_order(weight_quant, params)
    assert order is not None
    assert order.tolist() == [1, 0]


def test_reordered_gptq_restores_block_layout(monkeypatch):
    module = _load_candidate()
    dense = torch.arange(256, dtype=torch.float32).reshape(2, 128)
    state = {
        "gptq_block_order": torch.tensor([1, 0], dtype=torch.int16),
        "h_inv": torch.eye(128),
        "importance": torch.arange(128, dtype=torch.float32) + 1,
        "gram": None,
        "offsets": torch.tensor([0], dtype=torch.int8),
        "error_threshold": 0.0,
        "accept_margin": 0.0,
        "max_refine_ratio": 0.0,
        "max_refine_blocks": 0,
    }
    captured = {}

    def fake_gptq(x, h_inv, **kwargs):
        captured["dense"] = x
        captured["h_inv"] = h_inv
        return {
            "scale_factor": torch.zeros(2, 2, 1, 1, 1),
            "scale_lv2": torch.ones(2, 2, 8, 1, 1),
            "scale_lv3": torch.ones(2, 2, 8, 2, 1),
            "sign": torch.ones(2, 2, 8, 2, 4),
            "mant": torch.ones(2, 2, 8, 2, 4),
        }

    monkeypatch.setattr(module, "_activation_gptq_quantize", fake_gptq)
    result = module._static_actorder_reordered_gptq(dense, state)

    assert result is not None
    expected_order = torch.arange(64, 128, dtype=torch.long)
    assert torch.equal(captured["dense"][:, :64], dense[:, expected_order])
    assert torch.equal(captured["dense"][:, 64:], dense[:, :64])
    assert result["mant"].shape == (2, 2, 8, 2, 4)
