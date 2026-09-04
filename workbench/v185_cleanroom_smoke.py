from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluator.reference_hif4 import validate_hif4_params, validate_state


SOURCE = ROOT / "solutions" / "20260904_v185_cleanroom-robust-operator_scoreNA_timeNA" / "solution.py"


def load_module():
    spec = importlib.util.spec_from_file_location("v185_cleanroom", SOURCE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def nv_pair(rows: int, channels: int, seed: int):
    generator = torch.Generator().manual_seed(seed)
    carrier = torch.randn(rows, channels, generator=generator, dtype=torch.float32).clamp(-6, 6)
    scale = torch.rand(rows, channels // 16, generator=generator, dtype=torch.float32) * 0.1 + 0.01
    return carrier, scale


def main():
    module = load_module()
    weight = nv_pair(128, 128, 1)
    calibration = [nv_pair(32, 128, 2), nv_pair(48, 128, 3)]
    result = module.hif4_calibration_and_quantize_weight(*weight, calibration)
    validate_hif4_params(result["weight_params"], (128, 128))
    validate_state(result["activation_state"])
    activation = nv_pair(24, 128, 4)
    activation_params = module.hif4_dynamic_quantize_activation(
        *activation, result["activation_state"]
    )
    validate_hif4_params(activation_params, (24, 128))

    q_heads, kv_heads, head_dim = 4, 2, 64
    qkv_calibration = []
    for index, tokens in enumerate((32, 48)):
        qkv_calibration.append({
            "q": nv_pair(tokens, q_heads * head_dim, 10 + index),
            "k": nv_pair(tokens, kv_heads * head_dim, 20 + index),
            "v": nv_pair(tokens, kv_heads * head_dim, 30 + index),
        })
    states = module.hif4_calibration_attention(
        qkv_calibration, q_heads, kv_heads, head_dim
    )
    for state in states.values():
        validate_state(state)
    q = nv_pair(40, q_heads * head_dim, 40)
    k = nv_pair(40, kv_heads * head_dim, 41)
    v = nv_pair(40, kv_heads * head_dim, 42)
    q_params = module.hif4_dynamic_quantize_q(*q, q_heads, head_dim, states["q_state"])
    k_params = module.hif4_dynamic_quantize_k(*k, kv_heads, head_dim, states["k_state"])
    v_params = module.hif4_dynamic_quantize_v(*v, kv_heads, head_dim, states["v_state"])
    validate_hif4_params(q_params, (40, q_heads * head_dim))
    validate_hif4_params(k_params, (40, kv_heads * head_dim))
    validate_hif4_params(v_params, (40, kv_heads * head_dim))
    print("v185 clean-room six-API smoke: OK")


if __name__ == "__main__":
    main()
