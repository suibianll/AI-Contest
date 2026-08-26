"""Dev diagnostic: does L1 trigger for linear weights, and does it improve
plain reconstruction MSE (the evaluator's linear metric)?

Run: .venv/Scripts/python.exe artifacts/diag_l1_weights.py
"""

import importlib.util

import torch


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ev = load_module("evaluator/real_data_eval.py", "ev")
sol = load_module("solution.py", "sol")

LAYER_COUNT = 12
HIDDEN = 768
DEVICE = "cuda"

model, weights, calibration, tests, q_heads, head_dim = ev.collect_real_data(
    "models/gpt2", LAYER_COUNT, 128, 2, 2, device=DEVICE
)


def weight_of(layer_index, name):
    return weights[layer_index][name].to(
        device=DEVICE, dtype=torch.float32
    )


def quantize(weight, l1_on):
    sol._L1_DATA_DRIVEN_SCALE = l1_on
    return sol._dense_to_hif4(
        weight,
        importance=None,
        group_gram=None,
        search_offsets=sol._WEIGHT_OFFSETS,
        error_threshold=sol._WEIGHT_REFINE_ERROR_THRESHOLD,
        accept_margin=sol._WEIGHT_REFINE_ACCEPT_MARGIN,
        max_refine_ratio=sol._WEIGHT_REFINE_MAX_RATIO_SMALL,
        max_refine_blocks=sol._WEIGHT_REFINE_MAX_BLOCKS,
    )


for name in ("q", "k", "fc"):
    for layer_index in (0, 5, 11):
        weight = weight_of(layer_index, name)
        params_off = quantize(weight, False)
        params_on = quantize(weight, True)
        hat_off = sol._dequantize_hif4(params_off)
        hat_on = sol._dequantize_hif4(params_on)
        mse_off = float((hat_off - weight).square().mean())
        mse_on = float((hat_on - weight).square().mean())
        scale_changed = int(
            (params_off["scale_factor"] != params_on["scale_factor"]).sum()
        )
        mant_changed = int(
            (params_off["mant"] != params_on["mant"]).sum()
        )
        print(
            f"{name} L{layer_index}: plain MSE off={mse_off:.6e} "
            f"on={mse_on:.6e} ratio={mse_on / max(mse_off, 1e-30):.4f} "
            f"| blocks scale_changed={scale_changed} "
            f"mantissa_changed={mant_changed}"
        )
sol._L1_DATA_DRIVEN_SCALE = True
